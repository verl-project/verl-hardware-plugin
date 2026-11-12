# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""verl's sync replay buffer for the colocated run: a 0.1 s poll and a check of every sampled batch before anything
trains on it.

Select it with ``trainer.v1.sampler.custom_sampler.path=pkg://verl_hardware_plugin.accelerators.tpu.rollout.inproc.sampler``
and ``trainer.v1.sampler.custom_sampler.name=FastPollReplayBufferSync``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import log, warn

POLL_INTERVAL = 0.1


def expected_rollout_n(rb: Any, val: bool) -> Optional[int]:
    """rollout.n (val: rollout.val_kwargs.n) as the driver's sampler can see it: the trainer config behind verl's
    ``refill_fn`` (the trainer's own bound method), else ``sampler_kwargs.rollout_n`` (training only); None when
    unknown."""
    owner: Any = getattr(getattr(rb, "refill_fn", None), "__self__", None)
    try:
        r = owner.config.actor_rollout_ref.rollout
        return int(r.val_kwargs.n if val else r.n)
    except Exception:  # noqa: BLE001
        pass
    try:
        v = (getattr(rb, "sampler_kwargs", None) or {}).get("rollout_n")
        if v is not None and not val:
            return int(v)
    except Exception:  # noqa: BLE001
        pass
    return None


def batch_guard(rb: Any, batch: Any, global_steps: int, partition_id: str, batch_size: int) -> tuple:
    """The sampled batch, checked before anything trains on it: every sampled prompt group has trajectories for all
    rollout.n sessions (verl's sync sampler keeps a group whose sessions failed, with the surviving trajectories
    only), none of them failed or aborted (a sync-mode rollout aborts none), and, for training, every trajectory's
    weight version equals the last sync's (staleness 0, span 1). A training batch whose rollout.n cannot be resolved
    is refused: the session check would be skipped silently. Returns (problems, info)."""
    keys = list(getattr(batch, "keys", None) or [])
    tags = list(getattr(batch, "tags", None) or [])
    val = partition_id == "val"
    n = expected_rollout_n(rb, val)
    groups: dict = {}
    for k in keys:
        parts = str(k).split("_")
        groups.setdefault(parts[0], set()).add(parts[1] if len(parts) > 1 else "?")
    problems = []
    if len(groups) != int(batch_size):
        problems.append(
            f"{len(groups)} prompt groups carry trajectories, {batch_size} were sampled (a group lost all of them)"
        )
    short: dict = {}
    if n is not None:
        want = {str(i) for i in range(n)}
        short = {u: len(want - sess) for u, sess in groups.items() if not want <= sess}
        if short:
            problems.append(
                f"{len(short)} group(s) miss {sum(short.values())} of their rollout.n={n} sessions, e.g. "
                f"{sorted(short.items())[:3]}"
            )
    elif not val:
        problems.append(
            "rollout.n cannot be resolved for this training batch (no trainer config, no sampler_kwargs.rollout_n)"
        )
    failed = sorted(set(groups) & set((getattr(rb, "failure_keys", None) or {}).get(partition_id, ()) or ()))
    if failed:
        problems.append(
            f"{len(failed)} sampled group(s) ended in status 'failure' (a session raised), e.g. {failed[:3]}"
        )
    aborted = sum(
        1 for t in tags if isinstance(t, dict) and t.get("response_len") is not None and int(t["response_len"]) == 0
    )
    if aborted:
        problems.append(f"{aborted} aborted trajectories (response_len 0) in a sync-mode batch")
    status = sorted(
        {str(t.get("status")) for t in tags if isinstance(t, dict) and t.get("status") not in (None, "success")}
    )
    if status:
        problems.append(f"trajectory status {status}")
    info: dict[str, Any] = {
        "groups": len(groups),
        "sampled": int(batch_size),
        "trajectories": len(keys),
        "rollout_n": n,
        "short_groups": len(short),
        "failed_groups": len(failed),
        "aborted": aborted,
    }
    if not val:
        stal, spans, unknown = [], [], 0
        for t in tags:
            mn, mx = (t.get("min_global_steps"), t.get("max_global_steps")) if isinstance(t, dict) else (None, None)
            if mn is None or mx is None:
                unknown += 1
                continue
            stal.append((int(global_steps) - 1) - int(mx))
            spans.append(int(mx) - int(mn) + 1)
        if unknown:
            problems.append(f"{unknown} trajectories carry no weight version (min/max_global_steps None)")
        if stal and (max(stal) != 0 or min(stal) != 0):
            problems.append(f"trajectory_staleness min {min(stal)} max {max(stal)} (sync mode: 0)")
        if spans and max(spans) != 1:
            problems.append(f"trajectory_spans max {max(spans)} (sync mode: 1)")
        info.update(
            {
                "staleness": [min(stal), max(stal)] if stal else None,
                "spans": [min(spans), max(spans)] if spans else None,
                "no_version": unknown,
            }
        )
    return problems, info


_FAST_POLL_CLS = None


def _fast_poll_class():
    """verl's sync ReplayBuffer with a 0.1 s poll (its default is 2 s) and :func:`batch_guard` on every sampled batch.
    Exact: the same samples in the same order; only the trainer notices completion sooner."""
    global _FAST_POLL_CLS
    if _FAST_POLL_CLS is None:
        from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

        class FastPollReplayBufferSync(ReplayBuffer):
            def __init__(self, *args, poll_interval: float = POLL_INTERVAL, **kwargs):
                if kwargs.get("trainer_mode", args[0] if args else None) != "sync":
                    raise RuntimeError("FastPollReplayBufferSync serves trainer_mode=sync only")
                super().__init__(*args, poll_interval=poll_interval, **kwargs)

            def sample(self, global_steps: int, partition_id: str, batch_size: int):
                """verl's sync sample, then :func:`batch_guard` before anything trains on the batch: a short, failed,
                aborted or stale batch fails the step ('STEP FAILED before update_actor')."""
                batch, metrics = super().sample(
                    global_steps=global_steps, partition_id=partition_id, batch_size=batch_size
                )
                problems, info = batch_guard(self, batch, global_steps, partition_id, batch_size)
                log(f"batch step {global_steps} {partition_id}: {json.dumps(info, sort_keys=True, default=str)}")
                if problems:
                    line = (
                        f"STEP FAILED before update_actor: the {partition_id} batch of step {global_steps}: "
                        + "; ".join(problems)
                    )
                    warn(line)
                    raise RuntimeError(line)
                return batch, metrics

        FastPollReplayBufferSync.__module__ = __name__
        _FAST_POLL_CLS = FastPollReplayBufferSync
    return _FAST_POLL_CLS


def __getattr__(name: str):  # verl's custom-sampler loader imports this module and reads the class by name
    if name == "FastPollReplayBufferSync":
        return _fast_poll_class()
    raise AttributeError(name)
