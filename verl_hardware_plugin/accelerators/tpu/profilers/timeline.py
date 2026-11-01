# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Opt-in timeline prints of the TPU trainer and rollout, for the RL timeline report.

With ``VERL_TPU_TIMELINE=1`` in the job's runtime env (or exported in the driver shell; ``PlatformTPU`` forwards all
``VERL_TPU_*`` variables to the workers), the TPU components of the plugin print one line per finished span::

    VERL_TIMELINE{"name":"update_actor","group":"trainer_workers","lane":"rank0","ts":1791857954.85,"dur":13.87}

Ray forwards these lines into the job log. ``python3 -m verl_hardware_plugin.accelerators.tpu.profilers.make_report``
turns the log into a self-contained HTML report (with **Timeline** and **HBM Memory** tabs) of what the trainer chips
and the rollout chips do in every step. Nothing is printed (and no hook does any extra work) when the variable is unset.

Groups, lanes and span names:

* ``trainer`` / ``loop``: ``update_weights`` (``initial_weight_sync`` before step 1) and its ``sync.*`` phases, printed
  by the driver-side weight sync (``update_tpu_sync_weights``) with the step number. The report rebuilds the other
  phases of the trainer loop (``gen`` = waiting for rollout, ``ref``, ``adv``, ``update_actor``, ...) from the
  ``timing_s/*`` step metrics, anchored at the step's ``update_weights``.
* ``trainer_workers`` / ``rank<r>``: what trainer rank ``r`` executes: ``update_actor`` (``fwd_bwd`` and
  ``optimizer_step``), ``compute_ref_log_prob`` / ``compute_log_prob`` (forward only), and its side of the weight
  sync, ``update_weights`` (``gather``, ``bind``) and ``release_tpu_sync``. Only rank 0 prints by default; set
  ``VERL_TPU_TIMELINE_RANKS`` to a comma-separated list of ranks or to ``all``.
* ``rollout_replica_<i>`` / ``requests``: one ``request`` per ``generate`` call, with the time parked at the closed
  submission gate (``admit``), the time to the first token (``first``), the prompt / generated tokens (``in`` /
  ``out``), the finish reason (``stop``) and the weight version that generated it (``wv``).
* ``rollout_replica_<i>`` / ``server``: ``drain`` (submission gate closed for a weight sync, engine still running the
  admitted requests), ``paused`` (engine paused until the gate reopens), ``resume_engine`` and the ``rpc.*`` calls of
  the weight sync, plus a ``weights`` instant whenever a new weight version is installed.
"""

import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

MARKER = "VERL_TIMELINE"
ENV_VAR = "VERL_TPU_TIMELINE"
RANKS_ENV_VAR = "VERL_TPU_TIMELINE_RANKS"
HBM_PERIOD_ENV_VAR = "VERL_TPU_TIMELINE_HBM_PERIOD"

TRAINER = "trainer"
TRAINER_LOOP = "loop"
TRAINER_WORKERS = "trainer_workers"
SERVER_LANE = "server"
REQUESTS_LANE = "requests"
HBM_LANE = "hbm"

_FALSE_VALUES = ("", "0", "false", "no", "off")
_hbm_sampler_lock = threading.Lock()
_hbm_sampler_started: set[str] = set()
_GIB = float(1024**3)


def enabled() -> bool:
    """Whether ``VERL_TPU_TIMELINE`` is set to a true value in this process."""
    return os.environ.get(ENV_VAR, "0").strip().lower() not in _FALSE_VALUES


def rollout_group(replica_rank: Any) -> str:
    """Timeline group of a rollout replica."""
    return f"rollout_replica_{replica_rank if replica_rank is not None else 0}"


def _write(event: dict) -> None:
    # One whitespace-free token that contains digits: Ray's log deduplication ignores such words, so it never folds
    # the lines of different workers into "[repeated Nx across cluster]".
    line = MARKER + json.dumps(event, separators=(",", ":"), default=str).replace(" ", "\\u0020")
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def emit(
    name: str,
    ts: float,
    dur: float,
    group: str = TRAINER,
    lane: str = TRAINER_LOOP,
    step: Optional[int] = None,
    **args: Any,
) -> None:
    """Print one finished span that started at epoch time ``ts`` and lasted ``dur`` seconds.

    Args:
        name: Span name, e.g. ``update_actor``.
        ts: Start, in seconds since the epoch (``time.time()``).
        dur: Duration in seconds.
        group: Timeline group (one block of lanes in the report), e.g. ``trainer_workers``.
        lane: Lane within the group, e.g. ``rank0``.
        step: Trainer step the span belongs to, when known.
        **args: Extra details shown in the report's tooltip.
    """
    if not enabled():
        return
    event: dict[str, Any] = {
        "name": name,
        "group": group,
        "lane": lane,
        "ts": round(ts, 6),
        "dur": round(max(dur, 0.0), 6),
    }
    if step is not None:
        event["step"] = int(step)
    if args:
        event["args"] = args
    _write(event)


def instant(name: str, group: str, lane: str, ts: Optional[float] = None, **args: Any) -> None:
    """Print a point event (e.g. a new weight version installed on a rollout replica)."""
    if not enabled():
        return
    event: dict[str, Any] = {"ph": "i", "name": name, "group": group, "lane": lane, "ts": round(ts or time.time(), 6)}
    if args:
        event["args"] = args
    _write(event)


def meta(name: str, group: str, **args: Any) -> None:
    """Print a metadata record (e.g. a rollout replica's ``max_num_seqs``) that the report attaches to ``group``."""
    if not enabled():
        return
    event: dict[str, Any] = {"ph": "M", "name": name, "group": group, "lane": "meta", "ts": round(time.time(), 6)}
    if args:
        event["args"] = args
    _write(event)


@contextmanager
def span(
    name: str,
    group: str = TRAINER,
    lane: Optional[str] = TRAINER_LOOP,
    step: Optional[int] = None,
    **args: Any,
) -> Iterator[dict]:
    """Time the body and print it as one span when it ends (also on an exception).

    A no-op when timeline prints are disabled or ``lane`` is ``None`` (e.g. :func:`trainer_lane` on a rank that does
    not print).

    Yields:
        A dict of extra details: entries added inside the body are printed with the span.
    """
    if not enabled() or lane is None:
        yield {}
        return
    details = dict(args)
    start = time.time()
    try:
        yield details
    finally:
        emit(name, start, time.time() - start, group=group, lane=lane, step=step, **details)


def _trainer_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0") or 0)


def trainer_lane() -> Optional[str]:
    """``rank<r>`` if this trainer rank prints timeline spans, else ``None``.

    Only rank 0 prints by default (every rank runs the same SPMD program); ``VERL_TPU_TIMELINE_RANKS`` selects others,
    e.g. ``0,3`` or ``all``.
    """
    if not enabled():
        return None
    rank = _trainer_rank()
    ranks = os.environ.get(RANKS_ENV_VAR, "0").strip().lower()
    if ranks not in ("all", "*") and str(rank) not in {r.strip() for r in ranks.split(",")}:
        return None
    return f"rank{rank}"


def counter(
    name: str,
    group: str,
    lane: str = HBM_LANE,
    ts: Optional[float] = None,
    **args: Any,
) -> None:
    """Print a counter sample event (``ph="C"``), e.g. 1 Hz local TPU chip HBM usage."""
    if not enabled():
        return
    event: dict[str, Any] = {"ph": "C", "name": name, "group": group, "lane": lane, "ts": round(ts or time.time(), 3)}
    if args:
        event["args"] = args
    _write(event)


def start_hbm_sampler(
    group: str,
    lane: str = HBM_LANE,
    period: Optional[float] = None,
    ports: tuple[int, ...] = (8431, 8432, 8433, 8434),
) -> bool:
    """Start a daemon thread that polls ``tpu_info`` local chip HBM usage every ``period`` seconds.

    Emits ``VERL_TIMELINE{"ph":"C","name":"hbm","group":group,"lane":lane,"ts":...,"args":{"used":[...],...}}``
    where ``used`` is the per-chip HBM usage in GiB (`1024^3` bytes), ``cap`` is per-chip HBM capacity in GiB,
    and ``duty`` is the per-chip duty cycle percentage. Safe to call multiple times; starts at most one sampler
    thread per ``group`` in the current process. Disabled if ``VERL_TPU_TIMELINE`` is off or
    ``VERL_TPU_TIMELINE_HBM_PERIOD=0``.
    """
    if not enabled():
        return False
    if period is None:
        try:
            period = float(os.environ.get(HBM_PERIOD_ENV_VAR, "1.0"))
        except ValueError:
            period = 1.0
    if period <= 0:
        return False
    with _hbm_sampler_lock:
        if group in _hbm_sampler_started:
            return False
        try:
            from tpu_info import device, metrics  # type: ignore[import-not-found]

            chip_type, count = device.get_local_chips()
            if not chip_type or not count:
                return False
        except Exception:
            return False
        _hbm_sampler_started.add(group)

    def _loop() -> None:
        while True:
            t0 = time.time()
            by_dev: dict[int, tuple[int, int, float]] = {}
            for port in ports:
                try:
                    usages = metrics.get_chip_usage(chip_type, addr=f"localhost:{port}")
                except Exception:
                    continue
                for u in usages:
                    dev_id = int(u.device_id)
                    used_b = int(u.memory_usage)
                    tot_b = int(u.total_memory)
                    duty_pct = float(getattr(u, "duty_cycle_pct", 0.0) or 0.0)
                    prev = by_dev.get(dev_id)
                    if prev is None or used_b > prev[0]:
                        by_dev[dev_id] = (used_b, tot_b, duty_pct)
            if by_dev:
                devs = sorted(by_dev)
                used_gib = [round(by_dev[d][0] / _GIB, 3) for d in devs]
                cap_gib = round(max(by_dev[d][1] for d in devs) / _GIB, 2)
                duty = [round(by_dev[d][2], 1) for d in devs]
                counter("hbm", group=group, lane=lane, ts=t0, used=used_gib, cap=cap_gib, duty=duty)
            elapsed = time.time() - t0
            time.sleep(max(0.05, period - elapsed))

    t = threading.Thread(target=_loop, name=f"verl-timeline-hbm-{group}", daemon=True)
    t.start()
    return True


def run_hbm_csv_monitor(
    out_path: str = "/tmp/hbm_monitor.csv",
    period: float = 1.0,
    ports: tuple[int, ...] = (8431, 8432, 8433, 8434),
) -> None:
    """Standalone sidecar loop that appends 1 Hz ``tpu_info`` HBM samples to ``out_path`` CSV.

    Useful when you also want to capture HBM build-up *before* ``TorchTitanTPUEngine.initialize()`` or
    ``TPUvLLMHttpServer.launch_server()`` finishes (e.g. during initial checkpoint loading)::

        python3 -m verl_hardware_plugin.accelerators.tpu.profilers.timeline [/tmp/hbm_monitor.csv] [period] [ports]
    """
    from tpu_info import device, metrics  # type: ignore[import-not-found]

    chip_type, count = device.get_local_chips()
    with open(out_path, "a", buffering=1) as fh:
        fh.write(f"# chip_type={chip_type} count={count} ports={list(ports)}\n")
        fh.write("unix_time,port,device_id,memory_usage_bytes,total_memory_bytes,duty_cycle_pct\n")
        while True:
            t0 = time.time()
            for port in ports:
                try:
                    usages = metrics.get_chip_usage(chip_type, addr=f"localhost:{port}")
                except Exception:
                    continue
                for u in usages:
                    fh.write(
                        f"{t0:.3f},{port},{u.device_id},{u.memory_usage},{u.total_memory},{u.duty_cycle_pct:.1f}\n"
                    )
            time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    _out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hbm_monitor.csv"
    _period = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    _ports = tuple(int(p) for p in (sys.argv[3] if len(sys.argv) > 3 else "8431,8432,8433,8434").split(","))
    run_hbm_csv_monitor(_out, _period, _ports)
