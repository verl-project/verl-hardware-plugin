# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Turn the Ray job log of a ``VERL_TPU_TIMELINE=1`` TPU run into a self-contained RL timeline & HBM report.

The plugin's TPU trainer engine, ``tpu-sync`` weight sync and vLLM rollout server print ``VERL_TIMELINE{...}`` lines
(see ``verl_hardware_plugin/accelerators/tpu/profilers/timeline.py``); the trainer logs ``step:N - timing_s/...``
metric lines. Both are extracted from the log and embedded into a copy of ``index.html``, which renders two tabs at the
same URL:

1. **Timeline**: lays out the trainer-loop phases from ``timing_s/*`` metrics (anchored at each step's weight sync)
   and draws every trainer, weight-sync, and rollout lane alongside 1 Hz HBM tracks.
2. **HBM Memory**: multi-panel per-chip TPU HBM build-up & RL phase correlation charts plus a per-step HBM breakdown
   table.

The report opens offline and can be shared as a single file.

Examples::

    python3 -m verl_hardware_plugin.accelerators.tpu.profilers.make_report job.log -o report.html --summary
    # fetch the log from a Ray dashboard first (e.g. a kubectl port-forward of the head service):
    python3 -m verl_hardware_plugin.accelerators.tpu.profilers.make_report --ray-address http://localhost:8265 \\
        --job <submission-id> --save-log job.log -o report.html --summary
    # only steps 10-20 (e.g. past compilation and warm-up):
    python3 -m verl_hardware_plugin.accelerators.tpu.profilers.make_report job.log \\
        --steps 10-20 -o report.html --summary

``--summary`` prints, for every step, where the trainer loop spends its time, what trainer rank 0 runs on its chips
(actor update, reference forward, weight sync, idle) and what the rollout replicas do (generating, draining for a
weight sync, paused, idle).
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

MARKER = "VERL_TIMELINE"
_STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+(.*)$")
_NUMPY_SCALAR_RE = re.compile(r"^np\.\w+\((.*)\)$")
_EMBED_TAG = '<script id="tl-embedded" type="application/json"></script>'

# Trainer-loop phases in the order verl's v1 trainer runs them within a step (``_step_once``, then
# ``save_checkpoint``); ``update_weights`` (printed by the plugin) ends the step.
LOOP_PHASES = [
    "switch_wait",
    "switch_to_trainer",
    "gen",
    "reward",
    "old_log_prob",
    "ref",
    "values",
    "adv",
    "update_critic",
    "update_actor",
    "save_checkpoint",
]
# What a trainer rank executes on its chips (top-level ``trainer_workers`` spans).
RANK_CLASS = {
    "update_actor": "actor",
    "compute_ref_log_prob": "ref",
    "compute_log_prob": "logp",
    "update_weights": "sync",
    "release_tpu_sync": "sync",
}

Intervals = list[tuple[float, float]]


_GIB = float(1024**3)


def parse_hbm_csv(
    spec: str,
    default_group: str = "trainer_workers",
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> list[dict]:
    """Parse ``[group=]path.csv`` produced by ``timeline.py`` sidecar mode into ``ph='C'`` timeline events.

    Each CSV line has ``unix_time,port,device_id,memory_usage_bytes,total_memory_bytes,duty_cycle_pct``.
    """
    if "=" in spec and not os.path.exists(spec):
        prefix, path = spec.split("=", 1)
        group = prefix if prefix in ("trainer_workers",) or prefix.startswith("rollout_replica_") else default_group
    else:
        group, path = default_group, spec
    buckets: dict[int, dict[int, tuple[int, int, float]]] = defaultdict(dict)
    with open(path, errors="replace") as fh:
        for line in fh:
            if not line or line.startswith("#") or line.startswith("unix_time"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                t_raw = float(parts[0])
                t_sec = int(round(t_raw))
                dev = int(parts[2])
                used_b = int(parts[3])
                tot_b = int(parts[4])
                duty = float(parts[5])
            except ValueError:
                continue
            if t_min is not None and t_raw < t_min:
                continue
            if t_max is not None and t_raw > t_max:
                continue
            prev = buckets[t_sec].get(dev)
            if prev is None or used_b > prev[0]:
                buckets[t_sec][dev] = (used_b, tot_b, duty)
    out: list[dict] = []
    for t_sec in sorted(buckets):
        by_dev = buckets[t_sec]
        devs = sorted(by_dev)
        out.append(
            {
                "ph": "C",
                "name": "hbm",
                "group": group,
                "lane": "hbm",
                "ts": float(t_sec),
                "args": {
                    "used": [round(by_dev[d][0] / _GIB, 3) for d in devs],
                    "cap": round(max(by_dev[d][1] for d in devs) / _GIB, 2),
                    "duty": [round(by_dev[d][2], 1) for d in devs],
                },
            }
        )
    return out


def _parse_value(text: str) -> Optional[float]:
    text = text.strip()
    match = _NUMPY_SCALAR_RE.match(text)  # numpy scalars print as np.float64(1.0)
    if match:
        text = match.group(1)
    try:
        return float(text)
    except ValueError:
        return None


def parse_log_lines(lines: Iterable[str]) -> tuple[list[dict], dict[int, dict[str, float]], int]:
    """Return (timeline events, per-step metrics, number of unparsable timeline lines) of a job log."""
    events: list[dict] = []
    metrics: dict[int, dict[str, float]] = defaultdict(dict)
    bad = 0
    for line in lines:
        i = line.find(MARKER + "{")
        if i >= 0:
            j = line.rfind("}")
            try:
                events.append(json.loads(line[i + len(MARKER) : j + 1]))
            except ValueError:
                bad += 1
            continue
        k = line.find("step:")
        m = _STEP_RE.search(line[max(0, k - 1) :]) if k >= 0 else None
        if not m or "timing_s/" not in line:  # e.g. the step-0 validation line
            continue
        rec = metrics[int(m.group(1))]
        for part in m.group(2).split(" - "):
            key, sep, val = part.partition(":")
            value = _parse_value(val) if sep else None
            if value is not None:
                rec[key.strip()] = value
    return events, dict(metrics), bad


def fetch_ray_job_log(address: str, job: str) -> str:
    url = f"{address.rstrip('/')}/api/jobs/{job}/logs"
    with urllib.request.urlopen(url, timeout=600) as r:
        return json.loads(r.read().decode("utf-8", "replace"))["logs"]


def build_html(template: str, payload: dict) -> str:
    if _EMBED_TAG not in template:
        raise ValueError("index.html has no empty tl-embedded <script> tag to fill")
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(_EMBED_TAG, f'<script id="tl-embedded" type="application/json">{data}</script>', 1)


# ----------------------------------------------------------------------------------------------------------------------
# Interval arithmetic for the text summary (the page computes the same in JavaScript)
# ----------------------------------------------------------------------------------------------------------------------
def union(intervals: Iterable[tuple[float, float]]) -> Intervals:
    out: Intervals = []
    for a, b in sorted(iv for iv in intervals if iv[1] > iv[0]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def intersect(xs: Intervals, ys: Intervals) -> Intervals:
    out, i, j = [], 0, 0
    while i < len(xs) and j < len(ys):
        lo, hi = max(xs[i][0], ys[j][0]), min(xs[i][1], ys[j][1])
        if hi > lo:
            out.append((lo, hi))
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(xs: Intervals, ys: Intervals) -> Intervals:
    out: Intervals = []
    for a, b in xs:
        cur = a
        for c, d in ys:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
        if cur < b:
            out.append((cur, b))
    return out


def length(xs: Intervals, a: float = float("-inf"), b: float = float("inf")) -> float:
    return sum(max(0.0, min(y, b) - max(x, a)) for x, y in xs)


# ----------------------------------------------------------------------------------------------------------------------
# Per-step summary
# ----------------------------------------------------------------------------------------------------------------------
def step_windows(events: list[dict], metrics: dict[int, dict[str, float]]) -> list[dict]:
    """Steps placed on the wall clock: each ends with its ``update_weights`` span and lasts ``timing_s/step``."""
    anchors = {
        e["step"]: e
        for e in events
        if e.get("name") == "update_weights" and e.get("group", "trainer") == "trainer" and e.get("step") is not None
    }
    steps = []
    for k in sorted(metrics):
        dur = metrics[k].get("timing_s/step")
        if not dur or k not in anchors:
            continue
        end = anchors[k]["ts"] + anchors[k]["dur"]
        steps.append({"step": k, "t0": end - dur, "t1": end, "metrics": metrics[k]})
    for cur, nxt in zip(steps, steps[1:] + [None], strict=True):
        cur["t_next"] = nxt["t0"] if nxt else cur["t1"] + cur["metrics"].get("timing_s/testing", 0.0)
    return steps


def parse_step_range(text: str) -> tuple[int, int]:
    """``"10-20"`` (or ``"10:20"``, ``"15"``) -> ``(first, last)``."""
    first, sep, last = text.replace(":", "-").partition("-")
    first_step, last_step = int(first), int(last) if sep else int(first)
    if last_step < first_step:
        raise ValueError(f"empty step range {text!r}")
    return first_step, last_step


def select_steps(
    events: list[dict], metrics: dict[int, dict[str, float]], first: int, last: int
) -> tuple[list[dict], dict[int, dict[str, float]]]:
    """Keep only steps ``first``..``last``: their metrics and the events inside their wall-clock window.

    The window runs from the start of step ``first`` to the start of the step after ``last`` (or the end of ``last``).
    Spans crossing its edges are clipped to it; a request clipped at the start keeps its gate-wait / first-token
    offsets relative to the new start. Replica metadata is kept, and the weight version a replica serves when the
    window opens is carried over as a ``weights`` instant at the window start.
    """
    windows = {w["step"]: w for w in step_windows(events, metrics)}
    if first not in windows or last not in windows:
        raise ValueError(f"steps {first}-{last} not in the log; steps with a weight-sync anchor: {sorted(windows)}")
    w0, w1 = windows[first]["t0"], windows[last]["t_next"]
    selected: list[dict] = []
    weights_before: dict[str, dict] = {}  # group -> last "weights" instant before the window
    for e in events:
        ph, ts = e.get("ph", "X"), e.get("ts")
        if ph == "M":  # the replica is up (at the latest) when the window opens
            selected.append({**e, "ts": round(max(ts, w0), 6)} if ts is not None else e)
            continue
        if ts is None:
            continue
        if ph in ("i", "C"):
            if w0 <= ts <= w1:
                selected.append(e)
            elif ph == "i" and ts < w0 and e.get("name") == "weights":
                grp = str(e.get("group") or "")
                prev = weights_before.get(grp)
                if prev is None or ts > prev["ts"]:
                    weights_before[grp] = e
            continue
        end = ts + e.get("dur", 0.0)
        if (end <= w0 or ts >= w1) if end > ts else not w0 <= ts <= w1:  # no overlap with the window
            continue
        if w0 <= ts and end <= w1:
            selected.append(e)
            continue
        start, stop = max(ts, w0), min(end, w1)
        args = dict(e.get("args") or {})
        for key in ("admit", "first"):  # offsets from the span start
            if key in args:
                args[key] = round(max(0.0, args[key] - (start - ts)), 4)
        args["clipped"] = True
        selected.append({**e, "ts": round(start, 6), "dur": round(stop - start, 6), "args": args})
    selected += [{**e, "ts": round(w0, 6)} for e in weights_before.values()]
    return selected, {k: v for k, v in metrics.items() if first <= k <= last}


def sampler_intervals(events: list[dict], group: str) -> dict[str, Intervals]:
    """Generating / draining / paused intervals of one rollout replica (same rules as the page's ``state`` row).

    ``generating`` includes ``gate_closed`` (still generating while the submission gate is closed for a weight sync);
    ``paused`` is disjoint from both.
    """
    in_engine, drain, paused = [], [], []
    for e in events:
        if e.get("group") != group or e.get("ph", "X") != "X":
            continue
        if e.get("name") == "request":
            admit = min((e.get("args") or {}).get("admit", 0.0), e["dur"])
            in_engine.append((e["ts"] + admit, e["ts"] + e["dur"]))
        elif e.get("name") == "drain":
            drain.append((e["ts"], e["ts"] + e["dur"]))
        elif e.get("name") == "paused":
            paused.append((e["ts"], e["ts"] + e["dur"]))
    busy, drain_u = union(in_engine), union(drain)
    paused_all = union(union(paused) + subtract(drain_u, busy))
    return {
        "generating": subtract(busy, paused_all),
        "gate_closed": subtract(intersect(busy, drain_u), paused_all),
        "paused": paused_all,
    }


def text_summary(events: list[dict], metrics: dict[int, dict[str, float]]) -> str:
    """Per-step table of the trainer loop, trainer rank 0's chips and the rollout replicas."""
    steps = step_windows(events, metrics)
    if not steps:
        if not metrics:
            return "no step metrics found"
        return "no update_weights timeline spans (VERL_TPU_TIMELINE unset?); timing_s/step per step: " + ", ".join(
            f"{k}:{v.get('timing_s/step', float('nan')):.1f}" for k, v in sorted(metrics.items())
        )
    rank_lanes = sorted({e["lane"] for e in events if e.get("group") == "trainer_workers"})
    rank_lane = "rank0" if "rank0" in rank_lanes else (rank_lanes[0] if rank_lanes else None)
    rank_spans: dict[str, Intervals] = defaultdict(list)
    for e in events:
        if e.get("group") == "trainer_workers" and e.get("lane") == rank_lane and e.get("name") in RANK_CLASS:
            rank_spans[RANK_CLASS[e["name"]]].append((e["ts"], e["ts"] + e["dur"]))
    rank_union = {k: union(v) for k, v in rank_spans.items()}
    rank_busy = union([iv for v in rank_spans.values() for iv in v])
    groups = sorted({e["group"] for e in events if str(e.get("group", "")).startswith("rollout_replica_")})
    replicas = {g: sampler_intervals(events, g) for g in groups}
    requests = [e for e in events if e.get("name") == "request"]

    cols = [
        ("step", 4),
        ("start>next", 10),
        ("t_s/step", 9),
        ("gen wait", 9),
        ("ref", 7),
        ("upd_actor", 9),
        ("upd_wts", 8),
        ("|chips:actor", 12),
        ("ref", 7),
        ("sync", 7),
        ("IDLE", 7),
        ("|rollout:gen", 12),
        ("gate", 7),
        ("paused", 7),
        ("IDLE", 7),
        ("tok/s", 8),
    ]
    out = ["".join(f"{name:>{w}}" for name, w in cols)]
    rows = []
    for st in steps:
        a, b, m = st["t0"], st["t_next"], st["metrics"]
        window = b - a
        row = [
            st["step"],
            window,
            m.get("timing_s/step", 0.0),
            m.get("timing_s/gen", 0.0),
            m.get("timing_s/ref", 0.0),
            m.get("timing_s/update_actor", 0.0),
            m.get("timing_s/update_weights", 0.0),
        ]
        if rank_lane:
            row += [length(rank_union.get(k, []), a, b) for k in ("actor", "ref", "sync")]
            row.append(window - length(rank_busy, a, b))
        else:
            row += [float("nan")] * 4
        if replicas:
            # "generating" includes the part behind a closed gate ("gate"); paused and generating are disjoint.
            gen = sum(length(r["generating"], a, b) for r in replicas.values()) / len(replicas)
            gate = sum(length(r["gate_closed"], a, b) for r in replicas.values()) / len(replicas)
            paused = sum(length(r["paused"], a, b) for r in replicas.values()) / len(replicas)
            tokens = sum((r.get("args") or {}).get("out", 0) for r in requests if a <= r["ts"] + r["dur"] < b)
            row += [gen, gate, paused, window - gen - paused, tokens / max(window, 1e-9)]
        else:
            row += [float("nan")] * 5
        rows.append(row)
        out.append(f"{row[0]:>4}" + "".join(f"{v:>{w}.2f}" for v, (_, w) in zip(row[1:], cols[1:], strict=True)))
    steady = [r for r in rows if r[0] >= 2] or rows
    means = [sum(r[i] for r in steady) / len(steady) for i in range(1, len(cols))]
    out.append(" avg" + "".join(f"{v:>{w}.2f}" for v, (_, w) in zip(means, cols[1:], strict=True)))
    out.append(
        f"(avg over steps {steady[0][0]}-{steady[-1][0]}; seconds per step; start>next = wall time from this step's "
        "start to the next one's; chips = trainer " + (rank_lane or "rank") + "; rollout = mean over "
        f"{len(replicas)} replica(s); gate = still generating with the submission gate closed for a weight sync)"
    )
    n_abort = sum(1 for r in requests if (r.get("args") or {}).get("stop") == "abort")
    out.append(f"{len(requests)} rollout requests ({n_abort} aborted by a weight sync and resumed)")
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="Ray job log file ('-' for stdin)")
    ap.add_argument("--ray-address", help="Ray dashboard address to fetch the log from, e.g. http://localhost:8265")
    ap.add_argument("--job", help="Ray job submission id (with --ray-address)")
    ap.add_argument("-o", "--output", help="output HTML file")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))
    ap.add_argument("--title", help="source name shown in the report (default: log file name / job id)")
    ap.add_argument("--save-log", help="also write the fetched log here (with --ray-address)")
    ap.add_argument("--save-events", help="also write the extracted events and metrics here (JSON)")
    ap.add_argument(
        "--steps", help="only keep these steps, e.g. 10-20: their metrics and what happened in their time window"
    )
    ap.add_argument(
        "--hbm-csv",
        action="append",
        default=[],
        metavar="[GROUP=]CSV",
        help="merge external 1 Hz HBM CSV file(s) from timeline.py (e.g. trainer_workers=trainer.csv)",
    )
    ap.add_argument("--summary", action="store_true", help="print a per-step text summary")
    args = ap.parse_args(argv)
    if not args.output and not args.summary and not args.save_events:
        ap.error("nothing to do: give -o/--output, --summary and/or --save-events")

    if args.ray_address:
        if not args.job:
            ap.error("--job is required with --ray-address")
        text = fetch_ray_job_log(args.ray_address, args.job)
        if args.save_log:
            with open(args.save_log, "w") as f:
                f.write(text)
        events, metrics, bad = parse_log_lines(text.splitlines())
        source = args.job
    elif args.log:
        with sys.stdin if args.log == "-" else open(args.log, errors="replace") as fh:
            events, metrics, bad = parse_log_lines(fh)
        source = os.path.basename(args.log)
    else:
        ap.error("give a log file or --ray-address/--job")

    if args.hbm_csv:
        span_ts = [
            e["ts"] + (e.get("dur") or 0.0)
            for e in events
            if isinstance(e.get("ts"), int | float) and e.get("ph") not in ("M", "C")
        ] + [e["ts"] for e in events if isinstance(e.get("ts"), int | float) and e.get("ph") not in ("M", "C")]
        t_min = (min(span_ts) - 240.0) if span_ts else None
        t_max = (max(span_ts) + 30.0) if span_ts else None
        for spec in args.hbm_csv:
            events.extend(parse_hbm_csv(spec, t_min=t_min, t_max=t_max))

    if args.steps:
        first, last = parse_step_range(args.steps)
        events, metrics = select_steps(events, metrics, first, last)
        source += f" (steps {first}-{last})"
    payload = {"source": args.title or source, "events": events, "metrics": metrics}
    if args.save_events:
        with open(args.save_events, "w") as f:
            json.dump(payload, f)
    if args.output:
        with open(args.template) as f:
            template = f.read()
        with open(args.output, "w") as f:
            f.write(build_html(template, payload))
        print(
            f"wrote {args.output}: {len(events)} timeline events, {len(metrics)} step metric rows"
            + (f", {bad} unparsable lines" if bad else ""),
            file=sys.stderr,
        )
    if args.summary:
        print(text_summary(events, metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
