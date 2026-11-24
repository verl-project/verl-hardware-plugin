# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Dynamic NUMA and physical-core CPU pinning for sub-host TPU vLLM rollout engines.

When several vLLM rollout engines share one TPU host (e.g. ``2 x TP=4`` or ``4 x TP=2`` on a
TPU 7x ``2x2x1`` host), all server actors, ``EngineCore`` processes, and TP workers otherwise
compete for the same host CPUs and cross NUMA boundaries when feeding PCIe-attached TPUs.

This module discovers the host's allowed CPUs (``os.sched_getaffinity(0)``), NUMA node CPU lists,
physical cores (excluding SMT siblings), and TPU PCIe NUMA affinity dynamically from Linux
``sysfs``, then partitions each NUMA node's cores into disjoint slices for the server /
``EngineCore`` and each TP worker:

* The server actor computes the engine's plan *before* narrowing its own affinity, exports the
  per-worker CPU slices in ``VERL_TPU_SUBHOST_WORKER_CPUS``, and pins all of its threads to the
  ``EngineCore`` slice (which ``EngineCoreProc`` inherits on spawn).
* Each spawned vLLM worker reads its rank's slice from ``VERL_TPU_SUBHOST_WORKER_CPUS`` inside
  ``_configure_tpu_process_env`` (before ``torch_tpu`` / ``libtpu`` starts) and pins all of its
  threads to that disjoint slice.
* Single-engine hosts (``TP=8`` or multi-host) never set the sub-host metadata variables and do
  nothing. Set ``VERL_TPU_SUBHOST_CPU_PINNING=0`` (or ``false`` / ``no`` / ``off``) to opt out.
"""

import logging
import os
import threading
from collections.abc import Collection, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Opt-out toggle (default enabled when unset or "1"; disabled by 0/false/no/off, case-insensitive).
SUBHOST_CPU_PINNING_ENV = "VERL_TPU_SUBHOST_CPU_PINNING"
# Per-engine layout set by tpu_vllm.subhost_engine_env on engines that share a host.
SUBHOST_SLOT_ENV = "VERL_TPU_SUBHOST_SLOT"
SUBHOST_DEVICES_PER_ENGINE_ENV = "VERL_TPU_SUBHOST_DEVICES_PER_ENGINE"
SUBHOST_ENGINES_PER_HOST_ENV = "VERL_TPU_SUBHOST_ENGINES_PER_HOST"
# Per-worker CPU ranges (semicolon-separated cpulists, indexed by worker rank) exported by
# TPUvLLMHttpServer before pinning itself, so spawned workers are not limited by the server's
# narrowed sched_getaffinity mask.
SUBHOST_WORKER_CPUS_ENV = "VERL_TPU_SUBHOST_WORKER_CPUS"

_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})
# Google PCI vendor ID and PCI-SIG base classes used by TPU PCIe endpoints:
# 0x12 (Processing Accelerators) and 0xff (Vendor-Specific / Unassigned, e.g. TPU vfio-pci endpoints).
_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TPU_PCI_CLASS_PREFIXES = ("0x12", "0xff")


@dataclass(frozen=True)
class SubhostCpuPlan:
    """CPU pinning plan for one sub-host rollout engine and its TP workers."""

    slot: int
    devices_per_engine: int
    engines_per_host: int
    numa_node: int
    sub_slot: int
    engines_on_node: int
    ec_cpus: tuple[int, ...]
    worker_cpus: tuple[tuple[int, ...], ...]
    used_pci_numa: bool
    used_smt: bool


def subhost_cpu_pinning_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether sub-host CPU pinning is enabled (default True; disabled by 0/false/no/off)."""
    env = os.environ if environ is None else environ
    value = env.get(SUBHOST_CPU_PINNING_ENV)
    if value is None:
        return True
    return value.strip().lower() not in _DISABLED_VALUES


def parse_cpu_list(text: str) -> list[int]:
    """Parse a Linux cpulist string (e.g. ``'0-55,112-167'``) into sorted unique CPU IDs."""
    stripped = text.strip()
    if not stripped:
        return []
    cpus: set[int] = set()
    for part in stripped.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"Invalid empty entry in CPU list {text!r}")
        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2:
                raise ValueError(f"Invalid CPU range {token!r} in {text!r}")
            start = int(bounds[0].strip())
            end = int(bounds[1].strip())
            if start < 0 or end < start:
                raise ValueError(f"Invalid CPU range {token!r} in {text!r}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(token)
            if cpu < 0:
                raise ValueError(f"Negative CPU ID {cpu} in {text!r}")
            cpus.add(cpu)
    return sorted(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Format CPU IDs as a compact Linux cpulist string (e.g. ``'0-7,12,16-19'``)."""
    ordered = sorted(set(cpus))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def format_worker_cpu_sets(worker_cpus: Sequence[Sequence[int]]) -> str:
    """Encode per-worker CPU slices as semicolon-separated cpulist strings."""
    return ";".join(format_cpu_list(cpus) for cpus in worker_cpus)


def parse_worker_cpu_sets(raw: str) -> list[list[int]]:
    """Decode ``VERL_TPU_SUBHOST_WORKER_CPUS`` into one sorted CPU list per worker rank."""
    stripped = raw.strip()
    if not stripped:
        return []
    result: list[list[int]] = []
    for entry in stripped.split(";"):
        cpus = parse_cpu_list(entry)
        if not cpus:
            raise ValueError(f"Empty worker CPU entry in {raw!r}")
        result.append(cpus)
    return result


def get_allowed_cpus() -> set[int] | None:
    """Return the current process's allowed CPUs from ``os.sched_getaffinity(0)``, or ``None``."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return set(os.sched_getaffinity(0))
    except OSError:
        return None


def read_numa_nodes(allowed_cpus: Collection[int], sysfs_root: Path = Path("/sys")) -> dict[int, list[int]]:
    """Read per-NUMA-node CPU lists from ``sysfs``, intersected with ``allowed_cpus``.

    Falls back to ``{0: sorted(allowed_cpus)}`` when ``sysfs`` has no usable NUMA nodes.
    """
    allowed = set(allowed_cpus)
    if not allowed:
        return {}
    node_dir = sysfs_root / "devices" / "system" / "node"
    nodes: dict[int, list[int]] = {}
    try:
        entries = list(node_dir.iterdir())
    except OSError:
        entries = []

    for entry in entries:
        name = entry.name
        if not name.startswith("node") or not name[4:].isdigit():
            continue
        node_id = int(name[4:])
        try:
            raw = (entry / "cpulist").read_text(encoding="utf-8")
            cpus = sorted(set(parse_cpu_list(raw)) & allowed)
        except (OSError, ValueError):
            continue
        if cpus:
            nodes[node_id] = cpus

    if not nodes:
        return {0: sorted(allowed)}
    return {node_id: nodes[node_id] for node_id in sorted(nodes)}


def read_physical_cores(node_cpus: Sequence[int], sysfs_root: Path = Path("/sys")) -> list[int]:
    """Return one representative CPU per physical core in ``node_cpus`` (lowest allowed SMT sibling).

    Falls back to keeping a CPU as-is whenever its ``thread_siblings_list`` is unavailable.
    """
    ordered = sorted(set(node_cpus))
    if not ordered:
        return []
    node_set = set(ordered)
    cpu_base = sysfs_root / "devices" / "system" / "cpu"
    representatives: set[int] = set()
    for cpu in ordered:
        siblings_file = cpu_base / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        try:
            siblings = set(parse_cpu_list(siblings_file.read_text(encoding="utf-8")))
            in_node = siblings & node_set
            representatives.add(min(in_node) if in_node else cpu)
        except (OSError, ValueError):
            representatives.add(cpu)
    return sorted(representatives)


def _pci_bdf_sort_key(name: str) -> tuple[int, tuple[int, ...], str]:
    """Sort key for PCI Bus/Device/Function names such as ``0000:c0:00.1``."""
    parts = name.replace(".", ":").split(":")
    if len(parts) == 4:
        try:
            return (0, tuple(int(p, 16) for p in parts), name)
        except ValueError:
            pass
    return (1, (), name)


def read_tpu_pci_numa_nodes(sysfs_root: Path = Path("/sys")) -> list[int]:
    """Read ``numa_node`` (``-1`` when unset/unreadable) for BDF-sorted Google TPU PCIe devices."""
    pci_dir = sysfs_root / "bus" / "pci" / "devices"
    try:
        entries = sorted(pci_dir.iterdir(), key=lambda p: _pci_bdf_sort_key(p.name))
    except OSError:
        return []

    tpu_nodes: list[int] = []
    for entry in entries:
        try:
            vendor = (entry / "vendor").read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if vendor != _GOOGLE_PCI_VENDOR_ID:
            continue
        # Exclude non-TPU Google PCIe devices (e.g. gVNIC 0x020000, NVMe 0x010802, PCIe bridges 0x060400).
        class_file = entry / "class"
        if class_file.exists():
            try:
                pci_class = class_file.read_text(encoding="utf-8").strip().lower()
            except OSError:
                continue
            if not pci_class.startswith(_TPU_PCI_CLASS_PREFIXES):
                continue
        try:
            numa_node = int((entry / "numa_node").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            numa_node = -1
        tpu_nodes.append(numa_node)
    return tpu_nodes


def _slot_numa_from_pci(
    devices_per_engine: int,
    engines_per_host: int,
    valid_nodes: set[int],
    tpu_pci_numa_nodes: Sequence[int],
) -> list[int] | None:
    """Map every engine slot to a NUMA node from BDF-sorted TPU PCIe devices, or return ``None``."""
    if not tpu_pci_numa_nodes or devices_per_engine <= 0:
        return None
    count = len(tpu_pci_numa_nodes)
    if count == engines_per_host * devices_per_engine:
        devs_per_slot = devices_per_engine
    elif devices_per_engine % 2 == 0 and count == engines_per_host * (devices_per_engine // 2):
        devs_per_slot = devices_per_engine // 2
    else:
        return None

    if any(nd < 0 or nd not in valid_nodes for nd in tpu_pci_numa_nodes):
        return None

    slot_numa: list[int] = []
    for s in range(engines_per_host):
        group = set(tpu_pci_numa_nodes[s * devs_per_slot : (s + 1) * devs_per_slot])
        if len(group) != 1:
            return None
        slot_numa.append(next(iter(group)))

    counts: dict[int, int] = {}
    for nd in slot_numa:
        counts[nd] = counts.get(nd, 0) + 1
    if max(counts.values()) - min(counts.values()) > 1:
        return None
    return slot_numa


def select_numa_node_for_slot(
    slot: int,
    devices_per_engine: int,
    engines_per_host: int,
    numa_nodes: Mapping[int, Sequence[int]],
    tpu_pci_numa_nodes: Sequence[int] = (),
) -> tuple[int, int, int, bool]:
    """Select ``(numa_node, sub_slot, engines_on_node, used_pci_numa)`` for engine ``slot``."""
    sorted_nodes = sorted(numa_nodes)
    if not sorted_nodes:
        raise ValueError("numa_nodes must not be empty")
    if engines_per_host <= 0 or not (0 <= slot < engines_per_host):
        raise ValueError(f"Invalid slot {slot} for engines_per_host={engines_per_host}")

    pci_slot_numa = _slot_numa_from_pci(
        devices_per_engine=devices_per_engine,
        engines_per_host=engines_per_host,
        valid_nodes=set(sorted_nodes),
        tpu_pci_numa_nodes=tpu_pci_numa_nodes,
    )
    if pci_slot_numa is not None:
        slot_numa = pci_slot_numa
        used_pci_numa = True
    else:
        num_nodes = len(sorted_nodes)
        slot_numa = [sorted_nodes[(s * num_nodes) // engines_per_host] for s in range(engines_per_host)]
        used_pci_numa = False

    node = slot_numa[slot]
    peers = [s for s, nd in enumerate(slot_numa) if nd == node]
    return node, peers.index(slot), len(peers), used_pci_numa


def compute_subhost_cpu_plan(
    slot: int,
    devices_per_engine: int,
    engines_per_host: int,
    allowed_cpus: Collection[int],
    sysfs_root: Path = Path("/sys"),
) -> SubhostCpuPlan | None:
    """Compute disjoint EngineCore + per-worker CPU slices for a sub-host rollout engine."""
    if engines_per_host <= 1 or devices_per_engine <= 0 or not (0 <= slot < engines_per_host):
        return None
    numa_nodes = read_numa_nodes(allowed_cpus, sysfs_root=sysfs_root)
    if not numa_nodes:
        return None

    tpu_pci_numa_nodes = read_tpu_pci_numa_nodes(sysfs_root=sysfs_root)
    numa_node, sub_slot, engines_on_node, used_pci_numa = select_numa_node_for_slot(
        slot=slot,
        devices_per_engine=devices_per_engine,
        engines_per_host=engines_per_host,
        numa_nodes=numa_nodes,
        tpu_pci_numa_nodes=tpu_pci_numa_nodes,
    )

    node_logical = numa_nodes[numa_node]
    node_phys = read_physical_cores(node_logical, sysfs_root=sysfs_root)

    phys_size = len(node_phys) // engines_on_node
    if phys_size >= devices_per_engine + 1:
        engine_pool = node_phys[sub_slot * phys_size : (sub_slot + 1) * phys_size]
        used_smt = False
    else:
        log_size = len(node_logical) // engines_on_node
        if log_size > phys_size:
            engine_pool = node_logical[sub_slot * log_size : (sub_slot + 1) * log_size]
            used_smt = True
        else:
            engine_pool = node_phys[sub_slot * phys_size : (sub_slot + 1) * phys_size]
            used_smt = False

    pool_size = len(engine_pool)
    if pool_size == 0:
        return None

    if pool_size >= devices_per_engine + 1:
        per_worker = max(1, (pool_size * 6 // 7) // devices_per_engine)
        while per_worker > 1 and per_worker * devices_per_engine >= pool_size:
            per_worker -= 1
        ec_count = pool_size - per_worker * devices_per_engine
        ec_cpus = tuple(engine_pool[:ec_count])
        worker_cpus = tuple(
            tuple(engine_pool[ec_count + r * per_worker : ec_count + (r + 1) * per_worker])
            for r in range(devices_per_engine)
        )
    else:
        # Very small container: keep engines isolated from one another while sharing the slice
        # between this engine's EngineCore and its workers.
        shared = tuple(engine_pool)
        ec_cpus = shared
        worker_cpus = tuple(shared for _ in range(devices_per_engine))

    return SubhostCpuPlan(
        slot=slot,
        devices_per_engine=devices_per_engine,
        engines_per_host=engines_per_host,
        numa_node=numa_node,
        sub_slot=sub_slot,
        engines_on_node=engines_on_node,
        ec_cpus=ec_cpus,
        worker_cpus=worker_cpus,
        used_pci_numa=used_pci_numa,
        used_smt=used_smt,
    )


def apply_process_affinity(cpus: Collection[int], proc_task_dir: Path = Path("/proc/self/task")) -> bool:
    """Pin the calling thread and all live threads in ``/proc/self/task`` to ``cpus``.

    Returns ``True`` if affinity was applied to at least the calling thread, else ``False``.
    """
    cpu_set = set(cpus)
    if not cpu_set or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, cpu_set)
    except OSError as exc:
        logger.warning(
            "Skipping sub-host CPU pinning (sched_setaffinity failed for %s): %s",
            format_cpu_list(cpu_set),
            exc,
        )
        return False

    try:
        entries = os.listdir(proc_task_dir)
    except OSError:
        return True

    self_tid = threading.get_native_id() if hasattr(threading, "get_native_id") else None
    for entry in entries:
        if not entry.isdigit():
            continue
        tid = int(entry)
        if tid == self_tid:
            continue
        try:
            os.sched_setaffinity(tid, cpu_set)
        except OSError:
            # Thread exited (ESRCH) or refused affinity; calling thread is already pinned.
            continue
    return True


def configure_subhost_server_cpu_pinning(
    environ: MutableMapping[str, str] | None = None,
    sysfs_root: Path = Path("/sys"),
    proc_task_dir: Path = Path("/proc/self/task"),
) -> SubhostCpuPlan | None:
    """Compute and apply CPU pinning for ``TPUvLLMHttpServer`` and export per-worker CPU slices.

    No-op when this engine does not share its host (sub-host env vars unset or
    ``engines_per_host <= 1``), when ``VERL_TPU_SUBHOST_CPU_PINNING`` is disabled, or when the
    host/container disallows CPU affinity.
    """
    env = os.environ if environ is None else environ
    raw_slot = env.get(SUBHOST_SLOT_ENV)
    raw_devs = env.get(SUBHOST_DEVICES_PER_ENGINE_ENV)
    raw_engines = env.get(SUBHOST_ENGINES_PER_HOST_ENV)
    if raw_slot is None or raw_devs is None or raw_engines is None:
        return None

    if not subhost_cpu_pinning_enabled(env):
        env.pop(SUBHOST_WORKER_CPUS_ENV, None)
        return None

    try:
        slot = int(raw_slot)
        devices_per_engine = int(raw_devs)
        engines_per_host = int(raw_engines)
    except ValueError:
        logger.warning(
            "Skipping sub-host CPU pinning: invalid env (%s=%r, %s=%r, %s=%r)",
            SUBHOST_SLOT_ENV,
            raw_slot,
            SUBHOST_DEVICES_PER_ENGINE_ENV,
            raw_devs,
            SUBHOST_ENGINES_PER_HOST_ENV,
            raw_engines,
        )
        return None

    if engines_per_host <= 1:
        return None
    if devices_per_engine <= 0 or not (0 <= slot < engines_per_host):
        logger.warning(
            "Skipping sub-host CPU pinning: out-of-range slot=%d devices_per_engine=%d engines_per_host=%d",
            slot,
            devices_per_engine,
            engines_per_host,
        )
        return None

    allowed = get_allowed_cpus()
    if not allowed:
        return None

    plan = compute_subhost_cpu_plan(
        slot=slot,
        devices_per_engine=devices_per_engine,
        engines_per_host=engines_per_host,
        allowed_cpus=allowed,
        sysfs_root=sysfs_root,
    )
    if plan is None:
        return None

    if not apply_process_affinity(plan.ec_cpus, proc_task_dir=proc_task_dir):
        env.pop(SUBHOST_WORKER_CPUS_ENV, None)
        return None

    worker_cpus_str = format_worker_cpu_sets(plan.worker_cpus)
    env[SUBHOST_WORKER_CPUS_ENV] = worker_cpus_str
    logger.info(
        "Pinned sub-host rollout server (slot %d/%d, NUMA node %d via %s%s) to CPUs %s; workers: %s",
        plan.slot,
        plan.engines_per_host,
        plan.numa_node,
        "PCIe" if plan.used_pci_numa else "block-fallback",
        ", SMT-fallback" if plan.used_smt else "",
        format_cpu_list(plan.ec_cpus),
        worker_cpus_str,
    )
    return plan


def apply_subhost_worker_cpu_pinning(
    rank: int,
    local_rank: int,
    environ: Mapping[str, str] | None = None,
    proc_task_dir: Path = Path("/proc/self/task"),
) -> list[int] | None:
    """Pin a spawned sub-host vLLM TPU worker to its rank's CPU slice from ``VERL_TPU_SUBHOST_WORKER_CPUS``."""
    env = os.environ if environ is None else environ
    if not subhost_cpu_pinning_enabled(env):
        return None
    raw = env.get(SUBHOST_WORKER_CPUS_ENV)
    if not raw:
        return None
    try:
        worker_cpu_sets = parse_worker_cpu_sets(raw)
    except ValueError as exc:
        logger.warning("Skipping sub-host worker CPU pinning (malformed %s=%r): %s", SUBHOST_WORKER_CPUS_ENV, raw, exc)
        return None

    if not (0 <= rank < len(worker_cpu_sets)):
        logger.warning(
            "Skipping sub-host worker CPU pinning: rank %d out of range for %s=%r",
            rank,
            SUBHOST_WORKER_CPUS_ENV,
            raw,
        )
        return None

    worker_cpus = worker_cpu_sets[rank]
    if not apply_process_affinity(worker_cpus, proc_task_dir=proc_task_dir):
        return None

    logger.info(
        "Pinned sub-host rollout worker rank %d (local_rank %d) to CPUs %s",
        rank,
        local_rank,
        format_cpu_list(worker_cpus),
    )
    return worker_cpus
