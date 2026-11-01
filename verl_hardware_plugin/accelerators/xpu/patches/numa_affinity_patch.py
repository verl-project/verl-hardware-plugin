# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatch verl's NUMA-affinity pinning to work on Intel XPU, with zero
verl-core changes.

The gap being closed: `verl/utils/distributed.py::set_numa_affinity()` pins
each worker to the CPU cores local to its GPU via
`pynvml.nvmlDeviceSetCpuAffinity()`. On XPU `pynvml` isn't installed, so the
call raises `ImportError` -- which that function already catches. It prints a
warning and returns. Nothing crashes, nothing fails a test; every XPU
training run so far simply ran un-pinned, paying cross-socket memory traffic
between each worker and its own GPU for the life of the job. A silent
performance tax, not a correctness bug.

There is no CPU-affinity equivalent anywhere in `torch.xpu` or `pyzes`
(pyzes covers only telemetry: temperature/power/clock/utilization/memory).
This patch builds one from the same primitives NVML uses internally on
Linux:

    zesDevicePciGetProperties -> PCI BDF
      -> /sys/bus/pci/devices/<bdf>/numa_node
      -> /sys/devices/system/node/nodeN/cpulist
      -> os.sched_setaffinity

Keyed per device index, not per pod: a single pod's two GPUs can sit on two
different NUMA nodes (observed on a 2x B60 pod: device 0 on node 0,
device 1 on node 1), so pinning "the pod" to one NUMA node would be wrong.

Dispatch shape, and why this patch is applied where it is
--------------------------------------------------------
`set_numa_affinity` is a plain module-level function, and both call sites
bind it *by value* at their own import time:

    verl/workers/engine_workers.py:39
        from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
    verl/model_merger/megatron_model_merger.py:44
        from verl.utils.distributed import set_numa_affinity

So reassigning `verl.utils.distributed.set_numa_affinity` only takes effect
for a given call site if it happens before that call site's own
`set_numa_affinity()` runs -- either because the patch landed before the
call site imported the name, or because `_rebind_importers()` below fixes
up an already-imported module's binding before it's used.

`apply()` is called from `PlatformXPU.__init__`, same convention as
`reduce_avg_allreduce_patch` -- only once verl has actually selected XPU for
this process, so a host where a different platform is selected (e.g.
`VERL_PLATFORM=nvidia` on a mixed box) never has this function touched at
all. In practice `PlatformXPU` is usually constructed while verl's engine
modules are still being imported (e.g. `verl/workers/engine/fsdp/
transformer_impl.py` calls `get_device_name()` at module scope), so the patch
typically lands before `engine_workers` binds the name. That ordering is an
implementation detail of verl-core's imports, not a contract, so
`_rebind_importers()` covers the case where `engine_workers` is imported
first.

**Known gap: the checkpoint-merge path is not covered.**
`MegatronModelMerger.__init__` calls `set_numa_affinity()` on line 154 and
`get_nccl_backend()` -- the first thing on that path to call
`get_platform()`, and therefore the first thing to construct `PlatformXPU`
-- only on line 155, one line later. By the time this patch would be
applied, that call has already run against the original, un-pinned
function. No apply site can fix this without changing verl-core: the
platform genuinely is not selected yet when the call happens. Checkpoint
merging is a short, one-shot CPU-bound utility, not a sustained training
loop, so the cost of staying un-pinned there is far smaller than on the
training path this patch does cover. (The training path itself is fine:
`initialize_global_process_group_ray()` on line 90 of `engine_workers.py`
goes through `get_device_name()`/`get_nccl_backend()`, constructing
`PlatformXPU` and running this patch, before line 92's call.)

Because `apply()` runs only once XPU is confirmed selected, the platform
check inside the patched function (`_platform_is_xpu()`) is redundant in the
common case -- but kept anyway as a cheap guard against the platform
singleton being reassigned later via `set_platform()`, and because
`PlatformXPU()` can be constructed and then discarded during auto-detection
probing on a non-XPU host (see `platform_manager._detect_platform_name()`);
in that case `apply()` still runs, but the wrapped function's own check
means it is a no-op for as long as some other platform stays selected.
"""

import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False

# Modules that do `from verl.utils.distributed import set_numa_affinity` and
# therefore hold their own binding. Rebound only if already imported and still
# pointing at the original function.
_IMPORTERS = (
    "verl.workers.engine_workers",
    "verl.model_merger.megatron_model_merger",
)


def _zes_device_pci_bdf(local_rank: int) -> str:
    """Return the PCI BDF (``domain:bus:device.function``) of GPU ``local_rank``.

    Enumerates via pyzes (Level Zero Sysman), respecting whatever device
    visibility mask (e.g. cgroup GPU limits) is already in effect for this
    process -- verified on real hardware to enumerate exactly the allocated
    GPU count, not the host's full GPU count.

    Requires pyzes >= 0.1.2. The whole PCI family (``zesDevicePciGetProperties``,
    ``zes_pci_properties_t``, ``zes_pci_address_t``) was added by
    oneapi-src/level-zero#462 and first released in 0.1.2 (2026-06-12); 0.1.1
    has no PCI API at all. Checked explicitly because the bare failure is an
    unhelpful ``AttributeError: module 'pyzes' has no attribute
    'zes_pci_properties_t'`` -- observed on real hardware against an image that
    shipped 0.1.1.
    """
    from ctypes import byref, c_uint32

    import pyzes as pz

    if not hasattr(pz, "zesDevicePciGetProperties"):
        try:
            from importlib.metadata import version

            installed = version("pyzes")
        except Exception:  # noqa: BLE001 - only used to improve the message
            installed = "unknown"
        raise RuntimeError(
            f"pyzes {installed} has no PCI API (zesDevicePciGetProperties was added in 0.1.2); "
            "NUMA affinity needs pyzes>=0.1.2"
        )

    os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")
    pz.zesInit(0)

    driver_count = c_uint32(0)
    pz.zesDriverGet(byref(driver_count), None)
    drivers = (pz.zes_driver_handle_t * driver_count.value)()
    pz.zesDriverGet(byref(driver_count), drivers)

    devices = []
    for drv in drivers:
        dev_count = c_uint32(0)
        pz.zesDeviceGet(drv, byref(dev_count), None)
        devs = (pz.zes_device_handle_t * dev_count.value)()
        pz.zesDeviceGet(drv, byref(dev_count), devs)
        devices.extend(list(devs))

    if local_rank >= len(devices):
        raise RuntimeError(f"local_rank {local_rank} out of range for {len(devices)} zes-visible device(s)")

    props = pz.zes_pci_properties_t()
    pz.zesDevicePciGetProperties(devices[local_rank], byref(props))
    addr = props.address
    return f"{addr.domain:04x}:{addr.bus:02x}:{addr.device:02x}.{addr.function:x}"


def _read_int(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return int(f.read().strip())


def _read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read().strip()


def _parse_cpulist(cpulist: str) -> set:
    """Parse a Linux sysfs cpulist (e.g. ``"0-3,8,10-11"``) into a set of CPU ids."""
    cpus: set = set()
    for part in cpulist.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _set_numa_affinity_xpu(local_rank: int) -> None:
    """Pin this process to the CPU cores local to XPU device ``local_rank``.

    Failure to pin (no pyzes, no NUMA topology on this box, etc.) is a
    warning, not an exception -- this is a performance optimization, not a
    correctness requirement, matching the contract of the core function being
    replaced. This function never raises.
    """
    try:
        bdf = _zes_device_pci_bdf(local_rank)
        numa_node = _read_int(f"/sys/bus/pci/devices/{bdf}/numa_node")
        if numa_node is None or numa_node < 0:
            logger.info(
                "[verl_hardware_plugin] No NUMA topology for device %d (%s); skipping affinity pinning",
                local_rank,
                bdf,
            )
            return
        cpulist = _read_text(f"/sys/devices/system/node/node{numa_node}/cpulist")
        if not cpulist:
            logger.warning(
                "[verl_hardware_plugin] numa_node=%d for device %d (%s) but no cpulist found",
                numa_node,
                local_rank,
                bdf,
            )
            return
        os.sched_setaffinity(0, _parse_cpulist(cpulist))
        logger.info(
            "[verl_hardware_plugin] Pinned rank to NUMA node %d (device %d, %s, cpus=%s)",
            numa_node,
            local_rank,
            bdf,
            cpulist,
        )
    except ImportError:
        logger.warning("[verl_hardware_plugin] pyzes not available, skipping NUMA affinity setup")
    except Exception as e:  # noqa: BLE001 - best-effort optimization, never fatal
        logger.warning("[verl_hardware_plugin] Failed to set NUMA affinity: %s", e)


def _resolve_local_rank() -> Optional[int]:
    """Resolve this process's local accelerator index, mirroring verl-core.

    Same logic as the function being replaced: under Ray, ask the runtime
    context for the accelerator ids assigned to this actor; otherwise fall
    back to ``LOCAL_RANK``.

    One caveat is carried over from core unchanged, deliberately: under Ray
    this is the id *Ray* assigned, which is a global device index, whereas
    `_zes_device_pci_bdf` enumerates only the devices visible to this
    process. Those agree when `RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK` is
    set (no mask applied, all devices visible) and can disagree when Ray sets
    `ZE_AFFINITY_MASK` per worker. The pynvml path this replaces has exactly
    the same property, so it is not a regression introduced here -- but it is
    the first thing to check if a run logs "out of range for N zes-visible
    device(s)".
    """
    import ray

    from verl.utils.device import get_resource_name

    if ray.is_initialized():
        return int(ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0])
    local_rank = os.environ.get("LOCAL_RANK")
    return int(local_rank) if local_rank is not None else None


def _platform_is_xpu() -> bool:
    """Whether verl actually selected XPU for this process.

    Checked at call time even though `apply()` only runs from
    `PlatformXPU.__init__`: see the end of this module's docstring.
    """
    from verl.plugin.platform import get_platform

    return get_platform().device_name == "xpu"


def _rebind_importers(original, replacement) -> None:
    """Rebind `set_numa_affinity` in modules that already from-imported it.

    Needed because `apply()` runs from `PlatformXPU.__init__`, which is not
    guaranteed to happen before `engine_workers` binds the original function.
    """
    for name in _IMPORTERS:
        module = sys.modules.get(name)
        if module is not None and getattr(module, "set_numa_affinity", None) is original:
            setattr(module, "set_numa_affinity", replacement)  # noqa: B010
            logger.info("[verl_hardware_plugin] Rebound set_numa_affinity in already-imported %s", name)


def apply() -> None:
    global _applied
    if _applied:
        return

    import verl.utils.distributed as distributed

    original_set_numa_affinity = distributed.set_numa_affinity

    def _patched_set_numa_affinity():
        # Anything unexpected here must fall back to core rather than raise:
        # the function being replaced swallows every failure by contract, and
        # both call sites are in a worker/merger constructor.
        try:
            is_xpu = _platform_is_xpu()
        except Exception as e:  # noqa: BLE001
            logger.debug("[verl_hardware_plugin] Platform probe failed (%s); deferring to core", e)
            return original_set_numa_affinity()

        if not is_xpu:
            return original_set_numa_affinity()

        try:
            local_rank = _resolve_local_rank()
        except Exception as e:  # noqa: BLE001
            logger.warning("[verl_hardware_plugin] Could not resolve local rank (%s); skipping NUMA affinity", e)
            return

        if local_rank is None:
            logger.warning("[verl_hardware_plugin] Ray not initialised and LOCAL_RANK unset; skipping NUMA affinity")
            return

        _set_numa_affinity_xpu(local_rank)

    distributed.set_numa_affinity = _patched_set_numa_affinity
    _rebind_importers(original_set_numa_affinity, _patched_set_numa_affinity)
    _applied = True
    logger.info("[verl_hardware_plugin] Patched verl.utils.distributed.set_numa_affinity for XPU")
