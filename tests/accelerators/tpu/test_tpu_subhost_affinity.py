# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for dynamic NUMA and physical-core pinning of sub-host TPU vLLM rollout engines."""

import logging
import multiprocessing
import os
import threading
from pathlib import Path
from unittest import mock

import pytest

from verl_hardware_plugin.accelerators.tpu.rollout import tpu_subhost_affinity as affinity


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_sysfs(
    root: Path,
    *,
    numa_cpulists: dict[int, str] | None = None,
    thread_siblings: dict[int, str] | None = None,
    pci_devices: list[tuple[str, str, str | None, int | None]] | None = None,
) -> Path:
    """Create a synthetic ``/sys`` tree under ``root``.

    ``pci_devices`` entries are ``(bdf, vendor, pci_class_or_None, numa_node_or_None)``.
    """
    sysfs = root / "sys"
    sysfs.mkdir(parents=True, exist_ok=True)
    if numa_cpulists is not None:
        for node_id, cpulist in numa_cpulists.items():
            _write_text(sysfs / "devices" / "system" / "node" / f"node{node_id}" / "cpulist", cpulist + "\n")
    if thread_siblings is not None:
        for cpu, siblings in thread_siblings.items():
            _write_text(
                sysfs / "devices" / "system" / "cpu" / f"cpu{cpu}" / "topology" / "thread_siblings_list",
                siblings + "\n",
            )
    if pci_devices is not None:
        for bdf, vendor, pci_class, numa_node in pci_devices:
            dev_dir = sysfs / "bus" / "pci" / "devices" / bdf
            _write_text(dev_dir / "vendor", vendor + "\n")
            if pci_class is not None:
                _write_text(dev_dir / "class", pci_class + "\n")
            if numa_node is not None:
                _write_text(dev_dir / "numa_node", f"{numa_node}\n")
    return sysfs


def _build_xeon_v7x_sysfs(
    root: Path,
    *,
    pci_devices: list[tuple[str, str, str | None, int | None]] | None = None,
    include_non_tpu_google_pci: bool = True,
) -> tuple[Path, set[int]]:
    """2-socket Xeon 8581C layout: NUMA 0 phys 0..55 (SMT 112..167), NUMA 1 phys 56..111 (SMT 168..223)."""
    numa_cpulists = {
        0: "0-55,112-167",
        1: "56-111,168-223",
    }
    thread_siblings: dict[int, str] = {}
    for phys in range(112):
        smt = phys + 112
        pair = f"{phys},{smt}"
        thread_siblings[phys] = pair
        thread_siblings[smt] = pair

    if pci_devices is None:
        pci_devices = [
            ("0000:00:03.0", "0x1ae0", "0x120000", 0),
            ("0000:00:03.1", "0x1ae0", "0x120000", 0),
            ("0000:00:04.0", "0x1ae0", "0x120000", 0),
            ("0000:00:04.1", "0x1ae0", "0x120000", 0),
            ("0000:c0:00.0", "0x1ae0", "0x120000", 1),
            ("0000:c0:00.1", "0x1ae0", "0x120000", 1),
            ("0000:c0:02.0", "0x1ae0", "0x120000", 1),
            ("0000:c0:02.1", "0x1ae0", "0x120000", 1),
        ]
    else:
        pci_devices = list(pci_devices)

    if include_non_tpu_google_pci:
        # Non-TPU Google PCIe devices (gVNIC NIC 0x020000, NVMe 0x010802, PCIe bridge 0x060400) and Intel bridge.
        pci_devices.extend(
            [
                ("0000:00:01.0", "0x8086", "0x060000", 0),
                ("0000:00:02.0", "0x1ae0", "0x020000", 0),
                ("0000:00:05.0", "0x1ae0", "0x010802", 0),
                ("0000:c0:01.0", "0x1ae0", "0x060400", 1),
            ]
        )

    sysfs = _build_sysfs(root, numa_cpulists=numa_cpulists, thread_siblings=thread_siblings, pci_devices=pci_devices)
    return sysfs, set(range(224))


# ---------------------------------------------------------------------------
# Parsing, formatting, and opt-out env toggle
# ---------------------------------------------------------------------------


def test_parse_and_format_cpu_list_roundtrip():
    assert affinity.parse_cpu_list("") == []
    assert affinity.parse_cpu_list("  \n") == []
    assert affinity.parse_cpu_list("0") == [0]
    assert affinity.parse_cpu_list("0-3, 10, 6-8, 2") == [0, 1, 2, 3, 6, 7, 8, 10]
    assert affinity.format_cpu_list([10, 0, 1, 2, 3, 6, 7, 8, 2]) == "0-3,6-8,10"
    assert affinity.format_cpu_list([]) == ""


@pytest.mark.parametrize("bad", ["5-3", "-1", "0--2", "1-2-3", "0,,1", "abc", "0-a"])
def test_parse_cpu_list_rejects_malformed_inputs(bad):
    with pytest.raises(ValueError):
        affinity.parse_cpu_list(bad)


def test_worker_cpu_sets_roundtrip_and_validation():
    sets = [list(range(8, 20)), list(range(20, 32)), list(range(32, 44)), list(range(44, 56))]
    encoded = affinity.format_worker_cpu_sets(sets)
    assert encoded == "8-19;20-31;32-43;44-55"
    assert affinity.parse_worker_cpu_sets(encoded) == sets
    assert affinity.parse_worker_cpu_sets("") == []
    for bad in ("8-19;", ";8-19", "8-19;;20-31", "8-19;bad"):
        with pytest.raises(ValueError):
            affinity.parse_worker_cpu_sets(bad)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, True),
        ("", True),
        ("1", True),
        ("true", True),
        ("YES", True),
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("No", False),
        ("off", False),
        (" Off ", False),
    ],
)
def test_subhost_cpu_pinning_enabled_toggle(value, expected):
    env = {} if value is None else {affinity.SUBHOST_CPU_PINNING_ENV: value}
    assert affinity.subhost_cpu_pinning_enabled(env) is expected


# ---------------------------------------------------------------------------
# Topology discovery and CPU plan computation
# ---------------------------------------------------------------------------


def test_plan_2x_tp4_on_two_socket_xeon_v7x_exact_layout(tmp_path):
    sysfs, allowed = _build_xeon_v7x_sysfs(tmp_path)
    plans = [
        affinity.compute_subhost_cpu_plan(
            slot=slot, devices_per_engine=4, engines_per_host=2, allowed_cpus=allowed, sysfs_root=sysfs
        )
        for slot in range(2)
    ]
    assert all(p is not None for p in plans)
    p0, p1 = plans[0], plans[1]
    assert p0 is not None and p1 is not None

    assert (p0.numa_node, p0.sub_slot, p0.engines_on_node, p0.used_pci_numa, p0.used_smt) == (0, 0, 1, True, False)
    assert p0.ec_cpus == tuple(range(0, 8))
    assert p0.worker_cpus == (
        tuple(range(8, 20)),
        tuple(range(20, 32)),
        tuple(range(32, 44)),
        tuple(range(44, 56)),
    )

    assert (p1.numa_node, p1.sub_slot, p1.engines_on_node, p1.used_pci_numa, p1.used_smt) == (1, 0, 1, True, False)
    assert p1.ec_cpus == tuple(range(56, 64))
    assert p1.worker_cpus == (
        tuple(range(64, 76)),
        tuple(range(76, 88)),
        tuple(range(88, 100)),
        tuple(range(100, 112)),
    )

    # All 10 processes (2 EngineCore + 8 workers) are strictly disjoint and cover all 112 physical cores.
    all_slices = [p0.ec_cpus, *p0.worker_cpus, p1.ec_cpus, *p1.worker_cpus]
    flat = [cpu for s in all_slices for cpu in s]
    assert len(flat) == len(set(flat)) == 112
    assert set(flat) == set(range(112))


def test_plan_4x_tp2_on_two_socket_xeon_v7x_exact_layout(tmp_path):
    sysfs, allowed = _build_xeon_v7x_sysfs(tmp_path)
    plans = [
        affinity.compute_subhost_cpu_plan(
            slot=slot, devices_per_engine=2, engines_per_host=4, allowed_cpus=allowed, sysfs_root=sysfs
        )
        for slot in range(4)
    ]
    assert all(p is not None for p in plans)

    expected = [
        (0, 0, 2, tuple(range(0, 4)), (tuple(range(4, 16)), tuple(range(16, 28)))),
        (0, 1, 2, tuple(range(28, 32)), (tuple(range(32, 44)), tuple(range(44, 56)))),
        (1, 0, 2, tuple(range(56, 60)), (tuple(range(60, 72)), tuple(range(72, 84)))),
        (1, 1, 2, tuple(range(84, 88)), (tuple(range(88, 100)), tuple(range(100, 112)))),
    ]
    all_slices = []
    for plan, (nd, sub_slot, on_node, ec, workers) in zip(plans, expected, strict=True):
        assert plan is not None
        assert (plan.numa_node, plan.sub_slot, plan.engines_on_node) == (nd, sub_slot, on_node)
        assert plan.used_pci_numa is True and plan.used_smt is False
        assert plan.ec_cpus == ec
        assert plan.worker_cpus == workers
        all_slices.extend([plan.ec_cpus, *plan.worker_cpus])

    # All 12 processes (4 EngineCore + 8 workers) are strictly disjoint and cover all 112 physical cores.
    flat = [cpu for s in all_slices for cpu in s]
    assert len(flat) == len(set(flat)) == 112
    assert set(flat) == set(range(112))


@pytest.mark.parametrize("devices_per_engine, engines_per_host", [(4, 2), (2, 4)])
def test_plan_accepts_vendor_specific_0xff_tpu_pci_class_and_excludes_non_tpu_google_devices(
    tmp_path, devices_per_engine, engines_per_host
):
    # Real TPU 7x hosts expose 8 Google (0x1ae0) PCIe endpoints with PCI class 0xff0000 alongside
    # non-TPU Google PCIe devices: gVNIC (0x020000), NVMe (0x010802), and PCIe bridges (0x060400).
    pci_devices = [
        ("0000:00:03.0", "0x1ae0", "0xff0000", 0),
        ("0000:00:03.1", "0x1ae0", "0xff0000", 0),
        ("0000:00:04.0", "0x1ae0", "0xff0000", 0),
        ("0000:00:04.1", "0x1ae0", "0xff0000", 0),
        ("0000:c0:00.0", "0x1ae0", "0xff0000", 1),
        ("0000:c0:00.1", "0x1ae0", "0xff0000", 1),
        ("0000:c0:02.0", "0x1ae0", "0xff0000", 1),
        ("0000:c0:02.1", "0x1ae0", "0xff0000", 1),
    ]
    sysfs, allowed = _build_xeon_v7x_sysfs(tmp_path, pci_devices=pci_devices, include_non_tpu_google_pci=True)
    assert affinity.read_tpu_pci_numa_nodes(sysfs_root=sysfs) == [0, 0, 0, 0, 1, 1, 1, 1]

    all_cpus = []
    for slot in range(engines_per_host):
        plan = affinity.compute_subhost_cpu_plan(slot, devices_per_engine, engines_per_host, allowed, sysfs_root=sysfs)
        assert plan is not None
        assert plan.used_pci_numa is True
        assert plan.used_smt is False
        assert plan.numa_node == (0 if slot < engines_per_host // 2 else 1)
        all_cpus.extend([*plan.ec_cpus, *(c for w in plan.worker_cpus for c in w)])
    assert len(all_cpus) == len(set(all_cpus)) == 112


@pytest.mark.parametrize("devices_per_engine, engines_per_host", [(4, 2), (2, 4)])
def test_plan_supports_four_chip_pcie_entries_and_minimal_sysfs(tmp_path, devices_per_engine, engines_per_host):
    # 4 PCIe chip entries (one per dual-core chip) with no ``class`` file present.
    pci_devices = [
        ("0000:00:03.0", "0x1ae0", None, 0),
        ("0000:00:04.0", "0x1ae0", None, 0),
        ("0000:c0:00.0", "0x1ae0", None, 1),
        ("0000:c0:02.0", "0x1ae0", None, 1),
    ]
    sysfs, allowed = _build_xeon_v7x_sysfs(tmp_path, pci_devices=pci_devices, include_non_tpu_google_pci=False)
    for slot in range(engines_per_host):
        plan = affinity.compute_subhost_cpu_plan(slot, devices_per_engine, engines_per_host, allowed, sysfs_root=sysfs)
        assert plan is not None
        assert plan.used_pci_numa is True
        assert plan.numa_node == (0 if slot < engines_per_host // 2 else 1)


def test_plan_follows_reversed_and_interleaved_pci_numa_order(tmp_path):
    # Reversed BDFs: first 4 TPU functions live on NUMA 1, last 4 on NUMA 0.
    reversed_pci = [
        ("0000:00:03.0", "0x1ae0", "0x120000", 1),
        ("0000:00:03.1", "0x1ae0", "0x120000", 1),
        ("0000:00:04.0", "0x1ae0", "0x120000", 1),
        ("0000:00:04.1", "0x1ae0", "0x120000", 1),
        ("0000:c0:00.0", "0x1ae0", "0x120000", 0),
        ("0000:c0:00.1", "0x1ae0", "0x120000", 0),
        ("0000:c0:02.0", "0x1ae0", "0x120000", 0),
        ("0000:c0:02.1", "0x1ae0", "0x120000", 0),
    ]
    sysfs_rev, allowed = _build_xeon_v7x_sysfs(tmp_path / "rev", pci_devices=reversed_pci)
    p0 = affinity.compute_subhost_cpu_plan(0, 4, 2, allowed, sysfs_root=sysfs_rev)
    p1 = affinity.compute_subhost_cpu_plan(1, 4, 2, allowed, sysfs_root=sysfs_rev)
    assert p0 is not None and p1 is not None
    assert (p0.numa_node, p0.used_pci_numa) == (1, True)
    assert (p1.numa_node, p1.used_pci_numa) == (0, True)
    assert p0.ec_cpus == tuple(range(56, 64))
    assert p1.ec_cpus == tuple(range(0, 8))

    # Interleaved 4xTP=2: slots 0,2 -> NUMA 0; slots 1,3 -> NUMA 1.
    interleaved_pci = [
        ("0000:00:03.0", "0x1ae0", "0x120000", 0),
        ("0000:00:03.1", "0x1ae0", "0x120000", 0),
        ("0000:00:04.0", "0x1ae0", "0x120000", 1),
        ("0000:00:04.1", "0x1ae0", "0x120000", 1),
        ("0000:c0:00.0", "0x1ae0", "0x120000", 0),
        ("0000:c0:00.1", "0x1ae0", "0x120000", 0),
        ("0000:c0:02.0", "0x1ae0", "0x120000", 1),
        ("0000:c0:02.1", "0x1ae0", "0x120000", 1),
    ]
    sysfs_int, allowed = _build_xeon_v7x_sysfs(tmp_path / "interleaved", pci_devices=interleaved_pci)
    plans = [affinity.compute_subhost_cpu_plan(s, 2, 4, allowed, sysfs_root=sysfs_int) for s in range(4)]
    assert [(p.numa_node, p.sub_slot, p.used_pci_numa) for p in plans if p is not None] == [
        (0, 0, True),
        (1, 0, True),
        (0, 1, True),
        (1, 1, True),
    ]


@pytest.mark.parametrize(
    "case_name, pci_devices, devices_per_engine, engines_per_host",
    [
        ("no_pci_devices", [], 4, 2),
        (
            "all_minus_one",
            [(f"0000:00:0{i}.0", "0x1ae0", "0x120000", -1) for i in range(8)],
            4,
            2,
        ),
        (
            # Half valid (4 on node 0) + half invalid (4 on -1) must NOT mis-match 4-chip mode!
            "half_invalid_numa",
            [(f"0000:00:0{i}.0", "0x1ae0", "0x120000", 0 if i < 4 else -1) for i in range(8)],
            4,
            2,
        ),
        (
            # Slot 0 spans NUMA 0 and NUMA 1.
            "mixed_numa_within_slot",
            [
                ("0000:00:03.0", "0x1ae0", "0x120000", 0),
                ("0000:00:03.1", "0x1ae0", "0x120000", 0),
                ("0000:00:04.0", "0x1ae0", "0x120000", 1),
                ("0000:00:04.1", "0x1ae0", "0x120000", 1),
                ("0000:c0:00.0", "0x1ae0", "0x120000", 0),
                ("0000:c0:00.1", "0x1ae0", "0x120000", 0),
                ("0000:c0:02.0", "0x1ae0", "0x120000", 1),
                ("0000:c0:02.1", "0x1ae0", "0x120000", 1),
            ],
            4,
            2,
        ),
        (
            # Unbalanced: 3 engines on NUMA 0, 1 engine on NUMA 1.
            "unbalanced_slots",
            [
                ("0000:00:03.0", "0x1ae0", "0x120000", 0),
                ("0000:00:03.1", "0x1ae0", "0x120000", 0),
                ("0000:00:04.0", "0x1ae0", "0x120000", 0),
                ("0000:00:04.1", "0x1ae0", "0x120000", 0),
                ("0000:c0:00.0", "0x1ae0", "0x120000", 0),
                ("0000:c0:00.1", "0x1ae0", "0x120000", 0),
                ("0000:c0:02.0", "0x1ae0", "0x120000", 1),
                ("0000:c0:02.1", "0x1ae0", "0x120000", 1),
            ],
            2,
            4,
        ),
    ],
)
def test_plan_falls_back_to_contiguous_numa_blocks_when_pci_unusable(
    tmp_path, case_name, pci_devices, devices_per_engine, engines_per_host
):
    sysfs, allowed = _build_xeon_v7x_sysfs(tmp_path / case_name, pci_devices=pci_devices)
    plans = [
        affinity.compute_subhost_cpu_plan(s, devices_per_engine, engines_per_host, allowed, sysfs_root=sysfs)
        for s in range(engines_per_host)
    ]
    assert all(p is not None and not p.used_pci_numa for p in plans)
    expected_nodes = [(s * 2) // engines_per_host for s in range(engines_per_host)]
    assert [p.numa_node for p in plans if p is not None] == expected_nodes
    # All slices across all engines on the host remain strictly disjoint.
    all_cpus = [cpu for p in plans if p is not None for group in (p.ec_cpus, *p.worker_cpus) for cpu in group]
    assert len(all_cpus) == len(set(all_cpus)) == 112


def test_plan_one_socket_56_core_host_without_numa_sysfs(tmp_path):
    # Single-socket host with no /sys/devices/system/node and no thread_siblings_list.
    sysfs = _build_sysfs(tmp_path)
    allowed = set(range(56))

    # 2 x TP=4 -> 28 cores each (EC=4, W0..3=6 each).
    tp4_plans = [affinity.compute_subhost_cpu_plan(s, 4, 2, allowed, sysfs_root=sysfs) for s in range(2)]
    assert all(p is not None for p in tp4_plans)
    assert tp4_plans[0].ec_cpus == tuple(range(0, 4))
    assert [len(w) for w in tp4_plans[0].worker_cpus] == [6, 6, 6, 6]
    assert tp4_plans[1].ec_cpus == tuple(range(28, 32))
    assert [len(w) for w in tp4_plans[1].worker_cpus] == [6, 6, 6, 6]

    # 4 x TP=2 -> 14 cores each (EC=2, W0..1=6 each).
    tp2_plans = [affinity.compute_subhost_cpu_plan(s, 2, 4, allowed, sysfs_root=sysfs) for s in range(4)]
    all_cpus = []
    for p in tp2_plans:
        assert p is not None
        assert len(p.ec_cpus) == 2
        assert [len(w) for w in p.worker_cpus] == [6, 6]
        all_cpus.extend([*p.ec_cpus, *(c for w in p.worker_cpus for c in w)])
    assert len(all_cpus) == len(set(all_cpus)) == 56


def test_plan_respects_cgroup_allowed_subset_and_upper_smt_siblings(tmp_path):
    # Container cgroup only allows upper SMT siblings of NUMA 0 (CPUs 112..167); NUMA 1 is disallowed.
    sysfs, _ = _build_xeon_v7x_sysfs(tmp_path)
    allowed = set(range(112, 168))
    plans = [affinity.compute_subhost_cpu_plan(s, 4, 2, allowed, sysfs_root=sysfs) for s in range(2)]
    for plan in plans:
        assert plan is not None
        assert plan.numa_node == 0
        assert plan.used_pci_numa is False  # NUMA 1 from PCIe is not in allowed_cpus
        assert plan.used_smt is False
        used = set(plan.ec_cpus) | {c for w in plan.worker_cpus for c in w}
        assert used <= allowed
    assert plans[0].ec_cpus == tuple(range(112, 116))
    assert plans[1].ec_cpus == tuple(range(140, 144))


def test_plan_falls_back_to_smt_then_shared_slice_then_skip_on_tiny_containers(tmp_path):
    # 4 physical cores (0..3) + 4 SMT siblings (4..7) on one node, running 2 x TP=2 (W=2, W+1=3).
    # phys_size = 4 // 2 = 2 < 3, so SMT fallback uses 8 // 2 = 4 logical CPUs per engine -> disjoint!
    siblings = {c: f"{c},{c + 4}" for c in range(4)}
    siblings.update({c + 4: f"{c},{c + 4}" for c in range(4)})
    sysfs = _build_sysfs(tmp_path, numa_cpulists={0: "0-7"}, thread_siblings=siblings)

    p0 = affinity.compute_subhost_cpu_plan(0, 2, 2, set(range(8)), sysfs_root=sysfs)
    p1 = affinity.compute_subhost_cpu_plan(1, 2, 2, set(range(8)), sysfs_root=sysfs)
    assert p0 is not None and p1 is not None
    assert p0.used_smt is True and p1.used_smt is True
    assert p0.ec_cpus == (0, 1) and p0.worker_cpus == ((2,), (3,))
    assert p1.ec_cpus == (4, 5) and p1.worker_cpus == ((6,), (7,))

    # Very tiny container (4 CPUs, no SMT) running 2 x TP=4 (W=4): M=2 < 5, so EC and workers share
    # the engine's 2-CPU slice while staying disjoint between slot 0 and slot 1.
    tiny_sysfs = _build_sysfs(tmp_path / "tiny", numa_cpulists={0: "0-3"})
    t0 = affinity.compute_subhost_cpu_plan(0, 4, 2, {0, 1, 2, 3}, sysfs_root=tiny_sysfs)
    t1 = affinity.compute_subhost_cpu_plan(1, 4, 2, {0, 1, 2, 3}, sysfs_root=tiny_sysfs)
    assert t0 is not None and t1 is not None
    assert t0.ec_cpus == (0, 1) and all(w == (0, 1) for w in t0.worker_cpus)
    assert t1.ec_cpus == (2, 3) and all(w == (2, 3) for w in t1.worker_cpus)

    # Fewer CPUs than engines_on_node (1 CPU for 2 engines) -> M=0 -> skip pinning cleanly.
    assert affinity.compute_subhost_cpu_plan(0, 2, 2, {0}, sysfs_root=tiny_sysfs) is None


# ---------------------------------------------------------------------------
# Process affinity application and server/worker lifecycle helpers
# ---------------------------------------------------------------------------


def test_apply_process_affinity_pins_all_threads_and_handles_esrch(tmp_path):
    task_dir = tmp_path / "task"
    for tid in ("100", "101", "102", "not-a-tid"):
        (task_dir / tid).mkdir(parents=True)

    calls: list[tuple[int, set[int]]] = []

    def fake_setaffinity(pid: int, mask: set[int]) -> None:
        calls.append((pid, set(mask)))
        if pid == 101:
            raise ProcessLookupError("thread exited")

    with (
        mock.patch.object(os, "sched_setaffinity", side_effect=fake_setaffinity),
        mock.patch.object(threading, "get_native_id", return_value=100),
    ):
        assert affinity.apply_process_affinity([4, 5], proc_task_dir=task_dir) is True

    # Calling thread (0) pinned first, self_tid (100) skipped, exited 101 ignored, live 102 pinned.
    assert calls[0] == (0, {4, 5})
    assert {pid for pid, _ in calls[1:]} == {101, 102}


def test_apply_process_affinity_falls_back_without_proc_and_handles_eperm(tmp_path):
    missing_proc = tmp_path / "missing_task"
    with mock.patch.object(os, "sched_setaffinity") as setaff:
        assert affinity.apply_process_affinity([0, 1], proc_task_dir=missing_proc) is True
    setaff.assert_called_once_with(0, {0, 1})

    with mock.patch.object(os, "sched_setaffinity", side_effect=PermissionError("EPERM")):
        assert affinity.apply_process_affinity([0, 1], proc_task_dir=missing_proc) is False


def test_configure_server_and_apply_worker_pinning_with_mocked_affinity(tmp_path, caplog):
    sysfs, allowed = _build_xeon_v7x_sysfs(tmp_path)
    missing_proc = tmp_path / "no_proc"
    env = {
        affinity.SUBHOST_SLOT_ENV: "1",
        affinity.SUBHOST_DEVICES_PER_ENGINE_ENV: "4",
        affinity.SUBHOST_ENGINES_PER_HOST_ENV: "2",
    }

    pinned: list[tuple[int, set[int]]] = []
    with (
        caplog.at_level(logging.INFO, logger=affinity.__name__),
        mock.patch.object(affinity, "get_allowed_cpus", return_value=allowed),
        mock.patch.object(os, "sched_setaffinity", side_effect=lambda pid, mask: pinned.append((pid, set(mask)))),
    ):
        plan = affinity.configure_subhost_server_cpu_pinning(environ=env, sysfs_root=sysfs, proc_task_dir=missing_proc)
        assert plan is not None
        assert env[affinity.SUBHOST_WORKER_CPUS_ENV] == "64-75;76-87;88-99;100-111"
        assert pinned == [(0, set(range(56, 64)))]

        worker_cpus = affinity.apply_subhost_worker_cpu_pinning(
            rank=2, local_rank=6, environ=env, proc_task_dir=missing_proc
        )
        assert worker_cpus == list(range(88, 100))
        assert pinned[-1] == (0, set(range(88, 100)))

    assert any("slot 1/2, NUMA node 1 via PCIe" in r.message and "56-63" in r.message for r in caplog.records)
    assert any("rank 2 (local_rank 6) to CPUs 88-99" in r.message for r in caplog.records)


def test_configure_server_noops_when_single_engine_disabled_or_eperm(tmp_path):
    sysfs, allowed = _build_xeon_v7x_sysfs(tmp_path)
    missing_proc = tmp_path / "no_proc"

    # Single-engine host (unset subhost env) or engines_per_host=1 -> no-op.
    with mock.patch.object(os, "sched_setaffinity") as setaff:
        assert affinity.configure_subhost_server_cpu_pinning(environ={}, sysfs_root=sysfs) is None
        single = {
            affinity.SUBHOST_SLOT_ENV: "0",
            affinity.SUBHOST_DEVICES_PER_ENGINE_ENV: "8",
            affinity.SUBHOST_ENGINES_PER_HOST_ENV: "1",
        }
        assert affinity.configure_subhost_server_cpu_pinning(environ=single, sysfs_root=sysfs) is None
        # Opt-out toggle disables pinning and clears any stale worker CPU env.
        disabled = {
            affinity.SUBHOST_SLOT_ENV: "0",
            affinity.SUBHOST_DEVICES_PER_ENGINE_ENV: "4",
            affinity.SUBHOST_ENGINES_PER_HOST_ENV: "2",
            affinity.SUBHOST_CPU_PINNING_ENV: "false",
            affinity.SUBHOST_WORKER_CPUS_ENV: "stale",
        }
        assert affinity.configure_subhost_server_cpu_pinning(environ=disabled, sysfs_root=sysfs) is None
        assert affinity.SUBHOST_WORKER_CPUS_ENV not in disabled
        assert affinity.apply_subhost_worker_cpu_pinning(0, 0, environ=disabled) is None
    setaff.assert_not_called()

    # Restricted container (EPERM): server does not raise and does not export SUBHOST_WORKER_CPUS_ENV.
    restricted = {
        affinity.SUBHOST_SLOT_ENV: "0",
        affinity.SUBHOST_DEVICES_PER_ENGINE_ENV: "4",
        affinity.SUBHOST_ENGINES_PER_HOST_ENV: "2",
    }
    with (
        mock.patch.object(affinity, "get_allowed_cpus", return_value=allowed),
        mock.patch.object(os, "sched_setaffinity", side_effect=PermissionError("EPERM")),
    ):
        assert (
            affinity.configure_subhost_server_cpu_pinning(
                environ=restricted, sysfs_root=sysfs, proc_task_dir=missing_proc
            )
            is None
        )
    assert affinity.SUBHOST_WORKER_CPUS_ENV not in restricted


def test_apply_worker_pinning_skips_malformed_or_out_of_range_env():
    with mock.patch.object(os, "sched_setaffinity") as setaff:
        assert affinity.apply_subhost_worker_cpu_pinning(0, 0, environ={}) is None
        assert (
            affinity.apply_subhost_worker_cpu_pinning(0, 0, environ={affinity.SUBHOST_WORKER_CPUS_ENV: "not-cpus"})
            is None
        )
        assert (
            affinity.apply_subhost_worker_cpu_pinning(2, 2, environ={affinity.SUBHOST_WORKER_CPUS_ENV: "0-1;2-3"})
            is None
        )
    setaff.assert_not_called()


# ---------------------------------------------------------------------------
# Real Linux multiprocessing ``spawn`` test (verifies child affinity breakout)
# ---------------------------------------------------------------------------


def _spawned_worker_entry(rank: int, local_rank: int, conn) -> None:
    inherited = sorted(os.sched_getaffinity(0))
    applied = affinity.apply_subhost_worker_cpu_pinning(rank, local_rank)
    final = sorted(os.sched_getaffinity(0))
    conn.send((inherited, applied, final))
    conn.close()


def _spawned_server_entry(sysfs_str: str, conn) -> None:
    ctx = multiprocessing.get_context("spawn")
    plan = affinity.configure_subhost_server_cpu_pinning(sysfs_root=Path(sysfs_str))
    server_affinity = sorted(os.sched_getaffinity(0))
    worker_results = []
    if plan is not None:
        procs_and_pipes = []
        for rank in range(plan.devices_per_engine):
            parent_c, child_c = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_spawned_worker_entry, args=(rank, rank, child_c))
            proc.start()
            child_c.close()
            procs_and_pipes.append((proc, parent_c))
        for proc, parent_c in procs_and_pipes:
            worker_results.append(parent_c.recv())
            parent_c.close()
            proc.join(timeout=30)
            assert proc.exitcode == 0
    conn.send((plan, server_affinity, worker_results))
    conn.close()


def test_real_multiprocessing_spawn_server_and_workers_get_disjoint_os_affinity(tmp_path):
    """Spawn a real server process and child workers to verify kernel sched_setaffinity inheritance and breakout."""
    allowed = affinity.get_allowed_cpus()
    if not allowed or len(allowed) < 6:
        pytest.skip("Need Linux sched_getaffinity with at least 6 allowed CPUs")

    # Build a synthetic 1-node sysfs containing the host's real allowed CPUs (no SMT collapsing)
    # so 2 engines x TP=2 each get >= 3 real CPUs: >=1 for EngineCore and >=1 per worker.
    sysfs = _build_sysfs(tmp_path, numa_cpulists={0: affinity.format_cpu_list(allowed)})
    ctx = multiprocessing.get_context("spawn")

    env_patch = {
        affinity.SUBHOST_SLOT_ENV: "0",
        affinity.SUBHOST_DEVICES_PER_ENGINE_ENV: "2",
        affinity.SUBHOST_ENGINES_PER_HOST_ENV: "2",
    }
    with mock.patch.dict(os.environ, env_patch):
        os.environ.pop(affinity.SUBHOST_CPU_PINNING_ENV, None)
        os.environ.pop(affinity.SUBHOST_WORKER_CPUS_ENV, None)
        parent_c, child_c = ctx.Pipe(duplex=False)
        server_proc = ctx.Process(target=_spawned_server_entry, args=(str(sysfs), child_c))
        server_proc.start()
        child_c.close()
        plan, server_affinity, worker_results = parent_c.recv()
        parent_c.close()
        server_proc.join(timeout=30)
        assert server_proc.exitcode == 0

    assert plan is not None
    assert server_affinity == list(plan.ec_cpus)
    assert len(worker_results) == 2

    seen = set(server_affinity)
    for rank, (inherited, applied, final) in enumerate(worker_results):
        # Child inherited the server's narrowed EngineCore affinity mask before worker pinning...
        assert inherited == list(plan.ec_cpus)
        # ...and expanded/switched into its own disjoint worker slice via VERL_TPU_SUBHOST_WORKER_CPUS.
        assert applied == list(plan.worker_cpus[rank])
        assert final == list(plan.worker_cpus[rank])
        assert seen.isdisjoint(final)
        seen.update(final)
