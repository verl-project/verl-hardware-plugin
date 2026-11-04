# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""The XPU `dist.all_reduce(op=AVG)` patch must cover verl's mixed "cpu:gloo,xpu:xccl" process groups."""

from unittest import mock

import pytest
import torch
import torch.distributed as dist

from verl_hardware_plugin.accelerators.xpu.patches import reduce_avg_allreduce_patch as xpu_patch_mod


@pytest.fixture(autouse=True)
def _reset_patch_state():
    original_all_reduce = dist.all_reduce
    xpu_patch_mod._applied = False
    yield
    dist.all_reduce = original_all_reduce
    xpu_patch_mod._applied = False


def _call_all_reduce(backend, op):
    fake_original = mock.MagicMock()
    tensor = torch.tensor([4.0, 8.0])
    with (
        mock.patch.object(xpu_patch_mod, "_xpu_available", return_value=True),
        mock.patch.object(dist, "all_reduce", fake_original),
        mock.patch.object(dist, "get_backend", return_value=backend),
        mock.patch.object(dist, "get_world_size", return_value=2),
    ):
        xpu_patch_mod.apply()
        dist.all_reduce(tensor, op=op, group="dp_group")
    return fake_original, tensor


@pytest.mark.parametrize("backend", ["xccl", "cpu:gloo,xpu:xccl"])
def test_avg_on_xccl_groups_becomes_sum_and_divide(backend):
    """verl's groups report "cpu:gloo,xpu:xccl"; AVG on them must be rewritten like on a plain xccl group."""
    fake_original, tensor = _call_all_reduce(backend, dist.ReduceOp.AVG)

    fake_original.assert_called_once_with(tensor, op=dist.ReduceOp.SUM, group="dp_group", async_op=False)
    assert tensor.tolist() == [2.0, 4.0]


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_avg_on_non_xccl_groups_is_untouched(backend):
    fake_original, tensor = _call_all_reduce(backend, dist.ReduceOp.AVG)

    fake_original.assert_called_once_with(tensor, op=dist.ReduceOp.AVG, group="dp_group", async_op=False)
    assert tensor.tolist() == [4.0, 8.0]


def test_sum_on_xccl_group_is_untouched():
    fake_original, tensor = _call_all_reduce("cpu:gloo,xpu:xccl", dist.ReduceOp.SUM)

    fake_original.assert_called_once_with(tensor, op=dist.ReduceOp.SUM, group="dp_group", async_op=False)
    assert tensor.tolist() == [4.0, 8.0]
