# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""The TPU `dist.all_reduce` patch must stay traceable by torch.compile (vLLM's compiled TP all-reduce)."""

import socket
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from verl_hardware_plugin.accelerators.tpu.patches import reduce_avg_allreduce_patch as tpu_patch_mod


@pytest.fixture(autouse=True)
def _reset_patch_state():
    original_all_reduce = dist.all_reduce
    original_all_gather_into_tensor = getattr(dist, "all_gather_into_tensor", None)
    tpu_patch_mod._applied = False
    yield
    dist.all_reduce = original_all_reduce
    if original_all_gather_into_tensor is not None:
        dist.all_gather_into_tensor = original_all_gather_into_tensor
    tpu_patch_mod._applied = False


def test_default_op_is_not_forwarded():
    """A call without `op` reaches the original all_reduce without `op`."""
    fake_original = mock.MagicMock()
    tensor = torch.tensor([3.0])
    with mock.patch.object(tpu_patch_mod, "_tpu_available", return_value=True):
        with mock.patch.object(dist, "all_reduce", fake_original):
            tpu_patch_mod.apply()
            dist.all_reduce(tensor, group="tp_group")

    fake_original.assert_called_once_with(tensor, group="tp_group", async_op=False)


def test_patched_all_reduce_is_traceable_by_torch_compile():
    """vLLM's compiled tensor-parallel all-reduce calls `dist.all_reduce(x, group=...)` inside torch.compile.

    Dynamo rewrites `all_reduce` into a functional collective and converts a ReduceOp to its string name only
    when the traced function is the current `dist.all_reduce` attribute. Forwarding `op=ReduceOp.SUM` from the
    wrapper therefore failed with "'RedOpType' object has no attribute 'lower'".
    """
    initialized_here = not dist.is_initialized()
    if initialized_here:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        with mock.patch.object(tpu_patch_mod, "_tpu_available", return_value=True):
            tpu_patch_mod.apply()

        def fn(x):
            y = x * 2
            dist.all_reduce(y, group=dist.group.WORLD)
            return y + 1

        torch._dynamo.reset()
        out = torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(4))
        assert out.tolist() == [3.0, 3.0, 3.0, 3.0]
    finally:
        torch._dynamo.reset()
        if initialized_here:
            dist.destroy_process_group()


def test_gloo_only_group_with_tpu_tensor_stages_via_cpu():
    class _FakeTPUTensor:
        device = mock.Mock(type="tpu")

        def __init__(self, val: float):
            self._cpu = torch.tensor([val])
            self.copied = None

        def detach(self):
            return self

        def cpu(self):
            return self._cpu.clone()

        def copy_(self, src):
            self.copied = src.clone()
            return self

    def _fake_orig_all_reduce(t, op=None, group=None, async_op=False):
        t.mul_(2.0)
        return "ok"

    fake_tpu_tensor = _FakeTPUTensor(6.0)
    with (
        mock.patch.object(tpu_patch_mod, "_tpu_available", return_value=True),
        mock.patch.object(dist, "all_reduce", side_effect=_fake_orig_all_reduce),
        mock.patch.object(dist, "get_backend", return_value="gloo"),
        mock.patch.object(dist, "get_world_size", return_value=2),
    ):
        tpu_patch_mod.apply()
        res = dist.all_reduce(fake_tpu_tensor, op=dist.ReduceOp.AVG, group="dp_gloo")

    assert res == "ok"
    assert fake_tpu_tensor.copied is not None
    assert torch.allclose(fake_tpu_tensor.copied, torch.tensor([6.0]))


def test_gloo_only_group_all_gather_into_tensor_stages_via_cpu():
    class _FakeTPUTensor:
        device = mock.Mock(type="tpu")

        def __init__(self, data: torch.Tensor):
            self._cpu = data.clone()

        def detach(self):
            return self

        def cpu(self):
            return self._cpu.clone()

        def copy_(self, src):
            self._cpu.copy_(src)
            return self

    def _fake_all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
        assert output_tensor.device.type == "cpu" and input_tensor.device.type == "cpu"
        output_tensor.copy_(torch.cat([input_tensor, input_tensor * 3], dim=0))
        return "ok"

    out, inp = _FakeTPUTensor(torch.zeros(4)), _FakeTPUTensor(torch.tensor([1.0, 2.0]))
    with (
        mock.patch.object(tpu_patch_mod, "_tpu_available", return_value=True),
        mock.patch.object(dist, "all_gather_into_tensor", side_effect=_fake_all_gather_into_tensor),
        mock.patch.object(dist, "get_backend", return_value="gloo"),
    ):
        tpu_patch_mod.apply()
        assert dist.all_gather_into_tensor(out, inp, group="dp_gloo") == "ok"
    assert out._cpu.tolist() == [1.0, 2.0, 3.0, 6.0]
