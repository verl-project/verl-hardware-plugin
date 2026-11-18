# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of ``VERL_TPU_ACTIVATION_OFFLOAD`` (``engines/activation_offload.py``).

On CPU nothing is offloaded (only device tensors are), so ``_should_offload`` is widened to exercise the pack/unpack
round trip and the buffer pool.
"""

import torch
from torch import nn

from verl_hardware_plugin.accelerators.tpu.engines import activation_offload


class _CpuOffload(activation_offload.HostActivationOffload):
    def _should_offload(self, t):
        if isinstance(t, nn.Parameter) or id(t) in self._param_ids:
            return False
        nbytes = t.numel() * t.element_size()
        return self.min_bytes <= nbytes <= self.max_bytes and t.is_contiguous()


def _step(model, x, offload=None):
    ctx = offload.hooks() if offload is not None else torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t)
    with ctx:
        y = torch.tanh(model(x)) * x
    y.square().sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    return grads


def test_offload_round_trip_matches_and_skips_params():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
    x = torch.randn(32, 64)
    ref = _step(model, x)
    offload = _CpuOffload(model.parameters(), min_bytes=1024, max_bytes=1 << 30, max_inflight=1, pin_memory=False)
    packed = []
    orig_pack = offload.pack

    def spy(t):
        out = orig_pack(t)
        packed.append((t, out))
        return out

    offload.pack = spy
    got = _step(model, x, offload)
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g, r)
    offloaded = [o for _, o in packed if isinstance(o, activation_offload._Offloaded)]
    assert offloaded, "activations should have been offloaded"
    assert all(not isinstance(o, activation_offload._Offloaded) for t, o in packed if isinstance(t, nn.Parameter))
    assert offload.offloaded_bytes > 0
    offload.end_micro_batch()
    pooled = offload.stats()["pooled_gib"]
    assert pooled > 0
    # Second micro-batch reuses the pooled buffers.
    _step(model, x, offload)
    offload.end_micro_batch()
    assert offload.stats()["pooled_gib"] == pooled


def test_default_policy_keeps_cpu_tensors_and_env(monkeypatch):
    offload = activation_offload.HostActivationOffload([], min_bytes=0, pin_memory=False)
    t = torch.ones(1024)
    assert offload.pack(t) is t and offload.unpack(t) is t
    monkeypatch.setenv(activation_offload.ACTIVATION_OFFLOAD_ENV, "1")
    assert activation_offload.activation_offload_enabled()
