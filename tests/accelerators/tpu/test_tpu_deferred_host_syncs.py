# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the TorchTitan engine's deferred host syncs in training passes.

The engine's real training path runs on CPU with a toy model in place of the decoder: ``forward_backward_batch`` with
static micro-batch shapes, ``forward_step`` and verl's bypass-mode ``ppo_loss``, two updates. The TPU primitives are
recorders: ``tpu_synchronize`` (tensors or device-wide, wait flag) and the micro-batch fence's device events. A torch
function mode counts the host reads (``item()``, ``bool()``, ``cpu()``, ...) of tensors that would live on the TPU.

The tests that compare the deferred host reads with the eager ones need a verl that has ``verl.utils.metric.deferred``
and are skipped otherwise; the others run with any supported verl.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import importlib.util
import logging
import random
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.utils.losses import ppo_loss
from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu as engine_module

VOCAB = 64
_RECORD_DEVICE_EVENT = engine_module._record_device_event  # before any test replaces it
needs_deferred = pytest.mark.skipif(
    engine_module._deferred is None, reason="needs a verl with verl.utils.metric.deferred"
)


def _nested(ts):
    return torch.nested.nested_tensor(ts, layout=torch.jagged)


def _mini_batch(n, seed):
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    lengths = [(rng.randint(2, 20), 40 if rng.random() < 0.15 else rng.randint(1, 40)) for _ in range(n)]
    data = TensorDict(
        {
            "input_ids": _nested([torch.randint(1, VOCAB, (p + r,), generator=gen) for p, r in lengths]),
            "position_ids": _nested([torch.arange(p + r) for p, r in lengths]),
            "prompts": _nested([torch.randint(1, VOCAB, (p,), generator=gen) for p, _ in lengths]),
            "responses": _nested([torch.randint(1, VOCAB, (r,), generator=gen) for _, r in lengths]),
            "response_mask": _nested([torch.ones(r, dtype=torch.int64) for _, r in lengths]),
            "loss_mask": _nested(
                [torch.cat([torch.zeros(p, dtype=torch.int64), torch.ones(r, dtype=torch.int64)]) for p, r in lengths]
            ),
            "old_log_probs": _nested([(-3.5 - torch.rand(r, generator=gen)) for _, r in lengths]),
            "advantages": _nested([torch.randn(r, generator=gen) for _, r in lengths]),
        },
        batch_size=[n],
    )
    tu.assign_non_tensor(
        data,
        use_dynamic_bsz=True,
        use_remove_padding=True,
        pad_mode=DatasetPadMode.NO_PADDING,
        max_token_len_per_gpu=256,  # 4 micro-batches per update of 24 sequences
        global_batch_size=n,
        temperature=1.0,
    )
    return data


def _loss_function():
    """verl's bypass-mode ``ppo_loss`` with token-level IS, as the actor binds it (``functools.partial``)."""
    correction = RolloutCorrectionConfig(
        rollout_is="token",
        rollout_is_threshold=3.0,
        rollout_is_batch_normalize=False,
        rollout_rs=None,
        bypass_mode=True,
        loss_type="ppo_clip",
    )
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=16,
        ppo_micro_batch_size_per_gpu=4,
        loss_agg_mode="token-mean",
        use_kl_loss=False,
        entropy_coeff=0.0,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=1.0e9,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=correction),
    )
    return functools.partial(ppo_loss, config)


class _ToyLM(torch.nn.Module):
    """logits[t] = (E[token_t] + P[min(position_t, 63)]) @ W^T, in place of the decoder and the LM head."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(7)
        self.emb = torch.nn.Parameter(0.3 * torch.randn(VOCAB, 16, generator=gen))
        self.pos = torch.nn.Parameter(0.1 * torch.randn(64, 16, generator=gen))
        self.out = torch.nn.Parameter(0.3 * torch.randn(VOCAB, 16, generator=gen))

    def forward(self, tokens, positions=None):
        return (self.emb[tokens] + self.pos[positions.clamp(max=63)]) @ self.out.t()


class _Recorder:
    """The TPU primitives the engine calls, logged with the current phase and micro-batch."""

    def __init__(self):
        self.log = []
        self.phase = "init"
        self.micro_batch = -1
        self.events_fail = False


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()

    def tpu_synchronize(tensors=None, *, wait=True, strict=False):
        kind = "tensor" if isinstance(tensors, torch.Tensor) else "device"
        rec.log.append(("sync", kind, bool(wait), rec.phase, rec.micro_batch))

    class _Event:
        def __init__(self, index):
            self.index = index

        def synchronize(self):
            rec.log.append(("wait", self.index, rec.phase, rec.micro_batch))

    def record_device_event():
        if rec.events_fail:
            raise RuntimeError("no device events on this backend")
        rec.log.append(("record", rec.micro_batch, rec.phase, rec.micro_batch))
        return _Event(rec.micro_batch)

    monkeypatch.setattr(engine_module, "tpu_synchronize", tpu_synchronize)
    monkeypatch.setattr(engine_module, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(engine_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(engine_module, "_record_device_event", record_device_event)
    monkeypatch.setattr(engine_module._MicroBatchFence, "_disabled", False)

    # One forward_step per micro-batch: count the micro-batches there.
    cls = engine_module.TorchTitanTPUEngineWithLMHead
    forward_step = cls.forward_step

    def counted_forward_step(self, *args, **kwargs):
        rec.micro_batch += 1
        rec.log.append(("mb", rec.phase, rec.micro_batch))
        return forward_step(self, *args, **kwargs)

    monkeypatch.setattr(cls, "forward_step", counted_forward_step)
    return rec


def _engine(model):
    engine = object.__new__(engine_module.TorchTitanTPUEngineWithLMHead)
    engine.engine_config = SimpleNamespace(
        engine_kwargs={"tpu": {"seq_bucket_size": 32, "static_micro_batch_len": 256, "static_max_response_len": 48}},
        tensor_parallel_size=1,
        context_parallel_size=1,
    )
    engine.parallel_dims = SimpleNamespace(cp_enabled=False, pp_enabled=False, tp_enabled=False)
    engine._use_simple_fsdp, engine._tpu_eager_mode, engine._simple_fsdp_loss_scale = True, None, None
    engine._tpu_kernels = frozenset()
    engine.module = [model]
    engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=VOCAB))
    engine.trainer = SimpleNamespace(
        train_context=contextlib.nullcontext,
        config=SimpleNamespace(parallelism=SimpleNamespace(spmd_backend=None)),
    )
    engine.get_data_parallel_group = lambda: None
    engine.get_data_parallel_size = lambda: 1

    def zero_grad():
        for p in model.parameters():
            p.grad = None

    engine.optimizer_zero_grad = zero_grad
    return engine


def _comparable(output):
    def value(x):
        return (type(x).__name__, x.hex() if isinstance(x, float) else repr(x))

    metrics = {k: [value(x) for x in getattr(v, "values", [v])] for k, v in output["metrics"].items()}
    log_probs = output["model_output"]["log_probs"]
    return [value(x) for x in output["loss"]], metrics, [t.clone() for t in log_probs.unbind()]


def _train(rec, monkeypatch, mode):
    """Two training updates in host-scalar mode ``mode``: outputs and gradients per update."""
    monkeypatch.setattr(engine_module, "_HOST_SCALARS_MODE", mode)
    rec.log.clear()
    model = _ToyLM()
    engine = _engine(model)
    loss_function = _loss_function()
    updates = []
    for update, seed in enumerate((11, 12)):
        rec.phase, rec.micro_batch = f"train{update}", -1
        output = engine.forward_backward_batch(_mini_batch(24, seed), loss_function, forward_only=False)
        grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        updates.append((_comparable(output), grads))
        engine.optimizer_zero_grad()
    return updates


def _micro_batches(rec, phase):
    return sorted({e[2] for e in rec.log if e[0] == "mb" and e[1] == phase})


def _device_syncs(rec, phase, index):
    return [e[2] for e in rec.log if e[0] == "sync" and e[1] == "device" and e[3] == phase and e[4] == index]


def _tensor_syncs(rec, phase, index):
    return [e[2] for e in rec.log if e[0] == "sync" and e[1] == "tensor" and e[3] == phase and e[4] == index]


# Tensor methods that copy a device value to the host (and make the host wait for the device).
_HOST_READS = {"item", "__bool__", "__float__", "__int__", "__index__", "tolist", "numpy", "cpu", "nonzero"}


class _DeviceReadCounter(TorchFunctionMode):
    """Counts, per micro-batch, the host reads of tensors that would live on the TPU while ``active``: the model
    parameters, the loss inputs the engine moves to the device, and every tensor computed from them."""

    def __init__(self, rec):
        super().__init__()
        self.rec = rec
        self.active = False
        self.reads = collections.Counter()
        self._device = {}

    def mark(self, *objs):
        for obj in objs:
            for t in tree_flatten(obj)[0]:
                if isinstance(t, torch.Tensor):
                    self._device[id(t)] = t
                    if t.is_nested:
                        self.mark(t.values())

    def _on_device(self, t):
        return isinstance(t, torch.Tensor) and self._device.get(id(t)) is t

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        if name in _HOST_READS and args and self._on_device(args[0]):
            if self.active:
                self.reads[self.rec.micro_batch] += 1
        elif any(self._on_device(t) for t in tree_flatten((args, kwargs))[0]):
            out = func(*args, **kwargs)
            self.mark(out)
            return out
        return func(*args, **kwargs)


def _forward_step_reads(rec, monkeypatch, mode):
    """Host reads of device values inside each training ``forward_step`` of one update in ``mode``."""
    counter = _DeviceReadCounter(rec)
    cls = engine_module.TorchTitanTPUEngineWithLMHead
    forward_step, prepare = cls.forward_step, engine_module.prepare_tpu_loss_inputs

    def counted_forward_step(self, *args, **kwargs):
        counter.active = True
        try:
            return forward_step(self, *args, **kwargs)
        finally:
            counter.active = False

    def prepare_and_mark(model_output, micro_batch):
        loss_model_output, loss_data = prepare(model_output, micro_batch)
        counter.mark(*(loss_data[k] for k in ("response_mask", "old_log_probs", "advantages") if k in loss_data))
        return loss_model_output, loss_data

    with monkeypatch.context() as m:
        m.setattr(cls, "forward_step", counted_forward_step)
        m.setattr(engine_module, "prepare_tpu_loss_inputs", prepare_and_mark)
        m.setattr(engine_module, "_HOST_SCALARS_MODE", mode)
        model = _ToyLM()
        counter.mark(*model.parameters())
        rec.log.clear()
        rec.phase, rec.micro_batch = "count", -1
        with counter:
            _engine(model).forward_backward_batch(_mini_batch(24, 11), _loss_function(), forward_only=False)
    return [counter.reads[index] for index in _micro_batches(rec, "count")]


def _assert_same_updates(reference, other):
    for (out_r, grads_r), (out_o, grads_o) in zip(reference, other, strict=True):
        assert out_o[0] == out_r[0]  # losses: type and bits
        assert out_o[1] == out_r[1]  # every metric value
        assert all(torch.equal(a, b) for a, b in zip(out_o[2], out_r[2], strict=True))  # log-probs
        assert all(torch.equal(grads_o[n], grads_r[n]) for n in grads_r)


def test_the_verl_import_is_optional():
    """``verl.utils.metric.deferred`` is used when verl has it; otherwise the engine reads host scalars eagerly."""
    has_module = importlib.util.find_spec("verl.utils.metric.deferred") is not None
    assert (engine_module._deferred is not None) == has_module
    t = torch.tensor([2.5])
    assert engine_module._host_scalar(t) == 2.5  # outside a training pass: t.item()
    assert torch.equal(engine_module._host_copy(t), t.cpu())


def test_training_micro_batches_drain_only_at_the_end_of_the_loop(recorder, monkeypatch):
    _train(recorder, monkeypatch, "cut")
    for phase in ("train0", "train1"):
        indices = _micro_batches(recorder, phase)
        assert len(indices) >= 2
        for index in indices[:-1]:
            assert _device_syncs(recorder, phase, index) == [False]
        assert _device_syncs(recorder, phase, indices[-1]) == [True]


def test_the_fence_bounds_the_run_ahead_to_one_micro_batch(recorder, monkeypatch):
    _train(recorder, monkeypatch, "cut")
    for phase in ("train0", "train1"):
        indices = _micro_batches(recorder, phase)
        assert len(indices) >= 4
        events = [e for e in recorder.log if e[0] in ("record", "wait") and e[2] == phase]
        recorded = [e[1] for e in events if e[0] == "record"]
        assert recorded == indices[:-1]
        # After recording micro-batch k the host waits for k - 1 before it dispatches k + 1.
        waits = [(e[1], e[3]) for e in events if e[0] == "wait"]
        assert waits == [(k - 1, k) for k in indices[1:-1]]


@needs_deferred
@pytest.mark.parametrize("mode", ["cut", "fuse"])
def test_deferred_modes_match_the_eager_reads_bitwise(recorder, monkeypatch, mode):
    eager = _train(recorder, monkeypatch, "off")
    deferred = _train(recorder, monkeypatch, mode)
    _assert_same_updates(eager, deferred)


@needs_deferred
def test_training_forward_steps_read_no_device_value_on_the_host(recorder, monkeypatch):
    """The eager path reads device values in every training micro-batch; the deferred modes read none of them there."""
    eager = _forward_step_reads(recorder, monkeypatch, "off")
    assert len(eager) >= 2 and min(eager) > 10
    for mode in ("cut", "fuse"):
        assert _forward_step_reads(recorder, monkeypatch, mode) == [0] * len(eager)


@needs_deferred
def test_cut_mode_materializes_where_the_host_reads_were(recorder, monkeypatch):
    """``cut`` keeps a non-blocking materialization at every former read, ``fuse`` and ``off`` only the loss's."""
    eager_reads = _forward_step_reads(recorder, monkeypatch, "off")[0]
    per_mode = {}
    for mode in ("off", "cut", "fuse"):
        _train(recorder, monkeypatch, mode)
        per_mode[mode] = _tensor_syncs(recorder, "train0", _micro_batches(recorder, "train0")[0])
    assert per_mode["off"] == per_mode["fuse"] == [False]  # tpu_synchronize(loss, wait=False) before backward
    assert len(per_mode["cut"]) == 1 + eager_reads and not any(per_mode["cut"])


def test_without_verl_deferred_host_scalars_the_reads_stay_eager(recorder, monkeypatch):
    """With a verl that lacks ``verl.utils.metric.deferred``: eager host reads, the same results, still no drains."""
    eager = _train(recorder, monkeypatch, "off")
    monkeypatch.setattr(engine_module, "_deferred", None)
    for mode in ("cut", "fuse", "off"):
        monkeypatch.setattr(engine_module, "_HOST_SCALARS_MODE", mode)
        with engine_module._host_scalar_scope() as scope:
            assert scope is None
    fallback = _train(recorder, monkeypatch, "cut")
    _assert_same_updates(eager, fallback)
    for phase in ("train0", "train1"):
        indices = _micro_batches(recorder, phase)
        for index in indices:
            assert _tensor_syncs(recorder, phase, index) == [False]  # only the loss's
        for index in indices[:-1]:
            assert _device_syncs(recorder, phase, index) == [False]
        assert _device_syncs(recorder, phase, indices[-1]) == [True]
    eager_reads = _forward_step_reads(recorder, monkeypatch, "off")
    assert min(eager_reads) > 10 and _forward_step_reads(recorder, monkeypatch, "cut") == eager_reads


def test_forward_only_passes_keep_their_reads_and_syncs(recorder, monkeypatch):
    """Forward-only passes (log-probs) open no host-scalar scope and use neither the fence nor the drains."""
    outputs = {}
    for mode in ("off", "cut"):
        monkeypatch.setattr(engine_module, "_HOST_SCALARS_MODE", mode)
        recorder.log.clear()
        recorder.phase, recorder.micro_batch = "infer", -1
        engine = _engine(_ToyLM())
        with torch.no_grad():
            output = engine.forward_backward_batch(_mini_batch(24, 13), None, forward_only=True)
        assert len(_micro_batches(recorder, "infer")) >= 2
        assert not [e for e in recorder.log if e[0] in ("sync", "record", "wait")]
        outputs[mode] = output
    assert outputs["cut"]["loss"] == outputs["off"]["loss"] and all(type(x) is float for x in outputs["cut"]["loss"])
    log_probs = [o["model_output"]["log_probs"].unbind() for o in (outputs["off"], outputs["cut"])]
    assert all(torch.equal(a, b) for a, b in zip(*log_probs, strict=True))


def test_without_device_events_every_micro_batch_drains(recorder, monkeypatch, caplog):
    reference = _train(recorder, monkeypatch, "cut")
    recorder.events_fail = True
    with caplog.at_level(logging.WARNING, logger=engine_module.logger.name):
        fallback = _train(recorder, monkeypatch, "cut")
    assert "draining after every micro-batch instead" in caplog.text
    for phase in ("train0", "train1"):
        for index in _micro_batches(recorder, phase)[:-1]:
            assert _device_syncs(recorder, phase, index) == [False, True]
    _assert_same_updates(reference, fallback)


def test_without_a_tpu_device_every_micro_batch_drains(recorder, monkeypatch, caplog):
    """Only TPU events are used: elsewhere (here: CPU) the fence drains after every micro-batch from the start."""
    reference = _train(recorder, monkeypatch, "cut")
    monkeypatch.setattr(engine_module, "_record_device_event", _RECORD_DEVICE_EVENT)
    with pytest.raises(RuntimeError, match="need a TPU device, not cpu"):
        _RECORD_DEVICE_EVENT()
    with caplog.at_level(logging.WARNING, logger=engine_module.logger.name):
        on_cpu = _train(recorder, monkeypatch, "cut")
    assert "draining after every micro-batch instead" in caplog.text
    for phase in ("train0", "train1"):
        for index in _micro_batches(recorder, phase)[:-1]:
            assert _device_syncs(recorder, phase, index) == [False, True]
    _assert_same_updates(reference, on_cpu)
