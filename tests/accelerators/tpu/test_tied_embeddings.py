# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for ``engines/tied_embeddings.py`` (``engine_kwargs.tpu.tie_embeddings``) and the tied weight's export.

Most tests use a small stand-in for a torchtitan ``Decoder`` and build it in the order of torchtitan's ``Trainer``:
on the meta device, then ``parallelize_fn``, ``to_empty()`` and ``init_weights()``. Their ``parallelize_fn`` is a
stand-in for SimpleFSDP's ``data_parallel``: it registers a new (sharded) ``Parameter`` on every module, which splits
a shared weight, and gives each module a ``weight`` property that all-gathers it. The last test builds torchtitan's TPU
Qwen3.5 instead and runs only where google-pytorch/torchtitan is installed.
"""

import dataclasses
import logging
import os
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from verl_hardware_plugin.accelerators.tpu.engines import tied_embeddings
from verl_hardware_plugin.accelerators.tpu.engines.tied_embeddings import (
    check_tied_embeddings,
    tie_embedding_weights,
    tie_model_spec,
    tied_weight_shape,
)
from verl_hardware_plugin.accelerators.tpu.engines.weight_export import TrainerWeights
from verl_hardware_plugin.accelerators.tpu.options import TPUEngineOptions

VOCAB, DIM = 8, 4


@dataclasses.dataclass
class _EmbeddingConfig:
    num_embeddings: int = VOCAB
    embedding_dim: int = DIM


@dataclasses.dataclass
class _LinearConfig:
    in_features: int = DIM
    out_features: int = VOCAB
    bias: bool = False


@dataclasses.dataclass
class _DecoderConfig:
    """The fields of torchtitan's ``Decoder.Config`` that tying reads."""

    tok_embeddings: _EmbeddingConfig = dataclasses.field(default_factory=_EmbeddingConfig)
    lm_head: _LinearConfig = dataclasses.field(default_factory=_LinearConfig)
    enable_weight_tying: bool = False

    def build(self) -> "_Decoder":
        return _Decoder(self)


@dataclasses.dataclass
class _ModelSpec:
    """The fields of torchtitan's ``ModelSpec`` that tying reads."""

    model: Any
    parallelize_fn: Any
    name: str = "tiny"


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


class _Decoder(nn.Module):
    """Ties like torchtitan's ``Decoder.__init__``, but its ``init_states`` does not re-tie (torchtitan's does), so
    the tests show that ``tie_model_spec`` restores the tie by itself."""

    def __init__(self, config: _DecoderConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.tok_embeddings.num_embeddings, config.tok_embeddings.embedding_dim)
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.lm_head = nn.Linear(config.lm_head.in_features, config.lm_head.out_features, bias=config.lm_head.bias)
        if config.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    def init_states(self, *, buffer_device: Optional[torch.device] = None) -> None:
        for module in self.modules():
            for p in module._parameters.values():
                if p is not None:
                    _local(p).zero_()

    def init_weights(self, **kwargs: Any) -> None:  # torchtitan's BaseModel alias, which its Trainer calls
        self.init_states(buffer_device=kwargs.get("buffer_device"))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lm_head(torch.tanh(self.proj(self.tok_embeddings(tokens))))


def _gathered_weight(module: nn.Module) -> torch.Tensor:
    weight = module._parameters["weight"]
    return weight.full_tensor() if isinstance(weight, DTensor) else weight * 1


def _split_like_data_parallel(mesh: Any = None):
    """A ``parallelize_fn``: one new ``Parameter`` per module (``Shard(0)`` DTensors on ``mesh``), read through a
    ``weight`` property that gathers it, as SimpleFSDP's ``data_parallel`` does."""
    from torch.distributed.tensor import Shard, distribute_tensor

    def parallelize(model: nn.Module, **kwargs: Any) -> nn.Module:
        for module in list(model.modules()):
            if module._parameters.get("weight") is None:
                continue
            weight = module._parameters["weight"].detach()
            new = distribute_tensor(weight, mesh, [Shard(0)]) if mesh is not None else weight.clone()
            module._parameters["weight"] = nn.Parameter(new)
            module.__class__ = type(
                f"Sharded{type(module).__name__}", (type(module),), {"weight": property(_gathered_weight)}
            )
        return model

    return parallelize


def _spec(tied: bool, parallelize_fn: Any) -> _ModelSpec:
    spec = _ModelSpec(model=_DecoderConfig(), parallelize_fn=parallelize_fn)
    return tie_model_spec(spec) if tied else spec


def _build_like_trainer(spec: Any) -> nn.Module:
    with torch.device("meta"):
        model = spec.model.build()
    model = spec.parallelize_fn(model, parallel_dims=None)
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    return model


def _load(model: nn.Module, embedding: torch.Tensor, proj: torch.Tensor, chunk: Optional[tuple[int, int]] = None):
    """Loads a tied checkpoint: the LM head gets the embedding (as torchtitan's ``from_hf`` does without one)."""
    values = {"tok_embeddings": embedding, "proj": proj, "lm_head": embedding}
    with torch.no_grad():
        for name, value in values.items():
            local = _local(getattr(model, name)._parameters["weight"])
            local.copy_(value.chunk(chunk[0])[chunk[1]] if chunk is not None else value)


def _grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: _local(module._parameters["weight"].grad).clone()
        for name, module in model.named_children()
        if module._parameters["weight"].grad is not None
    }


def _check_tied_matches_untied(tied: nn.Module, untied: nn.Module, tokens: torch.Tensor) -> None:
    """Same output, and the tied weight's gradient is the sum of the untied embedding's and LM head's."""
    out_tied, out_untied = tied(tokens), untied(tokens)
    torch.testing.assert_close(out_tied, out_untied, rtol=0, atol=0)
    out_tied.square().sum().backward()
    out_untied.square().sum().backward()
    g_tied, g_untied = _grads(tied), _grads(untied)
    shared = g_tied["lm_head"]
    assert shared.ne(0).any()
    torch.testing.assert_close(shared, g_untied["tok_embeddings"] + g_untied["lm_head"])
    torch.testing.assert_close(g_tied["proj"], g_untied["proj"], rtol=0, atol=0)
    # Both uses contribute to the one gradient.
    for part in ("tok_embeddings", "lm_head"):
        assert not torch.allclose(shared, g_untied[part])


def test_tied_weight_shape_and_unsupported_configs():
    assert tied_weight_shape(_DecoderConfig()) == (VOCAB, DIM)
    with pytest.raises(ValueError, match="differ in shape"):
        tied_weight_shape(_DecoderConfig(lm_head=_LinearConfig(out_features=VOCAB + 1)))
    with pytest.raises(ValueError, match="has a bias"):
        tied_weight_shape(_DecoderConfig(lm_head=_LinearConfig(bias=True)))
    with pytest.raises(ValueError, match="no enable_weight_tying field"):
        tied_weight_shape(SimpleNamespace(tok_embeddings=_EmbeddingConfig(), lm_head=_LinearConfig()))


def test_tie_model_spec_sets_weight_tying_and_wraps_parallelize_fn(caplog):
    def parallelize(model: nn.Module, **kwargs: Any) -> nn.Module:
        return model

    spec = _ModelSpec(model=_DecoderConfig(), parallelize_fn=parallelize)
    tied = tie_model_spec(spec)
    assert spec.model.enable_weight_tying and tied.model is spec.model
    assert tied.parallelize_fn is not parallelize and tied.parallelize_fn.__wrapped__ is parallelize
    assert tied.name == "tiny" and spec.parallelize_fn is parallelize
    assert "tie_word_embeddings=false" not in caplog.text

    with caplog.at_level(logging.WARNING, logger=tied_embeddings.__name__):
        tie_model_spec(_ModelSpec(model=_DecoderConfig(), parallelize_fn=parallelize), SimpleNamespace())
        assert "tie_word_embeddings=false" not in caplog.text
        tie_model_spec(
            _ModelSpec(model=_DecoderConfig(), parallelize_fn=parallelize), SimpleNamespace(tie_word_embeddings=False)
        )
    assert "tie_word_embeddings=false" in caplog.text


def test_tie_survives_parallelize_and_to_empty():
    """The parallelize stand-in and ``to_empty()`` each split the shared weight; one parameter is left at the end."""
    tied = _build_like_trainer(_spec(tied=True, parallelize_fn=_split_like_data_parallel()))
    untied = _build_like_trainer(_spec(tied=False, parallelize_fn=_split_like_data_parallel()))
    assert tied.tok_embeddings._parameters["weight"] is tied.lm_head._parameters["weight"]
    assert untied.tok_embeddings._parameters["weight"] is not untied.lm_head._parameters["weight"]
    # enable_weight_tying alone does not survive the build: the model's __init__ ties, the build splits.
    config_only = _ModelSpec(model=_DecoderConfig(enable_weight_tying=True), parallelize_fn=_split_like_data_parallel())
    split = _build_like_trainer(config_only)
    assert split.tok_embeddings._parameters["weight"] is not split.lm_head._parameters["weight"]

    num_untied = sum(p.numel() for p in untied.parameters())
    assert check_tied_embeddings([tied], torch.optim.AdamW(tied.parameters())) == num_untied - VOCAB * DIM
    with pytest.raises(RuntimeError, match="separate parameters"):
        check_tied_embeddings([untied])

    generator = torch.Generator().manual_seed(0)
    embedding, proj = torch.randn(VOCAB, DIM, generator=generator), torch.randn(DIM, DIM, generator=generator)
    _load(tied, embedding, proj)
    _load(untied, embedding, proj)
    _check_tied_matches_untied(tied, untied, torch.tensor([[0, 3, 5, 3, 7, 1]]))


def test_check_tied_embeddings_checks_the_optimizer():
    model = _DecoderConfig(enable_weight_tying=True).build()
    shared = model.lm_head.weight
    others = [model.proj.weight]
    container = SimpleNamespace(optimizers=[torch.optim.SGD(model.parameters(), lr=0.1)])  # OptimizersContainer
    assert check_tied_embeddings([model], container) == (VOCAB + DIM) * DIM
    with pytest.raises(RuntimeError, match="0 times"):
        check_tied_embeddings([model], torch.optim.SGD(others, lr=0.1))
    with pytest.raises(RuntimeError, match="2 times"):
        check_tied_embeddings(
            [model], SimpleNamespace(param_groups=[{"params": [shared, *others]}, {"params": [shared]}])
        )
    with pytest.raises(RuntimeError, match="no tok_embeddings and lm_head"):
        check_tied_embeddings([nn.Linear(2, 2)])


def test_tie_embedding_weights_refuses_what_cannot_be_shared(monkeypatch):
    model = _DecoderConfig().build()
    assert tie_embedding_weights(model) and model.tok_embeddings.weight is model.lm_head.weight
    assert not tie_embedding_weights(model)  # already one parameter

    model.lm_head.weight = nn.Parameter(torch.zeros(VOCAB, DIM + 1))
    with pytest.raises(RuntimeError, match="cannot tie"):
        tie_embedding_weights(model)

    model = _DecoderConfig().build()
    monkeypatch.setattr(tied_embeddings, "FSDPModule", nn.Embedding)  # FSDP2 manages tok_embeddings
    with pytest.raises(RuntimeError, match="FSDP2"):
        tie_embedding_weights(model)


def test_engine_exports_the_tied_weight_in_distinct_buffers_only_with_tie_embeddings(monkeypatch):
    """``get_per_tensor_param`` asks ``TrainerWeights`` for distinct buffers when ``tie_embeddings`` is set."""
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine

    monkeypatch.setattr(torchtitan_tpu, "get_device_id", lambda: "cpu")

    class _TiedAdapter:  # torchtitan's state-dict adapters leave lm_head.weight out of a tied model's HF state dict
        def to_hf(self, state_dict):
            names = {"tok_embeddings.weight": "model.embed_tokens.weight"}
            return {names.get(k, k): v for k, v in state_dict.items() if k != "lm_head.weight"}

    model = _DecoderConfig(enable_weight_tying=True).build()
    for tie, distinct in ((None, False), (False, False), (True, True)):
        engine = object.__new__(TorchTitanTPUEngine)
        engine.module = [model]
        engine._is_offload_param = False
        engine.checkpointer = SimpleNamespace(sd_adapter=_TiedAdapter())
        engine._torchtitan_name = "llama3"
        engine._tpu_options = TPUEngineOptions(tie_embeddings=tie)
        weights, _ = engine.get_per_tensor_param()
        assert weights._distinct_alias_buffers is distinct
        exported = dict(weights)
        assert sorted(exported) == ["lm_head.weight", "model.embed_tokens.weight", "proj.weight"]
        assert torch.equal(exported["lm_head.weight"], exported["model.embed_tokens.weight"])


def _sharded_worker(rank: int, world: int, init_file: str) -> None:
    """One rank of ``test_tie_with_sharded_parameters_and_the_tpu_sync_export``."""
    import torch.distributed as dist
    from torch.distributed.tensor import Replicate, init_device_mesh

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        tied = _build_like_trainer(_spec(tied=True, parallelize_fn=_split_like_data_parallel(mesh)))
        untied = _build_like_trainer(_spec(tied=False, parallelize_fn=_split_like_data_parallel(mesh)))
        shared = tied.lm_head._parameters["weight"]
        assert tied.tok_embeddings._parameters["weight"] is shared
        assert isinstance(shared, DTensor) and shared.to_local().shape == (VOCAB // world, DIM)
        num_untied = sum(p.numel() for p in untied.parameters())
        assert check_tied_embeddings([tied], torch.optim.AdamW(tied.parameters())) == num_untied - VOCAB * DIM

        generator = torch.Generator().manual_seed(0)  # the same full weights on every rank
        embedding, proj = torch.randn(VOCAB, DIM, generator=generator), torch.randn(DIM, DIM, generator=generator)
        _load(tied, embedding, proj, chunk=(world, rank))
        _load(untied, embedding, proj, chunk=(world, rank))
        _check_tied_matches_untied(tied, untied, torch.tensor([[rank, 3, 5, 3, 7, 1]]))

        # tpu-sync export: the tied weight under its two HF names. By default both names share one exported tensor;
        # with distinct_alias_buffers, local_shards() gives the second name its own copy of the same local shard.
        params = {"model.embed_tokens.weight": shared, "lm_head.weight": shared}
        [(_, a, _), (_, b, _)] = list(TrainerWeights(dict(params), [], "cpu").local_shards())
        assert a is b
        [(n1, a, s1), (n2, b, s2)] = list(
            TrainerWeights(dict(params), [], "cpu", distinct_alias_buffers=True).local_shards()
        )
        assert (n1, n2) == ("model.embed_tokens.weight", "lm_head.weight") and s1 == s2 == (world, rank)
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
        assert torch.equal(a, shared.to_local().to(torch.bfloat16))
        [(_, a), (_, b)] = list(TrainerWeights(dict(params), [], "cpu", distinct_alias_buffers=True))
        assert a is b  # plain iteration (verl, ray-plasma-store) is unchanged

        # Weights with different layouts cannot be one parameter.
        model = _DecoderConfig().build()
        model.tok_embeddings.weight = nn.Parameter(DTensor.from_local(torch.zeros(VOCAB, DIM), mesh, [Replicate()]))
        model.lm_head.weight = nn.Parameter(shared.detach())
        try:
            tie_embedding_weights(model)
            raise AssertionError("tie_embedding_weights must refuse differently sharded weights")
        except RuntimeError as e:
            assert "sharded differently" in str(e)
    finally:
        dist.destroy_process_group()


def test_tie_with_sharded_parameters_and_the_tpu_sync_export(tmp_path):
    """2 ranks: the tie survives sharding and ``to_empty()``, gradients add up, the export has distinct buffers."""
    torch.multiprocessing.start_processes(
        _sharded_worker, args=(2, str(tmp_path / "pg")), nprocs=2, join=True, start_method="fork"
    )


def _qwen3_5_worker(rank: int, world: int, init_file: str, dump_folder: str) -> None:
    """One rank of ``test_tied_qwen3_5_with_simple_fsdp``."""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torchtitan.components.checkpointer import ModelWrapper
    from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.tpu.qwen3_5 import model_registry

    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import get_tpu_simple_fsdp_parallelize_fn

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        vocab, seq_len = 512, 64
        parallel_dims = ParallelDims(dp_replicate=1, dp_shard=world, cp=1, tp=1, pp=1, ep=1, world_size=world)
        parallel_dims.build_mesh()

        def spec_and_model(tied: bool):
            spec = model_registry("debugmodel", seq_len=seq_len)
            spec.model.vocab_size = spec.model.tok_embeddings.num_embeddings = spec.model.lm_head.out_features = vocab
            spec = dataclasses.replace(spec, parallelize_fn=get_tpu_simple_fsdp_parallelize_fn("qwen3_5"))
            if tied:
                spec = tie_model_spec(spec)
            with torch.device("meta"):
                model = spec.model.build()
            model = spec.parallelize_fn(
                model,
                parallel_dims=parallel_dims,
                training=TrainingConfig(
                    max_context_length=seq_len, mixed_precision_param="float32", mixed_precision_reduce="float32"
                ),
                parallelism=ParallelismConfig(data_parallel_shard_degree=world, use_simple_fsdp=True),
                compile_config=CompileConfig(enable=False, backend="tpu"),
                ac_config=None,
                dump_folder=dump_folder,
            )
            model.to_empty(device="cpu")
            with torch.no_grad():
                model.init_weights(buffer_device=None)
            return spec, model

        spec_t, tied = spec_and_model(tied=True)
        spec_u, untied = spec_and_model(tied=False)
        shared = tied.lm_head._parameters["weight"]
        assert tied.tok_embeddings._parameters["weight"] is shared and isinstance(shared, DTensor)
        optimizers = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[ParamGroupConfig(pattern=r".*", optimizer_name="AdamW", optimizer_kwargs={"lr": 1e-3})],
        ).build(model_parts=[tied])
        dim = spec_t.model.dim
        assert check_tied_embeddings([tied], optimizers) == sum(p.numel() for p in untied.parameters()) - vocab * dim

        # Equal weights (a tied checkpoint loaded into both), different tokens per rank.
        with torch.no_grad():
            params_t = dict(tied.named_parameters())
            for name, p in untied.named_parameters():
                if name != "lm_head.weight":
                    _local(params_t[name]).copy_(_local(p))
            _local(untied.lm_head._parameters["weight"]).copy_(_local(untied.tok_embeddings._parameters["weight"]))
        # One packed row of two documents, as the engine feeds it.
        tokens = torch.randint(0, vocab, (seq_len,), generator=torch.Generator().manual_seed(rank))
        positions = torch.cat([torch.arange(seq_len // 4), torch.arange(seq_len - seq_len // 4)])
        out_t, out_u = tied(tokens, positions=positions), untied(tokens, positions=positions)
        torch.testing.assert_close(out_t, out_u, rtol=0, atol=0)
        out_t.float().square().mean().backward()
        out_u.float().square().mean().backward()
        g_u = {n: _local(p.grad) for n, p in untied.named_parameters() if p.grad is not None}
        g_t = {n: _local(p.grad) for n, p in tied.named_parameters() if p.grad is not None}
        assert "lm_head.weight" not in g_t and set(g_u) - set(g_t) == {"lm_head.weight"}  # one parameter
        torch.testing.assert_close(g_t["tok_embeddings.weight"], g_u["tok_embeddings.weight"] + g_u["lm_head.weight"])
        for name, grad in g_t.items():
            if name != "tok_embeddings.weight":
                torch.testing.assert_close(grad, g_u[name], rtol=0, atol=0)

        # HF layout: the tied model's state dict converts to the official tied layout (no lm_head.weight) ...
        adapter_t = spec_t.state_dict_adapter(spec_t.model, None)
        hf_t = adapter_t.to_hf(dict(tied.state_dict()))
        hf_u = spec_u.state_dict_adapter(spec_u.model, None).to_hf(dict(untied.state_dict()))
        assert "lm_head.weight" not in hf_t and set(hf_u) - set(hf_t) == {"lm_head.weight"}
        # ... and torchtitan reads that layout back into both modules.
        tt = adapter_t.from_hf(dict(hf_t))
        assert torch.equal(_local(tt["lm_head.weight"]), _local(tt["tok_embeddings.weight"]))

        # Weight sync: lm_head.weight is still sent, for a rollout with a separate LM head, from its own buffer.
        engine: Any = SimpleNamespace(checkpointer=SimpleNamespace(sd_adapter=adapter_t))
        engine._to_hf_named_params = lambda params: TorchTitanTPUEngine._to_hf_named_params(engine, params)
        chunks = TorchTitanTPUEngine._to_hf_named_param_chunks(engine, dict(tied.state_dict()))
        exported = {
            n: (t, s) for n, t, s in TrainerWeights(chunks, [], "cpu", distinct_alias_buffers=True).local_shards()
        }
        assert set(exported) == set(hf_u)
        (embed, embed_shard), (head, head_shard) = exported["model.embed_tokens.weight"], exported["lm_head.weight"]
        assert embed_shard == head_shard == (world, rank)
        assert torch.equal(embed, head) and embed.data_ptr() != head.data_ptr()

        # An optimizer step keeps the tie. The checkpoint holds the shared weight under both names, so it also loads
        # into an untied model (e.g. for offline conversion), whose embedding and LM head then both get it.
        before = _local(shared).clone()
        optimizers.step()
        assert tied.tok_embeddings._parameters["weight"] is shared and not torch.equal(_local(shared), before)
        checkpoint = os.path.join(dump_folder, "checkpoint")
        dcp.save(ModelWrapper([tied]).state_dict(), checkpoint_id=checkpoint)
        assert {"tok_embeddings.weight", "lm_head.weight"} <= set(
            FileSystemReader(checkpoint).read_metadata().state_dict_metadata
        )
        _, reloaded = spec_and_model(tied=False)
        wrapper = ModelWrapper([reloaded])
        state_dict = wrapper.state_dict()
        dcp.load(state_dict, checkpoint_id=checkpoint)
        wrapper.load_state_dict(state_dict)
        for module in (reloaded.tok_embeddings, reloaded.lm_head):
            assert torch.equal(_local(module._parameters["weight"]), _local(shared))
    finally:
        dist.destroy_process_group()


def test_tied_qwen3_5_with_simple_fsdp(tmp_path):
    """torchtitan's TPU Qwen3.5 with the plugin's SimpleFSDP parallelize_fn, tied, on 2 ranks (needs torchtitan)."""
    pytest.importorskip("torchtitan.experiments.tpu.qwen3_5")
    torch.multiprocessing.start_processes(
        _qwen3_5_worker,
        args=(2, str(tmp_path / "pg"), str(tmp_path / "dump")),
        nprocs=2,
        join=True,
        start_method="fork",
    )
