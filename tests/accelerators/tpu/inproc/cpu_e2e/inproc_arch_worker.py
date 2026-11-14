# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""The vLLM worker of the in-process architecture-coverage CPU tests. Test code.

vLLM's real ``LLMEngine`` / ``InprocClient`` / ``UniProcExecutor`` / scheduler / KV-cache config drive this worker in
the trainer rank's own process, as vllm-torchtpu's ``TPUWorker`` is driven on TPU. The MODEL is real: ``load_model``
builds vLLM's own model class for the checkpoint's architecture (bf16, on CPU) and fills it through that model's
``load_weights`` from the checkpoint; ``get_kv_cache_spec`` returns that model's own layer specs (full attention,
sliding window, Mamba / Gated DeltaNet state). Only the device side is mocked: ``init_device`` reproduces
``TPUWorker.init_device``'s process effects for a TP=1 engine inside the trainer's world (gloo), the KV pool is
``kv_cache_memory_bytes``, and ``execute_model`` / ``sample_tokens`` return one deterministic token per request (vLLM's
CPU platform in a source tree has no attention kernels, so the model never runs a forward: these tests are about the
weights the engine holds).
"""

from __future__ import annotations

import glob
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from vllm.config import set_current_vllm_config
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

TOKEN_LO, TOKEN_SPAN = 1000, 40000


def mock_token(prompt: list, n_out: int) -> int:
    return TOKEN_LO + (sum(prompt[-4:]) * 31 + 7919 * n_out + 13) % TOKEN_SPAN


class ArchModelWorker(WorkerBase):
    def __init__(self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False, **kwargs):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.reqs: dict = {}
        self._pending = None
        self.arch = None
        self.kv_spec_kinds: dict = {}

    # -- vllm-torchtpu TPUWorker.init_device's process effects for a TP=1 engine in a trainer rank
    def init_device(self) -> None:
        from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment

        os.environ.setdefault("TORCH_TPU_XPROF_SESSION_ID", str(time.time_ns()))
        os.environ.pop("WORLD_SIZE", None)
        pc = self.parallel_config
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(os.environ.get("LOCAL_WORLD_SIZE") or pc.world_size)
        with set_current_vllm_config(self.vllm_config):
            init_distributed_environment(
                world_size=pc.world_size,
                rank=self.rank,
                local_rank=int(self.local_rank),
                distributed_init_method=self.distributed_init_method,
                backend="gloo",
            )
            ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        self.model_runner = SimpleNamespace(
            model=None, vllm_config=self.vllm_config, get_model=lambda: self.model_runner.model, drafter=None
        )

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.utils.torch_utils import set_default_torch_dtype

        vc = self.vllm_config
        mc = vc.model_config
        cls, arch = ModelRegistry.resolve_model_cls(mc.architectures, mc)
        with set_current_vllm_config(vc), set_default_torch_dtype(mc.dtype), torch.device("cpu"):
            model = cls(vllm_config=vc, prefix="")
        files = sorted(glob.glob(os.path.join(mc.model, "*.safetensors")))
        with set_current_vllm_config(vc), torch.no_grad():
            model.load_weights(safetensors_weights_iterator(files, use_tqdm_on_load=False))
        model.eval()
        self.arch = arch
        self.model_runner.model = model

    def get_kv_cache_spec(self):
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

        out = {}
        for name, layer in get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).items():
            try:
                spec = layer.get_kv_cache_spec(self.vllm_config)
            except Exception:  # noqa: BLE001 - the CPU attention backend's hook: the layer's plain full-attention spec
                spec = FullAttentionSpec(
                    block_size=self.vllm_config.cache_config.block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=self.vllm_config.model_config.dtype,
                )
            if spec is not None:
                out[name] = spec
                self.kv_spec_kinds[name] = type(spec).__name__
        return out

    def get_supported_kv_cache_layouts(self) -> list:
        return ["NHD"]

    def set_kv_cache_layout(self, kv_cache_layout: str) -> None:
        self.kv_cache_layout = kv_cache_layout

    def determine_available_memory(self) -> int:
        return int(self.cache_config.kv_cache_memory_bytes or 2**28)

    def initialize_from_config(self, kv_cache_config) -> None:
        self.kv_cache_config = kv_cache_config

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def get_supported_tasks(self):
        return ("generate",)

    def check_health(self) -> None:
        return None

    def reset_mm_cache(self) -> None:
        return None

    def reset_encoder_cache(self) -> None:
        return None

    def take_draft_token_ids(self):
        return None

    def get_model(self):
        return self.model_runner.model

    def execute_model(self, scheduler_output):
        for rid in scheduler_output.finished_req_ids:
            self.reqs.pop(rid, None)
        for nr in scheduler_output.scheduled_new_reqs:
            sp = nr.sampling_params
            self.reqs[nr.req_id] = {
                "prompt": list(nr.prompt_token_ids or []),
                "computed": int(nr.num_computed_tokens),
                "out": 0,
                "logprobs": sp is not None and sp.logprobs is not None,
            }
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            if rid in self.reqs:
                self.reqs[rid]["computed"] = int(cached.num_computed_tokens[i])
        req_ids, sampled, lp_ids, lps, ranks = [], [], [], [], []
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            st = self.reqs[rid]
            after = st["computed"] + int(n)
            req_ids.append(rid)
            if after >= len(st["prompt"]):
                tok = mock_token(st["prompt"], st["out"])
                st["out"] += 1
                sampled.append([tok])
                lp_ids.append([tok])
                lps.append([-0.25 - 0.01 * (st["out"] % 7)])
            else:
                sampled.append([])
                lp_ids.append([0])
                lps.append([0.0])
            ranks.append(1)
            st["computed"] = after
        logprobs = None
        if any(self.reqs[r]["logprobs"] for r in req_ids):
            logprobs = LogprobsLists(
                np.array(lp_ids, dtype=np.int64), np.array(lps, dtype=np.float32), np.array(ranks, dtype=np.int64)
            )
        self._pending = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=logprobs,
        )
        return None

    def sample_tokens(self, grammar_output):
        out, self._pending = self._pending, None
        return out

    def shutdown(self) -> None:
        return None
