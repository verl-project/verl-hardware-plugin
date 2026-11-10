# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Rollout worker adapters that receive the trainer's weights over tpu-sync on TPU.

``TPUvLLMHttpServer`` passes :class:`vLLMTPUSyncWorkerExtension` to vLLM as ``worker_extension_cls``, and
``TPUSGLangHttpServer`` attaches :class:`SGLangTPUSyncWorker` to each SGLang TP scheduler process. When
``actor_rollout_ref.rollout.checkpoint_engine.backend=tpu-sync``, the driver
(``update_tpu_sync_weights`` in ``verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_checkpoint_engine``) calls
their methods on every rollout TP worker through ``collective_rpc``:

1. ``init_tpu_sync_on_worker``: allocate TP-sharded receive buffers for the tensors the trainer
   published and register them with the tpu-sync controller (first sync only).
2. ``install_tpu_sync_weights``: after the transfer, copy the received buffers to TPU HBM and fuse or
   transpose them into the rollout model's parameters (every sync).
3. ``get_model_weights_stats``: weight norms for the optional parity check.

``vLLMColocateWorkerExtension`` is imported lazily on first access to :class:`vLLMTPUSyncWorkerExtension` so SGLang
processes never import vLLM.
"""

import logging
import math
import os
import socket
import sys
import time
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import torch

from verl_hardware_plugin.accelerators.tpu.rollout import direct_install
from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_api
from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_checkpoint_engine as tpu_sync_engine
from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import ExactParityCheck, compute_tensor_stats
from verl_hardware_plugin.accelerators.tpu.weight_sync.vllm_layout import VllmParam
from verl_hardware_plugin.accelerators.tpu.weight_sync.weight_sync_backends import (
    get_parent_module,
    resolve_model_key,
    to_target_layout,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Row-parallel layers: vLLM / SGLang shard their input dimension (dim 1) across TP ranks; every other 2-D weight
# whose output dimension divides evenly is sharded on dim 0.
_ROW_PARALLEL_SUFFIXES = (".o_proj.weight", ".down_proj.weight")
# GQA key/value projections: sharded per KV head when KV heads are replicated (tp_size > num_kv_heads).
_KV_PROJ_SUFFIXES = (".self_attn.k_proj.weight", ".self_attn.v_proj.weight")


def get_ip(peer_address: Optional[str] = None) -> str:
    """Return this worker's routable IP address without importing vLLM unless already loaded."""
    if "vllm" in sys.modules:
        try:
            from vllm.utils.network_utils import get_ip as _vllm_get_ip

            ip = _vllm_get_ip()
            if isinstance(ip, str) and ip:
                return ip.strip("[]")
        except Exception:
            pass
    if "sglang" in sys.modules:
        try:
            from sglang.srt.utils.network import get_local_ip_auto

            ip = get_local_ip_auto()
            if isinstance(ip, str) and ip:
                return ip.strip("[]")
        except Exception:
            pass
    if peer_address:
        host, _, port_str = peer_address.rpartition(":")
        host = host.strip("[]")
        if host and port_str.isdigit():
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            try:
                with socket.socket(family, socket.SOCK_DGRAM) as sock:
                    sock.connect((host, int(port_str)))
                    return sock.getsockname()[0].strip("[]")
            except OSError:
                pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


class _SamplerSlice(NamedTuple):
    """How one rollout TP rank holds a trainer tensor (see ``_sampler_sharding``)."""

    sharding_spec: list[str]
    local_shape: list[int]
    mesh_shape: list[int]
    global_shard_indices: list[int]
    replicas: int


# vLLM and SGLang fuse these trainer tensors into one parameter: (fused suffix, source suffixes in concat order).
_FUSED_TARGETS = (
    (
        ".self_attn.qkv_proj.weight",
        (".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"),
    ),
    (
        ".self_attn.qkv_proj.bias",
        (".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"),
    ),
    (".mlp.gate_up_proj.weight", (".mlp.gate_proj.weight", ".mlp.up_proj.weight")),
    (".mlp.gate_up_proj.bias", (".mlp.gate_proj.bias", ".mlp.up_proj.bias")),
)


def _sampler_sharding(
    name: str,
    global_shape: list[int],
    tp_size: int,
    tp_rank: int = 0,
    num_kv_heads: Optional[int] = None,
    use_vllm_loaders: bool = False,
) -> _SamplerSlice:
    """Returns the slice of a trainer tensor that one rollout TP rank holds.

    With GQA and ``tp_size > num_kv_heads``, vLLM and SGLang replicate each KV head on ``tp_size // num_kv_heads``
    ranks, so a dim-0 split of k_proj / v_proj over ``tp_size`` would hand every rank a fraction of a head. Those are
    registered as one slice per KV head instead, and ``global_shard_indices`` names the head this rank holds
    (ranks sharing a head receive the same slice). E.g. Qwen3-32B (8 KV heads) at TP=32: k_proj [1024, 5120]
    -> mesh [8, 1], local [128, 5120], rank 13 -> slice 3.

    When ``use_vllm_loaders=True`` (hybrid or MoE models), every >=2D tensor whose dim 0 divides evenly is sharded
    on dim 0 (and all-gathered at install time), while 1-D tensors are received whole.
    """
    spec_axes = [""] * len(global_shape)
    local_shape = list(global_shape)
    tp_axis_size = tp_size  # number of slices along the "tp" mesh axis
    global_shard_indices: list[int] = []  # empty: tpu-sync derives this rank's slice from the mesh geometry
    if use_vllm_loaders:
        if len(global_shape) >= 2 and global_shape[0] % tp_size == 0 and tp_size > 1:
            spec_axes[0] = "tp"
            local_shape[0] = global_shape[0] // tp_size
    elif len(global_shape) == 2:
        if any(name.endswith(s) for s in _ROW_PARALLEL_SUFFIXES):
            # Row parallel: partition input dimension (dim 1) across TP ranks
            spec_axes = ["", "tp"]
            local_shape[1] = global_shape[1] // tp_size
        elif num_kv_heads is not None and num_kv_heads < tp_size and name.endswith(_KV_PROJ_SUFFIXES):
            # GQA with replicated KV heads (see above): one slice per KV head.
            spec_axes = ["tp", ""]
            tp_axis_size = num_kv_heads
            local_shape[0] = global_shape[0] // num_kv_heads
            global_shard_indices = [tp_rank // (tp_size // num_kv_heads)]
        elif global_shape[0] % tp_size == 0:
            # Column parallel or vocab parallel: partition output dimension (dim 0) across TP ranks
            spec_axes = ["tp", ""]
            local_shape[0] = global_shape[0] // tp_size
    mesh_shape = [tp_axis_size if axis == "tp" else 1 for axis in spec_axes]
    # Number of TP ranks holding this same slice: 1 for a plain split, tp_size // num_kv_heads for GQA k/v,
    # tp_size for unsharded tensors. The norm parity check divides by it when summing over ranks.
    replicas = tp_size // math.prod(mesh_shape)
    return _SamplerSlice(spec_axes, local_shape, mesh_shape, global_shard_indices, replicas)


def _slice_from_spec(spec: list[str], local_shape: list[int], tp_size: int) -> _SamplerSlice:
    """``_SamplerSlice`` of a tensor split evenly over the TP ranks along the ``"tp"`` dimension of ``spec``."""
    mesh_shape = [tp_size if axis == "tp" else 1 for axis in spec]
    return _SamplerSlice(spec, local_shape, mesh_shape, [], tp_size // math.prod(mesh_shape))


def _uses_vllm_weight_loader(vllm_model: Any, global_shapes: dict[str, list[int]]) -> bool:
    """Whether to install received weights via the engine's own checkpoint loaders (hybrid linear-attn or MoE)."""
    return (
        hasattr(type(vllm_model), "get_mamba_state_shape_from_config")
        or any(len(shape) not in (1, 2) for shape in global_shapes.values())
        or any(".linear_attn." in name or ".experts." in name for name in global_shapes)
    )


class TPUSyncWorkerMixin:
    """Shared tpu-sync P2P weight receiver and parameter installer for TPU rollout workers (vLLM and SGLang)."""

    def _get_vllm_model(self) -> Any:
        """Extract the underlying ``nn.Module`` from the rollout worker."""
        worker: Any = getattr(self, "worker", self)
        return worker.model_runner.model

    def _get_tp_size(self) -> Optional[int]:
        """Return this rollout replica's tensor-parallel world size."""
        worker: Any = getattr(self, "worker", self)
        parallel_config = getattr(getattr(worker, "vllm_config", None), "parallel_config", None)
        return getattr(parallel_config, "tensor_parallel_size", None)

    def _get_model_config(self) -> Any:
        """Return the rollout engine's model config (exposing ``hf_text_config`` and ``get_total_num_kv_heads``)."""
        worker: Any = getattr(self, "worker", self)
        return getattr(getattr(worker, "vllm_config", None), "model_config", None)

    def init_tpu_sync_on_worker(
        self,
        controller_address: str,
        global_shapes: dict[str, list[int]],
        parallelism: int = tpu_sync_engine.TPU_SYNC_DEFAULT_PARALLELISM,
        job_name: str = "sampler0",
        receive_into_vllm: bool = False,
    ) -> bool:
        """Initialize tpu-sync WeightSynchronizer listener and register with the central tpu-sync controller.

        ``controller_address`` is the ``host:port`` of the driver's tpu-sync controller and ``global_shapes`` the
        full shape of every tensor the trainer sends; the driver passes both, because rollout ``mp`` workers are
        not Ray processes. ``job_name`` is the tpu-sync job this worker's replica registers under (the driver
        names every rollout replica after its ``replica_rank``), so the ranks ``0..TP-1`` of several replicas do
        not collide on the controller. With ``receive_into_vllm`` the trainer sends every tensor in vLLM's layout
        (named after vLLM's parameters, see ``weight_sync/vllm_layout.py``) and this worker binds those parameters
        as the receive buffers: no staging buffers and nothing to install. Only the first call does work: the
        receive buffers, the synchronizer and the registration are reused by every later sync.
        """
        if getattr(self, "_tpu_sync_ws", None) is not None:
            return True

        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            logger.warning("tpu-sync Sampler: could not locate rollout model to bind parameters.")
            return False

        bind_ip = get_ip(controller_address).strip("[]")
        rank_val = getattr(self, "rank", 0)
        # The listener port is assigned by the OS, as on the trainer side, and registered with the controller
        # below; fixed ``base + rank`` ports collided when two replicas shared a host.
        listener_port = 0

        controller_addr, global_shapes_map = controller_address, global_shapes
        if not controller_addr or not global_shapes_map:
            raise RuntimeError(
                f"tpu-sync Sampler Rank {rank_val}: the driver passed no tpu-sync controller address "
                f"({controller_addr!r}) or no trainer tensor shapes ({len(global_shapes_map or {})} entries)"
            )

        # 2. Determine tensor parallel size from the rollout worker config
        tp_size = self._get_tp_size()
        if not tp_size or tp_size <= 0:
            raise RuntimeError(
                f"tpu-sync Sampler Rank {rank_val}: Unable to determine tensor_parallel_size (tp_size) "
                f"from worker.vllm_config.parallel_config!"
            )

        if receive_into_vllm:
            return self._init_receive_into_vllm(
                vllm_model, controller_addr, global_shapes_map, tp_size, parallelism, job_name, bind_ip, listener_port
            )

        # Models with a direct-install layout (``direct_install``: e.g. Qwen3.5, Qwen3 MoE) install each rank's slices
        # straight into the rollout parameters; other hybrid or MoE models go through the engine's loaders, which
        # needs every tensor whole on every rank.
        model_config = self._get_model_config()
        hf_text_config = getattr(model_config, "hf_text_config", None)
        layout, plan, no_plan_reasons = direct_install.select_layout(
            vllm_model, global_shapes_map, hf_text_config, tp_size, rank_val
        )
        use_vllm_loaders = layout is None and _uses_vllm_weight_loader(vllm_model, global_shapes_map)
        self._tpu_sync_layout = layout
        self._tpu_sync_plan = plan
        self._tpu_sync_use_vllm_loaders = use_vllm_loaders
        # The loaders need the trainer's tensors back in HF's layout (``DirectInstallLayout.loader_weights``).
        self._tpu_sync_hf_text_config = hf_text_config
        loader_error = direct_install.loader_error(global_shapes_map, hf_text_config) if layout is None else ""
        if loader_error:
            raise RuntimeError(f"tpu-sync Sampler Rank {rank_val}: {loader_error}")
        install_mode = (
            f"{layout.name} direct" if layout is not None else "vllm loaders" if use_vllm_loaders else "direct"
        )
        logger.info(f"tpu-sync Sampler Rank {rank_val}: install mode {install_mode}")
        if use_vllm_loaders:
            # Every sync then all-gathers the whole model on every rank and re-runs the engine's loaders (slow).
            logger.warning(
                f"tpu-sync Sampler Rank {rank_val}: installing through checkpoint loaders (TP all-gather of "
                f"every tensor); no direct install: {'; '.join(no_plan_reasons)}"
            )

        # 3. Allocate local TPU staging buffers matching Trainer un-fused layout and sharding specs
        # TODO(tpu): engine_kwargs.tpu-sync.receive_into_vllm binds vLLM's parameters directly instead (the trainer
        # exports in vLLM's layout, weight_sync/vllm_layout.py); dense models with FSDP == TP for now. The staging
        # tensors are a second full copy of this rank's TP shard (~1.9 GiB per chip for Qwen3-8B, ~7.6 GiB for
        # Qwen3-32B at TP=8) that must fit next to vLLM's preallocated weights and KV cache on every sync.
        staging_tensors = {}
        staging_sharded = {}
        variable_protos = []
        valid_params = []
        replicas = {}
        # Total KV heads of the model, to slice k_proj / v_proj per KV head when KV heads are replicated. rank_val is
        # this worker's TP rank: a single rollout replica without pipeline parallelism, as the registration below
        # ("sampler/<rank>" on mesh [1, tp_size]) already assumes.
        num_kv_heads = model_config.get_total_num_kv_heads()
        # Tensors received straight into the rollout weights instead of an HBM staging buffer (see
        # DirectInstallLayout.zero_copy_targets).
        zero_copy: dict[str, torch.Tensor] = {}
        if layout is not None and layout.zero_copy_targets is not None:
            init_sd = vllm_model.state_dict()

            def _find_init_key(k: str) -> Optional[str]:
                resolved = resolve_model_key(k, init_sd)
                return resolved if resolved in init_sd else None

            zero_copy = layout.zero_copy_targets(plan, init_sd, _find_init_key, global_shapes_map)

        for idx, (name, g_shape) in enumerate(sorted(global_shapes_map.items(), key=lambda x: x[0])):
            g_shape = list(g_shape)
            rule = layout.sharding_rule(name, g_shape, tp_size, plan) if layout is not None else None
            if rule is not None:
                shard = _slice_from_spec(*rule, tp_size)
            else:
                shard = _sampler_sharding(
                    name,
                    g_shape,
                    tp_size,
                    tp_rank=rank_val,
                    num_kv_heads=num_kv_heads,
                    use_vllm_loaders=use_vllm_loaders,
                )
            replicas[name] = shard.replicas

            # Allocate a local staging buffer with matching layout (the trainer sends bf16), or receive straight
            # into the rollout weight.
            t = zero_copy.get(name)
            if t is not None:
                if list(t.shape) != list(shard.local_shape):
                    raise RuntimeError(
                        f"tpu-sync Sampler Rank {rank_val}: {name} slice {shard.local_shape} != rollout weight "
                        f"{list(t.shape)}"
                    )
            else:
                device = torch.device(tpu_sync_engine.TPU_SYNC_DEVICE)
                t = torch.empty(shard.local_shape, dtype=torch.bfloat16, device=device)
            staging_tensors[name] = t
            staging_sharded[name] = "tp" in shard.sharding_spec
            valid_params.append((name, t))

            variable_protos.append(
                tpu_sync_api.variable_metadata(
                    name=name,
                    shape=g_shape,
                    mesh_shape=shard.mesh_shape,
                    layout=list(range(len(shard.local_shape) - 1, -1, -1)),
                    item_size=t.element_size(),
                    layer_idx=idx,
                    sharding_spec=shard.sharding_spec,
                    global_shard_indices=shard.global_shard_indices,
                )
            )

        self._tpu_sync_staging = staging_tensors
        self._tpu_sync_staging_sharded = staging_sharded
        self._sorted_vllm_params = valid_params
        self._tpu_sync_replicas = replicas
        if zero_copy:
            logger.info(
                f"tpu-sync Sampler Rank {rank_val}: {len(zero_copy)} tensors "
                f"({sum(t.numel() * t.element_size() for t in zero_copy.values()) / 2**30:.2f} GiB) are received "
                "straight into the rollout weights (no staging copy)"
            )

        tpu_sync_engine.tpu_synchronize()

        logger.info(
            f"tpu-sync Sampler Rank {rank_val}: binding {len(valid_params)} staging tensors to WeightSynchronizer "
            f"(sample: name={valid_params[0][0]}, shape={valid_params[0][1].shape}, "
            f"total numel={sum(t.numel() for _, t in valid_params)})"
        )

        self._tpu_sync_ws = tpu_sync_engine.create_torch_weight_synchronizer(
            [[t] for _, t in valid_params],
            local_port=0,
            parallelism=parallelism,
            listener_port=listener_port,
            bind_ip=bind_ip,
        )

        # Whether a tensor can skip the CPU (de)tiling pass is decided by tpu-sync's planner per transfer, from the
        # source and destination slices together, and delivered to this listener with the transfer. Do not set it
        # here from the local shape alone: both ends must make the same choice, and when they disagree (the
        # sender de-tiles to row-major, h2d() copies those bytes raw into tiled HBM) the weights come out permuted
        # while every norm stays unchanged. With full trainer tensors the local guess usually matches the planner,
        # but not once the trainer sends shards that are not 8-row aligned (e.g. Qwen3's embedding at 32+ FSDP
        # ranks).

        try:
            ctrl_client = tpu_sync_api.controller_client(controller_addr)
            unit_id = tpu_sync_api.unit_id(job_name, str(rank_val))
            ctrl_client.register_work_unit(
                unit_id,
                [f"{bind_ip}:{self._tpu_sync_ws.local_port}"],
                f"{bind_ip}:{self._tpu_sync_ws.listener_port}",
                mesh_shape=[1, tp_size],
                variables=variable_protos,
                mesh_axes=["fsdp", "tp"],
            )
        except Exception as e:
            logger.error(f"tpu-sync Sampler Rank {rank_val} failed to register with the tpu-sync controller: {e}")
            raise
        logger.info(
            f"tpu-sync Sampler Rank {rank_val} bound {len(valid_params)} dynamic staging tensors and registered "
            f"directly with the tpu-sync controller ({controller_addr}) as {job_name}/{rank_val}: "
            f"mesh_shape=[1, {tp_size}], "
            f"data_port={self._tpu_sync_ws.local_port}, listener_port={self._tpu_sync_ws.listener_port}"
        )

        return True

    @torch.no_grad()
    def describe_vllm_params(self) -> dict:
        """vLLM's parameters as this rank holds them, for the trainer's export in vLLM's layout (``receive_into_vllm``).

        Returns ``{"tp_size": TP, "params": {name: VllmParam.to_dict()}}`` over ``vllm_model.state_dict()``;
        ``flipped`` is vllm-torchtpu's ``_tpu_weight_flipped`` on the owning module (the weight is stored transposed,
        ``(in, out)``). A parameter of a fused module (one the model class's ``packed_modules_mapping`` lists, e.g.
        ``qkv_proj`` <- ``q_proj, k_proj, v_proj``) carries its HF source names in order, the module's global
        ``output_sizes`` and its ``num_kv_head_replicas``, so the trainer needs no model knowledge. Parameters that
        share storage (a tied ``lm_head``) are listed once.
        """
        vllm_model = self._get_vllm_model()
        worker: Any = getattr(self, "worker", self)
        tp_size = int(worker.vllm_config.parallel_config.tensor_parallel_size)
        model_sd = vllm_model.state_dict()
        modules = dict(vllm_model.named_modules())
        packed = dict(getattr(vllm_model, "packed_modules_mapping", None) or {})
        params: dict[str, dict] = {}
        seen_storage: set[int] = set()
        for key in sorted(model_sd):
            t = model_sd[key]
            t = t.to_local() if hasattr(t, "to_local") else t
            if t.data_ptr() in seen_storage:
                continue
            seen_storage.add(t.data_ptr())
            module = get_parent_module(key, modules)
            flipped = bool(getattr(module, "_tpu_weight_flipped", False))
            module_name, _, leaf = key.rpartition(".")
            owner, _, module_leaf = module_name.rpartition(".")
            sources: tuple[str, ...] = ()
            output_sizes: tuple[int, ...] = ()
            kv_replicas = 1
            if module_leaf in packed:
                prefix = f"{owner}." if owner else ""
                sources = tuple(f"{prefix}{src}.{leaf}" for src in packed[module_leaf])
                output_sizes = tuple(int(s) for s in getattr(module, "output_sizes", ()) or ())
                kv_replicas = int(getattr(module, "num_kv_head_replicas", 1) or 1)
            params[key] = VllmParam(tuple(t.shape), str(t.dtype), flipped, sources, output_sizes, kv_replicas).to_dict()
        return {"tp_size": tp_size, "params": params}

    def _init_receive_into_vllm(
        self,
        vllm_model: Any,
        controller_addr: str,
        global_shapes: dict[str, list[int]],
        tp_size: int,
        parallelism: int,
        job_name: str,
        bind_ip: str,
        listener_port: int,
    ) -> bool:
        """``init_tpu_sync_on_worker`` for ``receive_into_vllm``: bind vLLM's parameters as the receive buffers.

        ``global_shapes`` is keyed by vLLM parameter name and holds the trainer's virtual global shape of each (the
        rollout ranks' local weights concatenated along the TP dim). A dim whose global size is ``tp_size`` times
        the local size is the TP dim; equal sizes are replicated. No model knowledge is needed here.
        """
        rank_val = getattr(self, "rank", 0)
        model_sd = vllm_model.state_dict()
        bound: dict[str, torch.Tensor] = {}
        replicas: dict[str, int] = {}
        variable_protos = []
        problems = []
        for idx, (name, g_shape) in enumerate(sorted(global_shapes.items(), key=lambda x: x[0])):
            key = resolve_model_key(name, model_sd)
            if key not in model_sd:
                problems.append(f"{name}: no vLLM parameter")
                continue
            t = model_sd[key]
            t = t.to_local() if hasattr(t, "to_local") else t
            if not (t.is_contiguous() and t.storage_offset() == 0 and t.dim() == len(g_shape)):
                problems.append(f"{name}: vLLM weight is not a contiguous whole buffer of rank {len(g_shape)}")
                continue
            spec, mesh = [""] * len(g_shape), [1] * len(g_shape)
            for d, (g, l_) in enumerate(zip(g_shape, t.shape, strict=True)):
                if g == l_ * tp_size and tp_size > 1:
                    spec[d], mesh[d] = "tp", tp_size
                elif g != l_:
                    problems.append(f"{name}: global {list(g_shape)} vs vLLM local {list(t.shape)} at TP={tp_size}")
                    break
            else:
                if spec.count("tp") > 1:
                    problems.append(f"{name}: more than one TP dim in {list(g_shape)} vs {list(t.shape)}")
                    continue
                bound[name] = t
                replicas[name] = tp_size // math.prod(mesh)
                variable_protos.append(
                    tpu_sync_api.variable_metadata(
                        name=name,
                        shape=list(g_shape),
                        mesh_shape=mesh,
                        layout=list(range(len(g_shape) - 1, -1, -1)),
                        item_size=t.element_size(),
                        layer_idx=idx,
                        sharding_spec=spec,
                        global_shard_indices=[],
                    )
                )
        if problems:
            raise RuntimeError(
                f"tpu-sync Sampler Rank {rank_val}: receive_into_vllm cannot bind {len(problems)} tensors: "
                + "; ".join(problems[:8])
            )

        valid_params = list(bound.items())
        self._tpu_sync_staging = dict(bound)  # the parity and install paths read the received tensors from here
        self._tpu_sync_staging_sharded = {
            n: "tp" in v.sharding_spec for n, v in zip(bound, variable_protos, strict=True)
        }
        self._sorted_vllm_params = valid_params
        self._tpu_sync_replicas = replicas
        self._tpu_sync_bound_names = set(bound)
        self._tpu_sync_layout = None
        self._tpu_sync_plan = None
        self._tpu_sync_use_vllm_loaders = False
        tpu_sync_engine.tpu_synchronize()
        logger.info(
            f"tpu-sync Sampler Rank {rank_val}: receive_into_vllm binds {len(bound)} vLLM parameters "
            f"({sum(t.numel() * t.element_size() for t in bound.values()) / 2**30:.2f} GiB) as the receive buffers"
        )

        self._tpu_sync_ws = tpu_sync_engine.create_torch_weight_synchronizer(
            [[t] for _, t in valid_params],
            local_port=0,
            parallelism=parallelism,
            listener_port=listener_port,
            bind_ip=bind_ip,
        )
        ctrl_client = tpu_sync_api.controller_client(controller_addr)
        ctrl_client.register_work_unit(
            tpu_sync_api.unit_id(job_name, str(rank_val)),
            [f"{bind_ip}:{self._tpu_sync_ws.local_port}"],
            f"{bind_ip}:{self._tpu_sync_ws.listener_port}",
            mesh_shape=[1, tp_size],
            variables=variable_protos,
            mesh_axes=["fsdp", "tp"],
        )
        return True

    def install_tpu_sync_weights(self, exact_parity: bool = False) -> dict:
        """Install received weights from host staging buffer into TPU HBM via zero-copy H2D DMA
        and fuse/transpose them directly into the rollout model parameters.

        Args:
            exact_parity: Set by the driver on the step-0 sync with ``verify_parity=exact|all``: compare every
                received tensor bit for bit with the rollout parameter it overwrites (``ExactParityCheck``).

        Returns:
            Per-worker timings in seconds: ``total`` (whole install), ``h2d`` (pure ``_tpu_sync_ws.h2d()``)
            and ``sync`` (final TPU sync barrier), plus ``exact_parity`` (this rank's result) when requested.
            Empty if the synchronizer is not initialized.
        """
        if getattr(self, "_tpu_sync_ws", None) is None:
            logger.warning("tpu-sync Sampler: install_tpu_sync_weights called before _tpu_sync_ws was initialized.")
            return {}

        rank = getattr(self, "rank", 0)
        t_start = time.perf_counter()
        # receive_into_vllm: the H2D writes vLLM's parameters itself, so the step-0 exact check compares them with a
        # host copy taken before it (the checkpoint vLLM loaded).
        bound_names: set[str] = getattr(self, "_tpu_sync_bound_names", set())
        bound_before = (
            {name: self._tpu_sync_staging[name].detach().to("cpu", copy=True) for name in sorted(bound_names)}
            if exact_parity and bound_names
            else {}
        )
        t_h2d_start = time.perf_counter()
        # h2d() applies the skip_tiling plan the listener received with this transfer (see
        # init_tpu_sync_on_worker); nothing must override it here. With receive_into_vllm it writes vLLM's
        # parameters directly and nothing is installed below; it must stay inside the rollout pause.
        self._tpu_sync_ws.h2d()
        t_h2d = time.perf_counter() - t_h2d_start

        if bound_names:
            parity = ExactParityCheck() if exact_parity else None
            if parity is not None:
                for name, before in bound_before.items():
                    parity.check(name, before, self._tpu_sync_staging[name].detach().to("cpu", copy=True))
            t_sync_start = time.perf_counter()
            tpu_sync_engine.tpu_synchronize()
            t_sync = time.perf_counter() - t_sync_start
            t_total = time.perf_counter() - t_start
            logger.info(
                f"[TPU-SYNC TELEMETRY | Sampler Worker] Sampler Rank {rank}: install_tpu_sync_weights "
                f"(receive_into_vllm, nothing to install) completed in {t_total:.4f}s "
                f"(H2D={t_h2d:.4f}s, TPUSyncBarrier={t_sync:.4f}s)"
            )
            result: dict[str, Any] = {"total": t_total, "h2d": t_h2d, "sync": t_sync}
            if parity is not None:
                result["exact_parity"] = parity.result(rank, [])
            return result

        if getattr(self, "_tpu_sync_use_vllm_loaders", False):
            self._install_with_vllm_loaders()
            t_sync_start = time.perf_counter()
            tpu_sync_engine.tpu_synchronize()
            t_sync = time.perf_counter() - t_sync_start
            t_total = time.perf_counter() - t_start
            logger.info(
                f"[TPU-SYNC TELEMETRY | Sampler Worker] Sampler Rank {rank}: "
                f"install_tpu_sync_weights (load_weights) completed in {t_total:.4f}s "
                f"(H2D={t_h2d:.4f}s, TPUSyncBarrier={t_sync:.4f}s)"
            )
            result = {"total": t_total, "h2d": t_h2d, "sync": t_sync}
            if exact_parity:
                # Checkpoint loaders re-shard and re-layout the tensors themselves; there is no 1:1 target to compare.
                result["exact_parity"] = {"rank": rank, "skipped": "not supported with the vLLM-loader install"}
            return result

        # Unpack, fuse, and transpose staging tensors into rollout model parameters
        vllm_model = self._get_vllm_model()
        model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else vllm_model.model.state_dict()
        module_dict = dict(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else {}
        parity = ExactParityCheck() if exact_parity else None

        def find_model_key(k: str) -> Optional[str]:
            resolved = resolve_model_key(k, model_sd)
            return resolved if resolved in model_sd else None

        def copy_into(resolved_key: str, src: torch.Tensor) -> None:
            target_param = model_sd[resolved_key]
            target_local = target_param.to_local() if hasattr(target_param, "to_local") else target_param
            parent_mod = get_parent_module(resolved_key, module_dict)
            is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))
            adapted = to_target_layout(src, target_local, is_flipped)
            if parity is not None:
                parity.check(resolved_key, target_local, adapted)
            target_local.copy_(adapted)

        consumed_staging = set()
        parity_check = parity.check if parity is not None else None
        layout: Optional[direct_install.DirectInstallLayout] = getattr(self, "_tpu_sync_layout", None)
        fused_targets: tuple[tuple[str, tuple[str, ...]], ...] = _FUSED_TARGETS
        if layout is not None:
            # The model-specific tensors (e.g. GatedDeltaNet projections, expert stacks); see ``direct_install``.
            consumed_staging |= layout.install(
                self._tpu_sync_staging, model_sd, find_model_key, copy_into, self._tpu_sync_plan, parity_check
            )
            fused_targets = _FUSED_TARGETS + layout.fused_targets

        # 1. Handle fused projections (QKV and Gate-Up)
        for target_suffix, src_suffixes in fused_targets:
            primary_src = src_suffixes[0]
            for name in list(self._tpu_sync_staging.keys()):
                if name.endswith(primary_src):
                    layer_prefix = name[: -len(primary_src)]
                    layer_src_keys = [layer_prefix + s for s in src_suffixes]
                    if all(sk in self._tpu_sync_staging for sk in layer_src_keys):
                        resolved_key = find_model_key(layer_prefix + target_suffix)
                        if resolved_key is not None:
                            fused = torch.cat([self._tpu_sync_staging[sk] for sk in layer_src_keys], dim=0)
                            copy_into(resolved_key, fused)
                            consumed_staging.update(layer_src_keys)

        # 2. Handle all remaining non-fused parameters (o_proj, down_proj, layernorms, embeddings)
        unresolved = []
        for name, src_t in self._tpu_sync_staging.items():
            if name in consumed_staging:
                continue
            resolved_key = find_model_key(name)
            if resolved_key is not None:
                copy_into(resolved_key, src_t)
            else:
                unresolved.append(name)
        if layout is not None and unresolved:
            raise RuntimeError(f"tpu-sync direct install: no vLLM parameter for {len(unresolved)}: {unresolved[:8]}")

        # 3. Handle tied word embeddings. Only when the trainer did not send lm_head itself: for untied
        #    models (e.g. Qwen3-8B / 32B) the received lm_head must not be overwritten by embed_tokens.
        received_lm_head = any(k == "lm_head.weight" or k.endswith(".lm_head.weight") for k in self._tpu_sync_staging)
        if received_lm_head:
            pass
        elif (
            hasattr(vllm_model, "lm_head")
            and hasattr(vllm_model, "model")
            and hasattr(vllm_model.model, "embed_tokens")
        ):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.model.embed_tokens.weight)
        elif hasattr(vllm_model, "lm_head") and hasattr(vllm_model, "embed_tokens"):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.embed_tokens.weight)

        # TODO(tpu): fail the first sync if some rollout parameter received no weights (it would keep the checkpoint's
        #   value for the whole run), for every direct install; Qwen3.5 first needs to confirm which parameters
        #   the text-only trainer never sends (e.g. a vision tower).

        t_sync_start = time.perf_counter()
        tpu_sync_engine.tpu_synchronize()
        t_sync = time.perf_counter() - t_sync_start
        t_total = time.perf_counter() - t_start

        logger.info(
            f"[TPU-SYNC TELEMETRY | Sampler Worker] Sampler Rank {rank}: install_tpu_sync_weights "
            f"completed in {t_total:.4f}s (H2D={t_h2d:.4f}s, TPUSyncBarrier={t_sync:.4f}s)"
        )
        # Returned through collective_rpc so the orchestrator can log them as step metrics.
        result = {"total": t_total, "h2d": t_h2d, "sync": t_sync}
        if parity is not None:
            # Covers every tensor written through copy_into, and the regions that a direct-install layout writes
            # itself (e.g. the experts' w13 / w2), through the ``check`` passed to its install.
            result["exact_parity"] = parity.result(rank, unresolved)
        return result

    @torch.no_grad()
    def _install_with_vllm_loaders(self) -> None:
        """Load the received checkpoint-format (HF-named) tensors through vLLM's own weight loaders."""
        from vllm.distributed import get_tp_group

        tp_group = get_tp_group()
        staging = self._tpu_sync_staging
        sharded = self._tpu_sync_staging_sharded
        t_gather = 0.0

        def _weights():
            nonlocal t_gather
            for name in sorted(staging):
                t = staging[name]
                if sharded[name]:
                    t0 = time.perf_counter()
                    t = tp_group.all_gather(t, dim=0)
                    # Run every collective on its own: deferred (fused) collective programs can be cut at different
                    # points on different ranks and then never match up, which stalls the TP group.
                    tpu_sync_engine.tpu_synchronize(t)
                    t_gather += time.perf_counter() - t0
                yield name, t

        def _checkpoint_weights():
            # Back to HF's layout (e.g. Qwen3.5's split tensors re-fused, Qwen3 MoE's expert stacks split per expert).
            return direct_install.loader_weights(_weights(), getattr(self, "_tpu_sync_hf_text_config", None))

        worker: Any = getattr(self, "worker", self)
        model_runner = worker.model_runner
        if hasattr(model_runner, "reload_weights"):
            # vllm-torchtpu's dense linear method transposes weights to (k, n) in process_weights_after_loading,
            # guarded by a per-module "already flipped" flag that layerwise reload does not reset, so the fresh
            # weights would stay [n_out, n_in] and fail to copy into the live [n_in, n_out] tensors. Clear it first.
            for module in self._get_vllm_model().modules():
                if getattr(module, "_tpu_weight_flipped", False):
                    module._tpu_weight_flipped = False
            model_runner.reload_weights(weights_iterator=_checkpoint_weights(), is_checkpoint_format=True)
        else:
            self._get_vllm_model().load_weights(_checkpoint_weights())
        logger.info(f"Sampler Rank {getattr(self, 'rank', 0)}: TP all-gathers of the received tensors: {t_gather:.2f}s")

    def get_model_weights_stats(self) -> dict:
        """Computes deterministic parameter count, L1 norm, and L2 norm of this worker's received weights."""
        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            return {"error": "No model found on worker"}

        if not getattr(self, "_sorted_vllm_params", None):
            raise RuntimeError(
                f"get_model_weights_stats is only supported for tpu-sync after initialization, "
                f"but self._sorted_vllm_params is not initialized on worker rank {getattr(self, 'rank', 'unknown')}!"
            )

        res = compute_tensor_stats(self._sorted_vllm_params)
        res["rank"] = getattr(self, "rank", 0)
        # How many TP ranks hold each tensor's slice, so the parity check counts every slice once.
        res["replicas"] = dict(getattr(self, "_tpu_sync_replicas", None) or {})
        return res


def resolve_sglang_tp_rank_and_size(scheduler: Any) -> tuple[int, int]:
    """Resolve ``(tp_rank, tp_size)`` across SGLang 0.5.x ``Scheduler`` layouts.

    SGLang >= 0.5.12 stores ``scheduler.tp_group`` (a ``GroupCoordinator`` with ``rank_in_group`` and
    ``world_size``) and ``scheduler.tp_size`` without setting ``scheduler.tp_rank`` directly, while also
    exposing ``sglang.srt.runtime_context.get_parallel()``.
    """
    tp_group = getattr(scheduler, "tp_group", None)
    tp_worker = getattr(scheduler, "tp_worker", None)

    tp_rank = getattr(scheduler, "tp_rank", None)
    if tp_rank is None:
        tp_rank = getattr(tp_group, "rank_in_group", None)
    if tp_rank is None:
        try:
            from sglang.srt.runtime_context import get_parallel

            tp_rank = getattr(get_parallel(), "tp_rank", None)
        except Exception:
            pass
    if tp_rank is None:
        tp_rank = getattr(tp_worker, "tp_rank", None)
    if tp_rank is None:
        raise AttributeError(
            "Unable to determine SGLang scheduler tp_rank from scheduler, tp_group, or runtime_context"
        )

    tp_size = getattr(scheduler, "tp_size", None)
    if tp_size is None:
        tp_size = getattr(tp_group, "world_size", None)
    if tp_size is None:
        try:
            from sglang.srt.runtime_context import get_parallel

            tp_size = getattr(get_parallel(), "tp_size", None)
        except Exception:
            pass
    if tp_size is None:
        tp_size = getattr(tp_worker, "tp_size", None)
    if tp_size is None:
        raise AttributeError(
            "Unable to determine SGLang scheduler tp_size from scheduler, tp_group, or runtime_context"
        )

    return int(tp_rank), int(tp_size)


class SGLangTPUSyncWorker(TPUSyncWorkerMixin):
    """Weight-sync RPC adapter attached to each SGLang TP scheduler process on TPU.

    Supports both ``tpu-sync`` (``init_tpu_sync_on_worker``, ``install_tpu_sync_weights``,
    ``get_model_weights_stats``) and ``ray-plasma-store`` (``load_weights_from_ray_registry``).
    """

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler
        self.rank, self.tp_size = resolve_sglang_tp_rank_and_size(scheduler)
        self.model_runner = scheduler.tp_worker.model_runner

    def _get_tp_size(self) -> Optional[int]:
        return self.tp_size

    def _get_model_config(self) -> Any:
        return self.model_runner.model_config

    @torch.no_grad()
    def _install_with_vllm_loaders(self) -> None:
        """Load checkpoint-format tensors through SGLang's ``weight_updater`` / ``model.load_weights``."""
        from sglang.srt.distributed import tensor_model_parallel_all_gather

        staging = self._tpu_sync_staging
        sharded = self._tpu_sync_staging_sharded
        t_gather = 0.0

        def _weights():
            nonlocal t_gather
            for name in sorted(staging):
                t = staging[name]
                if sharded[name]:
                    t0 = time.perf_counter()
                    t = tensor_model_parallel_all_gather(t, dim=0)
                    tpu_sync_engine.tpu_synchronize(t)
                    t_gather += time.perf_counter() - t0
                yield name, t

        def _checkpoint_weights():
            return direct_install.loader_weights(_weights(), getattr(self, "_tpu_sync_hf_text_config", None))

        weight_updater = getattr(self.model_runner, "weight_updater", None)
        if weight_updater is not None and hasattr(weight_updater, "begin_weight_update"):
            weight_updater.begin_weight_update()
            try:
                self._get_vllm_model().load_weights(_checkpoint_weights())
            finally:
                weight_updater.end_weight_update(run_post_load=True)
        else:
            self._get_vllm_model().load_weights(_checkpoint_weights())
        logger.info(f"Sampler Rank {self.rank}: TP all-gathers of the received tensors: {t_gather:.2f}s")

    def load_weights_from_ray_registry(self, step_key: int) -> int:
        """Pull ``step_key`` weights from ``RayWeightRegistry`` when ``checkpoint_engine.backend=ray-plasma-store``."""
        from verl_hardware_plugin.accelerators.tpu.weight_sync.ray_checkpoint_engine import (
            load_weights_from_ray_registry,
        )

        return load_weights_from_ray_registry(self, step_key)


_VLLM_EXTENSION_CLS: Optional[type] = None


def __getattr__(name: str) -> Any:
    global _VLLM_EXTENSION_CLS
    if name == "vLLMTPUSyncWorkerExtension":
        if _VLLM_EXTENSION_CLS is None:
            from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

            class vLLMTPUSyncWorkerExtension(TPUSyncWorkerMixin, vLLMColocateWorkerExtension):
                """vLLM Worker Extension for Google Cloud TPU with tpu-sync P2P weight synchronization."""

            _VLLM_EXTENSION_CLS = vLLMTPUSyncWorkerExtension
            globals()[name] = _VLLM_EXTENSION_CLS
        return _VLLM_EXTENSION_CLS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
