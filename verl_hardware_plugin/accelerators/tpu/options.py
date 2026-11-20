# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Every TPU-specific option of the plugin: options that verl's own (GPU) config has no field for.

Each option lives in the verl config, in the per-backend ``engine_kwargs`` dict of the component it configures,
the same channel verl uses for its own backends:

* TorchTitan engine: ``actor_rollout_ref.{actor,ref}.torchtitan.engine_kwargs.tpu.*`` -> ``TPUEngineOptions``.
  verl's ``TorchtitanEngineConfig`` has no ``engine_kwargs`` field yet; the platform adds it
  (``engines.tpu_utils.extend_torchtitan_engine_config``)::

      +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_splash_attention=True
      +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.tpu_eager_mode=DEFER_AND_FUSE

* ``tpu-sync`` weight sync: ``actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.*`` ->
  ``TPUSyncOptions``. verl passes these kwargs to the checkpoint engine's constructor; the driver reads the same
  entry of the config::

      +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.parallelism=8

* vLLM rollout: no TPU-only options. ``actor_rollout_ref.rollout.engine_kwargs.vllm`` goes to vLLM as is.

The config reaches every worker through verl, so nothing needs to be forwarded through the environment. Both
parsers reject unknown keys and values of the wrong type, so a typo fails loudly. For compatibility the older
channels are still read, each with a one-time deprecation warning: the top-level ``torchtitan.<option>`` fields,
and the ``VERL_TPU_*`` / ``VERL_TPU_SYNC_RELEASE_BUFFERS`` environment variables (which still take precedence over the
config, as before).

Process-wide settings that are not part of any verl config stay environment variables, forwarded to every worker by
the platform: the libtpu / XLA flags (``LIBTPU_INIT_ARGS``, ``XLA_FLAGS``) and ``VERL_TPU_CHIPS_PER_HOST``
(``tpu_chips_per_host_override``).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Key of the TPU options in ``TorchtitanEngineConfig.engine_kwargs``.
TPU_ENGINE_KWARGS_KEY = "tpu"
# Parallel transfer streams per worker, unless engine_kwargs.tpu-sync.parallelism overrides it.
TPU_SYNC_DEFAULT_PARALLELISM = 8
# Consecutive tensors per transfer stage of the controller, unless engine_kwargs.tpu-sync.group_size overrides it.
TPU_SYNC_DEFAULT_GROUP_SIZE = 16

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def _to_plain_dict(value: Any) -> Any:
    """Recursively converts mappings and iterables into plain dicts and lists."""
    if isinstance(value, Mapping):
        return {str(k): _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_plain_dict(v) for v in value]
    return value


def _parse(name: str, kind: type, value: Any, source: str, label: str = "TPU") -> Any:
    """Converts ``value`` (a config value or an environment string) to ``kind``; ``None`` / ``""`` is unset."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        if kind is dict:
            if isinstance(value, Mapping):
                return _to_plain_dict(value)
            raise ValueError(f"expected a mapping, got {type(value).__name__}")
        if hasattr(kind, "parse"):
            return kind.parse(value)
        if kind is bool:
            if isinstance(value, bool):
                return value
            lowered = str(value).lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            raise ValueError(value)
        if kind is int:
            if isinstance(value, bool):
                raise ValueError(value)
            parsed = int(value)
            if parsed <= 0:
                raise ValueError(value)
            return parsed
        if kind is float:
            if isinstance(value, bool):
                raise ValueError(value)
            parsed_float = float(value)
            if not math.isfinite(parsed_float) or parsed_float <= 0:
                raise ValueError(value)
            return parsed_float
        # Strings are passed through; e.g. tpu_eager_mode="NONE" explicitly disables the eager mode.
        return str(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid {label} option {name}={value!r} from {source}") from e


def _parse_section(
    raw: Optional[Mapping[str, Any]], kinds: Mapping[str, type], source: str, label: str
) -> dict[str, Any]:
    """Parses one ``engine_kwargs`` section; raises on unknown keys."""
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(kinds))
    if unknown:
        raise ValueError(f"Unknown {label} option(s) {unknown} in {source}; supported: {sorted(kinds)}")
    return {name: _parse(name, kinds[name], value, source, label) for name, value in raw.items()}


def _apply_legacy_env(
    values: dict[str, Any], kinds: Mapping[str, type], env_vars: Mapping[str, str], target: str
) -> None:
    """Applies the deprecated environment variables over ``values``, warning once per variable."""
    for name, env_name in env_vars.items():
        raw_env = os.environ.get(env_name)
        env = _parse(name, kinds[name], raw_env, env_name)
        if env is not None:
            _warn_once(
                f"env:{env_name}",
                f"{env_name}={raw_env!r} overrides the config and is deprecated; set {target}.{name} instead.",
            )
            values[name] = env


# ---------------------------------------------------------------------------------------------------------------
# TorchTitan engine: actor_rollout_ref.{actor,ref}.torchtitan.engine_kwargs.tpu
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TPUEngineOptions:
    """TPU options of the TorchTitan engine. ``None`` means "use the default derived from the rest of the config".

    Attributes:
        use_splash_attention: Use the Pallas splash attention kernel (default: ``use_torch_compile`` or
            ``context_parallel_size > 1``).
        use_simple_fsdp: Shard with SimpleFSDP (default: ``use_torch_compile`` when PP is 1).
        use_fused_linear_cross_entropy: Compute log-probs with Tokamax's fused linear + softmax + cross-entropy
            Pallas kernel, without materializing the ``[tokens, vocab]`` logits (default: on for MoE models, where
            it is required and the engine raises if it cannot be used; else ``use_torch_compile`` when TP is 1).
        fuse_qkv: Fuse attention Q/K/V projections in the model (default: ``True``; ``False`` avoids
            the FSDP all-gather inside ``FusedQKVLinear._split_qkv_on_save`` during ``state_dict()``).
        tpu_eager_mode: ``torch_tpu`` eager mode, ``DEFER_AND_FUSE`` or ``DEFER_NEVER``
            (default: ``DEFER_AND_FUSE`` with ``use_torch_compile``).
        context_parallel_strategy: Context-parallel attention strategy when ``context_parallel_size > 1``:
            ``all_gather`` (load-balanced KV all-gather with Splash Attention, default) or ``ulysses`` (all-to-all).
        context_parallel_gdn_strategy: How the context-parallel GatedDeltaNet (Qwen3.5) moves its inputs between
            the token and head layouts: ``all_to_all`` or ``allgather_reducescatter`` (default: unset, torchtitan's
            ``GatedDeltaNetContextParallelTransform`` default). Passed to the transform as ``strategy``; needs
            ``context_parallel_size > 1``, a model with GatedDeltaNet layers, and a torchtitan whose transform has
            that field (an older one is rejected when this is set). A performance A/B option, independent of
            ``context_parallel_strategy``.
        context_parallel_gdn_reduce_dtype: dtype of both the forward ``out_proj`` reduce-scatter and the backward
            input-gradient reduce-scatter under ``context_parallel_gdn_strategy='allgather_reducescatter'``,
            ``float32`` or ``bfloat16`` (default: unset, the transform's default). Passed as ``reduce_dtype``, with
            the same torchtitan requirement; rejected under ``all_to_all``, when ``context_parallel_size <= 1``, or
            on models without GatedDeltaNet layers.
        optim_impl: torchtitan optimizer implementation (default: ``fused``, ``for-loop`` with TP).
        seq_bucket_size: Packed sequence length bucket in tokens (default: ``pad_to_length_bucket`` when
            ``pad_to_length`` is set, else 256).
        model_kernels: Optional torchtitan TPU kernels of the model (``engines.tpu_model_kernels``): ``auto``
            (default: what the installed torchtitan provides), ``none``, ``all``, or a comma-separated subset.
        ep_capacity_factor: Per-destination-rank token capacity multiplier of the TPU EP token dispatcher when
            ``expert_parallel_size > 1`` (default: ``4.0``). torchtitan's pretraining default (``1.25``) drops
            skewed RL / math tokens across 32-64 EP ranks while vLLM rollout is dropless.
        qwen3_pallas_experts: Legacy Qwen3-30B-A3B flag; Pallas is required, so ``False`` is rejected.
        moe_paired_gemm: Legacy Qwen3-30B-A3B alias used only when ``moe_paired_grouped_mm`` is unset.
        mixed_precision_param: TorchTitan mixed-precision parameter dtype, ``bfloat16`` or ``float32``;
            unset preserves its default and FP32 master parameters.
        fp32_logits: Preserve FP32 LM-head accumulators with BF16 operands; ``False`` requests BF16 logits.
            Either explicit value disables the automatic fused-linear-loss default so this head is used.
        rollout_expert_parallel: Export MoE experts in vLLM's expert-parallel layout so the rollout receives
            them straight into its expert weights with no HBM staging copy (default: ``False``, see
            ``rollout/moe_weights.py``; supported by Qwen3.5 and Qwen3 MoE). Temporary for large MoEs whose expert
            shard cannot afford a second HBM staging copy alongside weights + KV cache (e.g. Qwen3.5-397B-A17B); remove
            once ``tpu-sync`` stages on host DRAM and H2Ds directly into vLLM weights (the default ``False`` path
            already supports vLLM EP).
        rollout_fp8_moe: Trainer-side expert quantization for ``rollout_expert_parallel`` when the rollout runs
            vllm-torchtpu's quantized MoE (e.g. ``QUANTIZE_ON_LOAD_PREFIXES=experts`` with
            ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=QUANTIZE_ON_LOAD_PREFIXES``). Unset (default): when
            ``rollout_expert_parallel=True`` and the rollout reports quantized MoE via ``describe_vllm_params``, the
            trainer automatically quantizes each expert stack in the rollout's ``(quant_dtype, block_size)`` format and
            sends quantized weights (received zero-copy into ``w13_weight`` / ``w2_weight``) plus FP32 scales (copied
            into ``w*_weight_scale_inv``), halving expert wire bytes and avoiding a bf16 staging copy. ``True`` forces
            trainer-side quantization (and implies ``rollout_expert_parallel=True`` when unset); ``False`` disables
            trainer-side quantization.
        param_dtype: Storage dtype of the trained parameters, their gradients and (with the default AdamW) the
            optimizer states: ``float32`` (default) or ``bfloat16``. ``bfloat16`` halves the parameter and gradient
            memory (pure bf16 training, no fp32 master), e.g. for models whose fp32 training state does not fit.
        optimizer_host_offload: Keep the optimizer states in host memory (default: ``False``). Unlike verl's
            ``optimizer_offload``, the whole state never has to fit on the device. With ``fp32_master=True`` and
            ``AdamW``, FP32 masters and moments stay resident on CPU and step with CPU ``fused=True`` ``AdamW`` while
            pipelining pinned ``bfloat16`` gradient D2H and stepped-weight H2D one parameter chunk at a time.
        fp32_master: With ``param_dtype=bfloat16``, the optimizer steps FP32 master weights with FP32 AdamW states
            and casts the result into the bf16 parameters (default: ``False``). Without it, bf16 weights absorb no
            update smaller than half a bf16 ulp (lr 1e-6 updates mostly round to zero). With
            ``optimizer_host_offload`` the masters stay resident in host memory with the optimizer states.
        ep_dispatcher: MoE expert-parallel token dispatcher mode when ``expert_parallel_size > 1``: ``auto``
            (default: ``ragged`` when ``torch.ops.tpu._ragged_all_to_all`` is available and ``spmd_backend`` is not
            ``spmd_types``, else ``static``), ``ragged`` (:class:`TpuRaggedEPTokenDispatcher` sending ``1.0x`` real
            routed tokens + unpadded MegaBlox ``gmm``/``tgmm`` group counts), or ``static`` (uniform ``[ep * cap, D]``
            ``TpuQwen35TokenDispatcher``).
        enable_tpu_topo_mesh: Preserve physical 3D ICI torus ``(x, y, z)`` rank order on ``ParallelismConfig``
            (``torch.tpu.topology_aware_mesh`` + ``DeviceMesh(preserve_rank_order=True)``). ``None`` (default) enables
            it automatically whenever supported by the installed ``torchtitan`` and ``torch`` (``torch >= 2.15``).
        ep_token_chunks: With ``expert_parallel_size > 1`` and a torchtitan that has ``TpuQwen35RoutedExperts``
            (Qwen3.5 fast dropless EP), pipeline each MoE layer's dispatch -> experts -> combine over this many token
            chunks, overlapping one chunk's expert matmuls with the next chunk's EP collectives (default: unset, no
            chunking). Needs ``ep_dispatcher=static`` so torchtitan's ``TpuQwen35TokenDispatcher`` runs.
        ep_paired_grouped_mm: With ``expert_parallel_size > 1`` and a torchtitan whose ``TpuGroupedExperts`` has
            ``allow_ep_paired_grouped_mm``, keep the paired (w1 + w3) Pallas grouped matmul under EP (default:
            ``False``: unpaired under EP, as torchtitan builds without it require).
        ac_save_splash_and_routing: When ``True`` and activation checkpointing is enabled, use a TPU ``SelectiveAC``
            policy that saves only Pallas ``SplashAttention`` forward outputs (``(out, lse)``) and MoE router
            ``aten.topk`` assignments while recomputing all projections, norms, GatedDeltaNet recurrences and MoE
            collectives/matmuls (matching TorchTitan's ``tpu_selective`` ``("moe.routing",
            "attention.inner_attention")`` benchmark memory policy).
        static_micro_batch_len: Pad every packed micro-batch to at least this many tokens (rounded up to the
            sequence bucket), e.g. the ``use_dynamic_bsz`` token budget, so the trainer compiles one sequence shape
            instead of one per length bucket it meets (default: unset, off). When ``gdn_auto_row_slot_len`` is set,
            must be ``>= gdn_auto_row_slot_len`` and divisible by ``gdn_auto_row_slot_len``.
        static_max_response_len: Pad every micro-batch's response-length bucket to at least this many tokens (e.g.
            ``max_response_length``), so response-shaped tensors also keep one shape (default: unset, off).
        loss_row_bucketing: Pad the PPO loss inputs of every micro-batch with fully masked rows up to a common row
            count (``tpu_padded_num_seqs``: the most sequences in any micro-batch of the mini-batch on any rank,
            rounded up to a power of two), so the loss programs do not recompile whenever ``use_dynamic_bsz`` changes
            the number of sequences per micro-batch (default: ``True``). Applied only with verl's ``ppo_loss`` and a
            policy loss that ignores fully masked rows (``losses.loss_inputs.loss_rows_can_be_padded``); ``geo_mean``,
            ``bypass_mode`` and custom ``register_policy_loss`` losses keep the exact row count. ``False`` turns it off.
        static_micro_batch_num_seqs: With ``loss_row_bucketing``, pad the PPO loss inputs to at least this many rows
            (e.g. the largest number of sequences a micro-batch can hold), so the loss keeps one row shape
            (default: unset, the power-of-two bucket only).
        splash_max_block_size: Largest Splash attention block size, a power of two >= 128 (default: ``512``,
            torchtitan's). Long packed micro-batches run faster with ``1024``.
        mixed_precision_reduce: dtype of the FSDP gradient reduce-scatter, ``float32`` (default, torchtitan's) or
            ``bfloat16`` (half the traffic; a performance A/B option).
        moe_paired_grouped_mm: Qwen3 MoE: run the W1/W3 expert projections with torchtitan's paired Pallas grouped
            matmul, sharing the routed-token loads (default: ``False``).
        moe_native_weight_layout: Qwen3 MoE: store the expert weights in the kernel-native ``(E, K, N)`` layout;
            ``state_dict()`` / ``load_state_dict()`` still use the canonical layout and names (default: ``False``).
        cp_full_positions: Under context parallelism, pass the pre-``cp_shard`` unsharded 1-D ``positions`` to
            ``model.forward`` as ``cp_full_positions`` (``google-pytorch/torchtitan#216``), eliminating per-layer
            ``all_gather(positions)`` across both Ulysses and KV all-gather CP (default: unset, off). Fails loudly if
            the installed torchtitan lacks ``cp_full_positions`` or on 3-D MRoPE / multimodal batches.
        gdn_auto_row_slot_len: Dynamic per-micro-batch row-slot length (e.g. ``32768``) for models whose ``forward``
            accepts ``gated_delta_document_resets`` (``google-pytorch/torchtitan#216``). Micro-batches whose single-doc
            sequences start at position ``0`` and fit ``gdn_auto_row_slot_len`` rows are permuted into slots and run
            unsegmented (``set_unflatten_seq_len(slot_len)``, ``gated_delta_document_resets=False``); single-document
            micro-batches starting at position ``0`` run unsegmented at ``padded_len``; multi-document or continuation
            micro-batches run segmented. Ranks agree via bitwise-AND all-reduce so SPMD collectives share one compiled
            shape. Mutually exclusive with ``gdn_row_slot_len``; requires ``use_fused_linear_cross_entropy``,
            ``tensor_parallel_size == 1``, contiguous CP layout (no ``headtail`` balancer), and any configured
            ``static_micro_batch_len`` / ``static_log_prob_micro_batch_len`` to be ``>= gdn_auto_row_slot_len`` and
            divisible by ``gdn_auto_row_slot_len``.
        fuse_gdn_projections: Set ``fuse_input_projections=True`` on every GatedDeltaNet config in ``model_spec.model``
            before build (``google-pytorch/torchtitan#216``), fusing the six bias-free input projections into one
            matmul (default: unset, off). Requires ``tensor_parallel_size == 1`` (``TP == 1``) and fails loudly if TP is
            enabled or if no module config declares ``fuse_input_projections``.
        sort_micro_batches_by_slot: Order each mini-batch's micro-batches by ``(slot_category, -total_tokens, idx)``
            using the shared slot classifier so micro-batches compiling to the same slot / segmentation program run
            back to back (default: unset, off).
        static_log_prob_micro_batch_len: Static packed micro-batch length in tokens for forward-only
            (old / reference log-prob) passes, overriding ``static_micro_batch_len`` when ``forward_only=True``
            (default: unset, off). When ``gdn_auto_row_slot_len`` is set, must be ``>= gdn_auto_row_slot_len`` and
            divisible by ``gdn_auto_row_slot_len``.
        gdn_row_slot_len: Qwen3.5: start every sequence of a packed micro-batch at a multiple of this many tokens
            and run the GatedDeltaNet over rows of this length (one sequence per row) instead of restarting it at
            packed documents through Tokamax ``segment_ids`` (default: unset, off). Needs every sequence <= this
            length, ``static_micro_batch_len`` = sequences per micro-batch x this, and ``context_parallel_size == 1``
            (use ``gdn_auto_row_slot_len`` with ``context_parallel_strategy='ulysses'`` under CP).
        frozen_param_regex: Parameters whose FQN matches this regex (``re.search``) are not trained: their
            gradients are dropped before the gradient norm and the optimizer step, e.g. ``moe.router.gate`` freezes
            the MoE router (default: unset, train every parameter).
        trainer_mode: How the actor's training step runs: ``eager_blocks`` (default: TorchTitan ``Trainer``,
            per-TransformerBlock ``torch.compile`` and eager loss / backward) or ``graph_trainer`` (TorchTitan's
            ``TpuGraphTrainer`` traces forward + RL loss + backward of a micro-batch into one graph and compiles it
            whole, see ``engines/graph_trainer_mode.py``). ``graph_trainer`` needs ``use_simple_fsdp`` and
            ``static_micro_batch_len`` (one traced shape for every micro-batch) and trains actors only.
        graph_trainer: Settings passed straight to TorchTitan's ``TpuGraphTrainerCompileConfig(**cfg)``
            (e.g. ``memory_policy``, ``save_regions``, ``fsdp_barrier_policy``, ``graph_split``).
            Ignored when ``trainer_mode`` is not ``graph_trainer``.
    """

    use_splash_attention: Optional[bool] = None
    use_simple_fsdp: Optional[bool] = None
    use_fused_linear_cross_entropy: Optional[bool] = None
    fuse_qkv: Optional[bool] = None
    tpu_eager_mode: Optional[str] = None
    context_parallel_strategy: Optional[str] = None
    context_parallel_gdn_strategy: Optional[str] = None
    context_parallel_gdn_reduce_dtype: Optional[str] = None
    optim_impl: Optional[str] = None
    seq_bucket_size: Optional[int] = None
    model_kernels: Optional[str] = None
    ep_capacity_factor: Optional[float] = None
    qwen3_pallas_experts: Optional[bool] = None
    moe_paired_gemm: Optional[bool] = None
    mixed_precision_param: Optional[str] = None
    fp32_logits: Optional[bool] = None
    rollout_expert_parallel: Optional[bool] = None
    rollout_fp8_moe: Optional[bool] = None
    param_dtype: Optional[str] = None
    optimizer_host_offload: Optional[bool] = None
    fp32_master: Optional[bool] = None
    ep_dispatcher: Optional[str] = None
    enable_tpu_topo_mesh: Optional[bool] = None
    ep_token_chunks: Optional[int] = None
    ep_paired_grouped_mm: Optional[bool] = None
    ac_save_splash_and_routing: Optional[bool] = None
    static_micro_batch_len: Optional[int] = None
    static_max_response_len: Optional[int] = None
    loss_row_bucketing: Optional[bool] = None
    static_micro_batch_num_seqs: Optional[int] = None
    splash_max_block_size: Optional[int] = None
    mixed_precision_reduce: Optional[str] = None
    moe_paired_grouped_mm: Optional[bool] = None
    moe_native_weight_layout: Optional[bool] = None
    cp_full_positions: Optional[bool] = None
    gdn_auto_row_slot_len: Optional[int] = None
    fuse_gdn_projections: Optional[bool] = None
    sort_micro_batches_by_slot: Optional[bool] = None
    static_log_prob_micro_batch_len: Optional[int] = None
    gdn_row_slot_len: Optional[int] = None
    frozen_param_regex: Optional[str] = None
    trainer_mode: Optional[str] = None
    graph_trainer: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.static_micro_batch_num_seqs is not None and (
            isinstance(self.static_micro_batch_num_seqs, bool) or self.static_micro_batch_num_seqs < 1
        ):
            raise ValueError(
                f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.static_micro_batch_num_seqs must be >= 1, "
                f"got {self.static_micro_batch_num_seqs!r}"
            )


_ENGINE_KINDS: dict[str, Any] = {
    "use_splash_attention": bool,
    "use_simple_fsdp": bool,
    "use_fused_linear_cross_entropy": bool,
    "fuse_qkv": bool,
    "tpu_eager_mode": str,
    "context_parallel_strategy": str,
    "context_parallel_gdn_strategy": str,
    "context_parallel_gdn_reduce_dtype": str,
    "optim_impl": str,
    "seq_bucket_size": int,
    "model_kernels": str,
    "ep_capacity_factor": float,
    "qwen3_pallas_experts": bool,
    "moe_paired_gemm": bool,
    "mixed_precision_param": str,
    "fp32_logits": bool,
    "rollout_expert_parallel": bool,
    "rollout_fp8_moe": bool,
    "param_dtype": str,
    "optimizer_host_offload": bool,
    "fp32_master": bool,
    "ep_dispatcher": str,
    "enable_tpu_topo_mesh": bool,
    "ep_token_chunks": int,
    "ep_paired_grouped_mm": bool,
    "ac_save_splash_and_routing": bool,
    "static_micro_batch_len": int,
    "static_max_response_len": int,
    "loss_row_bucketing": bool,
    "static_micro_batch_num_seqs": int,
    "splash_max_block_size": int,
    "mixed_precision_reduce": str,
    "moe_paired_grouped_mm": bool,
    "moe_native_weight_layout": bool,
    "cp_full_positions": bool,
    "gdn_auto_row_slot_len": int,
    "fuse_gdn_projections": bool,
    "sort_micro_batches_by_slot": bool,
    "static_log_prob_micro_batch_len": int,
    "gdn_row_slot_len": int,
    "frozen_param_regex": str,
    "trainer_mode": str,
    "graph_trainer": dict,
}
assert set(_ENGINE_KINDS) == {f.name for f in dataclasses.fields(TPUEngineOptions)}

# String options with a fixed set of values (lower-cased, checked by get_tpu_engine_options).
_ENGINE_CHOICES: dict[str, tuple[str, ...]] = {
    "context_parallel_strategy": ("all_gather", "ulysses"),
    "context_parallel_gdn_strategy": ("all_to_all", "allgather_reducescatter"),
    "context_parallel_gdn_reduce_dtype": ("float32", "bfloat16"),
}

# engine_kwargs.tpu.trainer_mode values.
TRAINER_MODE_EAGER_BLOCKS = "eager_blocks"
TRAINER_MODE_GRAPH_TRAINER = "graph_trainer"
_TRAINER_MODES = (TRAINER_MODE_EAGER_BLOCKS, TRAINER_MODE_GRAPH_TRAINER)

# Deprecated channels, read for compatibility: option -> environment variable / top-level config field.
_ENGINE_LEGACY_ENV_VARS = {
    "use_splash_attention": "VERL_TPU_USE_SPLASH_ATTENTION",
    "use_simple_fsdp": "VERL_TPU_USE_SIMPLE_FSDP",
    "use_fused_linear_cross_entropy": "VERL_TPU_USE_FUSED_LINEAR_CROSS_ENTROPY",
    "tpu_eager_mode": "VERL_TPU_EAGER_MODE",
    "optim_impl": "VERL_TPU_OPTIM_IMPL",
    "seq_bucket_size": "VERL_TPU_SEQ_BUCKET_SIZE",
    "qwen3_pallas_experts": "VERL_TPU_QWEN3_PALLAS_EXPERTS",
    "moe_paired_gemm": "VERL_MOE_PAIRED_GEMM",
    "fp32_logits": "VERL_TPU_FP32_LOGITS",
}
_ENGINE_LEGACY_CONFIG_FIELDS = (
    "use_splash_attention",
    "use_simple_fsdp",
    "use_fused_linear_cross_entropy",
    "fuse_qkv",
    "tpu_eager_mode",
    "mixed_precision_param",
    "fp32_logits",
)


def get_tpu_engine_options(engine_config: Any) -> TPUEngineOptions:
    """Returns the TPU options of ``engine_config``: ``engine_kwargs.tpu``, then the deprecated channels.

    Precedence per option: ``VERL_TPU_*`` environment variable (deprecated) > ``engine_kwargs.tpu`` >
    top-level config field (deprecated) > ``None`` (the caller's default).

    Raises:
        ValueError: on an unknown key in ``engine_kwargs.tpu`` or a value of the wrong type.
    """
    engine_kwargs = getattr(engine_config, "engine_kwargs", None) or {}
    raw = engine_kwargs.get(TPU_ENGINE_KWARGS_KEY) if isinstance(engine_kwargs, Mapping) else None
    values = _parse_section(raw, _ENGINE_KINDS, f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}", "TPU engine")

    for name in _ENGINE_LEGACY_CONFIG_FIELDS:
        legacy = getattr(engine_config, name, None)
        if legacy is not None and values.get(name) is None:
            _warn_once(
                f"cfg:{name}",
                f"torchtitan.{name} is deprecated; "
                f"set torchtitan.engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.{name} instead.",
            )
            values[name] = _parse(name, _ENGINE_KINDS[name], legacy, f"torchtitan.{name}", "TPU engine")

    _apply_legacy_env(
        values, _ENGINE_KINDS, _ENGINE_LEGACY_ENV_VARS, f"torchtitan.engine_kwargs.{TPU_ENGINE_KWARGS_KEY}"
    )
    mixed_precision_param = values.get("mixed_precision_param")
    if mixed_precision_param not in (None, "bfloat16", "float32"):
        raise ValueError("TPU engine mixed_precision_param must be bfloat16, float32, or null")
    if values.get("param_dtype") not in (None, "float32", "bfloat16"):
        raise ValueError(
            f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.param_dtype must be float32 or bfloat16, "
            f"got {values['param_dtype']!r}"
        )
    if values.get("fp32_master") and values.get("param_dtype") != "bfloat16":
        raise ValueError(f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.fp32_master needs param_dtype=bfloat16")
    if values.get("rollout_fp8_moe") is True:
        if values.get("rollout_expert_parallel") is False:
            raise ValueError(
                f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.rollout_fp8_moe=True conflicts with "
                "rollout_expert_parallel=False"
            )
        if values.get("rollout_expert_parallel") is None:
            values["rollout_expert_parallel"] = True
    if values.get("ep_dispatcher") is not None:
        ep_disp = values["ep_dispatcher"].strip().lower()
        if ep_disp not in ("auto", "ragged", "static"):
            raise ValueError(
                f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.ep_dispatcher must be auto, ragged, or static, "
                f"got {values['ep_dispatcher']!r}"
            )
        values["ep_dispatcher"] = ep_disp
    for name, choices in _ENGINE_CHOICES.items():
        if values.get(name) is not None:
            choice = values[name].strip().lower()
            if choice not in choices:
                raise ValueError(
                    f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.{name} must be one of {choices}, got {values[name]!r}"
                )
            values[name] = choice
    if (
        values.get("context_parallel_gdn_reduce_dtype") is not None
        and values.get("context_parallel_gdn_strategy") == "all_to_all"
    ):
        raise ValueError(
            f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.context_parallel_gdn_reduce_dtype only applies when "
            "context_parallel_gdn_strategy='allgather_reducescatter', got context_parallel_gdn_strategy='all_to_all'"
        )
    for pos_int_key in ("gdn_auto_row_slot_len", "static_log_prob_micro_batch_len"):
        if values.get(pos_int_key) is not None and values[pos_int_key] < 1:
            raise ValueError(
                f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.{pos_int_key} must be >= 1, got {values[pos_int_key]!r}"
            )
    if values.get("gdn_auto_row_slot_len") and values.get("gdn_row_slot_len"):
        raise ValueError(
            f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.gdn_auto_row_slot_len and gdn_row_slot_len are mutually exclusive"
        )
    auto_slot_len = values.get("gdn_auto_row_slot_len")
    if auto_slot_len:
        for static_key in ("static_micro_batch_len", "static_log_prob_micro_batch_len"):
            static_val = values.get(static_key)
            if static_val is not None and (static_val < auto_slot_len or static_val % auto_slot_len != 0):
                raise ValueError(
                    f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.{static_key} ({static_val}) must be >= "
                    f"gdn_auto_row_slot_len ({auto_slot_len}) and divisible by gdn_auto_row_slot_len"
                )
    if values.get("ep_token_chunks") is not None and values["ep_token_chunks"] < 1:
        raise ValueError(
            f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.ep_token_chunks must be >= 1, got {values['ep_token_chunks']!r}"
        )
    if values.get("static_micro_batch_num_seqs") is not None and values["static_micro_batch_num_seqs"] < 1:
        raise ValueError(
            f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.static_micro_batch_num_seqs must be >= 1, "
            f"got {values['static_micro_batch_num_seqs']!r}"
        )
    _normalize_trainer_mode(values)
    return TPUEngineOptions(**values)


def _normalize_trainer_mode(values: dict[str, Any]) -> None:
    """Checks ``trainer_mode`` (lower-cased in place)."""
    key = f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}"
    val = values.get("trainer_mode")
    if val is None:
        return
    if not isinstance(val, str):
        raise ValueError(f"{key}.trainer_mode must be a string, got {type(val).__name__}")
    mode = val.strip().lower()
    if mode not in _TRAINER_MODES:
        raise ValueError(f"{key}.trainer_mode must be one of {_TRAINER_MODES}, got {val!r}")
    values["trainer_mode"] = mode


# ---------------------------------------------------------------------------------------------------------------
# tpu-sync weight sync: actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync
# ---------------------------------------------------------------------------------------------------------------


class ParityMode:
    """Values of ``engine_kwargs.tpu-sync.verify_parity`` (see ``weight_sync/tpu_sync_parity.py``).

    * ``off`` (default): no check.
    * ``norm``: on every sync from step 1, compare per-tensor L1/L2 norms between trainer and rollout.
    * ``exact``: on the step-0 sync, compare every received tensor bit for bit with the parameter vLLM loaded.
    * ``all``: both.

    A bool is accepted for compatibility: ``True`` means ``norm``, ``False`` means ``off``. An enabled check that
    finds a mismatch, or cannot collect the stats it needs, raises ``TPUSyncParityError`` and stops the run.
    """

    MODES = ("off", "norm", "exact", "all")
    _ALIASES = {"false": "off", "0": "off", "no": "off", "none": "off", "true": "norm", "1": "norm", "yes": "norm"}

    @classmethod
    def parse(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "norm" if value else "off"
        mode = str(value).strip().lower()
        mode = cls._ALIASES.get(mode, mode)
        if mode not in cls.MODES:
            raise ValueError(value)
        return mode


class ReceiveIntoVllmMode:
    """Values of ``engine_kwargs.tpu-sync.receive_into_vllm`` (see ``weight_sync/vllm_layout.py``).

    * ``auto`` (default): receive straight into vLLM's parameters when the rollout is vLLM and the trainer's
      weights can be exported in vLLM's layout without cross-rank exchange; fall back to rollout staging buffers
      otherwise (e.g. SGLang, MoE / hybrid linear-attention models, ``FSDP != TP``, or replicated GQA KV heads).
    * ``on``: require direct receive into vLLM's parameters; raise when unsupported.
    * ``off``: always use rollout staging buffers.

    A bool is accepted for compatibility: ``True`` means ``on``, ``False`` means ``off``.
    """

    MODES = ("auto", "on", "off")
    _ALIASES = {"false": "off", "0": "off", "no": "off", "none": "off", "true": "on", "1": "on", "yes": "on"}

    @classmethod
    def parse(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "on" if value else "off"
        mode = str(value).strip().lower()
        mode = cls._ALIASES.get(mode, mode)
        if mode not in cls.MODES:
            raise ValueError(value)
        return mode


@dataclasses.dataclass(frozen=True)
class TPUSyncOptions:
    """Options of the ``tpu-sync`` checkpoint engine.

    Attributes:
        parallelism: Parallel transfer streams per worker.
        group_size: Number of consecutive tensors the tpu-sync controller plans and schedules as one stage of the
            transfer (``1`` makes every tensor its own stage).
        verify_parity: Weight-sync check, one of ``ParityMode.MODES`` (``off`` by default): ``norm`` compares
            trainer and rollout weight norms after every sync, ``exact`` compares every tensor the rollout
            received on the step-0 sync bit for bit with the checkpoint vLLM loaded, ``all`` runs both. A failed
            check raises ``TPUSyncParityError`` before generation resumes.
        release_buffers_after_sync: Unbind the send buffers (a bf16 copy of this rank's weights) after every
            transfer, so training gets that HBM back between syncs.
        tie_word_embeddings: Whether ``lm_head`` is tied to the input embedding, so it need not be sent.
            ``None`` sends ``lm_head``, which is correct for tied and untied models alike.
        return_after_d2h: Return to the trainer as soon as its weights are copied to host memory, and finish the sync
            (the transfer to the rollout hosts, the install and the resume) in the background while training goes on.
            The next sync waits for it first. Applies from the second sync on and needs a ``tpu-sync-torch`` whose
            ``WeightSynchronizer.d2h()`` returns a future (verl#8626); with an older ``tpu-sync-torch`` the sync runs
            to the end as before. Relies on ``CheckpointEngineManager.wait_for_weight_update`` (verl#8530) before
            validation and at the end of training.
        receive_into_vllm: One of ``ReceiveIntoVllmMode.MODES`` (``auto`` by default). When active, the rollout
            receives every tensor straight into vLLM's parameters with no HBM staging copy: the trainer exports its
            weights in vLLM's layout (fused, transposed, one virtual tensor per vLLM parameter), so each rollout rank's
            TP slice is its vLLM weight. ``auto`` enables it for dense vLLM models whose trainer layout needs no
            cross-rank exchange (e.g. ``FSDP == TP``) and falls back to staging otherwise; ``on`` (``True``) requires
            it and ``off`` (``False``) disables it.
        direct_install_quantized_moe: Install quantized rollout MoE experts (e.g. vllm-torchtpu's FP8 MoE with
            ``QUANTIZE_ON_LOAD_PREFIXES=experts``) directly: each rollout rank requantizes the bf16 expert slices it
            received with the rollout's own quantizer and writes the live weights and scales (``True``, default).
            ``False`` installs them through vLLM's checkpoint loaders instead (every rank all-gathers every tensor;
            much slower), e.g. to rule out the direct install while debugging.
    """

    parallelism: int = TPU_SYNC_DEFAULT_PARALLELISM
    group_size: int = TPU_SYNC_DEFAULT_GROUP_SIZE
    verify_parity: str = "off"
    release_buffers_after_sync: bool = True
    tie_word_embeddings: Optional[bool] = None
    return_after_d2h: bool = True
    receive_into_vllm: str = "auto"
    direct_install_quantized_moe: bool = True

    @property
    def verify_norms(self) -> bool:
        """Whether the per-step norm check runs (``norm`` or ``all``)."""
        return self.verify_parity in ("norm", "all")

    @property
    def verify_exact(self) -> bool:
        """Whether the step-0 bit-exact check runs (``exact`` or ``all``)."""
        return self.verify_parity in ("exact", "all")


_SYNC_KINDS: dict[str, type] = {
    "parallelism": int,
    "group_size": int,
    "verify_parity": ParityMode,
    "release_buffers_after_sync": bool,
    "tie_word_embeddings": bool,
    "receive_into_vllm": ReceiveIntoVllmMode,
    "return_after_d2h": bool,
    "direct_install_quantized_moe": bool,
}
assert set(_SYNC_KINDS) == {f.name for f in dataclasses.fields(TPUSyncOptions)}

_SYNC_LEGACY_ENV_VARS = {"release_buffers_after_sync": "VERL_TPU_SYNC_RELEASE_BUFFERS"}


def get_tpu_sync_options(engine_kwargs: Optional[Mapping[str, Any]], backend: str = "tpu-sync") -> TPUSyncOptions:
    """Returns the ``tpu-sync`` options from the checkpoint engine's kwargs for the configured ``backend``.

    ``engine_kwargs`` is the backend's own entry (what verl passes to the engine constructor, or
    ``checkpoint_engine.engine_kwargs[backend]`` on the driver). Precedence per option:
    ``VERL_TPU_SYNC_RELEASE_BUFFERS`` (deprecated) > ``engine_kwargs.<backend>`` > the default.

    Raises:
        ValueError: on an unknown key or a value of the wrong type.
    """
    target = f"checkpoint_engine.engine_kwargs.{backend}"
    values = _parse_section(engine_kwargs, _SYNC_KINDS, target, "tpu-sync")
    _apply_legacy_env(values, _SYNC_KINDS, _SYNC_LEGACY_ENV_VARS, target)
    return TPUSyncOptions(**{name: value for name, value in values.items() if value is not None})


# ---------------------------------------------------------------------------------------------------------------
# Platform: process-wide settings outside the verl config
# ---------------------------------------------------------------------------------------------------------------


def tpu_chips_per_host_override() -> Optional[int]:
    """``VERL_TPU_CHIPS_PER_HOST``: chips per host of a worker pool, when it cannot be inferred from the pool size.

    The platform resolves the TPU topology before any verl config exists, so this stays an environment variable.
    """
    return _parse("VERL_TPU_CHIPS_PER_HOST", int, os.environ.get("VERL_TPU_CHIPS_PER_HOST"), "VERL_TPU_CHIPS_PER_HOST")
