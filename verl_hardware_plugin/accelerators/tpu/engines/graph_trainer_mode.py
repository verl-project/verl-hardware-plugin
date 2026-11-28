# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""``engine_kwargs.tpu.trainer_mode=graph_trainer``: TorchTitan's ``TpuGraphTrainer`` runs the actor training step.

In the default ``eager_blocks`` mode the engine builds TorchTitan's ``Trainer``, compiles every TransformerBlock with
``torch.compile(backend="tpu")`` and runs the LM-head log-prob, verl's loss and ``loss.backward()`` eagerly. In
``graph_trainer`` mode it builds ``TpuGraphTrainer`` (``compile.mode=aot_fx_trace``) instead: the first training
micro-batch is traced with ``make_fx`` into one FX graph (model forward, fused LM-head log-prob, PPO policy loss and
backward), the TPU pass pipeline (memory policy / remat, FSDP schedule) runs on it, and ``torch_tpu``'s
``StaticCompiler`` compiles it whole. Every later micro-batch reuses that executable. The HF load, SimpleFSDP, the
optimizer, LR scheduler, chunked optimizer step, grad clipping and the tpu-sync weight export are unchanged.

* Actor training only. ``old_log_probs`` must come from the rollout (``rollout_correction.bypass_mode=True``) and the
  KL loss must be off, so no forward-only pass runs: ``forward_only`` / ``infer_batch`` raises (TODO-3), and so does a
  reference (``forward_only``) engine configured with this mode.
* Policy loss: verl's vanilla dual-clip PPO (``loss_mode=vanilla``, or ``bypass_mode`` with ``loss_type=ppo_clip``
  and no rejection sampling), ``token-mean`` aggregation, no entropy term. Its tensor math is traced into the graph;
  verl's own ``ppo_loss`` then runs eagerly under ``no_grad`` on the per-token log-probs the graph returns, so the
  logged loss and metrics (``pg_clipfrac``, ``ppo_kl``, rollout-correction metrics, ...) are verl's.
* Several micro-batches per mini-batch: ``TpuGraphTrainer`` adds each micro-batch's gradients to ``param.grad``.
  Every micro-batch loss is normalized by the mini-batch's global token count, as in the eager engine, so the sum is
  the mini-batch's token mean.
* One sequence shape (``static_micro_batch_len``), traced once (TODO-2: cache one graph per bucket length).

Loss scaling: verl's ``token-mean`` loss is ``sum(pg * mask) / global_tokens * dp_size`` and the eager engine scales
it by ``1 / dp_size`` before ``backward()`` because SimpleFSDP's reduce-scatter sums gradients. The graph loss is
``sum(pg * mask) / global_tokens`` (``global_tokens`` a tensor input, so it is not frozen into the graph): both
factors are folded, so the summed gradient is the global token mean exactly as in the eager engine.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from typing import Any, Callable

import torch
from tensordict import TensorDict

from verl_hardware_plugin.accelerators.tpu.options import TRAINER_MODE_GRAPH_TRAINER, TPUEngineOptions

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Key of the detached per-token log-probs in the loss metrics the traced step returns (``trainer.last_loss_metrics``);
# verl's loss and metrics are computed eagerly from them.
LOG_PROBS_METRIC = "log_probs"
# TorchTitan's compile mode for the graph trainer: trace forward + loss + backward of a step into one FX graph.
TPU_COMPILE_MODE = "aot_fx_trace"
# Static sequence-count grid for traced sequence-level rollout correction (rollout_rs / sequence rollout_is).
MAX_PACKED_SEQS = 128

_TODO_2 = "graph_trainer mode traces one sequence shape and cannot retrace (TODO-2: one traced graph per bucket length)"
_TODO_3 = (
    "graph_trainer mode has no forward-only path yet (TODO-3): run with algorithm.rollout_correction.bypass_mode=True "
    "(old_log_probs = rollout log-probs) and the KL loss off, and keep trainer_mode=eager_blocks on the reference model"
)


def is_graph_trainer_mode(options: TPUEngineOptions) -> bool:
    """Whether ``engine_kwargs.tpu.trainer_mode`` selects TorchTitan's graph trainer."""
    return options.trainer_mode == TRAINER_MODE_GRAPH_TRAINER


def forward_only_error() -> NotImplementedError:
    return NotImplementedError(f"forward_only / infer_batch: {_TODO_3}")


def shape_change_error(traced_len: int, padded_len: int) -> RuntimeError:
    return RuntimeError(
        f"{_TODO_2}: traced with {traced_len} packed tokens, got {padded_len}. Raise "
        "engine_kwargs.tpu.static_micro_batch_len to at least the largest micro-batch (e.g. ppo_max_token_len_per_gpu)."
    )


def validate_graph_trainer_engine(
    engine_config: Any, options: TPUEngineOptions, *, use_simple_fsdp: bool, tpu_kernels: frozenset[str]
) -> None:
    """Raises ``ValueError`` listing every setting Phase 1 of ``trainer_mode=graph_trainer`` does not support."""
    # TODO: drop checks here as follow-ups land:
    # - forward_only (TODO-3): once TorchTitan's graph trainer can trace a forward-only log-prob step.
    # - static_micro_batch_len (TODO-2): once the traced step is cached per sequence shape (TorchTitan / plugin).
    # - pipeline / context / tensor parallel sizes: once the graph-trainer path supports and is validated with them.
    # - gdn_row_slot_len: once the traced step restores the packed token order (gdn_contig_src) before the LM head
    #   and traces with the shim row length set to the slot length (see _packed_stream_as_one_row).
    # use_simple_fsdp and param_offload stay: the graph trainer shards with SimpleFSDP and holds the live parameters.
    errors = []
    if getattr(engine_config, "forward_only", False):
        errors.append(f"it trains actors only, but this engine is forward_only ({_TODO_3})")
    if not use_simple_fsdp:
        errors.append(
            "it needs engine_kwargs.tpu.use_simple_fsdp=True (TorchTitan's graph trainer shards with SimpleFSDP)"
        )
    if not options.static_micro_batch_len:
        errors.append(f"it needs engine_kwargs.tpu.static_micro_batch_len ({_TODO_2})")
    if options.use_fused_linear_cross_entropy is False:
        errors.append(
            "it needs engine_kwargs.tpu.use_fused_linear_cross_entropy=True (the non-fused log-prob path "
            "materializes [tokens, vocab] logits and exhausts TPU HBM)"
        )
    for name in ("pipeline_parallel_size", "tensor_parallel_size"):
        size = int(getattr(engine_config, name, 1) or 1)
        if size > 1:
            errors.append(f"{name}={size} is not supported yet")
    if getattr(engine_config, "param_offload", False):
        errors.append("param_offload swaps parameter storages between steps; it is not supported yet")
    budget = options.kda_document_budget
    if "kda" in tpu_kernels and budget is not None and budget < 2:
        errors.append(f"kda_document_budget={budget} must be at least 2 (one sequence plus the padding tail)")
    if options.precompile_train_step:
        # The precompiled step runs on synthetic data: nothing it pins may depend on the real first step's data.
        if "kda" in tpu_kernels and budget is None:
            errors.append("precompile_train_step with the Tokamax KDA kernels needs kda_document_budget")
        if (options.graph_trainer or {}).get("optimizer_compile") == "in_graph":
            errors.append("precompile_train_step does not support graph_trainer.optimizer_compile='in_graph'")
    if options.gdn_row_slot_len and options.static_micro_batch_len:
        if int(options.static_micro_batch_len) % int(options.gdn_row_slot_len) != 0:
            errors.append(
                f"static_micro_batch_len={options.static_micro_batch_len} must be a multiple of "
                f"gdn_row_slot_len={options.gdn_row_slot_len}"
            )
    if errors:
        raise ValueError("engine_kwargs.tpu.trainer_mode=graph_trainer: " + "; ".join(errors))


def initial_kda_document_budget(options: TPUEngineOptions, num_documents: int) -> int:
    """The Tokamax KDA ``max_num_segments`` to pin for the run, set once before the first trace.

    ``engine_kwargs.tpu.kda_document_budget`` if set, else ``num_documents + 1`` (one segment for the padding tail).
    ``num_documents`` is the most documents in any micro-batch of the first step, across all ranks.
    """
    return int(options.kda_document_budget or num_documents + 1)


def check_kda_document_budget(budget: int, num_documents: int) -> None:
    """Raises if a micro-batch with ``num_documents`` documents does not fit the pinned ``budget``.

    The traced graph froze ``max_num_segments`` at ``budget``; more documents would share recurrent state.
    """
    if num_documents + 1 > budget:
        raise RuntimeError(
            f"trainer_mode=graph_trainer pinned the Tokamax KDA document budget (max_num_segments) to {budget}, but a "
            f"micro-batch packs {num_documents} documents and needs {num_documents + 1}. Set "
            "engine_kwargs.tpu.kda_document_budget to at least the most documents per micro-batch plus one."
        )


def resolve_memory_policy(options: TPUEngineOptions, activation_checkpoint: str) -> str:
    """TorchTitan graph-trainer memory policy: ``options.graph_trainer``, else analogue of ``activation_checkpoint``."""
    if options.graph_trainer and "memory_policy" in options.graph_trainer:
        return options.graph_trainer["memory_policy"]
    return {"full": "full", "selective": "default"}.get(activation_checkpoint, "none")


def _pallas_loss_config() -> Any:
    """Default ``PallasCrossEntropyLoss.Config`` (module moved to ``components/loss.py`` after 109bd618)."""
    try:
        from torchtitan.experiments.tpu.components.loss import PallasCrossEntropyLoss
    except ImportError:
        from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
    return PallasCrossEntropyLoss.Config()


def build_graph_trainer_config(
    trainer_config_kwargs: dict[str, Any], options: TPUEngineOptions, *, activation_checkpoint: str
) -> Any:
    """``TpuGraphTrainer.Config`` with the eager engine's ``Trainer.Config`` fields plus the graph-trainer settings.

    Eager activation checkpointing is off (``activation_checkpoint=None``): the graph trainer's memory policy pass
    chooses what to save and what to recompute in the traced graph instead. ``compile.enable=False`` keeps the TPU
    recipe parallelize_fn from compiling anything; ``compile.mode=aot_fx_trace`` makes ``forward_backward_step``
    trace and compile the whole step.

    ``compile`` accepts settings straight from ``engine_kwargs.tpu.graph_trainer``
    (passed through to ``TpuGraphTrainerCompileConfig(**cfg)``).

    ``loss`` is a default ``PallasCrossEntropyLoss.Config``: with the fused log-prob head, ``lm_head`` is a
    ``FusedLossLinearWrapper`` and TorchTitan's graph-trainer patches require a loss config whose ``soft_cap`` /
    ``output_multiplier`` match the head (``None`` / ``1.0`` for Qwen3). The plugin replaces ``trainer.loss_fn`` with
    ``GraphPolicyLoss`` before the first step, so this loss is never called.
    """
    from torchtitan.experiments.tpu.graph_trainer.config import TpuGraphTrainerCompileConfig
    from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
    from torchtitan.experiments.tpu.tpu_job_config import TPUConfig

    raw_user_cfg = dict(options.graph_trainer or {})
    user_cfg = dict(raw_user_cfg)

    # Normalize save_regions if passed as string or list
    if "save_regions" in user_cfg:
        sr = user_cfg["save_regions"]
        if isinstance(sr, str):
            user_cfg["save_regions"] = tuple(r.strip() for r in sr.split(",") if r.strip())
        elif isinstance(sr, list | set):
            user_cfg["save_regions"] = tuple(sr)

    if "save_last_n_layers" in user_cfg and isinstance(user_cfg["save_last_n_layers"], dict):
        flat_slnl: dict[str, int] = {}
        for k, v in user_cfg["save_last_n_layers"].items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_slnl[f"{k}.{sub_k}"] = int(sub_v)
            else:
                norm_k = "moe.gate_up" if k == "moe_gate_up" else str(k)
                flat_slnl[norm_k] = int(v)
        user_cfg["save_last_n_layers"] = flat_slnl

    raw_gs = raw_user_cfg.get("graph_split")
    user_set_bwd_part = isinstance(raw_gs, dict) and "backward_partition_mode" in raw_gs

    # Normalize graph_split if passed as a dict
    if "graph_split" in user_cfg and isinstance(user_cfg["graph_split"], dict):
        try:
            from torchtitan.experiments.tpu.graph_trainer.config import GraphSplitterConfig

            user_cfg["graph_split"] = GraphSplitterConfig(**user_cfg["graph_split"])
        except ImportError:
            pass

    memory_policy = user_cfg.pop("memory_policy", None) or resolve_memory_policy(options, activation_checkpoint)
    fsdp_barrier_policy = user_cfg.pop("fsdp_barrier_policy", None) or "none"
    gs_obj = user_cfg.get("graph_split")
    gs_enabled = bool(getattr(gs_obj, "enabled", False)) if gs_obj is not None else False
    enable_remat_barriers = user_cfg.pop(
        "enable_remat_barriers",
        (memory_policy == "tpu_selective" or fsdp_barrier_policy != "none") and not gs_enabled,
    )

    parallelism = trainer_config_kwargs.get("parallelism")
    ep_degree = int(getattr(parallelism, "expert_parallel_degree", 1) or 1)
    cp_degree = int(getattr(parallelism, "context_parallel_degree", 1) or 1)
    cp_strategy = options.context_parallel_strategy or "all_gather"

    compile_kwargs: dict[str, Any] = {
        "enable": False,
        "mode": TPU_COMPILE_MODE,
        "memory_policy": memory_policy,
        "enable_remat_barriers": enable_remat_barriers,
        "fsdp_barrier_policy": fsdp_barrier_policy,
    }
    compile_kwargs.update(user_cfg)
    if ep_degree > 1:
        compile_kwargs["compile_grad_norm"] = False
    if (ep_degree > 1 or (cp_degree > 1 and cp_strategy == "all_gather")) and gs_enabled and not user_set_bwd_part:
        compile_kwargs["graph_split"].backward_partition_mode = "by_layer"

    # bld_block_layout=False keeps packed [T, D] inputs for the plugin's Splash / varlen attention modules, as in
    # eager_blocks mode.
    tpu_config = TPUConfig(use_simple_fsdp=True, bld_block_layout=False)
    compile_config = TpuGraphTrainerCompileConfig(**compile_kwargs)

    kwargs = dict(trainer_config_kwargs)
    if "xla_flags" not in kwargs:
        try:
            from torchtitan.experiments.tpu import tpu_flags

            kwargs["xla_flags"] = tpu_flags.merge_flags(xla_enable_hlo_sharding_v3=False)
        except ImportError:
            pass
    kwargs.update(compile=compile_config, activation_checkpoint=None, tpu_config=tpu_config, loss=_pallas_loss_config())
    logger.warning("TPU trainer_mode=graph_trainer: TpuGraphTrainer (aot_fx_trace), compile_config=%s", compile_config)
    return TpuGraphTrainer.Config(**kwargs)


def _packed_stream_as_one_row(slot_len: int | None = None) -> Any:
    """Sets or clears TorchTitan's unflatten-shim row length while the step runs.

    ``TpuGraphTrainer.forward_backward_step`` sets the row length to ``training.max_context_length`` (the engine's
    ``max_seq_len``). When ``slot_len`` is None, clearing it treats the packed ``[T]`` micro-batch as one row so
    Qwen3.5's GatedDeltaNet resets only at document boundaries. When ``slot_len`` is set (``gdn_row_slot_len`` or
    fixed-slot bin packing), each ``slot_len``-token bin runs as an independent row.
    """
    from verl_hardware_plugin.accelerators.tpu.engines.gdn_slot_schedule import unflatten_seq_len_context

    return unflatten_seq_len_context(slot_len)


def build_rl_graph_trainer(config: Any, *, row_slot_len: int | None = None) -> Any:
    """Builds a ``TpuGraphTrainer`` that runs its step with the configured packed/bin row-slot length.

    See ``_packed_stream_as_one_row``. The traced step returns ``GraphPolicyLoss``'s metrics (the detached per-token
    log-probs) as ``trainer.last_loss_metrics``.
    """
    from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer

    shim_len = int(row_slot_len) if row_slot_len else None

    class _RLTpuGraphTrainer(TpuGraphTrainer):
        def __init__(self, cfg: Any) -> None:
            super().__init__(cfg)
            self.active_slot_len: int | None = shim_len
            self._slot_cache: dict[int | None, tuple[Any, Any]] = {}

        def _make_fx_forward_backward_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            cur_slot = self.active_slot_len
            self._traced_step, self._graph_runner = self._slot_cache.get(cur_slot, (None, None))
            with _packed_stream_as_one_row(cur_slot):
                out = super()._make_fx_forward_backward_step(*args, **kwargs)
            self._slot_cache[cur_slot] = (self._traced_step, self._graph_runner)
            return out

    return _RLTpuGraphTrainer(config)


def _plugin_preprocess_inputs(input_dict: dict[str, Any], **_: Any) -> tuple[Any, Any, dict[str, Any]]:
    """``Decoder.preprocess_inputs`` for the plugin's own inputs: no attention masks, as in the eager engine.

    TorchTitan builds varlen / flex attention masks from ``positions`` when the model's attention config is varlen or
    flex. The plugin swaps in its TPU attention modules (Splash or ``TPUVarlenAttention``), which derive the document
    boundaries from ``positions`` themselves, and the eager engine calls the model with ``positions`` only; building
    the masks would also put per-step, data-dependent shapes into the traced graph's inputs.
    """
    batch = dict(input_dict)
    inputs = batch.pop("input")
    labels = batch.pop("labels")
    return inputs, labels, batch


def prepare_model_for_graph_trainer(model_parts: list[torch.nn.Module], *, ep_enabled: bool = False) -> None:
    """Module FQN annotations for the graph passes, and the plugin's ``preprocess_inputs``; after the attention swap.

    TorchTitan's graph-trainer ``parallelize_qwen3`` annotates every module's forward with its FQN so that the memory
    policy / remat passes find the layer boundaries (``layers.<N>``). The plugin parallelizes with the TPU recipe
    instead (SimpleFSDP without per-block compile) and replaces the attention modules after the model is built, so it
    annotates here, once the module tree is final.
    """
    from torchtitan.experiments.graph_trainer.common_utils import annotate_module_fqns

    try:  # TorchTitan's graph-trainer Qwen3.5 parallelize: FQNs of the EP-chunked expert all-gathers.
        from torchtitan.experiments.tpu.graph_trainer.qwen3_5 import annotate_expert_unshard_fqns
    except ImportError:
        annotate_expert_unshard_fqns = None
    if ep_enabled:
        # As TorchTitan's graph-trainer parallelize_qwen3_5: mark the EP dispatch / combine regions for the passes.
        from torchtitan.experiments.graph_trainer.common_utils import annotate_moe_ep_regions

        annotate_moe_ep_regions()
    for model in model_parts:
        annotate_module_fqns(model)
        if ep_enabled and annotate_expert_unshard_fqns is not None:
            annotate_expert_unshard_fqns(model)
        model.preprocess_inputs = _plugin_preprocess_inputs  # type: ignore[method-assign]


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _parse_rs_specs(
    rollout_rs: str | None, rollout_rs_threshold: str | float | None
) -> tuple[tuple[str, float | None, float], ...]:
    """Parses ``(option_name, lower_bound_or_None, upper_bound)`` tuples for ``rollout_rs``."""
    if not rollout_rs:
        return ()
    opts = []
    for raw in str(rollout_rs).split(","):
        s = raw.strip()
        if s and s not in opts:
            opts.append(s)
    if not opts:
        return ()
    if rollout_rs_threshold is None:
        raise ValueError("rollout_rs_threshold must be set when rollout_rs is enabled")
    thresh_parts = [p.strip() for p in str(rollout_rs_threshold).split(",") if p.strip()]
    if len(thresh_parts) == 1 and len(opts) > 1:
        thresh_parts = thresh_parts * len(opts)
    if len(thresh_parts) != len(opts):
        raise ValueError(f"rollout_rs={rollout_rs!r} has {len(opts)} options, got thresholds {rollout_rs_threshold!r}")
    parsed: list[tuple[str, float | None, float]] = []
    for opt, t_str in zip(opts, thresh_parts, strict=True):
        if opt.endswith("k1"):
            if "_" not in t_str:
                raise ValueError(f"rollout_rs k1 option {opt!r} requires 'lower_upper' threshold, got {t_str!r}")
            lo_s, hi_s = t_str.split("_", 1)
            parsed.append((opt, math.log(float(lo_s)), math.log(float(hi_s))))
        else:
            parsed.append((opt, None, float(t_str)))
    return tuple(parsed)


def _parse_is_bounds(rollout_is_threshold: str | float | None) -> tuple[float | None, float]:
    if rollout_is_threshold is None:
        return None, 2.0
    s = str(rollout_is_threshold).strip()
    if "_" in s:
        lo_s, hi_s = s.split("_", 1)
        return float(lo_s), float(hi_s)
    return None, float(s)


@dataclasses.dataclass(frozen=True)
class GraphPolicyLossSpec:
    """The constants of the traced policy loss; they are baked into the graph, so they must not change per step."""

    clip_ratio_low: float
    clip_ratio_high: float
    clip_ratio_c: float
    temperature: float
    loss_type: str = "ppo_clip"
    rollout_is: str | None = None
    rollout_is_lower: float | None = None
    rollout_is_upper: float = 2.0
    rollout_rs_specs: tuple[tuple[str, float | None, float], ...] = ()

    @property
    def needs_seq_ids(self) -> bool:
        return bool(self.rollout_rs_specs) or self.rollout_is == "sequence"

    @classmethod
    def from_loss_function(
        cls,
        loss_function: Callable,
        *,
        temperature: float,
        calculate_entropy: bool,
        has_rollout_is_weights: bool,
    ) -> GraphPolicyLossSpec:
        """Reads verl's actor config from ``partial(ppo_loss, config=actor_config)``."""
        config = getattr(loss_function, "keywords", {}).get("config") if loss_function is not None else None
        func = getattr(loss_function, "func", loss_function)
        if config is None or getattr(func, "__name__", "") != "ppo_loss":
            raise ValueError(
                "trainer_mode=graph_trainer traces verl's PPO policy loss and needs the actor's "
                f"partial(ppo_loss, config=actor_config) loss function, got {loss_function!r}"
            )
        errors = []
        policy_loss = _config_get(config, "policy_loss")
        loss_mode = _config_get(policy_loss, "loss_mode", "vanilla")
        loss_type = "ppo_clip"
        rollout_is: str | None = None
        rollout_is_lower: float | None = None
        rollout_is_upper: float = 2.0
        rollout_rs_specs: tuple[tuple[str, float | None, float], ...] = ()
        if loss_mode == "bypass_mode":
            rollout_correction = _config_get(policy_loss, "rollout_correction")
            loss_type = str(_config_get(rollout_correction, "loss_type", "ppo_clip") or "ppo_clip")
            if loss_type not in ("ppo_clip", "reinforce"):
                errors.append(f"bypass_mode loss_type={loss_type!r} (only ppo_clip or reinforce)")
            if _config_get(rollout_correction, "rollout_is_batch_normalize", False):
                errors.append("rollout_is_batch_normalize=True is not supported yet")
            raw_is = _config_get(rollout_correction, "rollout_is")
            if raw_is is not None and str(raw_is) != "null":
                rollout_is = str(raw_is)
                if rollout_is not in ("token", "sequence"):
                    errors.append(f"rollout_is={rollout_is!r} (only token or sequence)")
                rollout_is_lower, rollout_is_upper = _parse_is_bounds(
                    _config_get(rollout_correction, "rollout_is_threshold", 2.0)
                )
            raw_rs = _config_get(rollout_correction, "rollout_rs")
            if raw_rs is not None and str(raw_rs) != "null":
                try:
                    rollout_rs_specs = _parse_rs_specs(
                        str(raw_rs), _config_get(rollout_correction, "rollout_rs_threshold")
                    )
                    for opt_name, _, _ in rollout_rs_specs:
                        if not (
                            opt_name in ("token_k1", "token_k2", "token_k3")
                            or opt_name.startswith(("seq_mean_k", "seq_sum_k"))
                        ):
                            errors.append(f"rollout_rs option {opt_name!r} is not supported yet")
                except ValueError as exc:
                    errors.append(str(exc))
        elif loss_mode == "vanilla":
            if has_rollout_is_weights:
                errors.append("rollout_is_weights with loss_mode=vanilla")
        else:
            errors.append(f"policy_loss.loss_mode={loss_mode!r} (only vanilla and bypass_mode)")
        if _config_get(config, "loss_agg_mode", "token-mean") != "token-mean":
            errors.append(f"loss_agg_mode={_config_get(config, 'loss_agg_mode')!r} (only token-mean)")
        if _config_get(config, "use_kl_loss", False):
            errors.append(f"use_kl_loss=True ({_TODO_3})")
        if calculate_entropy or float(_config_get(config, "entropy_coeff", 0.0) or 0.0) != 0.0:
            errors.append("an entropy term / calculate_entropy (needs the [tokens, vocab] logits)")
        clip_ratio = _config_get(config, "clip_ratio", 0.2)
        clip_ratio_low = _config_get(config, "clip_ratio_low")
        clip_ratio_high = _config_get(config, "clip_ratio_high")
        clip_ratio_c = float(_config_get(config, "clip_ratio_c", 3.0))
        if not clip_ratio_c > 1.0:
            errors.append(f"clip_ratio_c={clip_ratio_c} (dual-clip PPO needs > 1)")
        if not math.isfinite(temperature) or temperature <= 0:
            errors.append(f"temperature={temperature}")
        if errors:
            raise ValueError("trainer_mode=graph_trainer does not support: " + "; ".join(errors))
        return cls(
            clip_ratio_low=float(clip_ratio if clip_ratio_low is None else clip_ratio_low),
            clip_ratio_high=float(clip_ratio if clip_ratio_high is None else clip_ratio_high),
            clip_ratio_c=clip_ratio_c,
            temperature=float(temperature),
            loss_type=loss_type,
            rollout_is=rollout_is if loss_type == "reinforce" else None,
            rollout_is_lower=rollout_is_lower,
            rollout_is_upper=rollout_is_upper,
            rollout_rs_specs=rollout_rs_specs,
        )


def _cp_all_reduce_sum(x: torch.Tensor, cp_group: Any) -> torch.Tensor:
    """Trace-safe ``all_reduce(sum)`` over ``cp_group`` (no-op when ``cp_group`` is None)."""
    if cp_group is None:
        return x
    import torch.distributed._functional_collectives as funcol

    out = funcol.all_reduce(x, reduceOp="sum", group=cp_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        out = out.wait()
    return out


def _apply_traced_rollout_rs(
    log_ratio_safe: torch.Tensor,
    loss_mask: torch.Tensor,
    seq_ids: torch.Tensor,
    rs_specs: tuple[tuple[str, float | None, float], ...],
    cp_group: Any = None,
) -> torch.Tensor:
    """Static-shape ``[MAX_PACKED_SEQS, T]`` equivalent of ``compute_rollout_rejection_mask``."""
    if not rs_specs:
        return loss_mask
    seq_grid = torch.arange(MAX_PACKED_SEQS, device=seq_ids.device, dtype=seq_ids.dtype).unsqueeze(1)
    one_hot = ((seq_ids.unsqueeze(0) == seq_grid) & loss_mask.bool().unsqueeze(0)).to(log_ratio_safe.dtype)
    seq_counts = _cp_all_reduce_sum(one_hot.sum(dim=-1), cp_group).clamp_min(1.0)
    token_k1 = -log_ratio_safe
    token_k2 = 0.5 * log_ratio_safe * log_ratio_safe
    token_k3 = torch.exp(log_ratio_safe) - 1.0 - log_ratio_safe
    combined_mask = torch.ones_like(loss_mask, dtype=log_ratio_safe.dtype)
    for opt_name, lo_val, hi_val in rs_specs:
        if opt_name.endswith("k1"):
            stat_tok = token_k1
        elif opt_name.endswith("k2"):
            stat_tok = token_k2
        else:
            stat_tok = token_k3
        if opt_name.startswith("token_"):
            keep_tok = stat_tok <= hi_val if lo_val is None else ((stat_tok >= lo_val) & (stat_tok <= hi_val))
            combined_mask = combined_mask * keep_tok.to(combined_mask.dtype)
        else:
            seq_sum = _cp_all_reduce_sum((one_hot * stat_tok.unsqueeze(0)).sum(dim=-1), cp_group)
            seq_stat = seq_sum / seq_counts if opt_name.startswith("seq_mean_") else seq_sum
            seq_keep = seq_stat <= hi_val if lo_val is None else ((seq_stat >= lo_val) & (seq_stat <= hi_val))
            keep_tok_f = (one_hot * seq_keep.to(one_hot.dtype).unsqueeze(1)).sum(dim=0)
            combined_mask = combined_mask * keep_tok_f
    return (loss_mask * combined_mask).to(loss_mask.dtype)


class GraphPolicyLoss:
    """Trace-safe verl PPO / bypass REINFORCE policy loss, the ``loss_fn`` of ``TpuGraphTrainer`` (pure tensor ops).

    ``pred`` is the model output with the fused LM-head override, ``(hidden_states, lm_head_weight_t)``: log-probs
    come from the Tokamax kernel, with no ``[tokens, vocab]`` logits (graph mode always uses the fused path). ``labels``
    is the tuple ``(target_ids, old_log_probs, advantages, loss_mask[, seq_ids])`` of packed ``[T]`` tensors
    (``pack_rl_loss_targets`` / ``pack_rl_seq_ids``). Returns ``(loss, {LOG_PROBS_METRIC: log_probs.detach()})``;
    TorchTitan returns the metrics from the traced step as ``trainer.last_loss_metrics``.

    Same math as ``verl.trainer.ppo.core_algos`` (vanilla dual-clip PPO or ``bypass_mode`` ``reinforce`` with
    ``rollout_is`` and ``rollout_rs``) with ``token-mean`` ``agg_loss``, without the ``dp_size`` factor.
    """

    def __init__(
        self, spec: GraphPolicyLossSpec, tpu_kernels: frozenset[str] = frozenset(), cp_group: Any = None
    ) -> None:
        self.spec = spec
        self.tpu_kernels = tpu_kernels
        self.cp_group = cp_group

    def log_probs(self, pred: Any, target_ids: torch.Tensor) -> torch.Tensor:
        if not (isinstance(pred, tuple) and len(pred) == 2):
            raise TypeError(
                "trainer_mode=graph_trainer needs the fused LM head output (hidden_states, lm_head_weight_t), got "
                f"{type(pred).__name__}"
            )
        from verl_hardware_plugin.accelerators.tpu.engines import tpu_model_kernels
        from verl_hardware_plugin.accelerators.tpu.engines.fused_logprob import fused_linear_logprobs_tpu

        hidden_states, lm_head_weight_t = pred
        return fused_linear_logprobs_tpu(
            hidden_states,
            target_ids,
            lm_head_weight_t,
            temperature=self.spec.temperature,
            **tpu_model_kernels.lce_block_sizes(
                self.tpu_kernels,
                hidden_size=int(lm_head_weight_t.shape[0]),
                vocab_size=int(lm_head_weight_t.shape[1]),
            ),
        )

    def __call__(
        self, pred: Any, labels: tuple[torch.Tensor, ...], *, global_valid_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(labels) == 5:
            target_ids, old_log_probs, advantages, loss_mask, seq_ids = labels
        else:
            target_ids, old_log_probs, advantages, loss_mask = labels
            seq_ids = None
        log_prob = self.log_probs(pred, target_ids)
        spec = self.spec
        detached_log_ratio = torch.clamp(log_prob.detach() - old_log_probs, min=-20.0, max=20.0)
        effective_mask = loss_mask
        if spec.rollout_rs_specs:
            assert seq_ids is not None
            effective_mask = _apply_traced_rollout_rs(
                detached_log_ratio, loss_mask, seq_ids, spec.rollout_rs_specs, cp_group=self.cp_group
            )

        if spec.loss_type == "reinforce":
            if spec.rollout_is == "token":
                raw_w = torch.exp(detached_log_ratio) * loss_mask
                if spec.rollout_is_lower is not None:
                    kept = (raw_w >= spec.rollout_is_lower) & (raw_w <= spec.rollout_is_upper)
                    w = torch.where(kept, raw_w, torch.zeros_like(raw_w))
                else:
                    w = torch.clamp(raw_w, max=spec.rollout_is_upper)
                pg_losses = -advantages * log_prob * w
            elif spec.rollout_is == "sequence":
                assert seq_ids is not None
                seq_grid = torch.arange(MAX_PACKED_SEQS, device=seq_ids.device, dtype=seq_ids.dtype).unsqueeze(1)
                one_hot = ((seq_ids.unsqueeze(0) == seq_grid) & loss_mask.bool().unsqueeze(0)).to(
                    detached_log_ratio.dtype
                )
                seq_lr = torch.clamp(
                    _cp_all_reduce_sum(
                        (one_hot * (log_prob.detach() - old_log_probs).unsqueeze(0)).sum(dim=-1),
                        self.cp_group,
                    ),
                    -20.0,
                    20.0,
                )
                raw_w = (one_hot * torch.exp(seq_lr).unsqueeze(1)).sum(dim=0) * loss_mask
                if spec.rollout_is_lower is not None:
                    kept = (raw_w >= spec.rollout_is_lower) & (raw_w <= spec.rollout_is_upper)
                    w = torch.where(kept, raw_w, torch.zeros_like(raw_w))
                else:
                    w = torch.clamp(raw_w, max=spec.rollout_is_upper)
                pg_losses = -advantages * log_prob * w
            else:
                pg_losses = -advantages * log_prob
        else:
            negative_approx_kl = torch.clamp(log_prob - old_log_probs, min=-20.0, max=20.0)
            ratio = torch.exp(negative_approx_kl)
            pg_losses1 = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - spec.clip_ratio_low, 1.0 + spec.clip_ratio_high)
            clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
            pg_losses3 = -advantages * spec.clip_ratio_c
            clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
            pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        # verl_F.masked_sum: zero outside the mask first, so bucket-padding tokens cannot leak NaN / inf.
        masked = torch.where(effective_mask.bool(), pg_losses, torch.zeros_like(pg_losses)) * effective_mask
        loss = masked.sum() / torch.clamp(global_valid_tokens.to(masked.dtype), min=1.0)
        return loss, {LOG_PROBS_METRIC: log_prob.detach()}


def _sequence_lengths(micro_batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence prompt and response lengths (CPU) from the nested ``prompts`` / ``responses``.

    verl's V1 trainer (``trainer.use_v1=True``) hands the engine one row per sequence as nested (jagged) tensors.
    """
    prompts = micro_batch["prompts"]
    responses = micro_batch["responses"]
    if not getattr(prompts, "is_nested", False) or not getattr(responses, "is_nested", False):
        raise ValueError(
            "trainer_mode=graph_trainer needs nested (jagged) prompts / responses, as verl's V1 trainer "
            "(trainer.use_v1=True) provides"
        )
    return prompts.offsets().diff().cpu(), responses.offsets().diff().cpu()


def pack_rl_loss_targets(micro_batch: TensorDict, padded_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(old_log_probs, advantages, loss_mask)`` scattered into the packed, bucket-padded token layout (CPU, FP32).

    The model predicts response token ``r`` of sequence ``i`` at packed position ``seq_end_i - resp_len_i - 1 + r``
    (verl's ``no_padding_2_padding`` convention). Every other position (prompts, bucket padding) gets zeros, so the
    loss mask removes it. Shapes depend only on ``padded_len``, so the traced graph's inputs never change shape.
    """
    prompt_lens, response_lens = _sequence_lengths(micro_batch)
    if bool((prompt_lens <= 0).any()):
        raise ValueError(f"every sequence needs a prompt token, got prompt lengths {prompt_lens.tolist()}")
    seq_ends = (prompt_lens + response_lens).cumsum(dim=0)
    input_ids = micro_batch["input_ids"]
    num_tokens = input_ids.values().numel() if getattr(input_ids, "is_nested", False) else input_ids.numel()
    if int(seq_ends[-1]) != num_tokens or num_tokens > padded_len:
        raise ValueError(
            f"packed sequence lengths ({int(seq_ends[-1])} tokens) do not match input_ids ({num_tokens}) / the padded "
            f"length ({padded_len})"
        )
    fields = micro_batch.select("response_mask", "old_log_probs", "advantages").to_padded_tensor()
    width = max(int(t.shape[1]) for t in fields.values())
    cols = torch.arange(width, dtype=torch.int64).unsqueeze(0)
    valid = cols < response_lens.to(torch.int64).unsqueeze(1)
    index = (seq_ends - response_lens - 1).to(torch.int64).unsqueeze(1) + cols
    flat_index = index[valid]
    packed = []
    for key in ("old_log_probs", "advantages", "response_mask"):
        tensor = fields[key].detach().cpu().float()
        if tensor.shape[1] < width:
            tensor = torch.nn.functional.pad(tensor, (0, width - tensor.shape[1]))
        out = torch.zeros(padded_len, dtype=torch.float32)
        out[flat_index] = tensor[valid]
        packed.append(out)
    return packed[0], packed[1], packed[2]


def pack_rl_seq_ids(micro_batch: TensorDict, padded_len: int) -> torch.Tensor:
    """Sequence index ``0 .. num_seqs - 1`` per response token in the packed ``[padded_len]`` layout (CPU, int32)."""
    prompt_lens, response_lens = _sequence_lengths(micro_batch)
    num_seqs = int(prompt_lens.numel())
    if num_seqs > MAX_PACKED_SEQS:
        raise ValueError(
            f"trainer_mode=graph_trainer supports at most {MAX_PACKED_SEQS} sequences per micro-batch for "
            f"sequence-level rollout correction, got {num_seqs}"
        )
    seq_ends = (prompt_lens + response_lens).cumsum(dim=0)
    width = max(int(response_lens.max().item()), 1)
    cols = torch.arange(width, dtype=torch.int64).unsqueeze(0)
    valid = cols < response_lens.to(torch.int64).unsqueeze(1)
    index = (seq_ends - response_lens - 1).to(torch.int64).unsqueeze(1) + cols
    seq_rows = torch.arange(num_seqs, dtype=torch.int32).unsqueeze(1).expand(-1, width)
    out = torch.full((padded_len,), MAX_PACKED_SEQS, dtype=torch.int32)
    out[index[valid]] = seq_rows[valid]
    return out


def check_loss_consistency(
    graph_loss: float,
    verl_loss: float,
    dp_size: int,
    rtol: float = 1e-2,
    *,
    raise_on_mismatch: bool = True,
) -> bool:
    """Checks whether the traced loss (no ``dp_size`` factor) and verl's eager ``ppo_loss`` on the same log-probs agree.

    They compute the same token-mean PPO loss, verl's times ``dp_size``; a mismatch means the graph's loss inputs or
    scaling are wrong, so the gradients are too.

    Args:
        graph_loss: Traced graph scalar loss.
        verl_loss: Eager reference verl loss.
        dp_size: Data parallel size.
        rtol: Relative tolerance (default: 1%).
        raise_on_mismatch: If True (default), raises ``RuntimeError`` on mismatch. If False, logs a warning and
            returns False.

    Raises:
        RuntimeError: if ``raise_on_mismatch=True`` and the losses disagree beyond ``rtol``.
    """
    expected = graph_loss * dp_size
    tolerance = rtol * max(abs(verl_loss), abs(expected)) + 1e-6
    ok = math.isfinite(expected) and math.isfinite(verl_loss) and abs(expected - verl_loss) <= tolerance
    if not ok:
        message = (
            f"graph_trainer: traced loss x dp_size = {expected:.6g} but verl's ppo_loss on the same "
            f"log-probs = {verl_loss:.6g} (diff {abs(expected - verl_loss):.6g} > tolerance {tolerance:.6g})"
        )
        if raise_on_mismatch:
            raise RuntimeError(message)
        logger.warning(message)
    return ok
