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

Phase 1 scope:

* Actor training only. ``old_log_probs`` must come from the rollout (``rollout_correction.bypass_mode=True``) and the
  KL loss must be off, so no forward-only pass runs: ``forward_only`` / ``infer_batch`` raises (TODO-3), and so does a
  reference (``forward_only``) engine configured with this mode.
* Policy loss: verl's vanilla dual-clip PPO (``loss_mode=vanilla``, or ``bypass_mode`` with ``loss_type=ppo_clip``
  and no rejection sampling), ``token-mean`` aggregation, no entropy term. Its tensor math is traced into the graph;
  verl's own ``ppo_loss`` then runs eagerly under ``no_grad`` on the per-token log-probs the graph returns, so the
  logged loss and metrics (``pg_clipfrac``, ``ppo_kl``, rollout-correction metrics, ...) are verl's.
* One micro-batch per mini-batch per rank: TorchTitan's ``TpuGraphTrainer`` assigns ``param.grad`` instead of
  accumulating it, so more micro-batches raise (TODO-1). Set ``ppo_max_token_len_per_gpu`` (with ``use_dynamic_bsz``)
  and ``static_micro_batch_len`` large enough for one micro-batch. Its loss is normalized by the mini-batch's global
  token count, as in the eager engine.
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

# The traced step's auxiliary output (``_wrap_step_for_loss_aux``): the detached per-token
# log-probs of the micro-batch, from which verl's metrics are computed eagerly.
NUM_LOSS_AUX_OUTPUTS = 1
# TorchTitan's compile mode for the graph trainer: trace forward + loss + backward of a step into one FX graph.
TPU_COMPILE_MODE = "aot_fx_trace"

_TODO_1 = (
    "graph_trainer mode runs one micro-batch per mini-batch per rank (TODO-1: gradient accumulation across "
    "micro-batches)"
)
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


def multi_micro_batch_error(num_micro_batches: int) -> RuntimeError:
    return RuntimeError(
        f"{_TODO_1}, got {num_micro_batches}. Raise actor.ppo_max_token_len_per_gpu (use_dynamic_bsz) and "
        "engine_kwargs.tpu.static_micro_batch_len so that a mini-batch fits in one micro-batch per rank."
    )


def validate_graph_trainer_engine(
    engine_config: Any, options: TPUEngineOptions, *, use_simple_fsdp: bool, tpu_kernels: frozenset[str]
) -> None:
    """Raises ``ValueError`` listing every setting Phase 1 of ``trainer_mode=graph_trainer`` does not support."""
    # TODO: drop checks here as follow-ups land:
    # - forward_only (TODO-3): once TorchTitan's graph trainer can trace a forward-only log-prob step.
    # - static_micro_batch_len (TODO-2): once the traced step is cached per sequence shape (TorchTitan / plugin).
    # - pipeline / context / tensor / expert parallel sizes: once the graph-trainer path supports and is validated
    #   with them.
    # - kda: once the plugin pins the Tokamax KDA document budget for the run instead of setting it per batch.
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
    for name in ("pipeline_parallel_size", "context_parallel_size", "tensor_parallel_size", "expert_parallel_size"):
        size = int(getattr(engine_config, name, 1) or 1)
        if size > 1:
            errors.append(f"{name}={size} is not supported yet (FSDP only)")
    if getattr(engine_config, "param_offload", False):
        errors.append("param_offload swaps parameter storages between steps; it is not supported yet")
    if "kda" in tpu_kernels:
        errors.append("the Tokamax KDA kernels change their document budget per batch; that would freeze in the graph")
    if errors:
        raise ValueError("engine_kwargs.tpu.trainer_mode=graph_trainer: " + "; ".join(errors))


def resolve_memory_policy(options: TPUEngineOptions, activation_checkpoint: str) -> str:
    """TorchTitan graph-trainer memory policy: the option, else the analogue of ``activation_checkpoint``."""
    if options.graph_trainer_memory_policy:
        return options.graph_trainer_memory_policy
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
    trace and compile the whole step. Needs TorchTitan with ``TpuGraphTrainerCompileConfig`` (e.g. 109bd618, the
    verl TPU Dockerfile pin).

    ``loss`` is a default ``PallasCrossEntropyLoss.Config``: with the fused log-prob head, ``lm_head`` is a
    ``FusedLossLinearWrapper`` and TorchTitan's graph-trainer patches require a loss config whose ``soft_cap`` /
    ``output_multiplier`` match the head (``None`` / ``1.0`` for Qwen3). The plugin replaces ``trainer.loss_fn`` with
    ``GraphPolicyLoss`` before the first step, so this loss is never called.
    """
    from torchtitan.experiments.tpu.graph_trainer.config import TpuGraphTrainerCompileConfig
    from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
    from torchtitan.experiments.tpu.tpu_job_config import TPUConfig

    memory_policy = resolve_memory_policy(options, activation_checkpoint)
    save_regions = tuple(r.strip() for r in (options.graph_trainer_save_regions or "").split(",") if r.strip())
    fsdp_schedule = options.graph_trainer_fsdp_schedule or "none"
    # TorchTitan validates these combinations (save regions vs. memory policy, FSDP schedule vs. remat) in
    # TpuGraphTrainer.__init__, before the model is built.
    # As TorchTitan's Qwen3-30B-A3B SFT recipe: remat barriers with the selective policy; FSDP schedules require them.
    enable_remat_barriers = memory_policy == "tpu_selective" or fsdp_schedule != "none"

    # bld_block_layout=False keeps packed [T, D] inputs for the plugin's Splash / varlen attention modules, as in
    # eager_blocks mode.
    tpu_config = TPUConfig(use_simple_fsdp=True, bld_block_layout=False)
    compile_config = TpuGraphTrainerCompileConfig(
        enable=False,
        mode=TPU_COMPILE_MODE,
        memory_policy=memory_policy,
        save_regions=save_regions,
        enable_remat_barriers=enable_remat_barriers,
        fsdp_barrier_policy=fsdp_schedule,
    )

    kwargs = dict(trainer_config_kwargs)
    kwargs.update(compile=compile_config, activation_checkpoint=None, tpu_config=tpu_config, loss=_pallas_loss_config())
    logger.warning(
        "TPU trainer_mode=graph_trainer: TpuGraphTrainer (aot_fx_trace), memory_policy=%s, save_regions=%s, "
        "fsdp_schedule=%s",
        memory_policy,
        save_regions,
        fsdp_schedule,
    )
    return TpuGraphTrainer.Config(**kwargs)


def _wrap_step_for_loss_aux(trainer: Any, num_aux: int):
    """Context manager threading ``num_aux`` extra loss tensors out of upstream ``TpuGraphTrainer``'s FX step.

    TorchTitan's ``make_fwd_bwd_step`` drops any second tuple element returned by ``loss_fn`` instead of returning it
    as a graph output. Wrapping ``make_fwd_bwd_step`` during trace appends the auxiliary tensors (detached per-token
    log-probs) to the graph's outputs, and wrapping ``run_traced`` pops them onto ``trainer.last_loss_aux`` before
    ``TpuGraphTrainer`` unpacks ``[loss, *grads, *routing_metrics]``, leaving TorchTitan's pass pipeline untouched.
    """
    from contextlib import contextmanager

    from torchtitan.experiments.graph_trainer import make_fx_tracer
    from torchtitan.experiments.graph_trainer import trainer as trainer_mod

    @contextmanager
    def _ctx():
        orig_make_fwd_bwd = trainer_mod.make_fwd_bwd_step
        orig_run_traced = make_fx_tracer.run_traced

        def make_fwd_bwd_with_aux(model: Any, loss_fn: Callable, *args: Any, **kwargs: Any) -> Callable:
            captured_aux: list[tuple[torch.Tensor, ...]] = []

            def loss_fn_capturing_aux(*l_args: Any, **l_kwargs: Any) -> torch.Tensor:
                result = loss_fn(*l_args, **l_kwargs)
                if not (isinstance(result, tuple) and len(result) == 2):
                    raise ValueError(
                        "trainer_mode=graph_trainer loss_fn must return (loss, tuple_of_aux_tensors), "
                        f"got {type(result).__name__}"
                    )
                loss_val, aux = result
                if not isinstance(aux, tuple | list) or len(aux) != num_aux:
                    raise ValueError(f"Expected {num_aux} auxiliary loss outputs, got {aux!r}")
                captured_aux.append(tuple(aux))
                return loss_val

            base_step = orig_make_fwd_bwd(model, loss_fn_capturing_aux, *args, **kwargs)

            def fwd_bwd_step(*s_args: Any, **s_kwargs: Any) -> list[torch.Tensor]:
                captured_aux.clear()
                outs = list(base_step(*s_args, **s_kwargs))
                if len(captured_aux) != 1:
                    raise RuntimeError(f"Expected loss_fn to run once during trace, ran {len(captured_aux)} times")
                outs.extend(captured_aux[0])
                return outs

            return fwd_bwd_step

        def run_traced_with_aux(*r_args: Any, **r_kwargs: Any) -> Callable:
            runner = orig_run_traced(*r_args, **r_kwargs)

            def run(*call_args: Any, **call_kwargs: Any) -> list[torch.Tensor]:
                outs = list(runner(*call_args, **call_kwargs))
                trainer.last_loss_aux = tuple(outs[-num_aux:])
                return outs[:-num_aux]

            return run

        if trainer._traced_step is None:
            trainer_mod.make_fwd_bwd_step = make_fwd_bwd_with_aux  # type: ignore[assignment]
        make_fx_tracer.run_traced = run_traced_with_aux  # type: ignore[assignment]
        try:
            yield
        finally:
            trainer_mod.make_fwd_bwd_step = orig_make_fwd_bwd  # type: ignore[assignment]
            make_fx_tracer.run_traced = orig_run_traced  # type: ignore[assignment]

    return _ctx()


def _rl_graph_trainer_init(self: Any, cfg: Any) -> None:
    super(type(self), self).__init__(cfg)
    self.last_loss_aux = ()


def _rl_graph_trainer_make_fx_step(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
    with _wrap_step_for_loss_aux(self, NUM_LOSS_AUX_OUTPUTS):
        return super(type(self), self)._make_fx_forward_backward_step(*args, **kwargs)


def build_rl_graph_trainer(config: Any) -> Any:
    """Builds a ``TpuGraphTrainer`` whose traced step also returns the detached per-token log-probs (aux output)."""
    from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer

    cls = type(
        "_RLTpuGraphTrainer",
        (TpuGraphTrainer,),
        {
            "__init__": _rl_graph_trainer_init,
            "_make_fx_forward_backward_step": _rl_graph_trainer_make_fx_step,
        },
    )
    return cls(config)


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


def prepare_model_for_graph_trainer(model_parts: list[torch.nn.Module]) -> None:
    """Module FQN annotations for the graph passes, and the plugin's ``preprocess_inputs``; after the attention swap.

    TorchTitan's graph-trainer ``parallelize_qwen3`` annotates every module's forward with its FQN so that the memory
    policy / remat passes find the layer boundaries (``layers.<N>``). The plugin parallelizes with the TPU recipe
    instead (SimpleFSDP without per-block compile) and replaces the attention modules after the model is built, so it
    annotates here, once the module tree is final.
    """
    from torchtitan.experiments.graph_trainer.common_utils import annotate_module_fqns

    for model in model_parts:
        annotate_module_fqns(model)
        model.preprocess_inputs = _plugin_preprocess_inputs  # type: ignore[method-assign]


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


@dataclasses.dataclass(frozen=True)
class GraphPolicyLossSpec:
    """The constants of the traced policy loss; they are baked into the graph, so they must not change per step."""

    clip_ratio_low: float
    clip_ratio_high: float
    clip_ratio_c: float
    temperature: float

    @classmethod
    def from_loss_function(
        cls,
        loss_function: Callable,
        *,
        temperature: float,
        calculate_entropy: bool,
        has_rollout_is_weights: bool,
    ) -> GraphPolicyLossSpec:
        """Reads verl's actor config from ``partial(ppo_loss, config=actor_config)`` and checks Phase 1 supports it."""
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
        if loss_mode == "bypass_mode":
            rollout_correction = _config_get(policy_loss, "rollout_correction")
            loss_type = _config_get(rollout_correction, "loss_type", "ppo_clip")
            if loss_type != "ppo_clip":
                errors.append(f"bypass_mode loss_type={loss_type!r} (only ppo_clip)")
            if _config_get(rollout_correction, "rollout_rs") is not None:
                errors.append("rollout rejection sampling (rollout_rs) changes the loss mask")
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
        )


class GraphPolicyLoss:
    """Trace-safe verl vanilla dual-clip PPO loss, the ``loss_fn`` of ``TpuGraphTrainer`` (pure tensor ops).

    ``pred`` is the model output: ``(hidden_states, lm_head_weight_t)`` with the fused LM-head override (log-probs from
    the Tokamax kernel, no ``[tokens, vocab]`` logits), else the logits (``use_fused_linear_cross_entropy=False``:
    plain log-softmax, the fallback if the fused op does not trace). ``labels`` is the tuple
    ``(target_ids, old_log_probs, advantages, loss_mask)`` of packed ``[T]`` tensors (``pack_rl_loss_targets``).
    Returns ``(loss, (log_probs.detach(),))``; the second element becomes the graph's auxiliary output.

    Same math as ``verl.trainer.ppo.core_algos.compute_policy_loss_vanilla`` with ``token-mean`` ``agg_loss``, without
    the ``dp_size`` factor (see the module docstring).
    """

    def __init__(self, spec: GraphPolicyLossSpec, tpu_kernels: frozenset[str] = frozenset()) -> None:
        self.spec = spec
        self.tpu_kernels = tpu_kernels

    def log_probs(self, pred: Any, target_ids: torch.Tensor) -> torch.Tensor:
        temperature = self.spec.temperature
        if isinstance(pred, tuple):
            from verl_hardware_plugin.accelerators.tpu.engines import tpu_model_kernels
            from verl_hardware_plugin.accelerators.tpu.engines.fused_logprob import fused_linear_logprobs_tpu

            hidden_states, lm_head_weight_t = pred
            return fused_linear_logprobs_tpu(
                hidden_states,
                target_ids,
                lm_head_weight_t,
                temperature=temperature,
                functional_backward=True,
                **tpu_model_kernels.lce_block_sizes(
                    self.tpu_kernels,
                    hidden_size=int(lm_head_weight_t.shape[0]),
                    vocab_size=int(lm_head_weight_t.shape[1]),
                ),
            )
        logits = pred.reshape(-1, pred.shape[-1]).float() / temperature
        log_softmax = torch.log_softmax(logits, dim=-1)
        return torch.gather(log_softmax, -1, target_ids.reshape(-1, 1).long()).squeeze(-1)

    def __call__(
        self, pred: Any, labels: tuple[torch.Tensor, ...], *, global_valid_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        target_ids, old_log_probs, advantages, loss_mask = labels
        log_prob = self.log_probs(pred, target_ids)
        spec = self.spec
        negative_approx_kl = torch.clamp(log_prob - old_log_probs, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - spec.clip_ratio_low, 1.0 + spec.clip_ratio_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * spec.clip_ratio_c
        clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        # verl_F.masked_sum: zero outside the mask first, so bucket-padding tokens cannot leak NaN / inf.
        masked = torch.where(loss_mask.bool(), pg_losses, torch.zeros_like(pg_losses)) * loss_mask
        loss = masked.sum() / torch.clamp(global_valid_tokens.to(masked.dtype), min=1.0)
        return loss, (log_prob.detach(),)


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


def check_loss_consistency(graph_loss: float, verl_loss: float, dp_size: int, rtol: float = 1e-2) -> bool:
    """Warns if the traced loss (no ``dp_size`` factor) and verl's eager ``ppo_loss`` on the same log-probs disagree.

    They compute the same token-mean PPO loss, verl's times ``dp_size``; a mismatch means the graph's loss inputs or
    scaling are wrong, so the gradients are too.
    """
    expected = graph_loss * dp_size
    tolerance = rtol * max(abs(verl_loss), abs(expected)) + 1e-6
    ok = math.isfinite(expected) and math.isfinite(verl_loss) and abs(expected - verl_loss) <= tolerance
    if not ok:
        logger.warning(
            "graph_trainer: traced loss x dp_size = %.6g but verl's ppo_loss on the same log-probs = %.6g",
            expected,
            verl_loss,
        )
    return ok
