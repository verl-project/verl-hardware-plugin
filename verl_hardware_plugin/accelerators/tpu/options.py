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

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def _parse(name: str, kind: type, value: Any, source: str, label: str = "TPU") -> Any:
    """Converts ``value`` (a config value or an environment string) to ``kind``; ``None`` / ``""`` is unset."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        if kind is ParityMode:
            return ParityMode.parse(value)
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
        use_simple_fsdp: Shard with SimpleFSDP (default: ``use_torch_compile`` when PP and EP are 1).
        use_fused_linear_cross_entropy: Use the Pallas fused linear + cross-entropy kernel for log-probs
            (default: ``use_torch_compile`` when TP is 1).
        fuse_qkv: Fuse attention Q/K/V projections in the model (default: ``True``; ``False`` avoids
            the FSDP all-gather inside ``FusedQKVLinear._split_qkv_on_save`` during ``state_dict()``).
        tpu_eager_mode: ``torch_tpu`` eager mode, ``DEFER_AND_FUSE`` or ``DEFER_NEVER``
            (default: ``DEFER_AND_FUSE`` with ``use_torch_compile``).
        context_parallel_strategy: Context-parallel attention strategy when ``context_parallel_size > 1``:
            ``all_gather`` (load-balanced KV all-gather with Splash Attention, default) or ``ulysses`` (all-to-all).
        optim_impl: torchtitan optimizer implementation (default: ``fused``, ``for-loop`` with TP).
        seq_bucket_size: Packed sequence length bucket in tokens (default: ``pad_to_length_bucket`` when
            ``pad_to_length`` is set, else 256).
        model_kernels: Optional torchtitan TPU kernels of the model (``engines.tpu_model_kernels``): ``auto``
            (default: what the installed torchtitan provides), ``none``, ``all``, or a comma-separated subset.
        param_dtype: Storage dtype of the trained parameters, their gradients and (with the default AdamW) the
            optimizer states: ``float32`` (default) or ``bfloat16``. ``bfloat16`` halves the parameter and gradient
            memory (pure bf16 training, no fp32 master), e.g. for models whose fp32 training state does not fit.
        optimizer_host_offload: Keep the optimizer states in host memory and move them to the device one
            parameter chunk (4 layers) at a time inside the chunked optimizer step (default: ``False``). Unlike
            verl's ``optimizer_offload``, the whole state never has to fit on the device.
        rollout_expert_parallel: The rollout runs the MoE experts with vLLM expert parallelism (default:
            ``False``). ``tpu-sync`` then sends the experts in vLLM's expert-parallel layout, which the rollout
            receives straight into its expert weights (``rollout/qwen3_5_layout.py``).
        fp32_master: With ``param_dtype=bfloat16``, the optimizer steps FP32 master weights with FP32 AdamW states
            and casts the result into the bf16 parameters (default: ``False``). Without it, bf16 weights absorb no
            update smaller than half a bf16 ulp (lr 1e-6 updates mostly round to zero). With
            ``optimizer_host_offload`` the masters live in host memory with the optimizer states.
        ep_capacity_factor: Per-destination-rank token capacity multiplier of ``TpuQwen35TokenDispatcher`` when
            ``expert_parallel_size > 1`` (default: ``4.0``). torchtitan's pretraining default (``1.25``) drops
            skewed RL / math tokens across 32-64 EP ranks while vLLM rollout is dropless.
    """

    use_splash_attention: Optional[bool] = None
    use_simple_fsdp: Optional[bool] = None
    use_fused_linear_cross_entropy: Optional[bool] = None
    fuse_qkv: Optional[bool] = None
    tpu_eager_mode: Optional[str] = None
    context_parallel_strategy: Optional[str] = None
    optim_impl: Optional[str] = None
    seq_bucket_size: Optional[int] = None
    model_kernels: Optional[str] = None
    param_dtype: Optional[str] = None
    optimizer_host_offload: Optional[bool] = None
    rollout_expert_parallel: Optional[bool] = None
    fp32_master: Optional[bool] = None
    ep_capacity_factor: Optional[float] = None


_ENGINE_KINDS: dict[str, type] = {
    "use_splash_attention": bool,
    "use_simple_fsdp": bool,
    "use_fused_linear_cross_entropy": bool,
    "fuse_qkv": bool,
    "tpu_eager_mode": str,
    "context_parallel_strategy": str,
    "optim_impl": str,
    "seq_bucket_size": int,
    "model_kernels": str,
    "param_dtype": str,
    "optimizer_host_offload": bool,
    "rollout_expert_parallel": bool,
    "fp32_master": bool,
    "ep_capacity_factor": float,
}
assert set(_ENGINE_KINDS) == {f.name for f in dataclasses.fields(TPUEngineOptions)}

# Deprecated channels, read for compatibility: option -> environment variable / top-level config field.
_ENGINE_LEGACY_ENV_VARS = {
    "use_splash_attention": "VERL_TPU_USE_SPLASH_ATTENTION",
    "use_simple_fsdp": "VERL_TPU_USE_SIMPLE_FSDP",
    "use_fused_linear_cross_entropy": "VERL_TPU_USE_FUSED_LINEAR_CROSS_ENTROPY",
    "tpu_eager_mode": "VERL_TPU_EAGER_MODE",
    "optim_impl": "VERL_TPU_OPTIM_IMPL",
    "seq_bucket_size": "VERL_TPU_SEQ_BUCKET_SIZE",
}
_ENGINE_LEGACY_CONFIG_FIELDS = (
    "use_splash_attention",
    "use_simple_fsdp",
    "use_fused_linear_cross_entropy",
    "fuse_qkv",
    "tpu_eager_mode",
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
    if values.get("param_dtype") not in (None, "float32", "bfloat16"):
        raise ValueError(
            f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.param_dtype must be float32 or bfloat16, "
            f"got {values['param_dtype']!r}"
        )
    if values.get("fp32_master") and values.get("param_dtype") != "bfloat16":
        raise ValueError(f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.fp32_master needs param_dtype=bfloat16")
    return TPUEngineOptions(**values)


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


@dataclasses.dataclass(frozen=True)
class TPUSyncOptions:
    """Options of the ``tpu-sync`` checkpoint engine.

    Attributes:
        parallelism: Parallel transfer streams per worker.
        verify_parity: Weight-sync check, one of ``ParityMode.MODES`` (``off`` by default): ``norm`` compares
            trainer and rollout weight norms after every sync, ``exact`` compares every tensor the rollout
            received on the step-0 sync bit for bit with the checkpoint vLLM loaded, ``all`` runs both. A failed
            check raises ``TPUSyncParityError`` before generation resumes.
        release_buffers_after_sync: Unbind the send buffers (a bf16 copy of this rank's weights) after every
            transfer, so training gets that HBM back between syncs.
        tie_word_embeddings: Whether ``lm_head`` is tied to the input embedding, so it need not be sent.
            ``None`` sends ``lm_head``, which is correct for tied and untied models alike.
    """

    parallelism: int = TPU_SYNC_DEFAULT_PARALLELISM
    verify_parity: str = "off"
    release_buffers_after_sync: bool = True
    tie_word_embeddings: Optional[bool] = None

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
    "verify_parity": ParityMode,
    "release_buffers_after_sync": bool,
    "tie_word_embeddings": bool,
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
