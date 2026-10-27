# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""FL Environment Manager for verl-hardware-plugin.

Manages FlagGems and FlagCX environment variables for training and rollout phases.

Environment Variables:
    TRAINING_FL_FLAGGEMS_ENABLE: Enable FlagGems for training (true/false/1/0)
    USE_FLAGCX: Enable FlagCX communication (1/0)
    TRAINING_FL_FLAGOS_WHITELIST: FlagGems operator whitelist for training
    TRAINING_FL_FLAGOS_BLACKLIST: FlagGems operator blacklist for training
    VLLM_FL_FLAGOS_WHITELIST: FlagGems operator whitelist for rollout
    VLLM_FL_FLAGOS_BLACKLIST: FlagGems operator blacklist for rollout
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class FLEnvManager:
    """Lightweight FL environment variable manager."""

    TRAINING_ENV_KEYS = [
        "TE_FL_PREFER",
        "TE_FL_STRICT",
        "TE_FL_ALLOW_VENDORS",
        "TE_FL_DENY_VENDORS",
        "TE_FL_PER_OP",
        "TE_FL_PREFER_VENDOR",
        "TE_FL_PLUGIN_MODULES",
        "TE_FL_SKIP_CUDA",
        "TEFL_LOG_LEVEL",
        "TRAINING_FL_FLAGGEMS_ENABLE",
        "TRAINING_FL_FLAGOS_WHITELIST",
        "TRAINING_FL_FLAGOS_BLACKLIST",
    ]

    ROLLOUT_ENV_KEYS = [
        "VLLM_FL_PREFER_ENABLED",
        "VLLM_FL_PLATFORM",
        "VLLM_FL_PREFER",
        "VLLM_FL_OOT_ENABLED",
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_CONFIG",
    ]

    COMMON_ENV_KEYS = [
        "USE_FLAGCX",
        "FLAGCX_PATH",
    ]

    @classmethod
    def is_flaggems_enabled(cls) -> bool:
        return os.getenv("TRAINING_FL_FLAGGEMS_ENABLE", "").lower() in ("1", "true")

    @classmethod
    def is_flagcx_enabled(cls) -> bool:
        return os.getenv("USE_FLAGCX", "").lower() in ("1", "true")

    @classmethod
    def get_training_whitelist(cls) -> Optional[list[str]]:
        val = os.getenv("TRAINING_FL_FLAGOS_WHITELIST", "").strip()
        if not val:
            return None
        return [op.strip() for op in val.split(",") if op.strip()]

    @classmethod
    def get_training_blacklist(cls) -> Optional[list[str]]:
        val = os.getenv("TRAINING_FL_FLAGOS_BLACKLIST", "").strip()
        if not val:
            return None
        return [op.strip() for op in val.split(",") if op.strip()]

    @classmethod
    def get_rollout_whitelist(cls) -> Optional[list[str]]:
        val = os.getenv("VLLM_FL_FLAGOS_WHITELIST", "").strip()
        if not val:
            return None
        return [op.strip() for op in val.split(",") if op.strip()]

    @classmethod
    def get_rollout_blacklist(cls) -> Optional[list[str]]:
        val = os.getenv("VLLM_FL_FLAGOS_BLACKLIST", "").strip()
        if not val:
            return None
        return [op.strip() for op in val.split(",") if op.strip()]

    @classmethod
    def get_summary(cls) -> str:
        parts = []
        parts.append(f"FlagGems={'ON' if cls.is_flaggems_enabled() else 'OFF'}")
        parts.append(f"FlagCX={'ON' if cls.is_flagcx_enabled() else 'OFF'}")
        wl = cls.get_training_whitelist()
        if wl:
            parts.append(f"whitelist={wl}")
        bl = cls.get_training_blacklist()
        if bl:
            parts.append(f"blacklist={bl}")
        return ", ".join(parts)

    @classmethod
    def get_env_snapshot(cls, phase: str = "all") -> dict[str, str]:
        keys = list(cls.COMMON_ENV_KEYS)
        if phase in ("all", "training"):
            keys.extend(cls.TRAINING_ENV_KEYS)
        if phase in ("all", "rollout"):
            keys.extend(cls.ROLLOUT_ENV_KEYS)
        return {k: os.getenv(k, "") for k in keys if os.getenv(k)}


def may_enable_flag_gems(phase: str = "training") -> None:
    """Conditionally enable FlagGems based on environment configuration.

    Args:
        phase: Either "training" or "rollout", controls which whitelist/blacklist to use.
    """
    if not FLEnvManager.is_flaggems_enabled():
        logger.debug("FlagGems is not enabled (TRAINING_FL_FLAGGEMS_ENABLE not set)")
        return

    try:
        import flag_gems

        if phase == "training":
            whitelist = FLEnvManager.get_training_whitelist()
            blacklist = FLEnvManager.get_training_blacklist()
        else:
            whitelist = FLEnvManager.get_rollout_whitelist()
            blacklist = FLEnvManager.get_rollout_blacklist()

        if whitelist and blacklist:
            raise ValueError(f"Cannot set both whitelist and blacklist for {phase} phase.")

        record_path = os.environ.get(f"{'TRAINING' if phase == 'training' else 'ROLLOUT'}_FLAGGEMS_PATH")

        if whitelist:
            logger.info("[FlagGems][%s] Enable only: %s", phase, whitelist)
            flag_gems.only_enable(include=whitelist, record=True, once=True, path=record_path)
        elif blacklist:
            logger.info("[FlagGems][%s] Disable: %s", phase, blacklist)
            flag_gems.enable(unused=blacklist, record=True, once=True, path=record_path)
        else:
            logger.info("[FlagGems][%s] Enable all ops", phase)
            flag_gems.enable(record=True, once=True, path=record_path)

        logger.info("FlagGems version: %s", flag_gems.__version__)

    except ImportError:
        logger.warning(
            "FlagGems is not available but TRAINING_FL_FLAGGEMS_ENABLE is set. "
            "Please install FlagGems: pip install flag-gems"
        )
