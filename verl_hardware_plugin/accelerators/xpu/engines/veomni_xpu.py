# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""VeOmni engine for Intel XPU devices.

verl-core registers its VeOmni engine for ``device=["cuda", "npu"]`` only, so on
XPU ``EngineRegistry.get_engine_cls(model_type, "veomni")`` finds nothing. These
subclasses register the same engine under ``(xpu, intel)`` and add the same
oneCCL workaround the XPU FSDP engine applies (see ``fsdp_xpu.py``): VeOmni's
FSDP2 wrappers also default to ReduceOp.AVG, which oneCCL can abort on, so
force SUM + manual division on every FSDP2 module after initialization.

No core code is patched; this is ordinary subclassing.
"""

import importlib
import logging
import os
import sys

try:
    importlib.import_module("veomni.models.checkpoint_tensor_loading")
except ModuleNotFoundError:
    # VeOmni main moved this module to veomni.models.checkpoint.convert (VeOmni#1142,
    # same functions); verl main still imports the old path. Alias it so the unmodified
    # core engine module imports against VeOmni main.
    sys.modules["veomni.models.checkpoint_tensor_loading"] = importlib.import_module("veomni.models.checkpoint.convert")

from verl.workers.engine.base import EngineRegistry  # noqa: E402
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead, VeOmniEngineWithValueHead  # noqa: E402

from .fsdp_xpu import _force_sum_reduction_on_all_fsdp_modules  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _release_xpu_cache() -> None:
    """Release cached, unused device memory back to the driver.

    Called after each engine pass so the allocator's cache does not accumulate across phases
    and starve colocated processes such as the rollout engine. Set VERL_XPU_EMPTY_CACHE=0 to disable.
    """
    if os.getenv("VERL_XPU_EMPTY_CACHE", "1") == "0":
        return
    import torch

    torch.xpu.empty_cache()


def _release_xpu_cache_if_bloated() -> None:
    """Release cached device memory once the allocator's reserved pool exceeds a threshold.

    Batches with variable sequence lengths can make the caching allocator reserve new segments
    instead of reusing cached ones. The XPU runtime may fail an allocation without first freeing
    the cache and retrying, so the pool is trimmed proactively.

    Threshold: VERL_XPU_EMPTY_CACHE_RESERVED_GB (default 8; 0 trims after every micro-batch).
    Set VERL_XPU_EMPTY_CACHE=0 to disable.
    """
    if os.getenv("VERL_XPU_EMPTY_CACHE", "1") == "0":
        return
    import torch

    limit = float(os.getenv("VERL_XPU_EMPTY_CACHE_RESERVED_GB", "8")) * 1024**3
    if torch.xpu.memory_reserved() > limit:
        torch.xpu.empty_cache()


def _log_xpu_memory(tag: str) -> None:
    """Print device memory usage when VERL_XPU_MEMLOG=1 (debugging aid)."""
    if os.getenv("VERL_XPU_MEMLOG") != "1":
        return
    import torch

    gb = 1024**3
    free, total = torch.xpu.mem_get_info()
    print(
        f"[xpu-mem] pid={os.getpid()} dev={torch.xpu.current_device()} {tag}: "
        f"alloc={torch.xpu.memory_allocated() / gb:.2f}G "
        f"reserved={torch.xpu.memory_reserved() / gb:.2f}G max_alloc={torch.xpu.max_memory_allocated() / gb:.2f}G "
        f"dev_free={free / gb:.2f}G/{total / gb:.2f}G",
        flush=True,
    )


@EngineRegistry.register(model_type="language_model", backend=["veomni"], device="xpu", vendor="intel")
class VeOmniXPUEngineWithLMHead(VeOmniEngineWithLMHead):
    """VeOmni LM engine for Intel XPU with xccl communication backend."""

    def forward_step(self, *args, **kwargs):
        try:
            return super().forward_step(*args, **kwargs)
        finally:
            _release_xpu_cache_if_bloated()

    def forward_backward_batch(self, *args, **kwargs):
        _log_xpu_memory("before forward_backward_batch")
        try:
            return super().forward_backward_batch(*args, **kwargs)
        finally:
            _release_xpu_cache()
            _log_xpu_memory("after forward_backward_batch")

    def to(self, *args, **kwargs):
        _log_xpu_memory(f"before to({args})")
        out = super().to(*args, **kwargs)
        _log_xpu_memory(f"after to({args})")
        return out

    def initialize(self):
        super().initialize()
        _force_sum_reduction_on_all_fsdp_modules(self.module)


@EngineRegistry.register(model_type="value_model", backend=["veomni"], device="xpu", vendor="intel")
class VeOmniXPUEngineWithValueHead(VeOmniEngineWithValueHead):
    """VeOmni value-model engine for Intel XPU with xccl communication backend."""

    def initialize(self):
        super().initialize()
        _force_sum_reduction_on_all_fsdp_modules(self.module)
