# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Make verl's fused Triton lm_head + log-prob/entropy kernels fast on Intel XPU.

Why this is needed: `verl.utils.kernel.kernels` (the `triton` fused-kernel backend) loads its
matrix tiles with plain pointer arithmetic. Intel's Triton does not lower that style to 2D block
loads, so on a B60 the stock kernel runs about 1.3-1.5 s for 4096 tokens forward + backward
against about 0.12 s for a chunked vendor-GEMM version. The same kernel file already contains a
second code path built on tensor descriptors (`USE_TMA`), but it is enabled only for CUDA
devices of compute capability >= 9. With Triton >= 3.9 the Intel backend compiles tensor
descriptors to 2D block loads and reaches vendor-GEMM speed (93.7 vs 93.0 TFLOPS on B60).
On Triton 3.8 the same path is about 13x slower than block pointers, so it must stay off there.

What this patch changes in `verl.utils.kernel.kernels`, after the module is imported:
- `SUPPORT_CUDA_TMA = True`, which turns on the tensor-descriptor path (read at call time);
- the allocator that tensor descriptors need (`triton.set_allocator`), because the module only
  sets it at import time for CUDA;
- the one hard-coded autotune tile of the two matmul kernels (forward main loop and
  `d_logits_split_N`), which is tuned for NVIDIA and takes 485-707 ms on B60 instead of ~120 ms.

The tile comes from a per-device table. Only devices with a measured tile are touched; any other
XPU keeps verl's original behaviour. Add a row after measuring a device, do not copy a row.

Gates (all must hold, otherwise this is a no-op and logs why at INFO):
- `VERL_XPU_FUSED_LOGPROB_PATCH` is not "0";
- torch.xpu is available and every visible device name contains a key of `_TILES` with one common tile;
- Triton >= 3.9 and `tl.make_tensor_descriptor` exists.

`VERL_XPU_FUSED_LOGPROB_TILE="BM,BN,BK,warps,stages"` overrides the table (also lets an unlisted
device opt in for tuning).

Import order: verl imports the kernels module lazily (inside the model forward), usually after
`PlatformXPU.__init__`. If the module is not loaded yet, a one-shot import hook patches it
right after it executes; if it is already loaded it is patched at once.
"""

import importlib.abc
import importlib.machinery
import logging
import os
import re
import sys

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False

_KERNELS_MODULE = "verl.utils.kernel.kernels"
_TILED_KERNELS = (
    "efficient_entropy_kernel_general_mainloop",
    "efficient_entropy_backward_kernel_general_d_logits_split_N",
)
# device-name substring -> (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages)
# Best tile per device from a sweep of verl's kernels (Triton 3.9.0, fwd + bwd):
# - B60: best of 5 tiles at 4096 tokens (123 ms).
# - B70: tested by a maintainer on a B70; the same tile as B60 was kept.
# Other devices are not measured and keep verl's defaults.
_TILES = {"B60": (64, 128, 32, 16, 2), "B70": (64, 128, 32, 16, 2)}
_MIN_TRITON = (3, 9)


def _xpu_available() -> bool:
    import torch

    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _triton_version() -> tuple:
    import triton

    m = re.match(r"(\d+)\.(\d+)", triton.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _select_tile():
    """Return the tile tuple to use, or None when this device/stack must keep verl's defaults."""
    override = os.getenv("VERL_XPU_FUSED_LOGPROB_TILE")
    if override:
        tile = tuple(int(x) for x in override.split(","))
        if len(tile) != 5:
            raise ValueError("VERL_XPU_FUSED_LOGPROB_TILE must be 'BM,BN,BK,warps,stages'")
        return tile
    import torch

    # Look at every visible device, not just xpu:0: Ray may pin a worker to any physical index (see
    # ray_device_pin_patch) and this runs before that happens. Only a host whose devices all map to the same
    # tile is patched; a mixed host keeps verl's defaults rather than risk a tile that does not fit.
    names = [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())]
    tiles = {next((t for k, t in _TILES.items() if k in n), None) for n in names}
    if len(tiles) == 1 and None not in tiles:
        return tiles.pop()
    logger.info("fused_logprob_patch: no single measured tile for devices %r, keeping verl defaults", names)
    return None


def _wait_for_import(name: str) -> None:
    """Block until a concurrent import of `name` has finished (a no-op for a reentrant import)."""
    try:
        import importlib._bootstrap as bootstrap

        bootstrap._lock_unlock_module(name)  # type: ignore[attr-defined]
    except Exception:  # private API: if it is missing the caller's completeness check still protects us
        pass


def _patch_kernels_module(mod, tile) -> None:
    import torch
    import triton
    import triton.language as tl

    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="xpu", dtype=torch.int8)

    try:  # Triton >= 3.5 keeps the allocator in a ContextVar; give worker threads a default.
        import contextvars

        import triton.runtime._allocation as _alloc

        if isinstance(getattr(_alloc, "_allocator", None), contextvars.ContextVar):
            _alloc._allocator = contextvars.ContextVar(_alloc._allocator.name, default=alloc_fn)
    except (ImportError, AttributeError):
        pass
    triton.set_allocator(alloc_fn)

    bm, bn, bk, warps, stages = tile
    for name in _TILED_KERNELS:
        kernel = getattr(mod, name)
        base = dict(kernel.configs[0].kwargs)
        base.update(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk)
        kernel.configs = [triton.Config(base, num_warps=warps, num_stages=stages)]
        kernel.cache.clear()
    mod.SUPPORT_CUDA_TMA = hasattr(tl, "make_tensor_descriptor")
    logger.info("fused_logprob_patch: tensor-descriptor path on, tile %s", tile)


class _PatchAfterImport(importlib.abc.MetaPathFinder):
    """One-shot: run the patch right after `verl.utils.kernel.kernels` has been executed."""

    def __init__(self, tile):
        self._tile = tile

    def find_spec(self, fullname, path, target=None):
        if fullname != _KERNELS_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        loader, tile, finder = spec.loader, self._tile, self
        original_exec = loader.exec_module

        def exec_module(module):
            # Leave the hook installed until the module really runs, so a spec-only lookup
            # (importlib.util.find_spec) does not disable the patch.
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)
            original_exec(module)
            try:
                _patch_kernels_module(module, tile)
            except Exception:
                logger.warning("fused_logprob_patch failed; verl's original kernels stay in use", exc_info=True)

        loader.exec_module = exec_module
        return spec


def apply() -> None:
    global _applied
    if _applied:
        return
    if os.getenv("VERL_XPU_FUSED_LOGPROB_PATCH", "1") == "0":
        return
    if not _xpu_available():
        return
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return
    if _triton_version() < _MIN_TRITON or not hasattr(tl, "make_tensor_descriptor"):
        logger.info("fused_logprob_patch: needs Triton >= 3.9 with tensor descriptors (have %s)", triton.__version__)
        return
    tile = _select_tile()
    if tile is None:
        return
    mod = sys.modules.get(_KERNELS_MODULE)
    if mod is None:
        sys.meta_path.insert(0, _PatchAfterImport(tile))
        _applied = True
        return
    _wait_for_import(_KERNELS_MODULE)
    if not all(hasattr(mod, name) for name in _TILED_KERNELS):
        # Another thread is still importing it (or it is an unexpected module): do not claim success, so a
        # later apply() can retry instead of leaving the stock kernels in use for good.
        logger.warning("fused_logprob_patch: %s is not fully imported yet; patch not applied", _KERNELS_MODULE)
        return
    try:
        _patch_kernels_module(mod, tile)
    except Exception:
        logger.warning("fused_logprob_patch failed; verl's original kernels stay in use", exc_info=True)
        return
    _applied = True
