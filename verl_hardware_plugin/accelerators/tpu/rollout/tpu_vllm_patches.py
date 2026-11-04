# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for the TPU rollout.

The rollout runs on vLLM's ``mp`` executor and vllm-torchtpu's ``mp`` multi-host backend, which need no
patch to span several hosts. Each remaining patch fills a gap in vLLM (v0.29.0 - v0.32.0) and vllm-torchtpu
(ff860050 - 732c4346) and can be dropped once upstream closes it:

* ``TPUWorker.reset_encoder_cache``: no-op. verl resets vLLM's caches after every weight update,
  and vllm-torchtpu's worker does not implement this call. Upstream fix: implement it.
* ``VLLM_DISABLE_COMPILE_CACHE=1``: a reloaded compile-cache artifact can make vLLM run the model
  eagerly, which crashes libtpu (b/501165531). Upstream fix: that bug.
* ``multiprocessing.process.BaseProcess.__init__``: install these patches in the EngineCore process and
  in vLLM's worker processes, which do not import the plugin.
"""

import logging
import multiprocessing.process
import os

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False


def patch_vllm_for_tpu() -> None:
    """Install the patches in this process. Later calls return at once."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    patch_multiprocessing_for_tpu()
    # vLLM can reload a degenerate AOT compile artifact (num_artifacts=0) that silently runs the
    # model eagerly; eager execution then hits the XLA:TPU unaligned-DUS CHECK (b/501165531).
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    # Imported here: CPU tests and processes outside the rollout may lack vLLM or vllm-torchtpu.
    try:
        from vllm_torchtpu.worker.tpu_worker import TPUWorker
    except ImportError as exc:
        logger.debug("Skipping vLLM TPU patches (vllm / vllm_torchtpu not installed): %s", exc)
        return

    TPUWorker.reset_encoder_cache = _reset_encoder_cache
    _PATCHES_APPLIED = True
    logger.info("Successfully applied all TPU patches to vLLM and vllm-torchtpu.")


def _reset_encoder_cache(self) -> None:
    """``TPUWorker.reset_encoder_cache``, which verl calls after every weight update: a no-op.

    The encoder cache holds multimodal encoder outputs; text models have none to reset.
    """


class PickleableProcessWrapper:
    """Process target that runs ``patch_vllm_for_tpu`` in the child before ``target``.

    This is how the patches reach vLLM's EngineCore and its ``mp`` workers. Inside a Ray actor vLLM always
    uses the ``spawn`` start method, and a spawned child imports only the modules its pickled target names
    (``EngineCoreProc.run_engine_core``, which vllm-torchtpu replaces with its own wrapper, or
    ``WorkerProc.worker_main``), so without this wrapper it would never import the plugin.
    """

    def __init__(self, target):
        self.target = target

    def __call__(self, *args, **kwargs):
        patch_vllm_for_tpu()
        return self.target(*args, **kwargs)


def patch_multiprocessing_for_tpu() -> None:
    """Wrap the target of every ``multiprocessing`` process started from now on in ``PickleableProcessWrapper``."""
    base_process = multiprocessing.process.BaseProcess
    if getattr(base_process, "_tpu_patched", False):
        return
    original_init = base_process.__init__

    def patched_init(self, group=None, target=None, *args, **kwargs):
        if target is not None:
            target = PickleableProcessWrapper(target)
        original_init(self, group, target, *args, **kwargs)

    base_process.__init__ = patched_init  # type: ignore[method-assign]
    base_process._tpu_patched = True  # type: ignore[attr-defined]
