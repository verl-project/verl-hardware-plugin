# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""The in-process rollout's build harness for the CPU tests that run vLLM's real engine (one fresh interpreter per
case). Test code.

:func:`setup` makes the process a trainer rank of a one-rank gloo world, imports verl (the CPU test platform and the
plugin's registration, through ``VERL_USE_EXTERNAL_MODULES``) and replaces the two TPU-only steps of the build: the
torch_tpu runtime check and the vllm-torchtpu worker patches (the logits-at-load hook goes on the test worker class,
``inproc_arch_worker.ArchModelWorker`` or a subclass). :class:`_Lines` keeps every line the package logs.
"""

from __future__ import annotations

import logging
import os
import tempfile

LOGGER = "verl_hardware_plugin.accelerators.tpu.rollout.inproc"


class _Lines(logging.Handler):
    """Every line the package logs, in order."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def mark(self) -> int:
        return len(self.lines)

    def since(self, i: int) -> list[str]:
        return self.lines[i:]


def pick(lines: list[str], key: str):
    return next((x for x in lines if key in x), None)


def setup(case: dict, cap: _Lines):
    """The gloo world, verl (the CPU platform and the plugin's registration) and the two TPU-only replacements."""
    import torch.distributed as dist

    store = tempfile.NamedTemporaryFile(prefix="inproc-pg-", dir=os.environ["TMPDIR"], delete=False)
    dist.init_process_group("gloo", init_method=f"file://{store.name}", rank=0, world_size=1)
    logging.getLogger(LOGGER).addHandler(cap)
    import verl  # noqa: F401  (the CPU platform + verl_hardware_plugin through VERL_USE_EXTERNAL_MODULES: registration)
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import rollout

    logging.getLogger(LOGGER).setLevel(logging.INFO)  # (the package sets VERL_LOGGING_LEVEL's level at its import)

    import inproc_arch_worker as W

    rollout.TPUInProcRollout._check_runtime_up = lambda self: None

    def _patch_vllm_in_process(self):
        from verl_hardware_plugin.accelerators.tpu.rollout.inproc.compile_guard import LOGITS_AT_LOAD

        if not LOGITS_AT_LOAD.install(W.ArchModelWorker):
            raise RuntimeError("compute_logits at load: ArchModelWorker has no load_model of its own to hook")

    rollout.TPUInProcRollout._patch_vllm_in_process = _patch_vllm_in_process
    return rollout, W
