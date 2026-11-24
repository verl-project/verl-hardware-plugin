# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""In-process (same-process) colocation of the vLLM rollout with the trainer on TPU. Opt-in, default off.

A TPU TensorCore can be opened by one process only, there is no cross-process device-memory sharing, and
vllm-torchtpu has no sleep mode, so verl's colocation with process separation cannot share TPU chips. The colocated
rollout builds one TP=1 vLLM engine inside each trainer process.

So far the package holds library modules only: nothing imports it and no registration reads its switch
(``VERL_TPU_ROLLOUT_INPROC``, :func:`inproc_enabled`), so no run is affected.
"""

from __future__ import annotations

from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import (
    INPROC_ENV,
    check_switches,
    inproc_enabled,
    logger,
)

__all__ = ["INPROC_ENV", "check_switches", "inproc_enabled", "logger"]
