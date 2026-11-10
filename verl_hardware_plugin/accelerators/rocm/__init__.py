# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""AMD ROCm modules.

The ROCm platform itself is built into verl (``verl/plugin/platform/platform_rocm.py``, registered as ``amd``), so
this package registers no platform; ``registration.py`` adds the ROCm-specific components: the ``mori`` checkpoint
engine and the Primus training engines (Megatron and MaxText/JAX) with their Hydra config groups.
"""
