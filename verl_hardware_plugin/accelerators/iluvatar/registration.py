# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Iluvatar registration declaration."""

from verl_hardware_plugin.registration.backend import BackendRegistration

BACKEND = BackendRegistration(
    package=__package__,
    platform=".platform_cuda_iluvatar",
    engines=((".engines.fsdp_iluvatar",), (".engines.megatron_iluvatar",)),
)
