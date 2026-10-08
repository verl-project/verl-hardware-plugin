# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Biren SUPA registration declaration."""

from verl_hardware_plugin.registration.backend import BackendRegistration

BACKEND = BackendRegistration(
    package=__package__,
    platform=".platform_supa",
    engines=((".engines.fsdp_supa",), (".engines.megatron_supa",)),
)
