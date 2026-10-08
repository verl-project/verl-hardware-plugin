# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Enflame GCU registration declaration."""

from verl_hardware_plugin.registration.backend import BackendRegistration

BACKEND = BackendRegistration(
    package=__package__,
    platform=".platform_enflame",
    engines=((".engines.fsdp_enflame",), (".engines.megatron_enflame",)),
)
