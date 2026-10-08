# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Moore Threads MUSA registration declaration."""

from verl_hardware_plugin.registration.backend import BackendRegistration

BACKEND = BackendRegistration(
    package=__package__,
    platform=".platform_musa",
    engines=((".engines.fsdp_musa",), (".engines.megatron_musa",)),
)
