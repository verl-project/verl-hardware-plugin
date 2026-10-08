# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Cross-accelerator FlagOS registration declaration."""

from verl_hardware_plugin.registration.backend import BackendRegistration

BACKEND = BackendRegistration(
    package=__package__,
    engines=((".engines.fsdp_flagos",), (".engines.megatron_flagos",)),
)
