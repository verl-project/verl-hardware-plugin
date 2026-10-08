# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""MetaX registration declaration."""

from verl_hardware_plugin.registration.backend import BackendRegistration

BACKEND = BackendRegistration(
    package=__package__,
    platform=".platform_cuda_metax",
    engines=((".engines.fsdp_metax",), (".engines.megatron_metax",)),
)
