# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""AWS Trainium platform registration, independent of the training and serving SDKs."""

import os

from .platform_neuron import PlatformNeuron
from .runtime import ensure_neuron, isolate_checkpoint_imports, process_role

__all__ = ["PlatformNeuron"]

if os.environ.get("VERL_PLATFORM") == "aws":
    isolate_checkpoint_imports()
    if process_role() == "trainer":
        ensure_neuron()
