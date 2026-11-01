# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Glue between the TPU engine and verl's losses (verl core: ``verl/workers/utils/losses.py``).

The engines stay algorithm-agnostic; the code here is the only TPU code that knows the SFT / PPO loss inputs.
"""
