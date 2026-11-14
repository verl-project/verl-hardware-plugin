# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Exercise the logging adapter for the released verl trainer."""

import logging
from unittest.mock import Mock

from verl_hardware_plugin.accelerators.neuron.compat_verl import (
    install_separate_async_logging_compatibility,
)


def test_stable_trainer_logging_is_idempotent(monkeypatch):
    logger = logging.getLogger("verl.trainer.ppo.v1.trainer_separate_async")
    original = Mock()
    monkeypatch.setattr(logger, "info", original)
    install_separate_async_logging_compatibility()
    wrapped = logger.info
    install_separate_async_logging_compatibility()
    assert logger.info is wrapped
    logger.info("standalone %s", 4, flush=True, extra={"step": 0})
    original.assert_called_once_with("standalone %s", 4, extra={"step": 0})
