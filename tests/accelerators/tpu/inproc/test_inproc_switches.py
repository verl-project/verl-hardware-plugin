# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the in-process rollout's switches (``rollout/inproc/common.py``): validated. A malformed value
refuses; a ``VERL_TPU_INPROC_*`` variable the package does not read is named. No gate reads the feature switch
yet: nothing imports the package.
"""

import pytest

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import common


@pytest.mark.parametrize(
    "name, value",
    [
        (common.INTEGRITY_ENV, "flag"),
        (common.PARITY_EVERY_ENV, "five"),
        (common.PARITY_EVERY_ENV, "-1"),
        (common.LAST_STEP_ENV, "x"),
        (common.GC_RSS_GIB_ENV, "-2"),
        (common.GC_RSS_GIB_ENV, "lots"),
    ],
)
def test_malformed_switch_values_refuse(name, value):
    with pytest.raises(RuntimeError, match=name):
        common.check_switches({common.INPROC_ENV: "1", name: value})


def test_switch_defaults():
    env = {common.INPROC_ENV: "1"}
    assert common.check_switches(env) == []
    assert common.integrity_mode(env) == "fatal" and common.parity_every(env) == 5 and common.last_step(env) is None
    assert common.gc_rss_limit_bytes(env) == 8 * 2**30
    assert common.gc_rss_limit_bytes({common.GC_RSS_GIB_ENV: "0"}) == 0
    assert common.integrity_mode({common.INTEGRITY_ENV: "report"}) == "report"


def test_unread_variables_are_named():
    env = {common.INPROC_ENV: "1", "VERL_TPU_INPROC_SOMETHING": "1", common.LAST_STEP_ENV: "8", "VERL_TPU_OTHER": "1"}
    assert common.check_switches(env) == ["VERL_TPU_INPROC_SOMETHING"]
