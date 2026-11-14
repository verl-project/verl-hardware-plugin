# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the in-process (colocated) TPU rollout (``verl_hardware_plugin.accelerators.tpu.rollout.inproc``).

Run them one file per process (several start child interpreters or patch process-wide state), for example::

    for f in tests/accelerators/tpu/inproc/test_*.py; do VLLM_TARGET_DEVICE=cpu pytest -q "$f"; done

Dependencies: the ``dev`` extra, and the ``inproc-test`` extra (vLLM on its CPU platform) for every test that builds
or drives vLLM; without it those tests are skipped.

Optional inputs; a test that needs one is skipped without it:

* ``INPROC_TEST_VLLM_TORCHTPU_SRC``: a vllm-torchtpu source directory (the one holding ``vllm_torchtpu``): the
  env_override review, and the tests that run vllm-torchtpu's runner code (the call-site pins, the engine entry).

``INPROC_TEST_WORKERS`` (default 2) sets how many child interpreters run at once.
"""
