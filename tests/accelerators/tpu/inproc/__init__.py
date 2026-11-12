# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the in-process (colocated) TPU rollout (``verl_hardware_plugin.accelerators.tpu.rollout.inproc``).

Run them one file per process (several start child interpreters or patch process-wide state), for example::

    for f in tests/accelerators/tpu/inproc/test_*.py; do VLLM_TARGET_DEVICE=cpu pytest -q "$f"; done

Dependencies: the ``dev`` extra, and the ``inproc-test`` extra (vLLM on its CPU platform, TransferQueue) for every
test that builds or drives vLLM or verl's replay buffer; without it those tests are skipped.

Optional inputs; a test that needs one is skipped without it:

* ``INPROC_TEST_VLLM_TORCHTPU_SRC``: a vllm-torchtpu source directory (the one holding ``vllm_torchtpu``): the
  env_override review, and the tests that run vllm-torchtpu's runner code (the call-site pins, the engine entry, the
  recompile guard on the substitution kernel, the precompile grid);
* ``INPROC_TEST_CPU_E2E=1`` with ``INPROC_TEST_TOKENIZER`` (a local HF tokenizer directory whose vocabulary covers
  token ids 1000-41000): the end-to-end and serving tests that build real vLLM engines on CPU (minutes);
* ``INPROC_DEFAULT_OFF_BASELINE``: a plugin tree without the feature, for the default-off differential.

``INPROC_TEST_WORKERS`` (default 2) sets how many child interpreters run at once; ``INPROC_TEST_EVIDENCE_DIR`` and
``INPROC_TEST_DUMP`` keep the children's records.
"""
