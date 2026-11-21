# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Run one case of ``inproc_serve_child.py`` in a fresh interpreter, as a trainer rank of a one-rank world on CPU.
Test code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
LIBTPU_ARGS = "--xla_tpu_dvfs_p_state=7 --xla_tpu_use_dynamic_smem_negotiation=true"


def run_case(tmp_path, case: dict, port: int, extra: Optional[dict] = None) -> dict:
    """The case's RESULT (the child's environment is built here, from nothing but the import path); ``extra`` adds
    variables. ``INPROC_TEST_DUMP`` keeps every RESULT."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    path = os.pathsep.join([HERE] + [p for p in sys.path if p and p != HERE])
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "TMPDIR": str(tmp_path),
        "CUDA_VISIBLE_DEVICES": "",
        "VLLM_TARGET_DEVICE": "cpu",
        "VERL_PLATFORM": "cpu",
        "VERL_USE_EXTERNAL_MODULES": "inproc_cpu_platform,verl_hardware_plugin",
        "INPROC_TEST_PATH": path,
        "PYTHONPATH": path,
        "VERL_TPU_ROLLOUT_INPROC": "1",
        "VERL_TPU_INPROC_ROLE": "trainer",
        "VERL_LOGGING_LEVEL": "INFO",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "TPU_VISIBLE_DEVICES": "0",
        "TPU_VISIBLE_CHIPS": "0",
        "TORCH_TPU_TOPOLOGY": "1,1,1,1",
        "ALLOW_MULTIPLE_LIBTPU_LOAD": "0",
        "LIBTPU_INIT_ARGS": LIBTPU_ARGS,
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "1",
        "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES": "0",
        "TPU_ACCELERATOR_TYPE": "tpu7x",
        "INPROC_TEST_CASE": json.dumps(case),
        **(extra or {}),
    }
    for k in ("ONEDNN_JIT_PROFILE", "DNNL_JIT_PROFILE"):
        if k in os.environ:
            env[k] = os.environ[k]
    child = os.path.join(HERE, "inproc_serve_child.py")
    p = subprocess.run([sys.executable, "-I", "-B", child], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    assert line, f"the case printed no RESULT (rc {p.returncode}): {p.stderr[-3000:]}"
    res = json.loads(line[-1][len("RESULT ") :])
    dump = os.environ.get("INPROC_TEST_DUMP")
    if dump:
        os.makedirs(dump, exist_ok=True)
        with open(os.path.join(dump, f"serve.{case['mode']}.{port}.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
    assert "error" not in res, res.get("traceback") or res["error"]
    return res
