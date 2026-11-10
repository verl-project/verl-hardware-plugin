# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Primus MaxText YAML -> MaxText config keys, and the per-process JAX setup."""

from __future__ import annotations

import os
import socket
from typing import Any, Optional

from ..yaml_config import ensure_primus_importable, load_module_params

# Keys Primus' own launcher consumes; MaxText's pyconfig rejects them.
_PRIMUS_ONLY = ("file_sink_level", "stderr_sink_level", "sink_level", "trainable", "model", "override_model")


def load_maxtext_params(config) -> dict[str, Any]:
    params = load_module_params(
        config.primus_config_path,
        module=config.primus_module,
        overrides=config.primus_overrides,
        framework="maxtext",
        primus_path=config.primus_path,
    )
    for key in _PRIMUS_ONLY:
        params.pop(key, None)
    # One rank per GPU; verl starts the JAX distributed system before MaxText looks at it.
    params["hardware"] = "gpu_multiprocess"
    # A zero eval batch makes the log-prob step slice every sequence away.
    per_device = params.get("per_device_batch_size")
    if per_device is not None and float(params.get("eval_per_device_batch_size") or 0) <= 0:
        params["eval_per_device_batch_size"] = per_device
    params.setdefault("gradient_accumulation_steps", 1)
    return params


def maxtext_path(config) -> str | None:
    path = config.maxtext_path or os.environ.get("MAXTEXT_PATH")
    if path is None and (config.primus_path or os.environ.get("PRIMUS_PATH")):
        candidate = os.path.join(config.primus_path or os.environ["PRIMUS_PATH"], "third_party", "maxtext")
        if os.path.isdir(os.path.join(candidate, "src")):
            path = candidate
    return path


def prepare_jax_env(config) -> None:
    """Apply Primus' MaxText environment defaults, keeping memory free for the torch weight sync."""
    fraction = str(config.mem_fraction)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = fraction
    ensure_primus_importable(config.primus_path)
    from primus.backends.maxtext.maxtext_adapter import MaxTextAdapter

    adapter = MaxTextAdapter()
    adapter.apply_env_defaults()
    # apply_env_defaults uses setdefault, so the fraction above wins.
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = fraction
    path = maxtext_path(config)
    if path:
        adapter.setup_backend_path(path)


def initialize_jax_distributed() -> None:
    """Join every verl worker of this group into one JAX job, one local GPU per process.

    verl has already initialized torch.distributed (gloo on CPU), so rank 0 publishes the
    coordinator address through it.
    """
    import jax
    import torch.distributed as dist

    if jax.distributed.is_initialized():
        return
    rank, world = dist.get_rank(), dist.get_world_size()
    address: list[Optional[str]] = [None]
    if rank == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        address[0] = f"{os.environ.get('MASTER_ADDR', socket.gethostbyname(socket.gethostname()))}:{port}"
    dist.broadcast_object_list(address, src=0, device="cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    jax.distributed.initialize(
        coordinator_address=address[0],
        num_processes=world,
        process_id=rank,
        local_device_ids=[local_rank],
    )
