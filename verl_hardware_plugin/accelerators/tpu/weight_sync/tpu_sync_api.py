# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""The parts of the tpu-sync library API that the plugin uses, under tpu-sync names.

The tpu-sync library (https://github.com/google/tpu-sync) still names its controller, unit ids, memory types and
metadata protos after its former code name. This module is the only place in the plugin that spells those names; the
rest of the plugin calls everything tpu-sync. The imports are lazy: tpu-sync is only needed once a weight sync
starts, and the CPU tests install a fake ``tpu_sync`` package.
"""

from __future__ import annotations

from typing import Any


def _controller_module() -> Any:
    from tpu_sync.rpc import raiden_controller

    return raiden_controller


def new_controller_server() -> tuple[Any, Any]:
    """``(controller, server)``: a tpu-sync controller on a free port and its (not yet started) RPC server."""
    module = _controller_module()
    controller = module.RaidenController(port=0)
    return controller, module.RaidenControllerServer(controller)


def controller_client(address: str) -> Any:
    """A client of the tpu-sync controller at ``host:port`` (``register_shards`` / ``unregister_shards``)."""
    return _controller_module().RaidenControllerClientFacade(address)


def unit_id(job_name: str, job_replica_id: str, data_name: str = "weights") -> Any:
    """The id of one transfer unit (a rank of a job) on the tpu-sync controller."""
    from tpu_sync.api.common import RaidenId

    return RaidenId(job_name=job_name, job_replica_id=job_replica_id, data_name=data_name)


def dram_memory_type() -> Any:
    """The destination memory type of a transfer into host DRAM."""
    return _controller_module().RaidenMemoryType.DRAM


def variable_metadata(**fields: Any) -> Any:
    """The metadata proto of one registered tensor (name, shape, layout, sharding)."""
    from tpu_sync.rpc import raiden_service_pb2

    return raiden_service_pb2.VariableMetadataProto(**fields)
