# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Collective-communication shim for PlatformXPU.get_collective_module().

verl.utils.rendezvous.ray_backend's create_nccl_communicator_in_ray() expects
a module shaped like cupy.cuda.nccl: a module-level get_unique_id() (called
once, by rank 0, before any communicator exists) and a
NcclCommunicator(ndev, commId, rank) class.

XCCL has no raw unique-id primitive, so this shim bootstraps a
torch.distributed process group with backend="cpu:gloo,xpu:xccl" instead.
get_unique_id() only reserves a (host, port) pair; the TCPStore
itself is created internally by init_process_group()'s own tcp:// rendezvous,
not by this module.
"""

import socket

import torch.distributed as dist


def get_unique_id():
    import ray

    host = ray.util.get_node_ip_address()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        port = s.getsockname()[1]
    return (host, port)


class NcclCommunicator:
    """Shape-compatible stand-in for cupy.cuda.nccl.NcclCommunicator, backed by a
    hybrid cpu:gloo,xpu:xccl torch.distributed process group."""

    def __init__(self, ndev, commId, rank):
        host, port = commId
        dist.init_process_group(
            backend="cpu:gloo,xpu:xccl",
            init_method=f"tcp://{host}:{port}",
            rank=rank,
            world_size=ndev,
        )
        self._rank = rank

    def rank_id(self) -> int:
        return self._rank

    @property
    def _pg(self):
        return dist.group.WORLD
