# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ECCL checkpoint engine for Enflame GCU.

Same wire format as the NCCL/HCCL checkpoint engines: the trainer broadcasts
bucketed weights to the rollout workers, with the per-bucket tensor metadata
sent separately over zeromq. The collective runs through
``ray.util.collective``; the backend follows the platform (``eccl`` by default,
``flagcx`` when ``USE_FLAGCX=1``), which is how torch_gcu exposes inter-process
communication (there is no cupy-nccl equivalent).

Device operations go through ``torch.gcu`` directly instead of
``verl.utils.device``: the in-tree GCU helpers (``is_torch_gcu_available`` and
friends) were removed from verl main, and this engine lives in the plugin.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator, Optional

import ray
import ray.util.collective as collective
import torch
import zmq

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.plugin.platform import get_platform
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Device type reported by the Enflame platform; buffers are allocated as "<device>:<index>".
_DEVICE_TYPE = "gcu"


def _gcu_available() -> bool:
    """Whether a real GCU is usable in this process.

    Importing ``torch_gcu`` is not enough: the same image may contain it on a
    CUDA host, where this engine must not claim the ``nccl`` key. The probe lives
    in ``platform_enflame`` so registration and the engine share one definition.
    """
    try:
        from verl_hardware_plugin.accelerators.enflame.platform_enflame import gcu_available

        return gcu_available()
    except Exception:  # pragma: no cover - a probe must never raise at import time
        return False


# Require an actual GCU, not just an importable torch_gcu: this engine owns the
# "nccl" key, and registering it on a CUDA host that happens to have torch_gcu
# installed would replace the working NCCL engine with ECCL.
if not _gcu_available():
    raise ImportError("ECCLCheckpointEngine is unavailable because no Enflame GCU device is available.")


def _gcu_module() -> Any:
    """Return the ``torch.gcu`` module used for device operations."""
    return torch.gcu


@dataclass
class MasterMetadata:
    zmq_ip: str
    zmq_port: int


class BroadcastOperation:
    """Broadcast one bucket from rank 0 over ECCL.

    The collective runs on the calling thread rather than a ``run_in_executor``
    thread, matching the MLU/HCCL engines and avoiding cross-thread stream issues
    on the GCU runtime. Do not move it to an executor unless that change has
    first been verified on real hardware.

    Args:
        rank (int): The rank of the current process.
        group_name (str): The name of the ECCL process group.
        bucket (torch.Tensor): The uint8 buffer to broadcast.
        metadata (dict[str, TensorMeta] | None): Metadata to publish on rank 0.
        socket (zmq.Socket): The zeromq socket used for metadata.
        topic (str): The zeromq topic.
    """

    def __init__(
        self,
        rank: int,
        group_name: str,
        bucket: torch.Tensor,
        metadata: dict[str, TensorMeta] | None,
        socket: zmq.Socket,  # type: ignore[name-defined]
        topic: str,
    ) -> None:
        self.rank = rank
        self.group_name = group_name
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic

        self._run()

    def _run(self):
        # broadcast tensor meta via zeromq PUB/SUB
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
        else:
            self.socket.recv_string()
            self.metadata = self.socket.recv_pyobj()

        # broadcast tensor via ECCL
        collective.broadcast(self.bucket, src_rank=0, group_name=self.group_name)

    async def wait_for_complete(self) -> dict[str, TensorMeta]:
        """Wait for the broadcast operation to complete.

        Returns:
            dict[str, TensorMeta]: The bucket meta after broadcast.
        """
        assert self.metadata is not None
        return self.metadata


# The registry key and the collective backend are separate concepts: "nccl" is
# only a compatibility alias that configs/scripts may set explicitly, while
# "eccl"/"flagcx" is what ray.util.collective receives. On Ascend verl's HCCL
# engine takes the "nccl" alias too, so those scripts run unchanged; this does
# the same for Enflame GCU. verl's own NCCL engine cannot be imported here
# (cupy, hard-coded "cuda"), so nothing is shadowed. If it ever becomes usable
# on GCU, drop this alias or it will override it. The module-level guard above
# keeps both registrations off non-GCU hosts.
@CheckpointEngineRegistry.register("nccl")
@CheckpointEngineRegistry.register("eccl")
class ECCLCheckpointEngine(CheckpointEngine):
    """ECCL checkpoint engine with collective communication.

    The registry names ``eccl`` and ``nccl`` are verl config keys, not
    collective backend names. The backend passed to
    ``collective.init_collective_group`` follows
    ``PlatformENFLAME.communication_backend_name()``: ``"eccl"`` by default and
    ``"flagcx"`` when ``USE_FLAGCX=1``, so the FlagCX/FlagOS path keeps working.

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
            two buffers to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
        group_name (str): The name of the ECCL process group. Defaults to "default".
        rebuild_group (bool): Whether to rebuild the process group in each update. Defaults to False.
        is_master (bool): Whether the current process is the master process. Defaults to False.
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to
            torch.bfloat16.
    """

    def __init__(
        self,
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.device = _gcu_module().current_device()
        # Keep the platform's choice: USE_FLAGCX=1 must not be silently forced
        # back to ECCL. The registry name stays "eccl"/"nccl"; only this
        # argument changes.
        platform = get_platform()
        if platform.device_name == "gcu":
            self.collective_backend = platform.communication_backend_name()
        else:
            self.collective_backend = "eccl"
            logger.warning(
                "Platform %r did not resolve to GCU; falling back to the 'eccl' collective backend. "
                "Confirm VERL_PLATFORM=enflame so USE_FLAGCX and the platform backend selection take effect.",
                platform.device_name,
            )
        # Per-instance state, set by init_process_group. `collective.is_group_initialized`
        # is process-global, so a group created by another instance must not be
        # mistaken for this instance's own registration.
        self.rank: Optional[int] = None
        self.world_size: Optional[int] = None
        self.send_buf: Optional[torch.Tensor] = None
        self.recv_buf: Optional[torch.Tensor] = None

        # start zeromq server for broadcasting bucket tensor metadata
        self.is_master = is_master
        self.topic = "bucket_metadata"
        if self.is_master:
            self._start_zmq_server()

    def _buffer_device(self) -> str:
        return f"{_DEVICE_TYPE}:{self.device}"

    def prepare(self) -> Optional[MasterMetadata]:
        device = self._buffer_device()
        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=device)
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=device)

        return MasterMetadata(zmq_ip=self.ip, zmq_port=self.zmq_port) if self.is_master else None

    def finalize(self):
        """Destroy the ECCL process group if rebuild_group is True."""
        # The zeromq socket is intentionally kept: CheckpointEngineManager calls
        # prepare() + init_process_group() for every weight update, so recreating
        # the SUB socket here would discard an unclosed connection and expose the
        # ZeroMQ slow-joiner race again (the receiver could block in recv_string).
        if self.rebuild_group:
            if self.rank is not None and self.rank >= 0:
                collective.destroy_collective_group(self.group_name)
            self.rank = None
            self.world_size = None

        self.send_buf = None
        self.recv_buf = None
        _gcu_module().empty_cache()

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        assert metadata, "build_topology requires the master metadata produced by actor rank 0"
        actor_wg_kwargs = {
            "rank": [0] + [-1] * (actor_wg_world_size - 1),
            "world_size": [rollout_world_size + 1] * actor_wg_world_size,
            "master_metadata": [metadata[0]] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "master_metadata": [metadata[0]] * rollout_world_size,
        }
        return actor_wg_kwargs, rollout_kwargs

    def _start_zmq_server(self):
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.zmq_port, _ = get_free_port(self.ip)

        context = zmq.Context()  # type: ignore[attr-defined]
        self.socket = context.socket(zmq.PUB)  # type: ignore[attr-defined]
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)  # type: ignore[attr-defined]
        else:
            address = f"tcp://{self.ip}:{self.zmq_port}"

        self.socket.bind(address)

    def _connect_zmq_client(self, metadata: MasterMetadata):
        assert not self.is_master, "Master process should not connect to other processes."
        context = zmq.Context()  # type: ignore[attr-defined]
        self.socket = context.socket(zmq.SUB)  # type: ignore[attr-defined]
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)  # type: ignore[attr-defined]
        else:
            address = f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"

        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)  # type: ignore[attr-defined]

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Initialize the ECCL process group.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
            master_metadata (MasterMetadata): The metadata of the master process.
        """
        # For actor workers other than rank 0, their rank should be -1.
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            return

        if self.rebuild_group or not collective.is_group_initialized(self.group_name):
            try:
                collective.init_collective_group(world_size, rank, self.collective_backend, self.group_name)
            except Exception:
                logger.error(
                    "Failed to initialize the checkpoint engine collective group with backend=%s "
                    "(from the platform communication_backend_name(); USE_FLAGCX=1 selects flagcx). "
                    "If the current Ray build does not expose this collective backend, set USE_FLAGCX=0 "
                    "to use ECCL, or contact the vendor for a Ray build that supports it.",
                    self.collective_backend,
                )
                raise
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank is not None, (
                f"ECCL group {self.group_name!r} is already initialized but was not created by this engine "
                "instance; set rebuild_group=True to rebuild it"
            )
            assert self.rank == rank, f"rank {rank} is not equal to self.rank {self.rank}"
            assert self.world_size == world_size, (
                f"world_size {world_size} is not equal to self.world_size {self.world_size}"
            )

        # Reuse the subscription so later updates do not race a new connection.
        if self.rank > 0 and not hasattr(self, "socket"):
            self._connect_zmq_client(master_metadata)

        # The barrier is the group's first collective, so it is where the backend
        # decides the rank space. Log the inputs here: a regression (the backend
        # asking for device_count() * world_size ranks) is otherwise only visible
        # as an ECCL "configuration invalid" from groupEnd().
        logger.info(
            "init_process_group rank: %s, world_size: %s, visible GCUs: %s, device: %s",
            self.rank,
            self.world_size,
            _gcu_module().device_count(),
            self.device,
        )

        collective.barrier(self.group_name)

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
            global_steps: Optional trainer step/version associated with this weight update.
        """
        assert self.rank is not None and self.rank <= 0, "Trainer workers other than rank 0 should not send weights."

        # For actor rank other than 0, consume weights without sending.
        if self.rank < 0:
            for _name, _weight in weights:
                pass
            return

        assert self.send_buf is not None and self.recv_buf is not None, "prepare() allocates the buffers"
        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op: Optional[BroadcastOperation] = None

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
            # fill the tensor bucket
            if offset + tensor_meta.chunk_size > self.bucket_size:
                _gcu_module().synchronize()

                # wait previous broadcast op finish
                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                broadcast_op = BroadcastOperation(
                    rank=self.rank,
                    group_name=self.group_name,
                    bucket=send_buf,
                    metadata={"bucket_meta": bucket_meta, "is_last": False},
                    socket=self.socket,
                    topic=self.topic,
                )

                # swap send_buf and recv_buf
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            assert offset + tensor_meta.chunk_size <= self.bucket_size
            assert tensor_meta.name not in bucket_meta

            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            send_buf[offset : offset + tensor_meta.chunk_size] = chunk
            offset += tensor_meta.chunk_size

        # broadcast last bucket
        _gcu_module().synchronize()
        if broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        broadcast_op = BroadcastOperation(
            rank=self.rank,
            group_name=self.group_name,
            bucket=send_buf,
            metadata={"bucket_meta": bucket_meta, "is_last": True},
            socket=self.socket,
            topic=self.topic,
        )
        await broadcast_op.wait_for_complete()

        # `wait_for_complete()` only guarantees the ECCL kernel is enqueued, so
        # synchronize before the buffers can be freed.
        _gcu_module().synchronize()

        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        async for name, weight in merge_weight_chunks(self._receive_weight_chunks(), self.bucket_size):
            yield name, weight

    async def _receive_weight_chunks(self) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        """Receive the weight chunks of the model.

        Yields:
            A tuple of the weight chunk metadata and the chunk itself.
        """
        assert self.rank is not None and self.rank > 0, "Rank 0 should not receive weights."
        assert self.send_buf is not None and self.recv_buf is not None, "prepare() allocates the buffers"
        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        # receive first bucket
        start_time = time.time()
        broadcast_op = BroadcastOperation(
            rank=self.rank,
            group_name=self.group_name,
            bucket=recv_buf,
            metadata=None,
            socket=self.socket,
            topic=self.topic,
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata["bucket_meta"])

        # wait for the ECCL broadcast kernel to finish before yielding tensors;
        # otherwise a non-blocking clone could race the buffer reuse.
        _gcu_module().synchronize()

        # swap send_buf and recv_buf
        send_buf, recv_buf = recv_buf, send_buf
        while not metadata["is_last"]:
            # 1. receive next bucket
            broadcast_op = BroadcastOperation(
                rank=self.rank,
                group_name=self.group_name,
                bucket=recv_buf,
                metadata=None,
                socket=self.socket,
                topic=self.topic,
            )

            # 2. yield tensor from send_buf
            for tensor_meta in metadata["bucket_meta"].values():
                tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
                yield tensor_meta, tensor

            # 3. wait for next bucket broadcast finish
            metadata = await broadcast_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(metadata["bucket_meta"])

            # 4. swap send_buf and recv_buf
            _gcu_module().synchronize()  # sync non-blocking copy
            send_buf, recv_buf = recv_buf, send_buf

        # yield tensor from send_buf
        for tensor_meta in metadata["bucket_meta"].values():
            tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
            yield tensor_meta, tensor

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
