# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import AsyncGenerator, Generator

import ray
import torch
import torch.distributed as dist
import zmq
from torch._C._distributed_c10d import BroadcastOptions, ProcessGroupXCCL

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

if not (hasattr(torch, "xpu") and dist.is_available()):
    raise ImportError("XCCLCheckpointEngine needs a torch build with XPU support.")

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_STORE_TIMEOUT_S = 300


@dataclass
class MasterMetadata:
    """Endpoint of the broadcast source (actor rank 0), handed to every worker in the group."""

    zmq_ip: str
    zmq_port: int
    store_port: int
    group_id: str
    multi_sender: bool


@dataclass
class WorkerMetadata:
    """What each worker reports from `prepare()` for `build_topology` to place it in the group.

    Only the source rank fills `master`; every worker reports `node_id` so it can be matched
    against the source's.
    """

    node_id: str
    master: MasterMetadata | None = None


def _broadcast(pg: dist.ProcessGroup, tensor: torch.Tensor) -> None:
    """Broadcast `tensor` from group rank 0."""
    opts = BroadcastOptions()
    opts.rootRank = 0
    pg.broadcast([tensor], opts).wait()


# Offsets inside a bucket are rounded up to this many bytes. The receiver rebuilds a weight with
# ``chunk.view(dtype)``, which needs the chunk to start at a multiple of the element size, and a bucket can mix dtypes.
# 16 covers every torch dtype.
_ALIGN = 16


def _place(offset: int, size: int, bucket_size: int) -> tuple[int, bool]:
    """Where the next chunk of ``size`` bytes starts in a bucket that is filled up to ``offset``.

    Returns ``(start, flush)``. When ``flush`` is true the current bucket has to be sent first and the chunk starts
    a new one at 0. The sender and the relay both lay buckets out with this function, so their bucket boundaries
    cannot drift apart.
    """
    start = -(-offset // _ALIGN) * _ALIGN
    if start + size > bucket_size:
        return 0, True
    return start, False


def _wait_for_subscribers(socket: zmq.Socket, subscribed: int, expected: int, timeout_s: float = 60.0) -> int:
    """Read the subscription frames of an XPUB socket until ``expected`` consumers are subscribed.

    ZeroMQ silently drops a PUB message that has no matching subscription yet, so publishing right after the
    consumers called ``connect`` can lose a bucket's metadata and leave them blocked in ``recv``. Only the publisher
    can tell that a subscription has arrived, and an XPUB socket hands it the subscription frames.

    ZeroMQ reports a consumer leaving only when it was the last one on the topic, so a consumer that disconnects
    while others stay subscribed is not seen here; consumers restarting mid-run is not supported by this engine.

    Returns the updated number of live subscriptions.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        # Drain what is already queued first, so an unsubscribe that is already waiting is applied before
        # deciding whether enough consumers are subscribed.
        if not socket.poll(0, zmq.POLLIN):
            if subscribed >= expected:
                return subscribed
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0 or not socket.poll(remaining_ms, zmq.POLLIN):
                raise TimeoutError(
                    f"only {subscribed} of {expected} consumers subscribed to the bucket metadata stream "
                    f"within {timeout_s:.0f}s"
                )
        frame = socket.recv()
        if frame[:1] == b"\x01":
            subscribed += 1
        elif frame[:1] == b"\x00":
            subscribed -= 1


class BroadcastOperation:
    """Async broadcast operation with XCCL in separate thread.

    Args:
        rank (int): The rank of the current process.
        pg (dist.ProcessGroup): The XCCL process group to broadcast on.
        bucket (torch.Tensor): The tensor to broadcast.
        metadata (dict[str, TensorMeta] | None): The metadata of the tensor; None on the receiving side until
            it arrives over ZeroMQ.
        socket (zmq.Socket): The zeromq socket to communicate with master.
        topic (str): The topic to subscribe.
    """

    def __init__(
        self,
        rank: int,
        pg: dist.ProcessGroup,
        bucket: torch.Tensor,
        metadata: dict[str, TensorMeta] | None,
        socket: zmq.Socket,
        topic: str,
    ) -> None:
        self.rank = rank
        self.pg = pg
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _run(self):
        # broadcast tensor meta via zeromq PUB/SUB
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
        else:
            self.socket.recv_string()
            self.metadata = self.socket.recv_pyobj()
            assert self.metadata is not None
            self.bucket = self.bucket[: self.metadata["length"]]

        # broadcast tensor via XCCL
        _broadcast(self.pg, self.bucket)

    async def wait_for_complete(self) -> dict[str, TensorMeta]:
        """Wait for the broadcast operation to complete.
        (This does not guarantee that the XCCL kernel has finished, only that it has been enqueued.)

        Returns:
            dict[str, TensorMeta]: The bucket meta after broadcast.
        """
        await self._task
        assert self.metadata is not None
        return self.metadata


# "nccl" is the name the disaggregated trainer configs ship with. On Ascend verl's HCCL engine takes that name too, so
# the stock configs run unchanged; this does the same for XPU. verl's own NCCL engine cannot be imported here (cupy,
# hard-coded "cuda"), so nothing is shadowed. If it ever becomes usable on XPU, drop this alias or it will override it.
@CheckpointEngineRegistry.register("nccl")
@CheckpointEngineRegistry.register("xccl")
class XCCLCheckpointEngine(CheckpointEngine):
    """XCCL (oneCCL) checkpoint engine for Intel GPUs, the counterpart of the NCCL engine.

    The process group is a torch `ProcessGroupXCCL` over a TCPStore hosted by actor rank 0, so it needs
    neither `ray.util.collective`, cupy nor vLLM, and it leaves the default torch.distributed world alone.

    Registered as backend ``"xccl"`` and, like verl's HCCL engine on Ascend, also as ``"nccl"`` when the plugin loads
    on an XPU host (see ``../registration.py``). The one-step-off and fully-async trainer configs already default to
    ``checkpoint_engine.backend: nccl``, so on XPU they need no override; ``backend=xccl`` selects it explicitly.
    ``engine_kwargs`` are looked up under whichever name is configured. The transport is confined to
    ``_start_store``, ``_create_process_group``, ``_destroy_process_group`` and ``_broadcast``, which is where a
    different transport (a Ray collective group, a vLLM communicator) would plug in.

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
            two buffer to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
        group_name (str): The name of the XCCL process group. Defaults to "default".
        rebuild_group (bool): Whether to rebuild the XCCL process group in each update. Defaults to False.
        is_master (bool): Whether the current process is the master process. Defaults to False.
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to torch.bfloat16.
        multi_sender (bool): Whether to also admit the source's node-local actor workers into the
            broadcast group as relays, widening the fan-out at the root. Defaults to True, as in verl's NCCL
            engine. Set it to False to keep the group at one sender (actor rank 0) plus the rollout workers.
    """

    def __init__(
        self,
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
        multi_sender: bool = True,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.multi_sender = multi_sender
        self.pg: dist.ProcessGroup | None = None
        self._store: dist.TCPStore | None = None
        self.rank = -1
        self.world_size = -1
        self.send_buf: torch.Tensor | None = None
        self.recv_buf: torch.Tensor | None = None
        self._zmq_endpoint: tuple[str, int] | None = None
        self._subscribers = 0

        # start zeromq server for broadcasting bucket tensor metadata
        self.is_master = is_master
        self.topic = "bucket_metadata"
        if self.is_master:
            self._start_zmq_server()
            self._start_store()

    @staticmethod
    def get_node_id() -> str:
        """Identity of the node this GPU belongs to, used as a proxy for intra-node reachability.

        GPUs on the same node share the intra-node fabric, and a node never spans more than one such
        domain, so matching on node id can only ever under-select peers, never over-select them.
        """
        return ray.get_runtime_context().get_node_id()

    def prepare(self) -> WorkerMetadata:
        # The transfer buffers are allocated in `init_process_group`, once it is known that this rank takes part:
        # with the default single sender every trainer rank but 0 sits the broadcast out and needs none.
        # A fresh id per round keeps a rebuilt group's keys apart from the previous one's in the store.
        master = (
            MasterMetadata(
                zmq_ip=self.ip,
                zmq_port=self.listen_port,
                store_port=self._master_store().port,
                group_id=uuid.uuid4().hex,
                multi_sender=self.multi_sender,
            )
            if self.is_master
            else None
        )
        return WorkerMetadata(node_id=self.get_node_id(), master=master)

    def finalize(self):
        """Destroy the XCCL process group if rebuild_group is True."""
        if self.rebuild_group:
            self._destroy_process_group()
            self.rank = -1
            self.world_size = -1

        self.send_buf = None
        self.recv_buf = None

        torch.xpu.empty_cache()

    @staticmethod
    def _single_sender_ranks(actor_wg_world_size: int) -> list[int]:
        """Only actor rank 0 joins the group; every other actor worker sits the broadcast out."""
        return [0] + [-1] * (actor_wg_world_size - 1)

    @staticmethod
    def _multi_sender_ranks(actor_wg_world_size: int, metadata: list[WorkerMetadata]) -> list[int]:
        """Actor rank 0 joins, plus every actor worker on its node, which are its intra-node peers.

        Those peers carry no data the rollout needs. They join so XCCL has somewhere to fan a
        bucket out to over the intra-node fabric, from which it can push on over each peer's own NIC. A
        peer on any other node would have to pull a full copy over the fabric to contribute nothing, so it
        gets rank -1 instead.
        """
        source_node = metadata[0].node_id

        # Ranks must stay contiguous, so number the survivors as we walk the actor workers.
        ranks, next_rank = [], 0
        for i in range(actor_wg_world_size):
            if i == 0 or metadata[i].node_id == source_node:
                ranks.append(next_rank)
                next_rank += 1
            else:
                ranks.append(-1)
        return ranks

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[WorkerMetadata]):
        master = metadata[0].master
        assert master is not None, "actor rank 0 must be the checkpoint engine master"

        if master.multi_sender:
            actor_ranks = cls._multi_sender_ranks(actor_wg_world_size, metadata)
        else:
            actor_ranks = cls._single_sender_ranks(actor_wg_world_size)

        # Multi-sender degrades to a single sender when rank 0 has no actor peers on its node.
        num_senders = sum(rank >= 0 for rank in actor_ranks)
        world_size = num_senders + rollout_world_size
        logger.info(
            f"build_topology: {num_senders} of {actor_wg_world_size} actor workers send, world_size {world_size}"
        )

        actor_wg_kwargs = {
            "rank": actor_ranks,
            "world_size": [world_size] * actor_wg_world_size,
            "master_metadata": [master] * actor_wg_world_size,
            "num_senders": [num_senders] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(num_senders, world_size)),
            "world_size": [world_size] * rollout_world_size,
            "master_metadata": [master] * rollout_world_size,
            "num_senders": [num_senders] * rollout_world_size,
        }
        return actor_wg_kwargs, rollout_kwargs

    def _start_zmq_server(self):
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port, _ = get_free_port(self.ip)

        context = zmq.Context()
        # XPUB is a PUB socket that also reports the consumers' subscriptions, see `_wait_for_subscribers`.
        self.socket = context.socket(zmq.XPUB)
        # report every consumer's subscription, not only the first one per topic
        self.socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.listen_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.listen_port}"

        self.socket.bind(address)

    def _master_store(self) -> dist.TCPStore:
        assert self._store is not None, "only the master hosts the TCPStore"
        return self._store

    def _start_store(self):
        """Host the TCPStore the XCCL process group rendezvouses on (actor rank 0 only)."""
        # Port 0 lets the OS pick one, which avoids the race of probing for a free port first.
        self._store = dist.TCPStore(
            self.ip, 0, is_master=True, timeout=timedelta(seconds=_STORE_TIMEOUT_S), wait_for_workers=False
        )

    def _create_process_group(self, rank: int, world_size: int, metadata: MasterMetadata) -> dist.ProcessGroup:
        """An XCCL process group of its own, independent of the default torch.distributed world."""
        store = self._store
        if store is None:
            store = dist.TCPStore(
                metadata.zmq_ip.strip("[]"),
                metadata.store_port,
                is_master=False,
                timeout=timedelta(seconds=_STORE_TIMEOUT_S),
            )
        prefix_store = dist.PrefixStore(f"xccl_ckpt_{self.group_name}_{metadata.group_id}", store)
        pg = dist.ProcessGroup(prefix_store, rank, world_size)
        backend = ProcessGroupXCCL(prefix_store, rank, world_size, ProcessGroupXCCL.Options())
        pg._set_default_backend(dist.ProcessGroup.BackendType.XCCL)
        pg._register_backend(torch.device("xpu"), dist.ProcessGroup.BackendType.XCCL, backend)
        return pg

    def _destroy_process_group(self):
        pg, self.pg = self.pg, None
        shutdown = getattr(pg, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception as e:
                logger.warning(f"XCCL process group shutdown failed: {e}")

    def _allocate_buffers(self):
        """Allocate the transfer buffers for a rank that is in the group; `finalize` frees them again.

        The sender and the consumers double-buffer. A relay only broadcasts into one buffer it never reads, so it
        gets a single one.
        """
        is_relay = 0 < self.rank < self.num_senders
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="xpu")
        self.send_buf = None if is_relay else torch.zeros(self.bucket_size, dtype=torch.uint8, device="xpu")

    def _connect_zmq_client(self, metadata: MasterMetadata):
        assert not self.is_master, "Master process should not connect to other processes."
        endpoint = (metadata.zmq_ip, metadata.zmq_port)
        if self._zmq_endpoint == endpoint:
            # Keep the subscription from the previous sync: it is already known to the publisher, and a fresh
            # socket per sync would leak one.
            return
        previous = getattr(self, "socket", None)
        if previous is not None:
            previous.close(linger=0)
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.SUB)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"

        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self._zmq_endpoint = endpoint

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata, num_senders: int):
        """Initialize the XCCL process group.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
            master_metadata (MasterMetadata): The endpoint of the broadcast source.
            num_senders (int): How many actor-side ranks the group starts with. Ranks below this
                send or relay weights; ranks at or above it consume them.
        """
        self.num_senders = num_senders

        # Actor workers left out of the group are given rank -1.
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            return

        if self.rebuild_group or self.pg is None:
            self._destroy_process_group()
            self.pg = self._create_process_group(rank, world_size, master_metadata)
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank, f"rank {rank} is not equal to self.rank {self.rank}"
            assert self.world_size == world_size, (
                f"world_size {world_size} is not equal to self.world_size {self.world_size}"
            )

        self._allocate_buffers()

        # Only consumers read the bucket metadata stream; relays never touch the payload.
        if self.rank >= num_senders:
            self._connect_zmq_client(master_metadata)
        # barrier, which also builds the lazily created communicator while every rank is here
        signal = torch.ones(1, dtype=torch.int32, device="xpu")
        self.pg.allreduce([signal]).wait()

        # The barrier says nothing about the ZeroMQ side, which is a separate connection: make sure every
        # consumer's subscription has reached the publisher before the first bucket's metadata goes out.
        if self.is_master:
            self._subscribers = _wait_for_subscribers(self.socket, self._subscribers, world_size - num_senders)

        logger.info(f"init_process_group rank: {self.rank}, world_size: {self.world_size}")

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        assert self.rank < self.num_senders, "Rollout workers should not send weights."

        # Actor workers left out of the group still have to walk the generator: producing each
        # weight is itself collective over the actor group.
        if self.rank < 0:
            for _name, _weight in weights:
                pass
            return

        if self.rank > 0:
            await self._relay_weights(weights)
            return

        assert self.send_buf is not None and self.recv_buf is not None, "init_process_group allocates the buffers"
        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op = None

        # In the case of multi senders, the broadcast and all-gather kernels must
        # be queued in a deterministic order, otherwise the XCCL kernel may deadlock.
        # Moreover the wait_for_complete() function only waits for the XCCL kernel to be enqueued,
        # not for the kernel to finish.
        pipelined = self.num_senders == 1

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
            # fill the tensor bucket
            start, flush = _place(offset, tensor_meta.chunk_size, self.bucket_size)
            if flush:
                torch.xpu.synchronize()

                # wait previous broadcast op finish
                if pipelined and broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                broadcast_op = BroadcastOperation(
                    rank=self.rank,
                    pg=self.pg,
                    bucket=send_buf[:offset],
                    metadata={"bucket_meta": bucket_meta, "is_last": False, "length": offset},
                    socket=self.socket,
                    topic=self.topic,
                )

                # swap send_buf and recv_buf
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
            offset = start

            assert offset + tensor_meta.chunk_size <= self.bucket_size
            assert tensor_meta.name not in bucket_meta

            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            send_buf[offset : offset + tensor_meta.chunk_size] = chunk
            offset += tensor_meta.chunk_size

            # keep the relays in step: no-op once the bucket's broadcast has already been drained
            if not pipelined and broadcast_op is not None:
                await broadcast_op.wait_for_complete()

        # broadcast last bucket
        torch.xpu.synchronize()
        if pipelined and broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        broadcast_op = BroadcastOperation(
            rank=self.rank,
            pg=self.pg,
            bucket=send_buf[:offset],
            metadata={"bucket_meta": bucket_meta, "is_last": True, "length": offset},
            socket=self.socket,
            topic=self.topic,
        )
        await broadcast_op.wait_for_complete()

        # the wait_for_complete() function just waits for the XCCL kernel to be enqueued,
        # not for the kernel to finish, hence we need to synchronize to make sure the
        # buffer does not get freed before the kernel finishes.
        torch.xpu.synchronize()

        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    async def _relay_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Join every broadcast as a node-local relay for rank 0, dropping the payload.

        A relay holds nothing the rollout needs, so it only has to match the root bucket for
        bucket. It recomputes the boundaries from its own copy of the weight stream under the same
        rule `send_weights` uses above, which keeps the two in step without a round trip -- and it
        has to be derived locally rather than read off the wire, because pulling a weight is itself
        collective over the actor group, so blocking on the wire would deadlock against rank 0's
        own gathers.
        """
        assert self.recv_buf is not None, "init_process_group allocates the buffers"
        start_time = time.time()
        offset = 0

        # a single buffer suffices: nothing reads the bucket, so successive broadcasts can share it
        async for tensor_meta, _ in split_weight_chunks(weights, self.bucket_size, meta_only=True):
            start, flush = _place(offset, tensor_meta.chunk_size, self.bucket_size)
            if flush:
                _broadcast(self.pg, self.recv_buf[:offset])
            offset = start + tensor_meta.chunk_size

        # relay last bucket
        _broadcast(self.pg, self.recv_buf[:offset])

        # wait for the enqueued XCCL kernels, so finalize() cannot free the buffer under them
        torch.xpu.synchronize()

        logger.info(f"Rank {self.rank} relay weights done, time cost: {time.time() - start_time:.2f}s")

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

    async def _receive_weight_chunks(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive the weight chunks of the model.

        Yields:
            A tuple of the name of the weight tensor and the chunk itself.
        """
        assert self.rank > 0, "Rank 0 should not receive weights."
        assert self.send_buf is not None and self.recv_buf is not None, "init_process_group allocates the buffers"
        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        # receive first bucket
        start_time = time.time()
        broadcast_op = BroadcastOperation(
            rank=self.rank,
            pg=self.pg,
            bucket=recv_buf,
            metadata=None,
            socket=self.socket,
            topic=self.topic,
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += metadata["length"]
        total_params += len(metadata["bucket_meta"])

        # wait for the XCCL broadcast kernel to finish before we yield the tensors
        # otherwise if the buffer is clone using a non-blocking copy, it may
        # lead to data corruption
        torch.xpu.synchronize()

        # swap send_buf and recv_buf
        send_buf, recv_buf = recv_buf, send_buf
        while not metadata["is_last"]:
            # 1. receive next bucket
            broadcast_op = BroadcastOperation(
                rank=self.rank,
                pg=self.pg,
                bucket=recv_buf,
                metadata=None,
                socket=self.socket,
                topic=self.topic,
            )

            # 2. yield tensor from send_buf
            for name, tensor_meta in metadata["bucket_meta"].items():
                tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
                yield tensor_meta, tensor

            # 3. wait for next bucket broadcast finish
            metadata = await broadcast_op.wait_for_complete()
            total_bytes += metadata["length"]
            total_params += len(metadata["bucket_meta"])

            # 4. swap send_buf and recv_buf
            torch.xpu.synchronize()  # sync non-blocking copy
            send_buf, recv_buf = recv_buf, send_buf

        # yield tensor from send_buf
        for name, tensor_meta in metadata["bucket_meta"].items():
            tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
            yield tensor_meta, tensor

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
