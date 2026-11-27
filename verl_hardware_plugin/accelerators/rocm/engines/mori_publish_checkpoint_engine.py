# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""MoRI publish checkpoint engine: the trainer publishes a snapshot of the weights in its own registered device
memory and returns; rollout workers pull it with one-sided RDMA READ over MoRI-IO (https://github.com/ROCm/mori).

This is verl's decoupled publish / commit mode (`support_publish_weights = True`): `send_weights` depends on no
rollout worker, so a slow or absent one never holds the trainer, and the snapshot stays unchanged until the next
`send_weights`, which the manager starts only after every rollout worker committed this one. The `mori` engine in
this package is the coupled counterpart (RDMA WRITE pushed down a chain of rollout workers).

Publishing. P = min(num_stripes, trainer ranks) trainer ranks lead one stripe each. All trainer ranks iterate the same
full-tensor generator (the trainer's collectives run inside it) and give each tensor to the stripe with the fewest
bytes so far, so they agree without talking. A leader packs its stripe into chunks of bucket_size bytes of registered
device memory, allocated on first use and kept across syncs, then writes the manifest (what is where, and the MoRI
MemoryDesc of every chunk) and, last, a status record into its header region:

    header[0:_STATUS_BYTES]   magic | sync id | manifest length | crc32(manifest) | crc32(the fields before)
    header[_STATUS_BYTES:]    manifest (JSON)

The status record names the sync (`build_topology` gives every worker of one update the same id), so a reader tells
this publication from the previous one whatever `global_steps` is, and its checksums let a reader that raced the
writer see a torn record as "not published yet".

Pulling. A rollout worker polls every stripe's status record until it names this sync and reads the manifests. It
then pulls in rounds: round k reads chunk k of every stripe, one stripe per scratch slot and one read at a time,
starting at a different stripe per worker so that the readers spread over the trainers' NICs; it checks each chunk
against the checksum in its manifest; and it yields a copy of each tensor, stripe by stripe. The yield order depends
on the manifests alone, so every rollout worker yields the same tensors in the same order: verl's SGLang adapter
loads one bucket across the TP ranks of a replica and needs them to agree. Last, it reads every status record again:
had one changed, that stripe would have been overwritten while it was read. Yielded tensors are copies, so a consumer
may keep any number of them for as long as it likes (the SGLang adapter batches them before loading).

One read at a time per worker is the default (reads_in_flight): with 8 workers each reading 8 chunks of 1 GiB at
once, the trainers' NICs stopped answering (CQE status 12, transport retry counter exceeded, on MI355X / Pollara
with MoRI 1.2.4).

shortcut: every rollout GPU pulls the whole model from the trainers (N rollout GPUs x model bytes over the trainers'
NICs); the `mori` engine's one-copy-per-node chain with XGMI fan-out is the upgrade when the pull time matters.
shortcut: full-tensor export only; the `mori` engine's export="shard" (no all-gather on the trainer) is not ported.

Args (engine_kwargs["mori_publish"]):
    num_stripes (int): Max trainer ranks that hold a part of the snapshot. Default 8.
    reads_in_flight (int): Chunk reads a rollout worker has in flight at once, at most num_stripes. Default 1,
        which is what the trainers' NICs are known to sustain with 8 rollout workers (see above). On that setup
        (8 trainer and 8 rollout GPUs, 1 GiB chunks) 2 worked in the one run that tried it and pulled a 30B model
        in 5.3 s instead of 6.5 s, while the rollout decoded more slowly meanwhile; 4 failed on the first update.
    float_dtype (str): If set (e.g. "bfloat16"), floating tensors wider than it are cast to it before they are
        snapshotted. Set it to the dtype the rollout holds its weights in: the rollout's loader casts to that dtype
        anyway, so the installed bytes are the same, and an fp32 trainer then publishes half the bytes. Leave it
        unset if the rollout keeps any floating parameter wider than this, e.g. the fp32 norm weights of a dense
        Qwen3 under SGLang's true-on-policy mode: those would be rounded to this dtype first.
    timeout_s (float): Max wait for a publication to appear and for any single read. Default 600.
    device (str): "cuda" (default) or "cpu" (tests only).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import socket
import struct
import time
import uuid
import zlib
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Generator

import torch

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    split_weight_chunks,
)
from verl.workers.rollout.utils import ensure_async_iterator
from verl_hardware_plugin.accelerators.rocm.engines.mori_checkpoint_engine import (
    _ALIGN,
    _DTYPES,
    _align,
    _alloc_region,
    _node_ip,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_MAGIC = b"MORIPUB1"
# magic, sync id (hex), manifest bytes, crc32 of the manifest, crc32 of the fields before
_STATUS = struct.Struct("<8s32sQII")
_STATUS_BYTES = _align(_STATUS.size)
_UNPUBLISHED = "0" * 32  # sync id of a header whose stripe is being repacked
_POLL_S = 0.002  # between looks at a status record that does not name this sync yet
# engines with a read that may still be landing: the NIC writes into their regions whatever Python does with them
_KEEPALIVE: list = []


def _pack_status(sync_id: str, manifest: bytes) -> bytes:
    fields = struct.pack("<8s32sQI", _MAGIC, sync_id.encode(), len(manifest), zlib.crc32(manifest))
    return fields + struct.pack("<I", zlib.crc32(fields))


def _unpack_status(raw: bytes) -> SimpleNamespace | None:
    """The status record in `raw`, or None for anything else, such as a record read while it was being written."""
    magic, sync_id, manifest_len, manifest_crc, crc = _STATUS.unpack(raw[: _STATUS.size])
    if magic != _MAGIC or zlib.crc32(raw[: _STATUS.size - 4]) != crc:
        return None
    return SimpleNamespace(sync_id=sync_id.decode(), manifest_len=manifest_len, manifest_crc=manifest_crc)


@CheckpointEngineRegistry.register("mori_publish")
class MoriPublishCheckpointEngine(CheckpointEngine):
    """MoRI-IO checkpoint engine in verl's publish / commit mode (trainer-side snapshot, rollout-side RDMA READ).
    See the module docstring.

    Args:
        bucket_size (int): Size of a snapshot chunk; a reader holds num_stripes scratch slots of this size.
        is_master (bool): Unused; accepted for signature compatibility with the other engines.
    """

    support_publish_weights = True

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        num_stripes: int = 8,
        header_bytes: int = 16 << 20,
        reads_in_flight: int = 1,
        qp_per_transfer: int = 4,
        num_nics_per_transfer: int = 1,
        enable_chunking: bool = True,
        host: str | None = None,
        timeout_s: float = 600.0,
        device: str = "cuda",
        float_dtype: str | None = None,
    ):
        self.is_master = is_master
        self.chunk_bytes = max(_ALIGN, bucket_size // _ALIGN * _ALIGN)
        assert int(reads_in_flight) >= 1, f"reads_in_flight must be >= 1, got {reads_in_flight}"
        self.reads_in_flight = int(reads_in_flight)
        self.num_stripes = max(1, int(num_stripes))
        self.header_bytes = _align(header_bytes)
        assert self.header_bytes > _STATUS_BYTES, f"header_bytes must exceed {_STATUS_BYTES}, got {header_bytes}"
        self.qp_per_transfer = qp_per_transfer
        self.num_nics_per_transfer = num_nics_per_transfer
        self.enable_chunking = enable_chunking
        self.host = host
        self.timeout_s = timeout_s
        self.device = device
        self.float_dtype = None if float_dtype is None else _DTYPES[float_dtype]

        # created once, on first prepare(), and kept across syncs
        self._engine: Any = None  # mori.io.IOEngine
        self._key: str | None = None
        self._header: Any = None  # mine as a publisher; where a reader lands the header it reads
        self._header_mem: Any = None
        self._chunks: list[tuple[torch.Tensor, Any]] = []  # publisher: (region, mori.io.MemoryDesc), grown on demand
        self._scratch: Any = None  # reader: one slot of chunk_bytes per stripe
        self._scratch_mem: Any = None
        self._owners: list = []  # what keeps the regions' allocations alive
        self._peers: dict = {}  # key -> mori.io.EngineDesc of the publishers registered with MoRI
        # why this engine refuses all further use: a read that timed out may still land in my regions
        self._unusable: str | None = None

        # per sync (set by init_process_group, cleared by finalize)
        self._sync_id: str | None = None
        self._stripe = -1  # the stripe I publish, -1 for trainer ranks beyond P and for rollout workers
        self._num_stripes = 0
        self._publishers: list[SimpleNamespace] = []  # rollout workers: one per stripe
        self._read_offset = 0  # rollout workers: the stripe each round's reads start at

    # ------------------------------------------------------------------ setup
    def _region(self, nbytes: int) -> tuple[torch.Tensor, Any]:
        """`nbytes` of device memory registered with MoRI, and its MemoryDesc."""
        region, owner = _alloc_region(nbytes, self.device)
        self._owners.append(owner)
        return region, self._engine.register_torch_tensor(region)

    def _setup(self):
        from mori.io import BackendType, IOEngine, IOEngineConfig, RdmaBackendConfig

        ip = self.host or _node_ip()
        self._key = f"verl-mori-publish-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._engine = IOEngine(key=self._key, config=IOEngineConfig(host=ip, port=0))
        self._engine.create_backend(
            BackendType.RDMA,
            RdmaBackendConfig(
                qp_per_transfer=self.qp_per_transfer,
                # readers poll their own read statuses; nothing pops the inbound status a notification leaves
                enable_notification=False,
                enable_transfer_chunking=self.enable_chunking,
                num_nics_per_transfer=self.num_nics_per_transfer,
            ),
        )
        self._header, self._header_mem = self._region(self.header_bytes)
        logger.info(f"mori publish engine {self._key} ready: chunks of {self.chunk_bytes >> 20} MiB")

    def prepare(self) -> dict:
        """Create the MoRI engine and the header region on first use; return this worker's metadata."""
        self._check_usable()
        if self._engine is None:
            self._setup()
        return {
            "engine_desc": self._engine.get_engine_desc().pack(),
            "header": self._header_mem.pack(),
            "num_stripes": self.num_stripes,
            "chunk_bytes": self.chunk_bytes,
        }

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        """The first P trainer ranks publish one stripe each; every rollout worker reads all of them, the r-th
        starting each round at stripe r % P."""
        actors = metadata[:actor_wg_world_size]
        assert len({m["chunk_bytes"] for m in metadata}) == 1, "all workers need the same bucket size"
        num_stripes = min(min(m["num_stripes"] for m in metadata), actor_wg_world_size)
        publishers = [{"engine_desc": m["engine_desc"], "header": m["header"]} for m in actors[:num_stripes]]
        sync_id = uuid.uuid4().hex  # one id for every worker of this update: it names the publication
        actor_kwargs = {
            "stripe": [a if a < num_stripes else -1 for a in range(actor_wg_world_size)],
            "num_stripes": [num_stripes] * actor_wg_world_size,
            "publishers": [None] * actor_wg_world_size,
            "read_offset": [0] * actor_wg_world_size,
            "sync_id": [sync_id] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "stripe": [-1] * rollout_world_size,
            "num_stripes": [num_stripes] * rollout_world_size,
            "publishers": [publishers] * rollout_world_size,
            "read_offset": [r % num_stripes for r in range(rollout_world_size)],
            "sync_id": [sync_id] * rollout_world_size,
        }
        return actor_kwargs, rollout_kwargs

    def init_process_group(
        self, stripe: int, num_stripes: int, publishers: list[dict] | None, read_offset: int, sync_id: str
    ):
        """Trainer ranks learn their stripe; rollout workers register the publishers with MoRI. The trainer never
        learns who reads it."""
        from mori.io import EngineDesc, MemoryDesc

        self._check_usable()
        self._stripe, self._num_stripes, self._read_offset, self._sync_id = stripe, num_stripes, read_offset, sync_id
        self._publishers = []
        for meta in publishers or []:
            desc = EngineDesc.unpack(meta["engine_desc"])
            if desc.key not in self._peers:
                self._engine.register_remote_engine(desc)
                self._peers[desc.key] = desc
            self._publishers.append(SimpleNamespace(key=desc.key, header=MemoryDesc.unpack(meta["header"])))

    def finalize(self):
        """Keep the engine, the regions and the registrations for the next sync; only forget this sync's roles. The
        snapshot stays readable until the next send_weights overwrites it."""
        self._stripe, self._publishers, self._sync_id = -1, [], None

    def close(self):
        """Release the MoRI registrations. Not part of the per-sync API; for teardown, e.g. in tests."""
        if self._engine is not None and self._unusable is None:  # an unusable engine may still be written into
            for desc in self._peers.values():
                self._engine.deregister_remote_engine(desc)
            for mem in [self._header_mem, self._scratch_mem, *(mem for _, mem in self._chunks)]:
                if mem is not None:
                    self._engine.deregister_memory(mem)
            self._engine = self._header = self._scratch = None
            self._chunks, self._owners = [], []
        self._peers.clear()

    def _check_usable(self):
        if self._unusable is not None:
            raise RuntimeError(f"mori publish: engine unusable ({self._unusable}); recreate the workers")

    def _mark_unusable(self, reason: str, status) -> str:
        """Fail-stop: a read may still be landing, so nothing of this engine is reused, deregistered or freed.
        That includes the read's status object, which MoRI updates through a bare pointer when the read ends."""
        self._unusable = reason
        _KEEPALIVE.append((self, status))
        return reason

    def _device_done(self):
        """Wait for the copies queued on the device, which a NIC read does not see."""
        if self.device == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------ publish
    def _write_header(self, offset: int, blob: bytes):
        data = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
        self._header[offset : offset + len(blob)].copy_(data)

    @staticmethod
    def _checksum(chunk: torch.Tensor, used: int) -> int:
        """Wrapping sum of the 8-byte words in the first `used` bytes, plus the bytes after the last whole word.
        Cheap on the device, and different for a chunk read before its copies ran or after it was repacked."""
        words = used // 8 * 8
        return int((chunk[:words].view(torch.int64).sum() + chunk[words:used].sum(dtype=torch.int64)).item())

    async def _narrowed(self, weights):
        """The weights as they are snapshotted: floating tensors wider than `float_dtype` cast to it."""
        async for name, weight in ensure_async_iterator(weights):
            if (
                self.float_dtype is not None
                and weight.is_floating_point()
                and weight.dtype.itemsize > self.float_dtype.itemsize
            ):
                weight = weight.to(self.float_dtype)
            yield name, weight

    def _chunk(self, index: int) -> torch.Tensor:
        if index == len(self._chunks):
            self._chunks.append(self._region(self.chunk_bytes))
        return self._chunks[index][0]

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Every trainer rank must iterate all weights (the trainer's collectives run inside the generator); the
        leader of a stripe snapshots the tensors assigned to it and publishes them. Returns once the snapshot is
        readable, without waiting for any reader."""
        assert self._sync_id is not None, "init_process_group was not called"
        self._check_usable()
        if self._stripe < 0:  # not leading a stripe: only drive the generator
            async for _ in ensure_async_iterator(weights):
                pass
            return
        start = time.time()
        # a reader still on the previous publication must fail its closing check, not mix two versions
        self._write_header(0, _pack_status(_UNPUBLISHED, b""))
        self._device_done()

        loads, mine = [0] * self._num_stripes, False
        buckets: list[dict[str, Any]] = []
        metas: list[list] = []
        index, offset, nbytes = 0, 0, 0
        async for tensor_meta, piece in split_weight_chunks(self._narrowed(weights), self.chunk_bytes):
            if tensor_meta.chunk_offset == 0:  # a new tensor: to the least-loaded stripe, the same on every rank
                to = min(range(self._num_stripes), key=loads.__getitem__)
                loads[to] += tensor_meta.shape.numel() * tensor_meta.dtype.itemsize
                mine = to == self._stripe
            if not mine:
                continue
            at = _align(offset)
            if at + tensor_meta.chunk_size > self.chunk_bytes:
                buckets.append({"chunk": index, "used": offset, "metas": metas})
                index, at, metas = index + 1, 0, []
            self._chunk(index)[at : at + tensor_meta.chunk_size].copy_(piece, non_blocking=True)
            metas.append(
                [
                    tensor_meta.name,
                    list(tensor_meta.shape),
                    str(tensor_meta.dtype).removeprefix("torch."),
                    tensor_meta.chunk_offset,
                    tensor_meta.chunk_size,
                    at,
                ]
            )
            offset = at + tensor_meta.chunk_size
            nbytes += tensor_meta.chunk_size
        buckets.append({"chunk": index, "used": offset, "metas": metas})
        for bucket in buckets:
            bucket["checksum"] = (
                self._checksum(self._chunks[bucket["chunk"]][0], bucket["used"]) if bucket["used"] else 0
            )

        manifest = json.dumps(
            {
                "global_steps": global_steps,
                "chunks": [base64.b64encode(mem.pack()).decode("ascii") for _, mem in self._chunks[: index + 1]],
                "buckets": buckets,
            }
        ).encode()
        if _STATUS_BYTES + len(manifest) > self.header_bytes:
            raise ValueError(
                f"mori publish: the manifest of stripe {self._stripe} is {len(manifest)} bytes, more than the header "
                f"region holds; raise engine_kwargs.mori_publish.header_bytes above {_STATUS_BYTES + len(manifest)}"
            )
        self._write_header(_STATUS_BYTES, manifest)
        self._device_done()  # the status record goes out only once everything it names is in place
        self._write_header(0, _pack_status(self._sync_id, manifest))
        self._device_done()
        logger.info(
            f"mori publish stripe {self._stripe}: step {global_steps}, {len(buckets)} chunks, "
            f"{nbytes / 2**30:.2f} GiB published in {time.time() - start:.2f}s"
        )

    # ------------------------------------------------------------------ pull
    async def _read(self, local, local_offset: int, remote, remote_offset: int, size: int, what: str):
        self._check_usable()  # reads queued behind one that failed must not go out
        status = self._engine.read(
            local, local_offset, remote, remote_offset, size, self._engine.allocate_transfer_uid()
        )
        deadline = time.monotonic() + self.timeout_s
        while status.InProgress():
            if time.monotonic() > deadline:
                reason = self._mark_unusable(f"the read of {what} did not finish in {self.timeout_s}s", status)
                raise TimeoutError(f"mori publish: {reason}")
            try:
                await asyncio.sleep(0.0002)
            except asyncio.CancelledError:
                self._mark_unusable(f"the read of {what} was cancelled while in flight", status)
                raise
        if not status.Succeeded():
            # the queue pair of a failed read is in the error state: nothing more can be read through this engine
            reason = self._mark_unusable(f"the read of {what} failed: {status.Message()}", status)
            raise RuntimeError(f"mori publish: {reason}")

    async def _read_status(self, publisher, stripe: int) -> SimpleNamespace | None:
        await self._read(self._header_mem, 0, publisher.header, 0, _STATUS_BYTES, f"the status of stripe {stripe}")
        return _unpack_status(self._header[: _STATUS.size].cpu().numpy().tobytes())

    async def _read_manifest(self, publisher, stripe: int) -> tuple[SimpleNamespace, dict]:
        """Wait for stripe `stripe` of this sync to be published; return its status record and manifest."""
        from mori.io import MemoryDesc

        deadline = time.monotonic() + self.timeout_s
        while True:
            status = await self._read_status(publisher, stripe)
            if status is not None and status.sync_id == self._sync_id:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"mori publish: stripe {stripe} was not published within {self.timeout_s}s")
            await asyncio.sleep(_POLL_S)
        what = f"the manifest of stripe {stripe}"
        await self._read(self._header_mem, _STATUS_BYTES, publisher.header, _STATUS_BYTES, status.manifest_len, what)
        blob = self._header[_STATUS_BYTES : _STATUS_BYTES + status.manifest_len].cpu().numpy().tobytes()
        if zlib.crc32(blob) != status.manifest_crc:
            raise RuntimeError(f"mori publish: {what} does not match its status record")
        manifest = json.loads(blob)
        manifest["chunks"] = [MemoryDesc.unpack(base64.b64decode(desc)) for desc in manifest["chunks"]]
        return status, manifest

    def _mark(self):
        """Event after the copies queued so far on the current stream (a slot may be read into once it fired)."""
        if self.device != "cuda":
            return None
        event = torch.cuda.Event()
        event.record()
        return event

    @torch.no_grad()
    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Pull every stripe of this sync's publication and yield a copy of each tensor, in an order that is the
        same on every rollout worker."""
        assert self._publishers, "init_process_group was not called on a rollout worker"
        self._check_usable()
        stripes = range(self._num_stripes)
        if self._scratch is None:
            self._scratch, self._scratch_mem = self._region(self._num_stripes * self.chunk_bytes)
        start, nbytes, reading = time.time(), 0, 0.0
        in_flight = asyncio.Semaphore(self.reads_in_flight)

        async def read_chunk(c: int, bucket: dict):
            async with in_flight:
                what = f"chunk {bucket['chunk']} of stripe {c}"
                await self._read(
                    self._scratch_mem,
                    c * self.chunk_bytes,
                    published[c][1]["chunks"][bucket["chunk"]],
                    0,
                    bucket["used"],
                    what,
                )

        published = [await self._read_manifest(self._publishers[c], c) for c in stripes]
        for c, (_, manifest) in enumerate(published):
            if global_steps is not None and manifest["global_steps"] not in (None, global_steps):
                raise RuntimeError(
                    f"mori publish: stripe {c} holds step {manifest['global_steps']}, expected {global_steps}"
                )
        partial: list = [None] * self._num_stripes  # per stripe: a tensor that spans chunks, being put together
        copied = None  # the event after the copies out of the slots
        for k in range(max(len(manifest["buckets"]) for _, manifest in published)):
            while copied is not None and not copied.query():  # the copies out of the slots still run
                await asyncio.sleep(0.0001)
            buckets = {c: published[c][1]["buckets"][k] for c in stripes if k < len(published[c][1]["buckets"])}
            # reads_in_flight at a time (see the module docstring), each worker starting at its own stripe; every
            # read of the round ends before a failure is raised, so none is left landing in the slots
            read_start = time.time()
            order = sorted(buckets, key=lambda c: (c - self._read_offset) % self._num_stripes)
            outcomes = await asyncio.gather(
                *[read_chunk(c, buckets[c]) for c in order if buckets[c]["used"]], return_exceptions=True
            )
            reading += time.time() - read_start
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            for c, bucket in buckets.items():
                base = c * self.chunk_bytes
                slot = self._scratch[base : base + self.chunk_bytes]
                if bucket["used"] and self._checksum(slot, bucket["used"]) != bucket["checksum"]:
                    raise RuntimeError(
                        f"mori publish: chunk {bucket['chunk']} of stripe {c} does not match the checksum in its "
                        "manifest: it was read before it was complete, or changed after it was published"
                    )
                for name, shape, dtype, chunk_offset, chunk_size, offset in bucket["metas"]:
                    meta = TensorMeta(name, torch.Size(shape), _DTYPES[dtype], chunk_offset, chunk_size, offset)
                    piece = slot[offset : offset + chunk_size]
                    full = meta.shape.numel() * meta.dtype.itemsize
                    if chunk_size == full:
                        yield name, piece.clone().view(meta.dtype).view(meta.shape)
                        continue
                    if chunk_offset == 0:
                        partial[c] = torch.empty(full, dtype=torch.uint8, device=self._scratch.device)
                    partial[c][chunk_offset : chunk_offset + chunk_size].copy_(piece)
                    if chunk_offset + chunk_size == full:
                        yield name, partial[c].view(meta.dtype).view(meta.shape)
                        partial[c] = None
                nbytes += bucket["used"]
            copied = self._mark()
        for c, (status, _) in enumerate(published):
            if await self._read_status(self._publishers[c], c) != status:
                raise RuntimeError(f"mori publish: stripe {c} was republished while it was read")
        self._device_done()
        logger.info(
            f"mori publish: pulled {self._num_stripes} stripes, {nbytes / 2**30:.2f} GiB in {time.time() - start:.2f}s"
            f" ({reading:.2f}s in the chunk reads, the rest in the consumer and the copies)"
        )
