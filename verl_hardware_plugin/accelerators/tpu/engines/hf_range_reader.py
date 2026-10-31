# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""A HuggingFace safetensors reader for torchtitan's initial load that reads byte ranges instead of mmap slices.

torch's ``HuggingFaceStorageReader`` reads every ``ReadItem`` through ``safe_open(...).get_slice()``, i.e. mmap page
faults. On a gcsfuse mount each fault becomes a small, latency-bound GCS read: ~6 MB/s per reader, so loading
Qwen3.5-397B-A17B's 740 GB from GCS would take hours. Here every item is one ``pread`` of its bytes (an FSDP / EP
``Shard(0)`` block of a row-major tensor is one contiguous range), ~170 MB/s per stream, and the files are read in
parallel, each sequentially (one stream per file: concurrent reads of one gcsfuse file slow each other down).
"""

from __future__ import annotations

import json
import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _header(path: str) -> tuple[dict[str, Any], int]:
    """The safetensors header of ``path`` and the file offset where its tensor data starts."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size)), 8 + size


def _pread_into(fd: int, buf: memoryview, offset: int) -> None:
    done = 0
    while done < len(buf):
        n = os.preadv(fd, [buf[done:]], offset + done)
        if n <= 0:
            raise OSError(f"short read at offset {offset + done}")
        done += n


def read_item(fd: int, header: dict[str, Any], data_start: int, fqn: str, offsets, lengths) -> torch.Tensor:
    """The ``[offsets, offsets + lengths)`` block of tensor ``fqn`` of an open safetensors file."""
    meta = header[fqn]
    dtype = _DTYPES[meta["dtype"]]
    shape = list(meta["shape"])
    begin = data_start + meta["data_offsets"][0]
    itemsize = torch.empty((), dtype=dtype).element_size()
    row = math.prod(shape[1:]) * itemsize if shape else itemsize
    whole_rows = all(o == 0 and n == s for o, n, s in zip(offsets[1:], lengths[1:], shape[1:], strict=True))
    if shape and whole_rows:
        # A dim-0 block: one contiguous range.
        buf = bytearray(lengths[0] * row)
        _pread_into(fd, memoryview(buf), begin + offsets[0] * row)
        if not buf:
            return torch.empty([0, *shape[1:]], dtype=dtype)
        return torch.frombuffer(buf, dtype=dtype).reshape([lengths[0], *shape[1:]])
    buf = bytearray(math.prod(shape) * itemsize)
    _pread_into(fd, memoryview(buf), begin)
    full = torch.frombuffer(buf, dtype=dtype).reshape(shape) if buf else torch.empty(shape, dtype=dtype)
    return full[tuple(slice(o, o + n) for o, n in zip(offsets, lengths, strict=True))]


def make_range_read_hf_storage_reader(base_cls: type, path: str, threads: int) -> Any:
    """An instance of a ``base_cls`` (torch's ``HuggingFaceStorageReader``) subclass whose ``read_data`` preads."""

    class RangeReadHFStorageReader(base_cls):  # type: ignore[misc, valid-type]
        def read_data(self, plan, planner):
            per_file: dict[str, list[Any]] = {}
            for item in plan.items:
                per_file.setdefault(self.storage_data[item.storage_index].relative_path, []).append(item)

            def one_file(file_name: str, items: list[Any]) -> list[tuple[Any, torch.Tensor]]:
                header, data_start = _header(file_name)
                items.sort(key=lambda r: header[r.storage_index.fqn]["data_offsets"][0])
                fd = os.open(file_name, os.O_RDONLY)
                try:
                    return [
                        (
                            req,
                            read_item(fd, header, data_start, req.storage_index.fqn, req.storage_offsets, req.lengths),
                        )
                        for req in items
                    ]
                finally:
                    os.close(fd)

            # Host reads in threads; the device copies stay on the calling thread.
            with ThreadPoolExecutor(max(1, min(threads, len(per_file)))) as ex:
                futures = [ex.submit(one_file, f, items) for f, items in per_file.items()]
                for fut in as_completed(futures):
                    for req, tensor in fut.result():
                        target = planner.resolve_tensor(req).detach()
                        if target.size() != tensor.size():
                            raise AssertionError(
                                f"req {req.storage_index} mismatch sizes {target.size()} vs {tensor.size()}"
                            )
                        target.copy_(tensor)
                        planner.commit_tensor(req, target)
            done: torch.futures.Future = torch.futures.Future()  # DCP calls .wait() on it
            done.set_result(None)
            return done

    return RangeReadHFStorageReader(path)
