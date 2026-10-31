# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the pread-based HF safetensors reader of torchtitan's initial load."""

import os

import pytest
import torch
from safetensors.torch import save_file

from verl_hardware_plugin.accelerators.tpu.engines import hf_range_reader


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_read_item_blocks_match_slicing(tmp_path, dtype):
    tensors = {"a": torch.randn(8, 3, 5).to(dtype), "b": torch.randn(6, 4).to(dtype), "c": torch.randn(7).to(dtype)}
    path = str(tmp_path / "x.safetensors")
    save_file(tensors, path)
    header, start = hf_range_reader._header(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        cases = [
            ("a", [2, 0, 0], [4, 3, 5]),  # dim-0 block: one contiguous pread
            ("a", [0, 1, 0], [8, 2, 5]),  # not a dim-0 block: whole tensor, then sliced
            ("b", [0, 0], [6, 4]),
            ("c", [3], [2]),
            ("a", [8, 0, 0], [0, 3, 5]),  # empty block
        ]
        for name, offsets, lengths in cases:
            got = hf_range_reader.read_item(fd, header, start, name, offsets, lengths)
            want = tensors[name][tuple(slice(o, o + n) for o, n in zip(offsets, lengths, strict=False))]
            assert got.dtype == dtype and torch.equal(got, want), (name, offsets, lengths)
    finally:
        os.close(fd)


def test_dcp_load_through_range_reader(tmp_path):
    """dcp.load() with the range reader fills a state dict exactly like the HF checkpoint (single process)."""
    import json

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageReader

    tensors = {"w": torch.randn(6, 4).to(torch.bfloat16), "v": torch.randn(5).to(torch.bfloat16)}
    save_file({"w": tensors["w"]}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"v": tensors["v"]}, str(tmp_path / "model-00002-of-00002.safetensors"))
    index = {
        "metadata": {},
        "weight_map": {"w": "model-00001-of-00002.safetensors", "v": "model-00002-of-00002.safetensors"},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    state = {k: torch.zeros_like(v) for k, v in tensors.items()}
    reader = hf_range_reader.make_range_read_hf_storage_reader(HuggingFaceStorageReader, str(tmp_path), threads=4)
    dcp.load(state, storage_reader=reader, no_dist=True)
    for k, v in tensors.items():
        assert torch.equal(state[k], v), k
