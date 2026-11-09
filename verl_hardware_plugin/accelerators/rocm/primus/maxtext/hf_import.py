# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Read a Hugging Face Qwen3-MoE checkpoint into MaxText's scanned layout.

This is the inverse of ``export``. Each leaf is built from a block callback: a process reads
only the layers and experts its devices hold, so host memory stays at one device block.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np

_TOP = {
    ("token_embedder", "embedding"): ("model.embed_tokens.weight", False),
    ("decoder", "decoder_norm", "scale"): ("model.norm.weight", False),
    ("decoder", "logits_dense", "kernel"): ("lm_head.weight", True),
}

# (leaf path, HF suffix under model.layers.N, matrix?). Layers are axis 1 of each leaf.
_PER_LAYER = [
    (("decoder", "layers", "pre_self_attention_layer_norm", "scale"), "input_layernorm.weight", False),
    (("decoder", "layers", "post_self_attention_layer_norm", "scale"), "post_attention_layernorm.weight", False),
    (("decoder", "layers", "self_attention", "query", "kernel"), "self_attn.q_proj.weight", True),
    (("decoder", "layers", "self_attention", "key", "kernel"), "self_attn.k_proj.weight", True),
    (("decoder", "layers", "self_attention", "value", "kernel"), "self_attn.v_proj.weight", True),
    (("decoder", "layers", "self_attention", "out", "kernel"), "self_attn.o_proj.weight", True),
    (("decoder", "layers", "self_attention", "query_norm", "scale"), "self_attn.q_norm.weight", False),
    (("decoder", "layers", "self_attention", "key_norm", "scale"), "self_attn.k_norm.weight", False),
    (("decoder", "layers", "moe_block", "gate", "kernel"), "mlp.gate.weight", True),
]

# Experts are axis 0 and layers axis 1.
_EXPERTS = [
    (("decoder", "layers", "moe_block", "wi_0"), "gate_proj"),
    (("decoder", "layers", "moe_block", "wi_1"), "up_proj"),
    (("decoder", "layers", "moe_block", "wo"), "down_proj"),
]

Block = tuple  # tuple of slices, one per axis
Place = Callable[[object, Callable[[Block], np.ndarray]], object]


class HFReader:
    """Tensor access by HF name. Returns numpy in the requested dtype."""

    def __init__(self, path: str):
        self.path = path
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as handle:
                self.files = json.load(handle)["weight_map"]
        else:
            single = os.path.join(path, "model.safetensors")
            if not os.path.exists(single):
                raise FileNotFoundError(f"No safetensors checkpoint under {path}")
            from safetensors import safe_open

            with safe_open(single, framework="pt") as handle:
                self.files = {name: "model.safetensors" for name in handle.keys()}  # noqa: SIM118
        self._local = threading.local()
        self._pool: Optional[ThreadPoolExecutor] = None

    def get_many(self, names: list, dtype) -> list:
        """Read several tensors concurrently; network filesystems need many requests in flight."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=int(os.environ.get("MAXTEXT_HF_READ_THREADS", "16")))
        return list(self._pool.map(lambda name: self.get(name, dtype), names))

    def get(self, name: str, dtype) -> np.ndarray:
        import torch
        from safetensors import safe_open

        shard = self.files[name]
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = {}
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(self.path, shard), framework="pt")
        tensor = handles[shard].get_tensor(name)
        if np.dtype(dtype).name == "bfloat16":
            import ml_dtypes

            tensor = tensor.to(torch.bfloat16).contiguous()
            return tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
        return tensor.to(torch.float32).numpy().astype(dtype)


def _set(tree: dict, path: tuple, value) -> None:
    for key in path[:-1]:
        tree = tree.setdefault(key, {})
    tree[path[-1]] = value


def _get(tree: dict, path: tuple):
    for key in path:
        tree = tree[key]
    return tree


def _indices(axis: slice, size: int) -> range:
    return range(*axis.indices(size))


def load_stacked_from_hf(reader: HFReader, like: dict, place: Place) -> dict:
    """Build a pytree shaped like ``like`` from HF tensors.

    ``place(old_leaf, read_block)`` returns the replacement leaf, normally
    ``jax.make_array_from_callback(old.shape, old.sharding, read_block)``;
    ``read_block(index)`` returns the host block for a tuple of slices.
    """
    out: dict = {}
    for path, (name, matrix) in _TOP.items():
        old = _get(like, path)

        def read_top(block, old=old, name=name, matrix=matrix, path=path):
            array = reader.get(name, old.dtype)
            array = array.T if matrix else array
            _check(path, array.shape, old.shape)
            return np.ascontiguousarray(array[block])

        _set(out, path, place(old, read_top))

    for path, suffix, matrix in _PER_LAYER:
        old = _get(like, path)

        def read_layers(block, old=old, suffix=suffix, matrix=matrix):
            # Leaf is (d0, layers, ...): one HF tensor per layer, reshaped to the leaf minus axis 1.
            per_layer = (old.shape[0],) + tuple(old.shape[2:])
            rest = (block[0],) + tuple(block[2:])
            rows = []
            for layer in _indices(block[1], old.shape[1]):
                array = reader.get(f"model.layers.{layer}.{suffix}", old.dtype)
                rows.append((array.T if matrix else array).reshape(per_layer)[rest])
            return np.ascontiguousarray(np.stack(rows, axis=1))

        _set(out, path, place(old, read_layers))

    for path, proj in _EXPERTS:
        old = _get(like, path)

        def read_experts(block, old=old, proj=proj):
            experts = list(_indices(block[0], old.shape[0]))
            layers = list(_indices(block[1], old.shape[1]))
            host = None
            for j, layer in enumerate(layers):
                names = [f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight" for e in experts]
                for i, array in enumerate(reader.get_many(names, old.dtype)):
                    value = array.T[tuple(block[2:])]
                    if host is None:
                        host = np.empty((len(experts), len(layers)) + value.shape, dtype=np.dtype(old.dtype))
                    host[i, j] = value
            return host

        _set(out, path, place(old, read_experts))
    return out


def _check(path: tuple, got: tuple, want: tuple) -> None:
    if tuple(got) != tuple(want):
        raise ValueError(f"{'/'.join(path)}: checkpoint gives {tuple(got)}, model expects {tuple(want)}")
