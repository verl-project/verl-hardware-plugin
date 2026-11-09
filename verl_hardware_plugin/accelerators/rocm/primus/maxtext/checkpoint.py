# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Weights-only checkpoints in Hugging Face safetensors layout.

Optimizer state is not saved. A resumed run restarts Adam from zero moments.
The files are written to ``<dir>.tmp`` and renamed, so a crash never leaves a
half-written checkpoint under the final name.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterable

_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
)


def write_hf_checkpoint(
    tensors: Iterable, out_dir: str, source_dir: str | None = None, shard_bytes: int = 4 * 2**30
) -> dict:
    """Write ``(name, torch.Tensor)`` pairs to sharded safetensors under ``out_dir``.

    Returns ``{"tensors": n, "bytes": total}``. Tensors may be on any device.
    """
    import torch
    from safetensors.torch import save_file

    out_dir = out_dir.rstrip("/")
    tmp = out_dir + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    shards: list[tuple[str, list[str]]] = []
    current: dict = {}
    size = 0
    total = 0
    count = 0

    def flush():
        nonlocal current, size
        if not current:
            return
        name = f"part-{len(shards):05d}.safetensors"
        save_file(current, os.path.join(tmp, name), metadata={"format": "pt"})
        shards.append((name, list(current)))
        current, size = {}, 0

    for name, tensor in tensors:
        host = tensor.detach().to("cpu", copy=True).contiguous() if isinstance(tensor, torch.Tensor) else tensor
        current[name] = host
        nbytes = host.numel() * host.element_size()
        size += nbytes
        total += nbytes
        count += 1
        if size >= shard_bytes:
            flush()
    flush()

    final = {}
    weight_map = {}
    for index, (temp_name, names) in enumerate(shards):
        final_name = f"model-{index + 1:05d}-of-{len(shards):05d}.safetensors"
        os.replace(os.path.join(tmp, temp_name), os.path.join(tmp, final_name))
        final[final_name] = names
        for tensor_name in names:
            weight_map[tensor_name] = final_name
    with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as handle:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, handle)
    if source_dir:
        for name in _METADATA_FILES:
            path = os.path.join(source_dir, name)
            if os.path.isfile(path):
                shutil.copy2(path, os.path.join(tmp, name))

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    os.replace(tmp, out_dir)
    # The trainer runs as root in a container; keep the files removable by the submitting user.
    for base, dirs, files in os.walk(out_dir):
        for name in dirs:
            os.chmod(os.path.join(base, name), 0o777)
        for name in files:
            os.chmod(os.path.join(base, name), 0o666)
    os.chmod(out_dir, 0o777)
    return {"tensors": count, "bytes": total}


def prune_checkpoints(root: str, current_step: int, keep: int | None) -> list[str]:
    """Delete all but the newest ``keep`` ``global_step_N`` folders, counting the current one.

    The current folder is never removed. Returns the removed paths.
    """
    if not keep or keep < 1 or not os.path.isdir(root):
        return []
    steps = []
    for entry in os.listdir(root):
        match = re.fullmatch(r"global_step_(\d+)", entry)
        if match and os.path.isdir(os.path.join(root, entry)):
            steps.append(int(match.group(1)))
    keep_steps = set(sorted(set(steps) | {current_step}, reverse=True)[:keep]) | {current_step}
    removed = []
    for step in steps:
        if step not in keep_steps:
            path = os.path.join(root, f"global_step_{step}")
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed
