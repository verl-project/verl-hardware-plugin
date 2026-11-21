# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Tiny-width checkpoints that keep each architecture's real layer structure and weight naming (CPU tests of the
in-process rollout).

Only the widths are tiny (hidden 64, head_dim 16, intermediate 128); the layer pattern, the tensor names, the fused /
stacked parameters vLLM builds from them and the tie setting are the architecture's own:

* ``qwen3_5`` / ``qwen3_5_tied``: Qwen3.5 dense text (``Qwen3_5ForCausalLM``, ``qwen3_5_text``), 8 layers = two periods
  of three Gated DeltaNet (linear attention) layers + one full-attention layer (``full_attention_interval`` 4), the
  attention output gate, partial rotary. The untied variant has as many value as key heads (as Qwen3.5-0.8B: 16 / 16);
  the tied one twice as many value heads (the ratio of the larger hybrid configs), so both GDN q / k / v row splits are
  installed.
* ``qwen3`` / ``qwen3_tied``: dense Qwen3 (``Qwen3ForCausalLM``: q / k RMS norms), 2 layers.
* ``llama`` / ``llama_tied``: Llama (``LlamaForCausalLM``), 2 layers.
* ``gemma3`` / ``gemma3_tied``: Gemma 3 text (``Gemma3ForCausalLM``, ``gemma3_text``), 6 layers = 5 sliding-window + 1
  global layer, one KV head (the 1B's multi-query attention), q / k norms, pre / post feed-forward norms; tied is
  Gemma's default.
* ``mistral`` / ``mistral_tied``: Mistral (``MistralForCausalLM``), 2 layers, no sliding window.
* ``qwen2`` / ``qwen2_tied``: Qwen2 / Qwen2.5 (``Qwen2ForCausalLM``: q / k / v biases, fused into vLLM's
  ``qkv_proj.bias``), 2 layers.
* ``qwen3_moe``: Qwen3 MoE (``Qwen3MoeForCausalLM``), 2 sparse layers of 4 experts, top-2 (not a tested layout:
  colocation refuses it).
* ``phi3``: Phi-3 (``Phi3ForCausalLM``: q / k / v and gate / up already fused in the checkpoint): a dense architecture
  that is not a tested layout (refused at build).

Every checkpoint is bf16 safetensors, seeded, with the tokenizer files copied from ``tokenizer_dir`` (any HF tokenizer
whose vocabulary covers the mock worker's token ids 1000-41000; vocab_size 248320 = the Qwen3.5 tokenizer's).
"""

from __future__ import annotations

import json
import os
import shutil

VOCAB = 248320
EOS = 248044
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
)
COMMON = {
    "vocab_size": VOCAB,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "eos_token_id": EOS,
    "pad_token_id": EOS,
    "bos_token_id": None,
}


def _qwen3_5(tied: bool) -> dict:
    n = 8
    return {
        **COMMON,
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "num_hidden_layers": n,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4 if tied else 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(n)],
        "attn_output_gate": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25},
        "tie_word_embeddings": tied,
        "hidden_act": "silu",
    }


def _dense(arch: str, model_type: str, tied: bool, **extra) -> dict:
    return {
        **COMMON,
        "architectures": [arch],
        "model_type": model_type,
        "num_hidden_layers": 2,
        "tie_word_embeddings": tied,
        "rope_theta": 1e6,
        "hidden_act": "silu",
        "attention_bias": False,
        **extra,
    }


def _gemma3(tied: bool) -> dict:
    n = 6
    return {
        **COMMON,
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3_text",
        "num_hidden_layers": n,
        "num_key_value_heads": 1,
        "layer_types": ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(n)],
        "sliding_window": 512,
        "query_pre_attn_scalar": 16,
        "hidden_activation": "gelu_pytorch_tanh",
        "final_logit_softcapping": None,
        "attn_logit_softcapping": None,
        "tie_word_embeddings": tied,
        "bos_token_id": 2,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
    }


def _qwen3_moe() -> dict:
    return {
        **COMMON,
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
        "rope_theta": 1e6,
        "hidden_act": "silu",
        "attention_bias": False,
    }


SPECS = {
    "qwen3_5": _qwen3_5(False),
    "qwen3_5_tied": _qwen3_5(True),
    "qwen3": _dense("Qwen3ForCausalLM", "qwen3", False),
    "qwen3_tied": _dense("Qwen3ForCausalLM", "qwen3", True),
    "llama": _dense("LlamaForCausalLM", "llama", False, mlp_bias=False),
    "llama_tied": _dense("LlamaForCausalLM", "llama", True, mlp_bias=False),
    "gemma3": _gemma3(False),
    "gemma3_tied": _gemma3(True),
    "mistral": _dense("MistralForCausalLM", "mistral", False, sliding_window=None),
    "mistral_tied": _dense("MistralForCausalLM", "mistral", True, sliding_window=None),
    "qwen2": _dense("Qwen2ForCausalLM", "qwen2", False, use_sliding_window=False),
    "qwen2_tied": _dense("Qwen2ForCausalLM", "qwen2", True, use_sliding_window=False),
    "qwen3_moe": _qwen3_moe(),
    "phi3": _dense("Phi3ForCausalLM", "phi3", False, resid_pdrop=0.0, embd_pdrop=0.0, attention_dropout=0.0),
}
# The variants of each architecture family (the per-architecture table groups them)
FAMILIES = {
    "qwen3_5": ("qwen3_5", "qwen3_5_tied"),
    "qwen3": ("qwen3", "qwen3_tied"),
    "llama": ("llama", "llama_tied"),
    "gemma3": ("gemma3", "gemma3_tied"),
    "mistral": ("mistral", "mistral_tied"),
    "qwen2": ("qwen2", "qwen2_tied"),
    "qwen3_moe": ("qwen3_moe",),
    "phi3": ("phi3",),
}


def make_model(out_dir: str, key: str, tokenizer_dir: str, seed: int = 11) -> str:
    """Writes the tiny checkpoint ``key`` under ``out_dir/key`` (once) and returns its directory."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    d = os.path.join(out_dir, key)
    if os.path.isfile(os.path.join(d, "model.safetensors")):
        return d
    os.makedirs(d, exist_ok=True)
    for f in TOKENIZER_FILES:
        if os.path.exists(os.path.join(tokenizer_dir, f)):
            shutil.copy(os.path.join(tokenizer_dir, f), d)
    spec = SPECS[key]
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(spec, fh, indent=1)
    cfg = AutoConfig.from_pretrained(d)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    if bool(spec.get("tie_word_embeddings")) != (
        model.get_output_embeddings().weight is model.get_input_embeddings().weight
    ):
        raise RuntimeError(f"{key}: the HF model's tie does not follow the config")
    model.to(torch.bfloat16).save_pretrained(d, safe_serialization=True)
    with open(os.path.join(d, "config.json"), "w") as fh:  # the exact config (save_pretrained rewrites it)
        json.dump(spec, fh, indent=1)
    return d
