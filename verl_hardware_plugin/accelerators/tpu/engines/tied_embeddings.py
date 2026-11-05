# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Training with a tied input embedding and LM head (``engine_kwargs.tpu.tie_embeddings``).

Some HF checkpoints share one weight between the input embedding and the LM head (``tie_word_embeddings=true``, no
``lm_head.weight`` in the checkpoint), while torchtitan's config of the same model trains a separate LM head that
starts as a copy of the embedding, e.g. Qwen3.5-0.8B (torchtitan's Qwen3.5 configs leave ``enable_weight_tying``
off). With ``tie_embeddings=True`` the engine trains the tied model instead: one ``[vocab, dim]`` weight, whose
gradient is the sum of the embedding's and the LM head's, and one set of optimizer states for it.

torchtitan already has the pieces for the model itself: ``Decoder`` shares the weight when its config sets
``enable_weight_tying``, and the state-dict adapters handle the tied HF layout (Qwen3.5's ``from_hf`` fills
``lm_head.weight`` from ``embed_tokens`` when the checkpoint has none, and its ``to_hf`` leaves ``lm_head.weight``
out, so an HF export of the tied model has the official checkpoint's layout). Two steps of the engine's model build
can split the shared weight again, though:

* SimpleFSDP's ``data_parallel`` registers a new sharded ``Parameter`` on every module, one for the embedding and one
  for the LM head. ``tie_model_spec`` wraps the ``parallelize_fn`` to point the embedding back at the LM head's
  sharded parameter, so the model is tied again from there on.
* ``to_empty()``, which materializes the meta model after parallelizing, keeps a sharded (``DTensor``) parameter but
  builds a new plain ``Parameter`` per module (``nn.Module._apply``: meta and device tensors are not interchangeable),
  e.g. without data parallelism. The wrapped model therefore re-ties again when ``init_states`` starts, right after
  ``to_empty()``. torchtitan's ``Decoder.init_states`` re-ties as well (the engine runs it with SimpleFSDP's
  parametrization disabled, ``tpu_utils.init_states_without_simple_fsdp_parametrization``), but the tie does not
  depend on the model class doing so.

``check_tied_embeddings`` then verifies, once torchtitan's ``Trainer`` is built, that exactly one parameter backs
both names in every model part and that the optimizer holds it once, so a run that asks for the tie never trains
an untied model.

The weight sync is unchanged: ``TorchTitanTPUEngine._to_hf_named_params`` still sends ``lm_head.weight`` (the shared
weight under its second name) for rollouts that keep a separate LM head; the engine has ``TrainerWeights`` export it
in its own buffer.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
from collections import Counter
from typing import Any, Optional

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _embedding_and_head(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    embedding = getattr(model, "tok_embeddings", None)
    head = getattr(model, "lm_head", None)
    if not isinstance(embedding, nn.Module) or not isinstance(head, nn.Module):
        raise RuntimeError(
            f"tie_embeddings: {type(model).__name__} has no tok_embeddings and lm_head modules to tie (a pipeline "
            "stage holds only one of them, and torchtitan does not support weight tying with pipeline parallelism)"
        )
    for module, name in ((embedding, "tok_embeddings"), (head, "lm_head")):
        if module._parameters.get("weight") is None:
            raise RuntimeError(f"tie_embeddings: {name} has no 'weight' parameter")
    return embedding, head


def tied_weight_shape(model_config: Any) -> tuple[int, int]:
    """``(vocab, dim)`` of the weight that the embedding and the LM head of a torchtitan ``Decoder`` config share.

    Raises:
        ValueError: if the config cannot tie them: no ``enable_weight_tying`` field, an embedding and an LM head of
            different shapes, or an LM head with a bias.
    """
    name = type(model_config).__name__
    if not hasattr(model_config, "enable_weight_tying"):
        raise ValueError(f"tie_embeddings: torchtitan model config {name} has no enable_weight_tying field")
    embedding = getattr(model_config, "tok_embeddings", None)
    head = getattr(model_config, "lm_head", None)
    if embedding is None or head is None:
        raise ValueError(f"tie_embeddings: torchtitan model config {name} has no tok_embeddings and lm_head")
    embedding_shape = (int(embedding.num_embeddings), int(embedding.embedding_dim))
    head_shape = (int(head.out_features), int(head.in_features))
    if embedding_shape != head_shape:
        raise ValueError(
            f"tie_embeddings: the embedding {embedding_shape} and the LM head {head_shape} of {name} differ in shape"
        )
    if getattr(head, "bias", False):
        raise ValueError(f"tie_embeddings: the LM head of {name} has a bias")
    return embedding_shape


def tie_embedding_weights(model: nn.Module) -> bool:
    """Points ``model.tok_embeddings``'s weight at ``model.lm_head``'s weight parameter.

    Returns:
        ``True`` if it re-tied them, ``False`` if they already shared one parameter.

    Raises:
        RuntimeError: if the two weights cannot share one parameter: different shapes, different ``DTensor`` layouts,
            or parameters managed by FSDP2 (which ties shared parameters itself and keeps references to each).
    """
    embedding, head = _embedding_and_head(model)
    embedding_weight, head_weight = embedding._parameters["weight"], head._parameters["weight"]
    if embedding_weight is head_weight:
        return False
    if isinstance(embedding, FSDPModule) or isinstance(head, FSDPModule):
        raise RuntimeError("tie_embeddings: FSDP2 split tok_embeddings.weight and lm_head.weight; cannot re-tie them")
    if embedding_weight.shape != head_weight.shape:
        raise RuntimeError(
            f"tie_embeddings: cannot tie tok_embeddings.weight {tuple(embedding_weight.shape)} and lm_head.weight "
            f"{tuple(head_weight.shape)}"
        )
    layouts = [
        (p.device_mesh, tuple(p.placements)) if isinstance(p, DTensor) else None
        for p in (embedding_weight, head_weight)
    ]
    if layouts[0] != layouts[1]:
        raise RuntimeError(
            f"tie_embeddings: tok_embeddings.weight and lm_head.weight are sharded differently ({layouts[0]} vs "
            f"{layouts[1]}); cannot tie them"
        )
    # Set the entry directly: SimpleFSDP replaces ``weight`` with a property whose getter all-gathers the parameter,
    # so ``nn.Module.__setattr__`` / ``register_parameter`` (which read the attribute first) would run a collective.
    embedding._parameters["weight"] = head_weight
    return True


def _retie_on_init_states(model: nn.Module) -> None:
    """Make ``model.init_states`` re-tie the weights first, after ``to_empty()`` may have split them."""
    init_states = model.init_states

    def _init_states(*args: Any, **kwargs: Any) -> None:
        tie_embedding_weights(model)
        init_states(*args, **kwargs)

    model.init_states = _init_states  # type: ignore[method-assign]


def tie_model_spec(model_spec: Any, hf_text_config: Optional[Any] = None) -> Any:
    """Returns ``model_spec`` set up to train with a tied input embedding and LM head (see the module docstring).

    Sets ``enable_weight_tying`` on ``model_spec.model`` (in place) and returns a copy of the spec whose
    ``parallelize_fn`` re-ties the two weights after parallelizing and again before initializing them.

    Args:
        model_spec: torchtitan ``ModelSpec`` of a ``Decoder`` model.
        hf_text_config: HF (text) config of the checkpoint, to warn when it does not tie the weights itself.

    Raises:
        ValueError: if the model cannot tie them (``tied_weight_shape``).
    """
    vocab, dim = tied_weight_shape(model_spec.model)
    if getattr(hf_text_config, "tie_word_embeddings", None) is False:
        logger.warning(
            "tie_embeddings=True, but the checkpoint config has tie_word_embeddings=false: the shared weight is "
            "loaded from both embed_tokens.weight and lm_head.weight, which must be equal"
        )
    model_spec.model.enable_weight_tying = True
    parallelize_fn = model_spec.parallelize_fn

    @functools.wraps(parallelize_fn)
    def parallelize_tied(model: nn.Module, **kwargs: Any) -> nn.Module:
        model = parallelize_fn(model, **kwargs)
        if tie_embedding_weights(model):
            logger.info("tie_embeddings: re-tied tok_embeddings.weight to lm_head.weight after parallelizing")
        _retie_on_init_states(model)
        return model

    logger.info("tie_embeddings: tok_embeddings and lm_head share one %d x %d weight", vocab, dim)
    return dataclasses.replace(model_spec, parallelize_fn=parallelize_tied)


def check_tied_embeddings(model_parts: list[nn.Module], optimizers: Optional[Any] = None) -> int:
    """Verifies the tie once the model and its optimizer are built.

    Args:
        model_parts: torchtitan ``Trainer.model_parts``.
        optimizers: torchtitan ``OptimizersContainer`` (or a single ``torch.optim.Optimizer``), or ``None``.

    Returns:
        The number of trainable parameter elements, the shared weight counted once.

    Raises:
        RuntimeError: if ``tok_embeddings.weight`` and ``lm_head.weight`` are not one parameter in every model part,
            or the optimizer does not hold that parameter exactly once.
    """
    shared = []
    for part in model_parts:
        embedding, head = _embedding_and_head(part)
        if embedding._parameters["weight"] is not head._parameters["weight"]:
            raise RuntimeError(
                "tie_embeddings: tok_embeddings.weight and lm_head.weight are separate parameters after the model "
                "was built; the tie did not survive parallelizing and initializing the model"
            )
        shared.append(head._parameters["weight"])
    if optimizers is not None:
        held = Counter(
            id(p)
            for optimizer in getattr(optimizers, "optimizers", [optimizers])
            for group in optimizer.param_groups
            for p in group["params"]
        )
        for weight in shared:
            if held[id(weight)] != 1:
                raise RuntimeError(
                    f"tie_embeddings: the optimizer holds the shared embedding / LM head weight {held[id(weight)]} "
                    "times instead of once"
                )
    seen: set[int] = set()
    num_trainable = 0
    for part in model_parts:
        for p in part.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                num_trainable += p.numel()
    logger.warning(
        "tie_embeddings: tok_embeddings.weight is lm_head.weight (%s elements); %s trainable parameter elements",
        f"{shared[0].numel():,}" if shared else 0,
        f"{num_trainable:,}",
    )
    return num_trainable
