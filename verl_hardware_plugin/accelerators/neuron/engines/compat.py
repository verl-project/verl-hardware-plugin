# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Narrow compatibility seams for the pinned NeuronTorchTitan revision."""


def vocabulary_parallel_log_probs(logits, labels, group, global_vocab_size):
    """Use NeuronTorchTitan release-2.33.0's unreduced vocabulary-parallel loss.

    Its public compute_logprobs still gathers logits, so keep the dependency on
    this private primitive at the compatibility boundary.
    """
    from torchtitan.components.loss import _LossParallelCrossEntropy

    loss = _LossParallelCrossEntropy.apply(logits, labels, group, global_vocab_size, "none")
    return -loss.reshape(labels.shape)


def configure_torch_compile_for_neuron(recompile_limit: int) -> None:
    """Keep static-shape settings across the separate contexts of Ray worker RPCs."""
    import torch._dynamo

    config = torch._dynamo.config
    values = {
        "automatic_dynamic_shapes": False,
        "recompile_limit": max(config.recompile_limit, recompile_limit),
    }
    entries = getattr(config, "_config", {})
    for name, value in values.items():
        # As in the TPU helper, update both this context and defaults inherited by
        # future RPC contexts. An assignment alone only affects the init RPC.
        setattr(config, name, value)
        if name in entries:
            entries[name].default = value
