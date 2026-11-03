# Copyright (c) 2026 Google LLC. All rights reserved.
# Diagnostics adapted from verl: Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Licensed under the Apache License, Version 2.0.

"""Token-weighted PPO diagnostics, without changing verl's loss or other backends."""

from functools import partial
from typing import Any, Callable

import torch

from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding

_TOKEN_METRICS = ("actor/ppo_kl_token_mean", "actor/ppo_k3_token_mean", "actor/logprob_gap_abs_token_mean")


def _diagnostics_enabled(config: Any) -> bool:
    if config.global_batch_info.get("batch_num_tokens") is None:
        return False
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    if loss_mode == "bypass_mode":
        # Rejection sampling changes the mask counted by batch_num_tokens.
        correction = config.policy_loss.get("rollout_correction") or {}
        return correction.get("loss_type", "ppo_clip") == "ppo_clip" and correction.get("rollout_rs") is None
    return loss_mode == "vanilla"


def ppo_token_diagnostics(config, log_prob, old_log_prob, response_mask) -> dict[str, Metric]:
    """Global token means: SUM micro-batches, then MEAN DP ranks."""
    if not _diagnostics_enabled(config):
        return {}
    with torch.no_grad():
        gap = torch.where(response_mask.bool(), log_prob.float() - old_log_prob.float(), 0.0)
        # Clamp KL as PPO does; keep the absolute gap unmodified.
        log_ratio = gap.clamp(-20.0, 20.0)
        values = (-log_ratio, (log_ratio.exp() - log_ratio - 1.0).clamp_min(0.0), gap.abs())
        return {
            name: Metric(AggregationType.SUM, agg_loss(value, response_mask, "token-mean", **config.global_batch_info))
            for name, value in zip(_TOKEN_METRICS, values, strict=True)
        }


def call_tpu_loss_function(loss_function: Callable, model_output, data, dp_group=None):
    """Call the loss unchanged; fill missing diagnostics for core PPO callbacks."""
    loss, metrics = loss_function(model_output=model_output, data=data, dp_group=dp_group)
    if not isinstance(loss_function, partial) or all(name in metrics for name in _TOKEN_METRICS):
        return loss, metrics

    from verl.workers.utils.losses import ppo_loss

    if loss_function.func is not ppo_loss:
        return loss, metrics
    config = loss_function.keywords.get("config")
    if config is None and loss_function.args:
        config = loss_function.args[0]
    if config is None or not _diagnostics_enabled(config):
        return loss, metrics

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    response_fields = data.select("old_log_probs", "response_mask").to_padded_tensor()
    for name, metric in ppo_token_diagnostics(
        config, log_prob, response_fields["old_log_probs"], response_fields["response_mask"]
    ).items():
        metrics.setdefault(name, metric)
    return loss, metrics
