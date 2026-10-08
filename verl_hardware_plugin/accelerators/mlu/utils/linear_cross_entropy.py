# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import typing

import torch
import torch.distributed as dist

from verl_hardware_plugin.utils.function_wrapper import FUNCTION_WRAPPER


@FUNCTION_WRAPPER.register_wrapper(func_name="verl.utils.kernel.linear_cross_entropy.LinearCrossEntropy.forward")
def forward(
    ctx,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: typing.Optional[float] = 1.0,
    reduction: typing.Optional[str] = "none",
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
):
    """_summary_

    Args:
        ctx (_type_): _description_
        hidden (torch.Tensor): (batch_size, num_tokens, hidden_size) -> (batch_size * num_tokens, hidden_size)
        weight (torch.Tensor): (vocab_size, hidden_size)
        labels (torch.Tensor): (batch_size, num_tokens) -> (batch_size * num_tokens, )
        temperature (typing.Optional[float], optional): _description_. Defaults to 1.0.
        reduction (typing.Optional[str], optional): _description_. Defaults to "none".
        dist_process_group (typing.Optional[dist.ProcessGroup], optional): _description_. Defaults to None.

    Returns:
        typing.List[torch.Tensor]: _description_
    """
    assert isinstance(temperature, float), f"temperature must be a float, but got {type(temperature)}"
    assert isinstance(reduction, str), f"reduction must be a str, but got {type(reduction)}"
    with torch.mlu.cnpx.range("LinearCrossEntropy-forward"):
        # use mlu kernel
        from . import kernels_mlu as kernels

        REDUCTION = kernels.get_entropy_reduction_enum_number(reduction.lower())

        original_hidden_shape = hidden.shape
        if len(hidden.shape) != 2:
            hidden = hidden.view(-1, hidden.shape[-1])  # (batch_size * num_tokens, hidden_size)
        if len(labels.shape) != 1:
            labels = labels.view(-1)

        logprobs, entropy, _maximum, _accumulate, _entropy_b = kernels.efficient_entropy_forward(
            hidden, weight, labels, REDUCTION, temperature, dist_process_group
        )

        ctx.save_for_backward(hidden, weight, labels, _maximum, _accumulate, _entropy_b)
        ctx.original_hidden_shape = original_hidden_shape
        ctx.REDUCTION = REDUCTION
        ctx.dist_process_group = dist_process_group
        ctx.should_return_fp32_grad = False
        ctx.temperature = temperature
    return logprobs, entropy


@FUNCTION_WRAPPER.register_wrapper(func_name="verl.utils.kernel.linear_cross_entropy.LinearCrossEntropy.backward")
def backward(ctx, dlogprobs: torch.Tensor, dentropy: torch.Tensor):
    from . import kernels_mlu as kernels

    with torch.mlu.cnpx.range("LinearCrossEntropy-backward"):
        (hidden, weight, labels, _maximum, _accumulate, _entropy_b) = ctx.saved_tensors
        REDUCTION = ctx.REDUCTION
        dist_process_group = ctx.dist_process_group
        should_return_fp32_grad = ctx.should_return_fp32_grad
        temperature = ctx.temperature

        d_hidden, d_weight = kernels.efficient_entropy_backward(
            dlogprobs,
            dentropy,
            hidden,
            weight,
            labels,
            _maximum,
            _accumulate,
            _entropy_b,
            REDUCTION,
            should_return_fp32_grad,
            temperature,
            dist_process_group,
        )
        d_hidden = d_hidden.view(ctx.original_hidden_shape)

    return (d_hidden, d_weight, None, None, None, None)


FUNCTION_WRAPPER.func_wrapper_apply()
