# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Ragged expert-parallel token dispatch/combine and topology-mesh helpers for TPU MoE training.

Replaces dense ``[ep_size * cap, D]`` ``all_to_all_single`` padding (``4x`` ICI traffic and ``[4x N, D]`` HBM
gather buffers at ``ep_capacity_factor=4.0``) with ``torch.ops.tpu._ragged_all_to_all`` (``1.0x`` real routed
tokens over ICI) and passes unpadded ``local_expert_counts_e`` to Pallas MegaBlox ``gmm``/``tgmm`` so trailing
unused capacity tiles are skipped on TensorCore.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import math
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.autograd.function import once_differentiable
from torch.distributed._functional_collectives import AsyncCollectiveTensor

EP_DISPATCHER_MODES = ("auto", "ragged", "static")

# Unit-test hook: force CPU tensors through ``torch.ops.tpu._ragged_all_to_all`` (e.g. under ``torch.compile``).
_FORCE_TPU_CUSTOM_OP_ON_CPU = False

# Resolve optional ``torchtitan`` helpers once at module import time so ``torch.compile(..., fullgraph=True)``
# never encounters a failing ``import`` statement inside traced ``dispatch`` / ``combine`` regions.
try:
    from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh as _maybe_set_sparse_mesh
except ImportError:
    _maybe_set_sparse_mesh = None  # type: ignore[assignment]

try:
    from torchtitan.experiments.tpu.token_dispatcher import (
        scale_route_weights as _tt_scale_route_weights,
    )
    from torchtitan.experiments.tpu.token_dispatcher import (
        stable_grouped_reduce as _tt_stable_grouped_reduce,
    )
except ImportError:
    _tt_scale_route_weights = None  # type: ignore[assignment]
    _tt_stable_grouped_reduce = None  # type: ignore[assignment]


def has_tpu_ragged_all_to_all() -> bool:
    """Whether ``torch.ops.tpu._ragged_all_to_all`` is registered in the current PyTorch runtime."""
    tpu_ns = getattr(torch.ops, "tpu", None)
    return getattr(tpu_ns, "_ragged_all_to_all", None) is not None


def should_enable_tpu_topo_mesh(requested: Optional[bool], parallelism_cls: Any) -> Optional[bool]:
    """Resolve ``ParallelismConfig.enable_tpu_topo_mesh``, guarding ``DeviceMesh(preserve_rank_order=...)``.

    ``torchtitan``'s ``ParallelDims.build_mesh`` raises ``ValueError`` if ``enable_tpu_topo_mesh=True`` on PyTorch
    builds whose ``DeviceMesh.__init__`` lacks ``preserve_rank_order`` (``torch < 2.15``). When ``requested`` is
    ``None``, enable ``enable_tpu_topo_mesh`` automatically whenever both ``ParallelismConfig`` and ``DeviceMesh``
    support it so multi-host 3D ICI torus ``(x, y, z)`` rank coordinates are preserved instead of sorted by host ID.
    """
    fields = getattr(parallelism_cls, "__dataclass_fields__", None) or {}
    if "enable_tpu_topo_mesh" not in fields:
        return None
    if requested is not None:
        return bool(requested)
    try:
        from torch.distributed.device_mesh import DeviceMesh

        return "preserve_rank_order" in inspect.signature(DeviceMesh.__init__).parameters
    except Exception:
        return False


def expert_tile_alignment(inner_experts: Any) -> int:
    """Row-count multiple keeping ``TpuGroupedExperts.forward`` at ``pad_rows == 0`` for unpadded MegaBlox counts."""
    if inner_experts is None or not bool(getattr(inner_experts, "use_pallas_grouped_mm", False)):
        return 1
    align = max(1, int(getattr(inner_experts, "pallas_grouped_mm_tile_m", 1) or 1))
    for attr in ("pallas_grouped_mm_tgmm_tiling", "pallas_grouped_mm_w2_tgmm_tiling"):
        tiling = getattr(inner_experts, attr, None)
        if isinstance(tiling, tuple | list) and len(tiling) > 0 and int(tiling[0] or 0) > 0:
            align = math.lcm(align, int(tiling[0]))
    return align


def _wait_collective(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _cpu_ragged_all_to_all(
    operand: torch.Tensor,
    output: torch.Tensor,
    input_offsets: torch.Tensor,
    send_sizes: torch.Tensor,
    output_offsets: torch.Tensor,
    recv_sizes: torch.Tensor,
    pg: dist.ProcessGroup,
) -> torch.Tensor:
    """Exact CPU ``ProcessGroup`` emulation of ``jax.lax.ragged_all_to_all`` / ``torch.ops.tpu._ragged_all_to_all``."""
    world = dist.get_world_size(pg)
    in_off = input_offsets.detach().to(device="cpu", dtype=torch.int64)
    s_sz = send_sizes.detach().to(device="cpu", dtype=torch.int64)
    out_off = output_offsets.detach().to(device="cpu", dtype=torch.int64).contiguous()
    r_sz = recv_sizes.detach().to(device="cpu", dtype=torch.int64)

    # Sender j provided output_offsets[dst] = row in dst's output where j's slice lands.
    dst_offsets_from_sender = torch.empty_like(out_off)
    dist.all_to_all_single(dst_offsets_from_sender, out_off, group=pg)

    send_list = [operand.detach()[int(in_off[j]) : int(in_off[j] + s_sz[j])] for j in range(world)]
    flat_send = torch.cat(send_list, dim=0).contiguous() if send_list else operand.detach()[:0].contiguous()
    total_recv = int(r_sz.sum().item())
    flat_recv = operand.new_empty((total_recv, *operand.shape[1:]))
    dist.all_to_all_single(
        flat_recv,
        flat_send,
        output_split_sizes=r_sz.tolist(),
        input_split_sizes=s_sz.tolist(),
        group=pg,
    )

    res = output.detach().clone()
    cursor = 0
    for sender in range(world):
        count = int(r_sz[sender].item())
        if count > 0:
            dst_start = int(dst_offsets_from_sender[sender].item())
            res[dst_start : dst_start + count] = flat_recv[cursor : cursor + count]
            cursor += count
    return res


def _raw_ragged_all_to_all(
    operand: torch.Tensor,
    output: torch.Tensor,
    input_offsets: torch.Tensor,
    send_sizes: torch.Tensor,
    output_offsets: torch.Tensor,
    recv_sizes: torch.Tensor,
    pg_name: str,
    pg: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    use_custom_op = has_tpu_ragged_all_to_all() and (
        operand.device.type != "cpu" or _FORCE_TPU_CUSTOM_OP_ON_CPU or pg is None
    )
    if use_custom_op:
        # XLA TPU's Int64Rewriter does not implement an s64 -> s32 rewrite for StableHLO ``ragged_all_to_all``;
        # all four offset/size tensors MUST be ``torch.int32`` (``s32``).
        if input_offsets.dtype != torch.int32:
            input_offsets = input_offsets.to(torch.int32)
        if send_sizes.dtype != torch.int32:
            send_sizes = send_sizes.to(torch.int32)
        if output_offsets.dtype != torch.int32:
            output_offsets = output_offsets.to(torch.int32)
        if recv_sizes.dtype != torch.int32:
            recv_sizes = recv_sizes.to(torch.int32)
        # Pass ``operand.detach()`` so C++ ``AtenRaggedAllToAllAutograd`` skips its 2 eager ``pg->alltoall_base``
        # offset transposes (which fail on FakeTensors under ``fullgraph=True`` and add 4 extra collectives/layer).
        return torch.ops.tpu._ragged_all_to_all(
            operand.detach(),
            output.detach(),
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            pg_name,
        )
    if pg is None:
        raise RuntimeError("CPU fallback for ragged_all_to_all requires an initialized ProcessGroup")
    return _cpu_ragged_all_to_all(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, pg)


class _RaggedAllToAllFn(torch.autograd.Function):
    """Differentiable ``ragged_all_to_all`` using pre-transposed backward offsets (0 extra backward collectives)."""

    @staticmethod
    def forward(
        ctx: Any,
        operand: torch.Tensor,
        output_rows: int,
        input_offsets: torch.Tensor,
        send_sizes: torch.Tensor,
        output_offsets: torch.Tensor,
        recv_sizes: torch.Tensor,
        bwd_input_offsets: torch.Tensor,
        bwd_output_offsets: torch.Tensor,
        pg_name: str,
        pg: Optional[dist.ProcessGroup],
    ) -> torch.Tensor:
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(bwd_input_offsets, recv_sizes, bwd_output_offsets, send_sizes)
        ctx.operand_rows = int(operand.shape[0])
        ctx.pg_name = pg_name
        ctx.pg = pg
        out_init = operand.new_zeros((output_rows, *operand.shape[1:]))
        return _raw_ragged_all_to_all(
            operand,
            out_init,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            pg_name,
            pg,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: Optional[torch.Tensor]):
        if grad_output is None or not ctx.needs_input_grad[0]:
            return (None,) * 10
        bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz = ctx.saved_tensors
        grad_init = grad_output.new_zeros((ctx.operand_rows, *grad_output.shape[1:]))
        grad_operand = _raw_ragged_all_to_all(
            grad_output.contiguous(),
            grad_init,
            bwd_in_off,
            bwd_send_sz,
            bwd_out_off,
            bwd_recv_sz,
            ctx.pg_name,
            ctx.pg,
        )
        return grad_operand, None, None, None, None, None, None, None, None, None


class _PermuteValidRows(torch.autograd.Function):
    """Bijective row permutation on ``[0 .. R-1]`` that zeros invalid tail rows via ``torch.where`` (NaN-safe).

    When MegaBlox ``gmm``/``tgmm`` receives unpadded ``group_sizes`` (``sum == total_valid < R``), rows
    ``[total_valid : R]`` of its output/input-grad allocation are never written and may contain ``NaN`` bit patterns
    in recycled HBM. Using ``torch.where`` instead of ``x * mask`` prevents ``NaN * 0.0 -> NaN`` propagation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x_RD: torch.Tensor,
        fwd_idx_R: torch.Tensor,
        bwd_idx_R: torch.Tensor,
        valid_mask_R1: torch.Tensor,
    ) -> torch.Tensor:
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(bwd_idx_R, valid_mask_R1)
        if x_RD.shape[0] == 0:
            return x_RD.clone()
        zeros = torch.zeros_like(x_RD)
        clean_in = torch.where(valid_mask_R1, x_RD, zeros)
        permuted = torch.index_select(clean_in, 0, fwd_idx_R)
        return torch.where(valid_mask_R1, permuted, zeros)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out_RD: Optional[torch.Tensor]):
        if grad_out_RD is None or not ctx.needs_input_grad[0]:
            return None, None, None, None
        bwd_idx_R, valid_mask_R1 = ctx.saved_tensors
        if grad_out_RD.shape[0] == 0:
            return grad_out_RD.clone(), None, None, None
        zeros = torch.zeros_like(grad_out_RD)
        clean_grad = torch.where(valid_mask_R1, grad_out_RD, zeros)
        grad_x = torch.index_select(clean_grad, 0, bwd_idx_R)
        return torch.where(valid_mask_R1, grad_x, zeros), None, None, None


@dataclasses.dataclass(frozen=True, kw_only=True)
class TpuRaggedEPDispatchMetadata:
    """Static-shape metadata linking ``TpuRaggedEPTokenDispatcher.dispatch`` and ``combine``."""

    token_indices_experts_sorted_N: torch.Tensor
    topk_scores_experts_sorted_N: torch.Tensor
    num_routed: int
    r_static: int
    disp_in_off_EP: torch.Tensor
    disp_send_sz_EP: torch.Tensor
    disp_out_off_EP: torch.Tensor
    disp_recv_sz_EP: torch.Tensor
    comb_in_off_EP: torch.Tensor
    comb_out_off_EP: torch.Tensor
    perm_idx_R: torch.Tensor
    inv_perm_idx_R: torch.Tensor
    valid_mask_R1: torch.Tensor


@contextlib.contextmanager
def _maybe_sparse_mesh():
    if _maybe_set_sparse_mesh is None:
        yield
        return
    with _maybe_set_sparse_mesh():
        yield


def _default_local_reorder(
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    score_before_experts: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fallback ``_local_reorder`` matching ``TokenDispatcher._local_reorder`` for standalone tests."""
    top_k = int(topk_expert_ids_TK.shape[1])
    flat_experts_N = topk_expert_ids_TK.reshape(-1)
    flat_scores_N = topk_scores_TK.reshape(-1)
    order_N = torch.argsort(flat_experts_N.to(torch.int32), stable=True)
    token_indices_N = order_N // top_k
    scores_sorted_N = torch.index_select(flat_scores_N, 0, order_N)
    routed_ND = torch.index_select(x_TD, 0, token_indices_N)
    if score_before_experts:
        routed_ND = (routed_ND.to(torch.float32) * scores_sorted_N.unsqueeze(-1).to(torch.float32)).to(x_TD.dtype)
    return routed_ND, token_indices_N, scores_sorted_N


def _default_combine_reduce(
    routed_output_ND: torch.Tensor,
    token_indices_N: torch.Tensor,
    scores_sorted_N: torch.Tensor,
    x_TD: torch.Tensor,
    *,
    top_k: int,
    score_before_experts: bool,
) -> torch.Tensor:
    """Fallback local combine matching ``TpuAllToAllTokenDispatcher`` when ``torchtitan`` is not imported."""
    _ = top_k
    if not score_before_experts:
        routed_output_ND = (routed_output_ND.to(torch.float32) * scores_sorted_N.unsqueeze(-1).to(torch.float32)).to(
            routed_output_ND.dtype
        )
    out_TD = x_TD.new_zeros(x_TD.shape)
    out_TD.index_add_(0, token_indices_N, routed_output_ND.to(out_TD.dtype))
    return out_TD


class TpuRaggedEPTokenDispatcher:
    """Ragged ``all_to_all`` EP token dispatcher with unpadded MegaBlox expert counts and 0 host syncs.

    Wraps an existing ``TokenDispatcher`` (``TpuQwen35TokenDispatcher``, ``TpuAllToAllTokenDispatcher``, or
    ``AllToAllTokenDispatcher``) so ``EP == 1`` delegates directly to the base dispatcher while ``EP > 1``:

    1. Exchanges ``(ep_size, e)`` capped expert counts and ``(ep_size, 2)`` buffer offsets using two tiny uniform
       ``int32`` ``all_to_all_single``s (``< 1 KB``, 100% SPMD-symmetric across ranks with zero rank constants).
    2. Scatters ``1.0x`` real routed tokens directly from ``routed_input_ND`` ``[N, D]`` into a tile-aligned
       ``recv_buf_flat`` ``[r_static, D]`` via ``torch.ops.tpu._ragged_all_to_all`` (eliminating ``4x`` ICI padding
       bytes and both ``[4x N, D]`` ``send_buf_flat`` / ``send_back_flat`` HBM gather buffers).
    3. Permutes the ``total_valid`` received rows into expert-major order with NaN-safe ``_PermuteValidRows`` and,
       when ``unpadded_expert_counts=True`` (Pallas MegaBlox ``use_pallas_grouped_mm=True``), passes unpadded
       ``local_expert_counts_e`` (``sum == total_valid``) so ``gmm``/``tgmm`` skip trailing zero-row tiles.
    """

    def __init__(
        self,
        base_dispatcher: Any = None,
        *,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
        score_before_experts: Optional[bool] = None,
        capacity_factor: Optional[float] = 4.0,
        unpadded_expert_counts: bool = False,
        row_align: int = 1,
        ep_mesh: Any = None,
    ):
        self._base = base_dispatcher
        if num_experts is None or top_k is None:
            if base_dispatcher is None:
                raise ValueError("num_experts and top_k are required when base_dispatcher is None")
            num_experts = base_dispatcher.num_experts if num_experts is None else num_experts
            top_k = base_dispatcher.top_k if top_k is None else top_k
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        # TorchTitan's LocalTokenDispatcher / TpuAllToAllTokenDispatcher / TpuQwen35TokenDispatcher apply routing
        # scores after expert compute in combine() and do not define ``score_before_experts`` as an attribute.
        self.score_before_experts = bool(
            score_before_experts
            if score_before_experts is not None
            else getattr(base_dispatcher, "score_before_experts", False)
        )
        self.fuse_route_score_backward = bool(getattr(base_dispatcher, "fuse_route_score_backward", False))
        self.capacity_factor = capacity_factor
        self.unpadded_expert_counts = bool(unpadded_expert_counts)
        self.row_align = max(1, int(row_align))
        self.ep_mesh = ep_mesh if ep_mesh is not None else getattr(base_dispatcher, "ep_mesh", None)

    def wire_meshes(self, ep_mesh: Any = None, **kwargs: Any) -> None:
        self.ep_mesh = ep_mesh
        if self._base is not None and callable(getattr(self._base, "wire_meshes", None)):
            self._base.wire_meshes(ep_mesh=ep_mesh, **kwargs)

    def _reorder_local(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._base is not None and callable(getattr(self._base, "_local_reorder", None)):
            return self._base._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)
        return _default_local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK, self.score_before_experts)

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        if self.ep_mesh is None:
            if self._base is not None:
                return self._base.dispatch(x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E)
            raise RuntimeError("TpuRaggedEPTokenDispatcher with ep_mesh=None requires a base dispatcher")

        ep_size = int(self.ep_mesh.size())
        e = self.num_experts // ep_size
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._reorder_local(x_TD, topk_scores_TK, topk_expert_ids_TK)

        with _maybe_sparse_mesh():
            pg = self.ep_mesh.get_group()
            pg_name = pg.group_name
            num_routed = int(routed_input_ND.shape[0])
            if self.capacity_factor is None:
                cap = max(num_routed, 1)
            else:
                cap = max(
                    1,
                    min(
                        max(num_routed, 1),
                        math.ceil(float(self.capacity_factor) * num_routed / ep_size),
                    ),
                )
            raw_r = ep_size * cap
            r_static = ((raw_r + self.row_align - 1) // self.row_align) * self.row_align
            device = x_TD.device

            with torch.no_grad():
                local_counts_EP_e = num_local_tokens_per_expert_E.to(torch.int32).reshape(ep_size, e)
                if self.capacity_factor is not None:
                    local_cum_e = local_counts_EP_e.cumsum(dim=1, dtype=torch.int32).clamp(max=cap)
                    local_counts_capped_EP_e = torch.cat(
                        [local_cum_e[:, :1], local_cum_e[:, 1:] - local_cum_e[:, :-1]],
                        dim=1,
                    )
                else:
                    local_counts_capped_EP_e = local_counts_EP_e

                unclamped_send_EP = local_counts_EP_e.sum(dim=1, dtype=torch.int32)
                disp_send_sz_EP = local_counts_capped_EP_e.sum(dim=1, dtype=torch.int32)
                unclamped_starts_EP = unclamped_send_EP.cumsum(dim=0, dtype=torch.int32) - unclamped_send_EP
                disp_in_off_EP = torch.where(
                    disp_send_sz_EP > 0,
                    unclamped_starts_EP,
                    torch.zeros_like(unclamped_starts_EP),
                )

                # Collective 1 (uniform int32, ep_size * e): exchange capped (dst_rank, local_expert) counts.
                recv_counts_flat = _wait_collective(
                    funcol.all_to_all_single(
                        local_counts_capped_EP_e.reshape(-1),
                        output_split_sizes=None,
                        input_split_sizes=None,
                        group=pg,
                    )
                )
                t_mat = recv_counts_flat.to(torch.int32).reshape(ep_size, e)

                disp_recv_sz_EP = t_mat.sum(dim=1, dtype=torch.int32)
                recv_ends_EP = disp_recv_sz_EP.cumsum(dim=0, dtype=torch.int32)
                recv_starts_EP = recv_ends_EP - disp_recv_sz_EP
                comb_in_off_EP = torch.where(
                    disp_recv_sz_EP > 0,
                    recv_starts_EP,
                    torch.zeros_like(recv_starts_EP),
                )

                # Collective 2 (uniform int32, ep_size * 2): transpose (comb_in_off_EP, disp_in_off_EP) across peers
                # so both forward dispatch and backward/combine know remote offsets with 0 extra backward collectives.
                local_pair_EP_2 = torch.stack([comb_in_off_EP, disp_in_off_EP], dim=1)
                peer_pair_EP_2 = (
                    _wait_collective(
                        funcol.all_to_all_single(
                            local_pair_EP_2.reshape(-1),
                            output_split_sizes=None,
                            input_split_sizes=None,
                            group=pg,
                        )
                    )
                    .to(torch.int32)
                    .reshape(ep_size, 2)
                )
                disp_out_off_EP = torch.where(
                    disp_send_sz_EP > 0,
                    peer_pair_EP_2[:, 0].contiguous(),
                    torch.zeros_like(disp_send_sz_EP),
                )
                comb_out_off_EP = torch.where(
                    disp_recv_sz_EP > 0,
                    peer_pair_EP_2[:, 1].contiguous(),
                    torch.zeros_like(disp_recv_sz_EP),
                )

                # Map each contiguous received row q in [0 .. total_valid - 1] to its local expert [0 .. e - 1]
                # (and trailing rows >= total_valid to sentinel e) via 2-level sender + intra-sender prefix sums.
                total_valid = recv_ends_EP[-1]
                q_R = torch.arange(r_static, device=device, dtype=torch.int32)
                is_valid_R = q_R < total_valid
                sender_R = (
                    (q_R.unsqueeze(1) >= recv_ends_EP.unsqueeze(0))
                    .sum(dim=1, dtype=torch.int32)
                    .clamp(max=max(ep_size - 1, 0))
                )
                pos_in_sender_R = q_R - torch.index_select(recv_starts_EP, 0, sender_R)
                rank_local_ends_EP_e = t_mat.cumsum(dim=1, dtype=torch.int32)
                local_exp_R = (
                    pos_in_sender_R.unsqueeze(1) >= torch.index_select(rank_local_ends_EP_e, 0, sender_R)
                ).sum(dim=1, dtype=torch.int32)
                slot_expert_R = torch.where(is_valid_R, local_exp_R, torch.full_like(local_exp_R, e))
                perm_idx_R = torch.argsort(slot_expert_R, stable=True).to(torch.int32)
                inv_perm_idx_R = torch.argsort(perm_idx_R).to(torch.int32)
                valid_mask_R1 = is_valid_R.unsqueeze(1)

                local_expert_counts_e = t_mat.sum(dim=0, dtype=torch.int32)
                if self.unpadded_expert_counts:
                    num_global_tokens_per_local_expert_e = local_expert_counts_e
                else:
                    pad_rows = r_static - total_valid
                    num_global_tokens_per_local_expert_e = torch.cat(
                        [local_expert_counts_e[:-1], local_expert_counts_e[-1:] + pad_rows]
                    )

            recv_buf_flat = _RaggedAllToAllFn.apply(
                routed_input_ND.contiguous(),
                r_static,
                disp_in_off_EP,
                disp_send_sz_EP,
                disp_out_off_EP,
                disp_recv_sz_EP,
                comb_in_off_EP,
                comb_out_off_EP,
                pg_name,
                pg,
            )
            routed_input_RD = _PermuteValidRows.apply(
                recv_buf_flat,
                perm_idx_R,
                inv_perm_idx_R,
                valid_mask_R1,
            )

        metadata = TpuRaggedEPDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            num_routed=num_routed,
            r_static=r_static,
            disp_in_off_EP=disp_in_off_EP,
            disp_send_sz_EP=disp_send_sz_EP,
            disp_out_off_EP=disp_out_off_EP,
            disp_recv_sz_EP=disp_recv_sz_EP,
            comb_in_off_EP=comb_in_off_EP,
            comb_out_off_EP=comb_out_off_EP,
            perm_idx_R=perm_idx_R,
            inv_perm_idx_R=inv_perm_idx_R,
            valid_mask_R1=valid_mask_R1,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: Any,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_mesh is None:
            if self._base is not None:
                return self._base.combine(routed_output_RD, metadata, x_TD)
            raise RuntimeError("TpuRaggedEPTokenDispatcher with ep_mesh=None requires a base dispatcher")
        assert isinstance(metadata, TpuRaggedEPDispatchMetadata)

        with _maybe_sparse_mesh():
            pg = self.ep_mesh.get_group()
            pg_name = pg.group_name
            recv_back_flat = _PermuteValidRows.apply(
                routed_output_RD,
                metadata.inv_perm_idx_R,
                metadata.perm_idx_R,
                metadata.valid_mask_R1,
            )
            routed_output_ND = _RaggedAllToAllFn.apply(
                recv_back_flat.contiguous(),
                metadata.num_routed,
                metadata.comb_in_off_EP,
                metadata.disp_recv_sz_EP,
                metadata.comb_out_off_EP,
                metadata.disp_send_sz_EP,
                metadata.disp_in_off_EP,
                metadata.disp_out_off_EP,
                pg_name,
                pg,
            )

        if self._base is not None and _tt_stable_grouped_reduce is not None:
            if not self.score_before_experts:
                if self.fuse_route_score_backward and _tt_scale_route_weights is not None:
                    routed_output_ND = _tt_scale_route_weights(
                        routed_output_ND,
                        metadata.topk_scores_experts_sorted_N,
                    )
                else:
                    routed_output_ND = (
                        routed_output_ND.to(torch.float32)
                        * metadata.topk_scores_experts_sorted_N.reshape(-1, 1).to(torch.float32)
                    ).to(routed_output_ND.dtype)
            return _tt_stable_grouped_reduce(
                routed_output_ND,
                metadata.token_indices_experts_sorted_N,
                self.top_k,
            )

        return _default_combine_reduce(
            routed_output_ND,
            metadata.token_indices_experts_sorted_N,
            metadata.topk_scores_experts_sorted_N,
            x_TD,
            top_k=self.top_k,
            score_before_experts=self.score_before_experts,
        )


def resolve_ep_dispatcher_mode(requested: Optional[str], *, spmd_backend: Optional[str] = None) -> str:
    """Resolve ``engine_kwargs.tpu.ep_dispatcher`` (``auto`` | ``ragged`` | ``static``) to ``ragged`` or ``static``."""
    mode = (requested or "auto").strip().lower()
    if mode not in EP_DISPATCHER_MODES:
        raise ValueError(f"engine_kwargs.tpu.ep_dispatcher must be one of {EP_DISPATCHER_MODES}, got {requested!r}")
    if mode == "static":
        return "static"
    if mode == "ragged":
        if spmd_backend == "spmd_types":
            raise ValueError("engine_kwargs.tpu.ep_dispatcher='ragged' is not supported with spmd_backend='spmd_types'")
        return "ragged"
    # "auto": enable ragged EP dispatch whenever torch_tpu provides _ragged_all_to_all and spmd_types is inactive.
    if spmd_backend != "spmd_types" and has_tpu_ragged_all_to_all():
        return "ragged"
    return "static"


def install_tpu_ep_token_dispatcher(
    modules: Any,
    *,
    ep_dispatcher: Optional[str] = None,
    capacity_factor: Optional[float] = 4.0,
    spmd_backend: Optional[str] = None,
    compile_config: Any = None,
) -> tuple[str, int]:
    """Configure MoE EP token dispatchers across ``modules``, returning ``(resolved_mode, num_updated)``.

    * When ``resolved_mode == "ragged"``, wraps every ``RoutedExperts.token_dispatcher`` in
      :class:`TpuRaggedEPTokenDispatcher` with ``unpadded_expert_counts=bool(inner_experts.use_pallas_grouped_mm)``
      and ``row_align=expert_tile_alignment(inner_experts)``. Any ``TransformerBlock`` that was previously relaxed to
      ``fullgraph=False`` (because ``AllToAllTokenDispatcher`` lacked static shapes) is upgraded back to
      ``fullgraph=True``.
    * When ``resolved_mode == "static"``, sets ``capacity_factor`` on ``TpuQwen35TokenDispatcher`` instances.
    """
    resolved = resolve_ep_dispatcher_mode(ep_dispatcher, spmd_backend=spmd_backend)
    roots = modules if isinstance(modules, list | tuple) else [modules]
    count = 0

    if resolved == "static":
        seen_ids: set[int] = set()
        for root in roots:
            if not isinstance(root, nn.Module):
                continue
            for m in root.modules():
                for candidate in (m, getattr(m, "token_dispatcher", None)):
                    if candidate is None or id(candidate) in seen_ids:
                        continue
                    if isinstance(candidate, TpuRaggedEPTokenDispatcher) and candidate._base is not None:
                        candidate = candidate._base
                    if any(c.__name__ == "TpuQwen35TokenDispatcher" for c in type(candidate).__mro__):
                        seen_ids.add(id(candidate))
                        candidate.capacity_factor = capacity_factor  # type: ignore[attr-defined]
                        count += 1
        return resolved, count

    compile_enabled = bool(getattr(compile_config, "enable", False))
    compile_backend = getattr(compile_config, "backend", "tpu")

    for root in roots:
        if not isinstance(root, nn.Module):
            continue
        root_count = 0
        for parent in root.modules():
            td = getattr(parent, "token_dispatcher", None)
            if td is None:
                continue
            if not isinstance(td, TpuRaggedEPTokenDispatcher) and not (
                callable(getattr(td, "dispatch", None)) and callable(getattr(td, "combine", None))
            ):
                continue
            inner_experts = getattr(parent, "inner_experts", None)
            use_pallas = bool(getattr(inner_experts, "use_pallas_grouped_mm", False))
            align = expert_tile_alignment(inner_experts)
            if isinstance(td, TpuRaggedEPTokenDispatcher):
                td.capacity_factor = capacity_factor
                td.unpadded_expert_counts = use_pallas
                td.row_align = align
            else:
                parent.token_dispatcher = TpuRaggedEPTokenDispatcher(
                    td,
                    capacity_factor=capacity_factor,
                    unpadded_expert_counts=use_pallas,
                    row_align=align,
                )
            count += 1
            root_count += 1

        if compile_enabled and root_count > 0:
            layers = getattr(root, "layers", None)
            if layers is not None:
                for _, block in layers.named_children():
                    if getattr(block, "_tpu_compile_relaxed", False) and callable(getattr(block, "compile", None)):
                        has_ragged = any(
                            isinstance(getattr(m, "token_dispatcher", None), TpuRaggedEPTokenDispatcher)
                            for m in block.modules()
                        )
                        if has_ragged:
                            block.compile(backend=compile_backend, fullgraph=True)
                            block._tpu_compile_relaxed = False  # type: ignore[attr-defined]

    return resolved, count
