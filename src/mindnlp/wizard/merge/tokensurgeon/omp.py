# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
# Ported from MergeKit to MindSpore for Wizard

import logging
from typing import Optional, Tuple

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
from mindspore import Tensor  # pylint: disable=import-error

from .rope_helpers import apply_rope, estimate_pos_id_best

LOG = logging.getLogger(__name__)


def batch_omp(
    targets: Tensor,
    candidate_points: Tensor,
    k: int,
    eps: float = 1e-8,
    reorthogonalize_interval: int = 256,
) -> Tuple[Tensor, Tensor]:
    """
    Batched Orthogonal Matching Pursuit (OMP) to select `k` points from
    `candidate_points` that best approximate each target in `targets`.

    Args:
        targets: (B, D) tensor of target vectors.
        candidate_points: (N, D) tensor of candidate points.
        k: Number of points to select (sparsity level).
        eps: Tolerance for numerical stability.
        reorthogonalize_interval: Number of iterations between reorthogonalization steps.

    Returns:
        selected_indices: (B, k) tensor of indices selected for each target.
        coeff: (B, k) tensor of coefficients for each selected point.
    """
    B, D = targets.shape
    N, _ = candidate_points.shape
    if k > N:
        raise ValueError(f"Cannot select {k} points from {N} candidates")
    work_dtype = (
        targets.dtype
        if targets.dtype in (mindspore.float32, mindspore.float64)
        else mindspore.float32
    )
    targets_work = targets.astype(work_dtype)
    points_work = candidate_points.astype(work_dtype)

    q = ops.zeros((B, D, k), dtype=work_dtype)
    r = ops.zeros((B, k, k), dtype=work_dtype)
    selected_indices = ops.zeros((B, k), dtype=mindspore.int64)
    mask = ops.zeros((B, N), dtype=mindspore.bool_)
    residuals = targets_work.copy()

    for t in range(k):
        rms_0 = ops.norm(residuals, axis=1).mean()  # pylint: disable=unexpected-keyword-arg
        abs_inner = ops.abs(ops.matmul(residuals, points_work.T))  # (B, N)
        abs_inner = ops.masked_fill(abs_inner, mask, -float("inf"))

        _, new_idx = ops.max(abs_inner, axis=1)  # (B,)
        selected_indices[:, t] = new_idx
        mask[ops.arange(B), new_idx] = True

        new_atom = points_work[new_idx]  # (B, D)
        if t == 0:
            r[:, 0, 0] = ops.norm(new_atom, axis=1)  # pylint: disable=unexpected-keyword-arg
            norm = ops.clamp(r[:, 0, 0], min=eps)
            q[:, :, 0] = new_atom / norm.unsqueeze(1)
        else:
            projections = ops.bmm(
                q[:, :, :t].swapaxes(1, 2), new_atom.unsqueeze(-1)
            ).squeeze(-1)  # (B, t)
            residual = new_atom - ops.bmm(
                q[:, :, :t], projections.unsqueeze(-1)
            ).squeeze(-1)  # (B, D)
            norm = ops.clamp(ops.norm(residual, axis=1), min=eps)  # pylint: disable=unexpected-keyword-arg
            r[:, :t, t] = projections
            r[:, t, t] = norm
            q[:, :, t] = residual / norm.unsqueeze(-1)

        if t > 0 and t % reorthogonalize_interval == 0:
            q_b = q[:, :, : t + 1]
            # MindSpore doesn't have batched QR directly; use manual Gram-Schmidt
            # or fall back to loop-based QR for small k
            for b_idx in range(B):
                q_single, r_single = ops.qr(q_b[b_idx])
                r[b_idx, : t + 1, : t + 1] = ops.matmul(
                    r_single, r[b_idx, : t + 1, : t + 1]
                )
                q[b_idx, :, : t + 1] = q_single

        qt_targets = ops.bmm(
            q[:, :, : t + 1].swapaxes(1, 2), targets_work.unsqueeze(-1)
        )  # (B, t+1, 1)
        approx = ops.bmm(q[:, :, : t + 1], qt_targets).squeeze(-1)
        residuals = targets_work - approx
        LOG.debug(f"OMP iteration {t}: RMS {rms_0} -> {ops.norm(residuals, axis=1).mean()}")

    # Get final coefficients via triangular solve
    rhs = ops.bmm(q[:, :, :k].swapaxes(1, 2), targets_work.unsqueeze(-1))
    # solve_triangular: R * x = rhs  =>  x = R^{-1} rhs
    # MindSpore: use ops.SolveTriangular or manual inverse for upper triangular
    r_upper = r[:, :k, :k]
    final_coeff = _batched_triangular_solve(r_upper, rhs).squeeze(-1)

    if LOG.isEnabledFor(logging.DEBUG):
        rt_approx = ops.bmm(
            final_coeff.unsqueeze(1), points_work[selected_indices]
        ).squeeze(1)
        residuals = targets_work - rt_approx
        LOG.debug(f"OMP final RMS: {ops.norm(residuals, axis=1).mean()}")

    return selected_indices, final_coeff


def _batched_triangular_solve(R: Tensor, b: Tensor) -> Tensor:
    """Solve R @ x = b for upper-triangular R in a batched fashion.

    Args:
        R: (B, k, k) upper triangular matrices
        b: (B, k, 1) right hand side vectors

    Returns:
        x: (B, k, 1) solution vectors
    """
    B, k, _ = R.shape
    x = ops.zeros_like(b)
    for i in range(k - 1, -1, -1):
        val = b[:, i, :] - ops.matmul(
            R[:, i:i+1, i+1:], x[:, i+1:, :]
        ).squeeze(1)
        x[:, i, :] = val / R[:, i, i].unsqueeze(-1).clamp(min=1e-12)
    return x


def batch_mp_resets(
    targets: Tensor,
    candidate_points: Tensor,
    k: int,
    eps: float = 1e-8,
    total_iterations: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Matching Pursuit with Resets
    """
    if total_iterations is None:
        total_iterations = k * 3
    if total_iterations < k:
        raise ValueError(
            f"total_iterations {total_iterations} must be greater than or equal to k {k}"
        )
    B, D = targets.shape
    N, _ = candidate_points.shape
    if k > N:
        raise ValueError(f"Cannot select {k} points from {N} candidates")
    work_dtype = (
        targets.dtype
        if targets.dtype in (mindspore.float32, mindspore.float64)
        else mindspore.float32
    )
    targets_work = targets.astype(work_dtype)
    points_work = candidate_points.astype(work_dtype)
    selected_indices = ops.zeros((B, k), dtype=mindspore.int64)
    mask = ops.zeros((B, N), dtype=mindspore.bool_)
    coeff = ops.zeros((B, k), dtype=work_dtype)
    residuals = targets_work.copy()

    iter_indices = list(range(k))
    while len(iter_indices) < total_iterations:
        import numpy as np
        honk = np.random.permutation(k).tolist()
        iter_indices.extend(honk)
    iter_indices = iter_indices[:total_iterations]

    for step, t in enumerate(iter_indices):
        if step < k:
            inner_products = ops.matmul(residuals, points_work.T)  # B x N
            inner_products = ops.masked_fill(inner_products, mask, -float("inf"))
            max_values, max_indices = ops.max(inner_products, axis=1)
            selected_points = points_work[max_indices]  # B x D
            norms_sq = ops.sum(selected_points ** 2, axis=1) + eps  # pylint: disable=unexpected-keyword-arg
            coeffs = max_values / norms_sq
            residuals = residuals - coeffs.unsqueeze(-1) * selected_points
            selected_indices[:, t] = max_indices
            coeff[:, t] = coeffs
            mask = mask.scatter(1, max_indices.unsqueeze(1), True)
        else:
            old_indices = selected_indices[:, t]
            old_coeffs = coeff[:, t]
            old_points = points_work[old_indices]
            residuals = residuals + old_coeffs.unsqueeze(-1) * old_points
            mask = mask.scatter(1, old_indices.unsqueeze(1), False)
            inner_products = ops.matmul(residuals, points_work.T)
            inner_products = ops.masked_fill(inner_products, mask, -float("inf"))
            new_max_values, new_max_indices = ops.max(inner_products, axis=1)
            new_points = points_work[new_max_indices]
            norms_sq = ops.sum(new_points ** 2, axis=1) + eps  # pylint: disable=unexpected-keyword-arg
            new_coeffs = new_max_values / norms_sq
            residuals = residuals - new_coeffs.unsqueeze(-1) * new_points
            selected_indices[:, t] = new_max_indices
            coeff[:, t] = new_coeffs
            mask = mask.scatter(1, new_max_indices.unsqueeze(1), True)

    return selected_indices, coeff


def batch_mp_rope(  # pylint: disable=too-many-positional-arguments
    targets: Tensor,
    points_a: Tensor,
    points_b: Tensor,
    k: int,
    num_heads_a: int,
    num_heads_b: int,
    eps: float = 1e-8,
    a_rope_base: float = 10000.0,
    b_rope_base: float = 10000.0,
    final_least_squares: bool = True,
) -> Tensor:
    B, D_a = targets.shape
    N, _ = points_a.shape
    _, D_b = points_b.shape
    assert (
        points_a.shape[0] == points_b.shape[0]
    ), "Number of points in A and B must match"
    if k > N:
        raise ValueError(f"Cannot select {k} points from {N} candidates")
    work_dtype = (
        targets.dtype
        if targets.dtype in (mindspore.float32, mindspore.float64)
        else mindspore.float32
    )
    out_dtype = targets.dtype
    points_a = points_a.astype(work_dtype)
    points_b = points_b.astype(work_dtype)
    targets = targets.astype(work_dtype)
    selected_indices = ops.zeros((B, k), dtype=mindspore.int64)
    coeffs = ops.zeros((B, k), dtype=work_dtype)
    pos_ids = ops.zeros((B, k), dtype=work_dtype)
    mask = ops.zeros((B, N), dtype=mindspore.bool_)
    residuals = targets.copy()

    for t in range(k):
        abs_inner = ops.abs(ops.matmul(residuals, points_a.T))  # (B, N)
        abs_inner = ops.masked_fill(abs_inner, mask, -float("inf"))

        _, new_idx = ops.max(abs_inner, axis=1)  # (B,)

        selected_indices[:, t] = new_idx
        mask[ops.arange(B), new_idx] = True
        new_atom = points_a[new_idx]

        pos_id = estimate_pos_id_best(
            new_atom,
            residuals,
            num_heads=num_heads_a,
            head_dim=D_a // num_heads_a,
            base=a_rope_base,
        ).squeeze(-1)
        pos_id_neg = estimate_pos_id_best(
            new_atom,
            -residuals,
            num_heads=num_heads_a,
            head_dim=D_a // num_heads_a,
            base=a_rope_base,
        ).squeeze(-1)
        pos_id = ops.where(
            ops.abs(pos_id) < ops.abs(pos_id_neg), pos_id, pos_id_neg
        )
        pos_ids[:, t] = pos_id
        new_atom = apply_rope(
            new_atom,
            pos_id.unsqueeze(-1),
            num_heads=num_heads_a,
            head_dim=D_a // num_heads_a,
            base=a_rope_base,
        )

        current_coeff = (residuals * new_atom).sum(axis=1) / (
            new_atom.pow(2).sum(axis=1).clamp(min=eps)
        )
        coeffs[:, t] = current_coeff

        residuals = residuals - current_coeff.unsqueeze(1) * new_atom

    if final_least_squares:
        roped_pts_a = apply_rope(
            points_a[selected_indices],
            pos_ids.unsqueeze(-1),
            num_heads=num_heads_a,
            head_dim=D_a // num_heads_a,
            base=a_rope_base,
        )
        coeffs = ops.lstsq(
            roped_pts_a.swapaxes(1, 2).astype(mindspore.float32),
            targets.unsqueeze(-1).astype(mindspore.float32),
        ).squeeze(-1)

    selected_points_b = points_b[selected_indices]
    atoms_b = apply_rope(
        selected_points_b,
        pos_ids.unsqueeze(-1),
        num_heads=num_heads_b,
        head_dim=D_b // num_heads_b,
        base=b_rope_base,
    )
    approx_b = (atoms_b * coeffs.unsqueeze(-1)).sum(axis=1)
    final_tensor = approx_b.astype(out_dtype)
    return selected_indices, coeffs, final_tensor, targets - residuals
