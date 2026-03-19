# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
# Ported from MergeKit to MindSpore for Wizard

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
from mindspore import Tensor  # pylint: disable=import-error


def llama_rope_rotationmat(theta: Tensor) -> Tensor:
    """
    Create a rotation matrix for RoPE as used in the `transformers` Llama implementation.

    Args:
        theta: Tensor of shape (..., n_heads, head_dim // 2) representing the angles for the rotation.
    """
    n_heads = theta.shape[-2]
    head_dim = theta.shape[-1] * 2
    theta_p = ops.cat([theta, theta], axis=-1)
    cos_theta = ops.cos(theta_p)
    sin_theta = ops.sin(theta_p)
    P = ops.zeros(
        tuple(list(theta.shape[:-1]) + [head_dim, head_dim]),
        dtype=theta.dtype,
    )
    idx = ops.arange(head_dim // 2)
    P[..., :, idx, idx] = cos_theta[..., :, idx]
    P[..., :, idx, head_dim // 2 + idx] = sin_theta[..., :, idx]
    P[..., :, head_dim // 2 + idx, idx] = -sin_theta[..., :, idx]
    P[..., :, head_dim // 2 + idx, head_dim // 2 + idx] = cos_theta[..., :, idx]
    return P


def _rope_inv_freq(base: float, dim: int) -> Tensor:
    return 1.0 / (
        base ** (ops.arange(0, dim, 2).astype(mindspore.float32) / dim)
    )


def estimate_theta(
    x_0: Tensor,
    x_1: Tensor,
    num_heads: int,
    head_dim: int,
) -> Tensor:
    """Estimate a set of per-head, per-dimension angles (theta) such that
    rotating x_0 by theta will least-squares approximate x_1.

    Args:
        x_0: Tensor of shape (..., n_heads*head_dim) representing the first input.
        x_1: Tensor of shape (..., n_heads*head_dim) representing the second input.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
    Returns:
        Tensor of shape (..., n_heads, head_dim // 2) representing the estimated theta values.
    """
    x0_reshaped = x_0.view(*x_0.shape[:-1], num_heads, head_dim)
    x1_reshaped = x_1.view(*x_1.shape[:-1], num_heads, head_dim)

    half_dim = head_dim // 2
    x0_i = x0_reshaped[..., :half_dim]
    x0_j = x0_reshaped[..., half_dim:]
    x1_i = x1_reshaped[..., :half_dim]
    x1_j = x1_reshaped[..., half_dim:]

    A_d = x0_i * x1_i + x0_j * x1_j
    B_d = x0_i * x1_j - x0_j * x1_i

    theta = ops.atan2(B_d, A_d)
    return theta


def estimate_position_id(
    x_0: Tensor,
    x_1: Tensor,
    num_heads: int,
    head_dim: int,
    base: float = 10000.0,
) -> Tensor:
    """
    Estimate a scalar position ID such that applying RoPE to x_0
    will least-squares approximate x_1.
    """
    x0_heads = x_0.view(*x_0.shape[:-1], num_heads, head_dim)
    x1_heads = x_1.view(*x_1.shape[:-1], num_heads, head_dim)

    split_idx = head_dim // 2
    x0_a = x0_heads[..., :split_idx]
    x0_b = x0_heads[..., split_idx:]
    x1_c = x1_heads[..., :split_idx]
    x1_d = x1_heads[..., split_idx:]

    numerator = x0_a * x1_d - x0_b * x1_c
    denominator = x0_a * x1_c + x0_b * x1_d
    theta = ops.atan2(numerator, denominator)

    inv_freq = _rope_inv_freq(base, head_dim)
    pos_i = theta / inv_freq
    weights = x0_a.pow(2) + x0_b.pow(2)
    sum_pos = (pos_i * weights).sum(axis=(-1, -2))
    sum_weights = weights.sum(axis=(-1, -2))
    pos_estimate = sum_pos / (sum_weights + 1e-8)
    return pos_estimate.unsqueeze(-1)


def estimate_position_id_projection(
    x_0: Tensor,
    x_1: Tensor,
    num_heads: int,
    head_dim: int,
    base: float = 10000.0,
) -> Tensor:
    inv_freq = _rope_inv_freq(base, head_dim)
    basis_vector = inv_freq.view(1, 1, head_dim // 2).expand(
        x_0.shape[:-1] + (num_heads, head_dim // 2)
    )
    basis_vector = basis_vector.reshape(*x_0.shape[:-1], -1)
    basis_vector_norm = ops.norm(basis_vector, axis=-1, keepdims=True)  # pylint: disable=unexpected-keyword-arg
    basis_vector = basis_vector / (basis_vector_norm + 1e-8)
    theta = estimate_theta(x_0, x_1, num_heads, head_dim)
    theta = theta.reshape(*x_0.shape[:-1], -1)
    projection = ops.sum(theta * basis_vector, axis=-1)  # pylint: disable=unexpected-keyword-arg
    f_norm = ops.norm(inv_freq)
    scaling_factor = ops.sqrt(Tensor(float(num_heads), dtype=mindspore.float32)) * f_norm
    pos_estimate = projection / (scaling_factor + 1e-8)
    return pos_estimate.unsqueeze(-1)


def apply_rope_theta(
    x: Tensor,
    theta: Tensor,
    num_heads: int,
    head_dim: int,
) -> Tensor:
    """
    Apply RoPE to the input tensor x using the given theta.
    """
    x_reshaped = x.view(*x.shape[:-1], num_heads, head_dim)

    half_dim = head_dim // 2
    x_i = x_reshaped[..., :half_dim]
    x_j = x_reshaped[..., half_dim:]

    cos_theta = ops.cos(theta)
    sin_theta = ops.sin(theta)

    x_i_rot = x_i * cos_theta - x_j * sin_theta
    x_j_rot = x_j * cos_theta + x_i * sin_theta

    rotated = ops.cat([x_i_rot, x_j_rot], axis=-1)
    return rotated.view(*x.shape)


def estimate_pos_id_best(
    x_0: Tensor,
    x_1: Tensor,
    num_heads: int,
    head_dim: int,
    base: float = 10000.0,
) -> Tensor:
    return estimate_position_id_projection(
        x_0,
        x_1,
        num_heads,
        head_dim,
        base=base,
    )


def apply_rope(
    x: Tensor,
    pos: Tensor,
    num_heads: int,
    head_dim: int,
    base: float = 10000.0,
) -> Tensor:
    """
    Apply RoPE to the input tensor x using the given position pos.
    """
    inv_freq = _rope_inv_freq(base, head_dim)
    theta = pos.unsqueeze(-1) * inv_freq

    return apply_rope_theta(
        x,
        theta,
        num_heads,
        head_dim,
    )
