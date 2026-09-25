"""GDN scan primitives for linear-context DFlash serving.

Mirrors SpecForge ``specforge.modeling.draft.linear_context`` (gated delta
step, identity-masked block commit). Training FLA gather is not used here:
serving keeps a per-request FP32 state and updates it incrementally.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)


def gated_delta_step(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    alpha = log_decay.exp()
    if alpha.ndim == 2:
        decayed = state * alpha[:, :, None, None]
    elif alpha.ndim == 3:
        decayed = state * alpha[:, :, :, None]
    else:
        raise ValueError(
            f"log_decay must be [B, H] or [B, H, K], got {tuple(log_decay.shape)}"
        )
    beta = beta[:, :, None, None]
    key_t_state = torch.einsum("bhk,bhkv->bhv", key, decayed)
    write = torch.einsum("bhk,bhv->bhkv", key, value)
    erase = torch.einsum("bhk,bhv->bhkv", key, key_t_state)
    return decayed + beta * (write - erase)


def gated_delta_scan(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    normalize_qk: bool = False,
) -> torch.Tensor:
    if normalize_qk:
        key = _l2norm(key)
    batch, seq_len, num_heads, key_dim = key.shape
    value_dim = value.shape[-1]
    if initial_state is None:
        state = key.new_zeros(batch, num_heads, key_dim, value_dim)
    else:
        state = initial_state.to(dtype=key.dtype)
    for time in range(seq_len):
        state = gated_delta_step(
            state,
            key[:, time],
            value[:, time],
            log_decay[:, time],
            beta[:, time],
        )
    return state


def apply_identity_mask(
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    commit_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = beta.shape[1]
    steps = torch.arange(seq_len, device=beta.device).view(1, seq_len)
    valid = steps < commit_lens.to(device=beta.device, dtype=steps.dtype).view(-1, 1)
    beta = beta * valid[:, :, None].to(dtype=beta.dtype)
    mask = valid
    while mask.ndim < log_decay.ndim:
        mask = mask.unsqueeze(-1)
    log_decay = log_decay * mask.to(dtype=log_decay.dtype)
    return log_decay, beta


def commit_block_masked(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    commit_lens: torch.Tensor,
    *,
    normalize_qk: bool = False,
) -> torch.Tensor:
    log_decay, beta = apply_identity_mask(log_decay, beta, commit_lens)
    return gated_delta_scan(
        key,
        value,
        log_decay,
        beta,
        initial_state=state,
        normalize_qk=normalize_qk,
    )


def _inverse_softplus(dt: torch.Tensor) -> torch.Tensor:
    return dt + torch.log(-torch.expm1(-dt))


def _init_A_log(num_channels: int) -> nn.Parameter:
    values = torch.empty(num_channels).uniform_(0.25, 16.0)
    parameter = nn.Parameter(values.log())
    parameter._no_weight_decay = True
    return parameter


def _init_dt_bias(num_channels: int) -> nn.Parameter:
    dt = torch.exp(
        torch.rand(num_channels) * (math.log(0.1) - math.log(1e-5)) + math.log(1e-5)
    ).clamp(min=1e-5)
    parameter = nn.Parameter(_inverse_softplus(dt))
    parameter._no_weight_decay = True
    return parameter


def _import_fla_chunk():
    try:
        from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

        return chunk_gated_delta_rule
    except Exception:
        pass
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return chunk_gated_delta_rule
    except Exception:
        return None


def fla_varlen_final_states(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    normalize_qk: bool,
) -> Optional[torch.Tensor]:
    """One FLA launch over packed prefixes. Returns None if the kernel is missing."""

    chunk_fn = _import_fla_chunk()
    if chunk_fn is None:
        return None
    dummy_q = torch.zeros_like(key)
    kwargs = {
        "q": dummy_q,
        "k": key,
        "v": value,
        "g": log_decay,
        "beta": beta,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "scale": 1.0,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": normalize_qk,
        "head_first": False,
    }
    try:
        output = chunk_fn(**kwargs)
    except TypeError:
        try:
            output = chunk_fn(
                dummy_q, key, value, log_decay, beta, **{
                    "scale": 1.0,
                    "output_final_state": True,
                    "use_qk_l2norm_in_kernel": normalize_qk,
                    "cu_seqlens": cu_seqlens,
                    "initial_state": initial_state,
                }
            )
        except Exception:
            return None
    except Exception:
        return None
    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
        return output[1].to(dtype=key.dtype)
    return None


class LinearContextScan(nn.Module):
    """Trainable GDN projections. Matches SpecForge LinearContextScan names."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        *,
        variant: str = "gdn",
        normalize_qk: bool = True,
    ) -> None:
        super().__init__()
        if variant != "gdn":
            raise NotImplementedError(
                f"SGLang linear-context serving implements GDN only, got {variant!r}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.variant = variant
        self.normalize_qk = normalize_qk
        self.k_proj = nn.Linear(hidden_size, num_heads * key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * value_dim, bias=False)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.A_log = _init_A_log(num_heads)
        self.dt_bias = _init_dt_bias(num_heads)

    def project(
        self, target_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = target_hidden.shape
        dtype = target_hidden.dtype
        key = self.k_proj(target_hidden).view(
            batch, seq_len, self.num_heads, self.key_dim
        )
        value = self.v_proj(target_hidden).view(
            batch, seq_len, self.num_heads, self.value_dim
        )
        beta = torch.sigmoid(self.beta_proj(target_hidden))
        gate = self.g_proj(target_hidden)
        scale = self.A_log.float().exp().view(1, 1, self.num_heads)
        shift = self.dt_bias.float().view(1, 1, self.num_heads)
        log_decay = (-scale * F.softplus(gate.float() + shift)).to(dtype=dtype)
        return key, value, log_decay, beta
