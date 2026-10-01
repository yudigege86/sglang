"""Linear-context DFlash draft model for SGLang.

Same DFLASH algorithm as stock ``DFlashDraftModel``, but context is a
fixed-size GDN state per request per layer plus dense B-token attention.
Weight names match SpecForge ``DFlashLinearDraftModel`` so HF exports load
without remapping fused QKV.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.speculative.dflash_linear_state import (
    LinearContextScan,
    commit_block_masked,
    fla_varlen_final_states,
    gated_delta_scan,
)
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)

LINEAR_CONTEXT_INJECTIONS = ("gated_residual", "independent", "qkv_conditioning")


def _rms(norm: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    orig = tensor.shape
    flat = tensor.reshape(-1, orig[-1])
    out = norm(flat)
    if isinstance(out, tuple):
        out = out[0]
    return out.reshape(orig)


def resolve_linear_context_settings(config) -> dict:
    method = dict(getattr(config, "dflash_config", None) or {})
    if not isinstance(method, dict):
        method = {}
    settings = dict(method.get("linear_context") or {})
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    settings.setdefault("variant", "gdn")
    settings.setdefault("injection", "gated_residual")
    settings.setdefault("context_residual", True)
    settings.setdefault("num_heads", min(8, kv_heads))
    settings.setdefault("key_dim", min(64, head_dim))
    settings.setdefault("value_dim", settings["key_dim"])
    settings.setdefault("normalize_qk", True)
    if settings["variant"] != "gdn":
        raise NotImplementedError(
            "SGLang linear-context serving implements GDN only, "
            f"got variant={settings['variant']!r}"
        )
    if settings["injection"] not in LINEAR_CONTEXT_INJECTIONS:
        raise ValueError(
            "linear_context.injection must be one of "
            f"{LINEAR_CONTEXT_INJECTIONS}, got {settings['injection']!r}"
        )
    settings["context_residual"] = bool(settings["context_residual"])
    return settings


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class BlockLocalAttention(nn.Module):
    """Dense bidirectional attention over one DFlash block. No context KV."""

    def __init__(self, config, *, retrieved_bias: bool = False):
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.head_dim = int(
            getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        )
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)
        if retrieved_bias:
            self.q_r_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_r_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.v_r_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        else:
            self.q_r_proj = None
            self.k_r_proj = None
            self.v_r_proj = None
        rms_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = RMSNorm(self.head_dim, eps=rms_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_eps)
        rope_theta, rope_scaling = get_rope_config(config)
        max_position = int(getattr(config, "max_position_embeddings", 32768))
        rope_is_neox_style = bool(
            getattr(config, "rope_is_neox_style", getattr(config, "is_neox_style", True))
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        retrieved: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        if retrieved is not None:
            if self.q_r_proj is None:
                raise ValueError("retrieved features require qkv_conditioning")
            query = query + self.q_r_proj(retrieved)
            key = key + self.k_r_proj(retrieved)
            value = value + self.v_r_proj(retrieved)
        flat_q = query.reshape(batch * seq_len, self.num_heads * self.head_dim)
        flat_k = key.reshape(batch * seq_len, self.num_kv_heads * self.head_dim)
        pos = positions.reshape(-1)
        q_by_head = flat_q.view(batch * seq_len, self.num_heads, self.head_dim)
        k_by_head = flat_k.view(batch * seq_len, self.num_kv_heads, self.head_dim)
        q_by_head = _rms(self.q_norm, q_by_head.reshape(-1, self.head_dim)).view_as(q_by_head)
        k_by_head = _rms(self.k_norm, k_by_head.reshape(-1, self.head_dim)).view_as(k_by_head)
        flat_q = q_by_head.reshape(batch * seq_len, -1)
        flat_k = k_by_head.reshape(batch * seq_len, -1)
        flat_q, flat_k = self.rotary_emb(pos, flat_q, flat_k)
        query = flat_q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = flat_k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        attn = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scaling
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn)


class BlockMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate = int(config.intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashLinearDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        del layer_id
        hidden_size = int(config.hidden_size)
        rms_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.settings = resolve_linear_context_settings(config)
        self.block_size = int(getattr(config, "block_size", 16) or 16)
        draft_cfg = parse_dflash_draft_config(draft_hf_config=config)
        if draft_cfg.block_size is not None:
            self.block_size = int(draft_cfg.block_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_eps)
        self.context_scan = LinearContextScan(
            hidden_size=hidden_size,
            num_heads=int(self.settings["num_heads"]),
            key_dim=int(self.settings["key_dim"]),
            value_dim=int(self.settings["value_dim"]),
            variant=self.settings["variant"],
            normalize_qk=bool(self.settings["normalize_qk"]),
        )
        recurrent_out = int(self.settings["num_heads"]) * int(self.settings["value_dim"])
        self.context_query = nn.Linear(
            hidden_size, int(self.settings["num_heads"]) * int(self.settings["key_dim"]), bias=False
        )
        self.horizon_embed = nn.Embedding(self.block_size, hidden_size)
        self.context_read_proj = nn.Linear(recurrent_out, hidden_size, bias=False)
        if self.settings["injection"] == "gated_residual":
            self.inject_gate = nn.Linear(2 * hidden_size, hidden_size, bias=True)
            self.inject_value = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.inject_gate = None
            self.inject_value = None
        if self.settings["context_residual"]:
            self.context_residual = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.context_residual = None
        self.self_attn = BlockLocalAttention(
            config, retrieved_bias=self.settings["injection"] == "qkv_conditioning"
        )
        self.mlp = BlockMLP(config)

    def _context_read(self, normalized: torch.Tensor, prefix_state: torch.Tensor) -> torch.Tensor:
        batch, num_anchors, block, _ = normalized.shape
        num_heads = int(self.settings["num_heads"])
        key_dim = int(self.settings["key_dim"])
        offsets = torch.arange(block, device=normalized.device)
        query_in = normalized + self.horizon_embed(offsets)
        query = self.context_query(query_in).view(batch, num_anchors, block, num_heads, key_dim)
        if self.settings["normalize_qk"]:
            query = query * torch.rsqrt(query.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        # Serving state is FP32; the draft forward is BF16.
        prefix_state = prefix_state.to(dtype=query.dtype)
        retrieved = torch.einsum("baqhk,bahkv->baqhv", query, prefix_state)
        retrieved = retrieved.reshape(batch, num_anchors, block, -1)
        return self.context_read_proj(retrieved)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        prefix_state: torch.Tensor,
    ) -> torch.Tensor:
        batch, packed, hidden_size = hidden_states.shape
        block = packed
        num_anchors = 1
        blocks = hidden_states.view(batch, num_anchors, block, hidden_size)
        residual = blocks
        normalized = _rms(self.input_layernorm, blocks)
        if prefix_state.ndim == 4:
            prefix_state = prefix_state.unsqueeze(1)
        retrieved = self._context_read(normalized, prefix_state)
        injection = self.settings["injection"]
        attn_retrieved = None
        if injection == "gated_residual":
            gate = torch.sigmoid(self.inject_gate(torch.cat([normalized, retrieved], dim=-1)))
            conditioned = normalized + gate * self.inject_value(retrieved)
        elif injection == "independent":
            conditioned = normalized
        else:
            conditioned = normalized
            attn_retrieved = retrieved
        flat = conditioned.reshape(batch * num_anchors, block, hidden_size)
        retrieved_flat = (
            None if attn_retrieved is None else attn_retrieved.reshape(batch * num_anchors, block, hidden_size)
        )
        pos = positions.view(batch * num_anchors, block)
        attn = self.self_attn(flat, pos, retrieved=retrieved_flat).view(
            batch, num_anchors, block, hidden_size
        )
        hidden = residual + attn
        if self.context_residual is not None:
            hidden = hidden + self.context_residual(retrieved)
        hidden = hidden + self.mlp(_rms(self.post_attention_layernorm, hidden))
        return hidden.view(batch, packed, hidden_size)


class DFlashLinearDraftModel(nn.Module):
    """SGLang draft model: GDN prefix state + dense B-token attention."""

    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        del quant_config, prefix
        self.config = config
        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.settings = resolve_linear_context_settings(config)
        self.layers = nn.ModuleList(
            [DFlashLinearDecoderLayer(config, layer_id=i) for i in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_eps)
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        if draft_config.num_target_layers is not None:
            target_num_layers = int(draft_config.num_target_layers)
        elif draft_config.target_layer_ids is not None:
            target_num_layers = max(draft_config.target_layer_ids) + 1
        else:
            target_num_layers = num_layers
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        self.num_context_features = len(target_layer_ids)
        self.fc = nn.Linear(self.num_context_features * hidden_size, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_eps)
        self.block_size = int(draft_config.resolve_block_size(default=16) or 16)
        self.ctx_state: Optional[torch.Tensor] = None
        # Stock DFlashWorkerV2 reads this; linear drafts have no DFlash2 selector.
        self.candidate_selector = None

    def set_block_size(self, block_size: int) -> None:
        """Adopt the block size the worker resolved (v0.5.19 DFlashWorkerV2)."""
        self.block_size = int(block_size)
        for layer in self.layers:
            layer.block_size = self.block_size

    def bind_context_state(self, ctx_state: torch.Tensor) -> None:
        self.ctx_state = ctx_state

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH-linear target_hidden feature dim mismatch. "
                f"Expected [N, {expected}], got {tuple(target_hidden.shape)}."
            )
        return _rms(self.hidden_norm, self.fc(target_hidden))

    def _state_rows(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        if self.ctx_state is None:
            raise RuntimeError("DFlashLinearDraftModel.ctx_state is not bound")
        return self.ctx_state[req_pool_indices.to(dtype=torch.long)]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del input_ids, get_embedding, pp_proxy_tensors
        if input_embeds is None:
            raise ValueError("DFlashLinearDraftModel requires input_embeds from the target")
        if input_embeds.numel() == 0:
            return LogitsProcessorOutput(next_token_logits=None, hidden_states=input_embeds)
        req_indices = forward_batch.req_pool_indices
        bs = int(req_indices.shape[0])
        hidden = input_embeds
        if hidden.ndim != 2:
            raise ValueError(f"input_embeds must be 2D, got {tuple(hidden.shape)}")
        if hidden.shape[0] % bs != 0:
            raise ValueError(
                f"input_embeds rows {hidden.shape[0]} not divisible by batch {bs}"
            )
        block = hidden.shape[0] // bs
        hidden = hidden.view(bs, block, -1)
        positions = positions.view(bs, block)
        prefix = self._state_rows(req_indices)
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(hidden, positions, prefix[:, layer_idx])
        hidden = _rms(self.norm, hidden)
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden.reshape(bs * block, -1),
        )

    def advance_state_varlen(
        self,
        target_hidden: torch.Tensor,
        req_indices: torch.Tensor,
        extend_lens: list[int],
    ) -> None:
        """Scan newly prefilled tokens into ``ctx_state[req_indices]``."""

        if target_hidden is None or target_hidden.numel() == 0:
            return
        fused = self.project_target_hidden(target_hidden)
        rows = req_indices.to(dtype=torch.long)
        for layer_idx, layer in enumerate(self.layers):
            scan = layer.context_scan
            packed_key = []
            packed_value = []
            packed_decay = []
            packed_beta = []
            lengths = []
            token_offset = 0
            for length in extend_lens:
                chunk = fused[token_offset : token_offset + int(length)]
                token_offset += int(length)
                if int(length) <= 0:
                    continue
                key, value, log_decay, beta = scan.project(chunk.unsqueeze(0))
                packed_key.append(key[0])
                packed_value.append(value[0])
                packed_decay.append(log_decay[0])
                packed_beta.append(beta[0])
                lengths.append(int(length))
            if not packed_key:
                continue
            key = torch.cat(packed_key, dim=0)
            value = torch.cat(packed_value, dim=0)
            log_decay = torch.cat(packed_decay, dim=0)
            beta = torch.cat(packed_beta, dim=0)
            cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=key.device)
            cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=key.device).cumsum(0)
            active = [i for i, length in enumerate(extend_lens) if int(length) > 0]
            active_rows = rows[active]
            initial = self.ctx_state[active_rows, layer_idx].to(dtype=key.dtype)
            finals = fla_varlen_final_states(
                key,
                value,
                log_decay,
                beta,
                cu,
                initial,
                normalize_qk=scan.normalize_qk,
            )
            if finals is None:
                token_offset = 0
                for local_i, length in enumerate(lengths):
                    chunk_key = key[token_offset : token_offset + length].unsqueeze(0)
                    chunk_value = value[token_offset : token_offset + length].unsqueeze(0)
                    chunk_decay = log_decay[token_offset : token_offset + length].unsqueeze(0)
                    chunk_beta = beta[token_offset : token_offset + length].unsqueeze(0)
                    updated = gated_delta_scan(
                        chunk_key,
                        chunk_value,
                        chunk_decay,
                        chunk_beta,
                        initial_state=initial[local_i : local_i + 1],
                        normalize_qk=scan.normalize_qk,
                    )
                    self.ctx_state[active_rows[local_i], layer_idx] = updated[0].to(
                        dtype=self.ctx_state.dtype
                    )
                    token_offset += length
            else:
                self.ctx_state[active_rows, layer_idx] = finals.to(dtype=self.ctx_state.dtype)

    def advance_state_block(
        self,
        target_hidden: torch.Tensor,
        req_indices: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Masked B-token GDN update from verify aux hidden states."""

        if target_hidden is None or target_hidden.numel() == 0:
            return
        if target_hidden.ndim == 2:
            rows = int(req_indices.shape[0])
            block = target_hidden.shape[0] // rows
            target_hidden = target_hidden.view(rows, block, -1)
        bs, block, width = target_hidden.shape
        fused = self.project_target_hidden(target_hidden.reshape(bs * block, width))
        fused = fused.view(bs, block, -1)
        rows = req_indices.to(dtype=torch.long)
        for layer_idx, layer in enumerate(self.layers):
            scan = layer.context_scan
            key, value, log_decay, beta = scan.project(fused)
            state = self.ctx_state[rows, layer_idx].to(dtype=key.dtype)
            updated = commit_block_masked(
                state,
                key,
                value,
                log_decay,
                beta,
                commit_lens,
                normalize_qk=scan.normalize_qk,
            )
            self.ctx_state[rows, layer_idx] = updated.to(dtype=self.ctx_state.dtype)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped = name[len("model.") :]
                if stripped in params_dict:
                    return stripped
            else:
                prefixed = f"model.{name}"
                if prefixed in params_dict:
                    return prefixed
            return None

        for name, loaded_weight in weights:
            resolved = resolve_param_name(name)
            if resolved is None:
                continue
            param = params_dict[resolved]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = [DFlashLinearDraftModel]
