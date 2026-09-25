"""DFLASH worker branch for DFlashLinearDraftModel.

Reuses stock DFlashWorkerV2 verify/accept/draft-block construction.
Replaces the concat-KV appends with per-request GDN state updates.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.speculative.dflash_linear_state import gated_delta_scan
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.models.dflash_linear import resolve_linear_context_settings

logger = logging.getLogger(__name__)


class DFlashLinearWorkerV2(DFlashWorkerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_compact_draft_cache:
            raise ValueError(
                "DFlashLinear does not support --speculative-dflash-draft-window-size"
            )
        if not bool(getattr(self.server_args, "disable_radix_cache", False)):
            raise ValueError(
                "DFlashLinear requires --disable-radix-cache "
                "(prefix-cache cloning of GDN state is not implemented)"
            )
        self.ctx_state: Optional[torch.Tensor] = None
        self.owner_gen: Optional[torch.Tensor] = None
        self._shadow_aux: dict[int, list[torch.Tensor]] = {}
        self._shadow_check = os.environ.get("SGLANG_DFLASH_LINEAR_SHADOW_CHECK", "0") == "1"

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        super().alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        pool = self.draft_model_runner.req_to_token_pool
        n_rows = int(pool.req_to_token.shape[0])
        settings = resolve_linear_context_settings(
            self.draft_model_runner.model_config.hf_config
        )
        n_layers = int(self.draft_model.config.num_hidden_layers)
        heads = int(settings["num_heads"])
        key_dim = int(settings["key_dim"])
        value_dim = int(settings["value_dim"])
        self.ctx_state = torch.zeros(
            n_rows,
            n_layers,
            heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.owner_gen = torch.full((n_rows,), -1, dtype=torch.int64)
        self.draft_model.bind_context_state(self.ctx_state)
        logger.info(
            "DFlashLinear state pool rows=%s layers=%s heads=%s key_dim=%s "
            "value_dim=%s bytes=%s",
            n_rows,
            n_layers,
            heads,
            key_dim,
            value_dim,
            self.ctx_state.numel() * self.ctx_state.element_size(),
        )

    def init_cuda_graphs(self):
        logger.info("DFlashLinear skips draft CUDA-graph capture (M1 eager path)")

    def clear_cache_pool(self):
        if self.ctx_state is not None:
            self.ctx_state.zero_()
        if self.owner_gen is not None:
            self.owner_gen.fill_(-1)
        self._shadow_aux.clear()

    def _ensure_owners(self, batch) -> None:
        if self.ctx_state is None or self.owner_gen is None:
            raise RuntimeError("DFlashLinear state pool is not allocated")
        pool = self.draft_model_runner.req_to_token_pool
        idxs = [int(x) for x in batch.req_pool_indices.detach().to("cpu").tolist()]
        prefix_lens = list(batch.prefix_lens)
        gens = getattr(pool, "req_generation", None)
        for i, idx in enumerate(idxs):
            if idx <= 0:
                continue
            gen = int(gens[idx]) if gens is not None else (int(self.owner_gen[idx]) + 1)
            if int(self.owner_gen[idx]) == gen:
                continue
            prefix = int(prefix_lens[i])
            if prefix > 0:
                raise RuntimeError(
                    "DFlashLinear cannot clone GDN state from a radix prefix hit "
                    f"(req_pool_idx={idx}, prefix_len={prefix}). "
                    "Restart with --disable-radix-cache."
                )
            self.ctx_state[idx].zero_()
            self.owner_gen[idx] = gen
            self._shadow_aux.pop(idx, None)

    def _commit_prefill_context(self, batch, logits_output, positions) -> None:
        del positions
        self._ensure_owners(batch)
        hidden = logits_output.hidden_states
        extend_lens = [int(x) for x in batch.extend_lens]
        self.draft_model.advance_state_varlen(
            hidden, batch.req_pool_indices, extend_lens
        )
        if self._shadow_check:
            self._shadow_append_prefill(batch, hidden, extend_lens)

    def _commit_verify_context(
        self,
        batch,
        hidden,
        verify_out_cache_loc,
        verify_out_cache_loc_2d,
        positions,
        commit_lens,
    ) -> None:
        del verify_out_cache_loc, verify_out_cache_loc_2d, positions
        self.draft_model.advance_state_block(
            hidden, batch.req_pool_indices, commit_lens
        )
        if self._shadow_check:
            self._shadow_append_verify(batch, hidden, commit_lens)

    def _shadow_append_prefill(self, batch, hidden, extend_lens) -> None:
        idxs = [int(x) for x in batch.req_pool_indices.detach().to("cpu").tolist()]
        offset = 0
        for i, length in enumerate(extend_lens):
            idx = idxs[i]
            chunk = hidden[offset : offset + length].detach()
            offset += length
            if length <= 0 or idx <= 0:
                continue
            self._shadow_aux.setdefault(idx, []).append(chunk.cpu())
            self._shadow_check_row(idx)

    def _shadow_append_verify(self, batch, hidden, commit_lens) -> None:
        idxs = [int(x) for x in batch.req_pool_indices.detach().to("cpu").tolist()]
        commit_cpu = commit_lens.detach().to("cpu").tolist()
        for i, idx in enumerate(idxs):
            if idx <= 0:
                continue
            n = int(commit_cpu[i])
            if n <= 0:
                continue
            self._shadow_aux.setdefault(idx, []).append(hidden[i, :n].detach().cpu())
            self._shadow_check_row(idx)

    def _shadow_check_row(self, idx: int) -> None:
        pieces = self._shadow_aux.get(idx)
        if not pieces:
            return
        aux = torch.cat(pieces, dim=0).to(device=self.device)
        fused = self.draft_model.project_target_hidden(aux).unsqueeze(0)
        for layer_idx, layer in enumerate(self.draft_model.layers):
            scan = layer.context_scan
            key, value, log_decay, beta = scan.project(fused)
            expected = gated_delta_scan(
                key,
                value,
                log_decay,
                beta,
                normalize_qk=scan.normalize_qk,
            )[0]
            got = self.ctx_state[idx, layer_idx].to(dtype=expected.dtype)
            if not torch.allclose(got, expected, rtol=1e-2, atol=1e-2):
                max_abs = float((got - expected).abs().max().item())
                raise RuntimeError(
                    f"DFlashLinear shadow check failed req={idx} layer={layer_idx} "
                    f"max_abs={max_abs}"
                )
