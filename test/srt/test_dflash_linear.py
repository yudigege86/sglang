"""CPU/GPU checks for DFlashLinearDraftModel vs SpecForge serving math."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

TINY = {
    "architectures": ["DFlashLinearDraftModel"],
    "model_type": "qwen3",
    "block_size": 4,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "num_target_layers": 4,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "vocab_size": 256,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "rope_theta": 10000.0,
    "layer_types": ["full_attention"],
    "dflash_config": {
        "mask_token_id": 0,
        "target_layer_ids": [1],
        "linear_context": {
            "variant": "gdn",
            "injection": "gated_residual",
            "num_heads": 2,
            "key_dim": 16,
            "value_dim": 16,
            "normalize_qk": True,
            "backend": "naive",
        },
    },
}


def _config():
    return json.loads(json.dumps(TINY))


class _HFConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return dict(self.__dict__)


def _as_hf_config(payload: dict):
    cfg = _HFConfig(**payload)
    cfg.dflash_config = dict(payload["dflash_config"])
    return cfg


class DFlashLinearServingMathTest(unittest.TestCase):
    def test_masked_commit_is_identity_when_commit_len_is_zero(self):
        from sglang.srt.speculative.dflash_linear_state import commit_block_masked

        torch.manual_seed(0)
        state = torch.randn(2, 2, 4, 4)
        key = torch.randn(2, 4, 2, 4)
        value = torch.randn(2, 4, 2, 4)
        log_decay = torch.randn(2, 4, 2)
        beta = torch.rand(2, 4, 2)
        out = commit_block_masked(
            state,
            key,
            value,
            log_decay,
            beta,
            torch.zeros(2, dtype=torch.long),
        )
        torch.testing.assert_close(out, state)

    def test_sglang_forward_reads_bound_state(self):
        from sglang.srt.models.dflash_linear import DFlashLinearDraftModel

        cfg = _as_hf_config(_config())
        model = DFlashLinearDraftModel(cfg)
        model.eval()
        rows, layers, heads, key_dim, value_dim = 3, 1, 2, 16, 16
        ctx = torch.zeros(rows, layers, heads, key_dim, value_dim)
        torch.manual_seed(1)
        ctx[1] = torch.randn_like(ctx[1])
        model.bind_context_state(ctx)
        batch = 1
        block = 4
        hidden = torch.randn(batch * block, 64)
        positions = torch.arange(8, 12)
        req = torch.tensor([1], dtype=torch.long)
        forward_batch = SimpleNamespace(req_pool_indices=req)
        with torch.no_grad():
            out = model(
                input_ids=torch.zeros(batch * block, dtype=torch.long),
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=hidden,
            )
        self.assertEqual(tuple(out.hidden_states.shape), (batch * block, 64))
        self.assertFalse(torch.isnan(out.hidden_states).any())

    @unittest.skipUnless(
        os.environ.get("SGLANG_DFLASH_LINEAR_COMPARE_SPECFORGE", "0") == "1",
        "set SGLANG_DFLASH_LINEAR_COMPARE_SPECFORGE=1 with SpecForge on PYTHONPATH",
    )
    def test_sglang_and_specforge_hidden_states_match(self):
        specforge_root = os.environ.get("SPECFORGE_ROOT")
        if specforge_root:
            sys.path.insert(0, specforge_root)
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from sglang.srt.models.dflash_linear import DFlashLinearDraftModel

        payload = _config()
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        try:
            sf_cfg = AutoDraftModelConfig.from_file(path)
            sf_model = AutoDraftModel.from_config(sf_cfg)
        finally:
            os.unlink(path)
        sg_model = DFlashLinearDraftModel(_as_hf_config(payload))
        sf_state = sf_model.state_dict()
        missing, unexpected = sg_model.load_state_dict(sf_state, strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(
            all("rotary" in name or "inv_freq" in name for name in missing),
            msg=f"unexpected missing keys {missing}",
        )
        torch.manual_seed(2)
        batch, block, start = 1, 4, 9
        noise = torch.randn(batch, block, 64)
        target_hidden = torch.randn(batch, start, 64)
        positions = torch.arange(start, start + block).unsqueeze(0)
        with torch.no_grad():
            expected = sf_model._teacher_force_block(
                target_hidden=target_hidden,
                noise_embedding=noise,
                position_ids=positions,
                start=start,
            )
            fused = sg_model.project_target_hidden(target_hidden.reshape(-1, 64))
            fused = fused.view(batch, start, -1)
            layer = sg_model.layers[0]
            key, value, log_decay, beta = layer.context_scan.project(fused)
            from sglang.srt.speculative.dflash_linear_state import gated_delta_scan

            state = gated_delta_scan(
                key, value, log_decay, beta, normalize_qk=layer.context_scan.normalize_qk
            )
            ctx = torch.zeros(2, 1, state.shape[1], state.shape[2], state.shape[3])
            ctx[1, 0] = state[0]
            sg_model.bind_context_state(ctx)
            forward_batch = SimpleNamespace(req_pool_indices=torch.tensor([1]))
            got = sg_model(
                input_ids=torch.zeros(batch * block, dtype=torch.long),
                positions=positions.reshape(-1),
                forward_batch=forward_batch,
                input_embeds=noise.reshape(batch * block, -1),
            ).hidden_states.view(batch, block, -1)
        torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
