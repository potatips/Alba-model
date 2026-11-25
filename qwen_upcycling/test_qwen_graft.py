import copy
import sys
import os
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_geo_muon import create_qwen_adamw_optimizer
from qwen_residual_graft import (
    apply_residual_graft_to_qwen,
    collect_graft_gate_stats,
    gor_v2_transform_u,
)


class _DummyNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x * rms * self.weight


class _DummyAttn(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache: bool = False,
        position_embeddings=None,
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (hidden_states.size(-1) ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        out = self.o_proj(torch.matmul(probs, v))
        return out


class _DummyMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2, bias=False)
        self.up_proj = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class _DummyDecoderLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.input_layernorm = _DummyNorm(d)
        self.post_attention_layernorm = _DummyNorm(d)
        self.self_attn = _DummyAttn(d)
        self.mlp = _DummyMLP(d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache: bool = False,
        position_embeddings=None,
    ) -> torch.Tensor:
        res = hidden_states
        h = res + self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class _DummyQwenBackbone(nn.Module):
    def __init__(self, n_layers: int, d: int):
        super().__init__()
        self.layers = nn.ModuleList([_DummyDecoderLayer(d) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, use_cache=False)
        return h


class _DummyQwenCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, n_layers: int = 6, d: int = 24):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.model = _DummyQwenBackbone(n_layers, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

        class _Cfg:
            hidden_size = d

        self.config = _Cfg()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = self.model(h)
        return self.lm_head(h)


class TestQwenResidualGraft(unittest.TestCase):
    def test_gor_v2_matches_projected_decomposition(self):
        torch.manual_seed(0)
        x = torch.randn(2, 8, 16)
        u = torch.randn(2, 8, 16)
        u_eff = gor_v2_transform_u(x, u, k=3.0)
        dot = (x * u).sum(dim=-1, keepdim=True)
        nx = (x * x).sum(dim=-1, keepdim=True) + 1e-7
        nu = (u * u).sum(dim=-1, keepdim=True) + 1e-7
        rho = dot / torch.sqrt(nx * nu)
        phi = torch.clamp(2.0 * torch.sigmoid(3.0 * rho), max=1.0)
        u_par = (dot / nx) * x
        u_perp = u - u_par
        ref = u_perp + phi * u_par
        self.assertLess((u_eff - ref).abs().max().item(), 1e-5)

    def test_step0_exact_identity_at_zero_gates(self):
        torch.manual_seed(42)
        base = _DummyQwenCausalLM(vocab=32, n_layers=6, d=24)
        bar = copy.deepcopy(base)
        bar_gor = copy.deepcopy(base)

        apply_residual_graft_to_qwen(bar, enable_gor_v2=False, num_blocks=3, use_sublayer_checkpointing=False)
        apply_residual_graft_to_qwen(bar_gor, enable_gor_v2=True, num_blocks=3, use_sublayer_checkpointing=False)

        base.eval()
        bar.eval()
        bar_gor.eval()

        ids = torch.randint(0, 32, (2, 10))
        with torch.no_grad():
            y_base = base(ids)
            y_bar = bar(ids)
            y_gor = bar_gor(ids)

        self.assertLess((y_base - y_bar).abs().max().item(), 1e-5)
        self.assertLess((y_base - y_gor).abs().max().item(), 1e-5)
        stats = collect_graft_gate_stats(bar_gor)
        self.assertEqual(stats["mean_abs_alpha_attn"], 0.0)
        self.assertEqual(stats["mean_abs_beta_gor"], 0.0)

    def test_two_step_training_wakes_all_graft_params_with_sublayer_ckpt(self):
        torch.manual_seed(7)
        model = _DummyQwenCausalLM(vocab=32, n_layers=6, d=24)
        apply_residual_graft_to_qwen(
            model,
            enable_gor_v2=True,
            num_blocks=3,
            use_sublayer_checkpointing=True,
        )
        opt = create_qwen_adamw_optimizer(model, lr=1e-4, graft_lr=1e-2)
        self.assertEqual(len(opt.param_groups), 3)
        self.assertAlmostEqual(opt.param_groups[2]["lr"], 1e-2)

        model.train()
        ids = torch.randint(0, 32, (2, 12))
        targets = torch.randint(0, 32, (2, 12))

        for step in range(2):
            opt.zero_grad()
            logits = model(ids)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
            loss.backward()
            if step == 0:
                self.assertGreater(model.residual_graft.alpha_attn.grad.abs().sum().item(), 0.0)
                self.assertGreater(model.residual_graft.beta_gor.grad.abs().sum().item(), 0.0)
            else:
                self.assertGreater(model.residual_graft.depth_queries.grad.abs().sum().item(), 0.0)
            opt.step()

        stats = collect_graft_gate_stats(model)
        self.assertGreater(stats["mean_abs_alpha_attn"], 0.0)
        self.assertGreater(stats["mean_abs_beta_gor"], 0.0)


if __name__ == "__main__":
    unittest.main()
