from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .residuals import (
    BlockAttnResConnector,
    GORAntiDiluteResidual,
    GORBoundedResidual,
    GORv2Residual,
    GORv3SubspaceConnector,
    PreNormResidual,
    RMSNorm,
    StdDualBankConnector,
    SwiGLU,
)


@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    dim: int = 384
    n_layers: int = 16
    n_heads: int = 6
    max_seq_len: int = 256
    hidden_dim: Optional[int] = None
    residual_type: str = "pre_norm"
    gor_k: float = 3.0
    attn_num_blocks: int = 8


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    qf = q.float()
    kf = k.float()
    cos_b = cos[None, : q.size(1), None, :]
    sin_b = sin[None, : k.size(1), None, :]

    q_even, q_odd = qf[..., 0::2], qf[..., 1::2]
    k_even, k_odd = kf[..., 0::2], kf[..., 1::2]

    q_rot_even = q_even * cos_b - q_odd * sin_b
    q_rot_odd = q_even * sin_b + q_odd * cos_b
    k_rot_even = k_even * cos_b - k_odd * sin_b
    k_rot_odd = k_even * sin_b + k_odd * cos_b

    q_out = torch.stack((q_rot_even, q_rot_odd), dim=-1).flatten(-2)
    k_out = torch.stack((k_rot_even, k_rot_odd), dim=-1).flatten(-2)
    return q_out.to(orig_dtype), k_out.to(orig_dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q, k, v = self.wqkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim)
        q, k = apply_rope(q, k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim))


class TransformerSublayerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = CausalSelfAttention(config.dim, config.n_heads)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = SwiGLU(config.dim, config.hidden_dim)


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.detach().float()
    yf = y.detach().float()
    xf = xf - xf.mean(dim=0, keepdim=True)
    yf = yf - yf.mean(dim=0, keepdim=True)
    k_xy = torch.linalg.matrix_norm(xf.t() @ yf, ord="fro").pow(2)
    k_xx = torch.linalg.matrix_norm(xf.t() @ xf, ord="fro")
    k_yy = torch.linalg.matrix_norm(yf.t() @ yf, ord="fro")
    denom = k_xx * k_yy
    if denom <= 1e-12:
        return 1.0
    return float(torch.clamp(k_xy / denom, 0.0, 1.0).item())


class FromScratchTransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.residual_type = config.residual_type
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerSublayerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        num_sublayers = 2 * config.n_layers
        if config.residual_type == "pre_norm":
            self.res_op = PreNormResidual()
            self.connector = None
        elif config.residual_type == "gor_v2":
            self.res_op = GORv2Residual(k=config.gor_k)
            self.connector = None
        elif config.residual_type == "gor_bounded":
            self.res_op = GORBoundedResidual(k=config.gor_k, max_phi=1.0)
            self.connector = None
        elif config.residual_type == "gor_antidilute":
            self.res_op = GORAntiDiluteResidual(k=config.gor_k, max_phi=1.0, max_boost=2.0)
            self.connector = None
        elif config.residual_type == "std_dual_bank":
            self.res_op = None
            self.connector = StdDualBankConnector(config.dim, num_sublayers=num_sublayers)
        elif config.residual_type == "gor_v3_subspace":
            self.res_op = None
            self.connector = GORv3SubspaceConnector(config.dim, num_sublayers=num_sublayers, k=config.gor_k)
        elif config.residual_type == "attn_res_block":
            self.res_op = None
            self.connector = BlockAttnResConnector(
                config.dim, num_sublayers=num_sublayers, num_blocks=config.attn_num_blocks
            )
        else:
            raise ValueError(f"Unknown residual_type: {config.residual_type}")

        cos, sin = precompute_rope_freqs(config.dim // config.n_heads, config.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_backbone_weights)

    def _init_backbone_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_residual_overhead_params(self) -> int:
        if self.connector is None:
            return 0
        return sum(p.numel() for p in self.connector.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        collect_geometry: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, object]]]:
        _, seq_len = input_ids.shape
        x = self.tok_embeddings(input_ids)
        cos = self.rope_cos[:seq_len].to(x.device)
        sin = self.rope_sin[:seq_len].to(x.device)

        geom: Optional[Dict[str, object]] = None
        if collect_geometry:
            geom = {
                "conflict_rates": [],
                "norm_growth": [],
                "dilution_ratios": [],
                "depth_renewal": [],
            }
            x0_norm_val = float(torch.linalg.vector_norm(x.detach().float(), dim=-1).mean().item())
            prev_h = x.detach().float().reshape(-1, self.config.dim)

        if self.connector is None:
            h = x
            x0_norm = torch.sqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) + 1e-7)
            for layer in self.layers:
                u_attn = layer.attn(layer.attn_norm(h), cos, sin)
                h_next = self.res_op(h, u_attn, x0_norm=x0_norm)
                if collect_geometry:
                    self._record_sublayer_geom(geom, h, u_attn, h_next, x0_norm_val, prev_h)
                    prev_h = h_next.detach().float().reshape(-1, self.config.dim)
                h = h_next

                u_ffn = layer.ffn(layer.ffn_norm(h))
                h_next = self.res_op(h, u_ffn, x0_norm=x0_norm)
                if collect_geometry:
                    self._record_sublayer_geom(geom, h, u_ffn, h_next, x0_norm_val, prev_h)
                    prev_h = h_next.detach().float().reshape(-1, self.config.dim)
                h = h_next
            out_h = h
        else:
            state = self.connector.init_state(x)
            for layer_idx, layer in enumerate(self.layers):
                sub_attn = 2 * layer_idx
                h_attn = self.connector.read(state, sub_attn)
                u_attn = layer.attn(layer.attn_norm(h_attn), cos, sin)
                self.connector.update(state, u_attn, sub_attn)
                if collect_geometry:
                    cur = state.s_fast if state.s_fast is not None else (state.partial if state.partial is not None else state.completed[-1])
                    self._record_sublayer_geom(geom, h_attn, u_attn, cur, x0_norm_val, prev_h)
                    prev_h = cur.detach().float().reshape(-1, self.config.dim)

                sub_ffn = 2 * layer_idx + 1
                h_ffn = self.connector.read(state, sub_ffn)
                u_ffn = layer.ffn(layer.ffn_norm(h_ffn))
                self.connector.update(state, u_ffn, sub_ffn)
                if collect_geometry:
                    cur = state.s_fast if state.s_fast is not None else (state.partial if state.partial is not None else state.completed[-1])
                    self._record_sublayer_geom(geom, h_ffn, u_ffn, cur, x0_norm_val, prev_h)
                    prev_h = cur.detach().float().reshape(-1, self.config.dim)
            out_h = self.connector.finalize(state)

        logits = self.lm_head(self.final_norm(out_h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, geom

    def _record_sublayer_geom(
        self,
        geom: Dict[str, object],
        h: torch.Tensor,
        u: torch.Tensor,
        h_next: torch.Tensor,
        x0_norm_val: float,
        prev_h_flat: torch.Tensor,
    ) -> None:
        hf = h.detach().float()
        uf = u.detach().float()
        dot = (hf * uf).sum(dim=-1)
        nh = torch.linalg.vector_norm(hf, dim=-1).clamp_min(1e-7)
        nu = torch.linalg.vector_norm(uf, dim=-1).clamp_min(1e-7)
        rho = dot / (nh * nu)
        geom["conflict_rates"].append(float((rho < 0.0).float().mean().item() * 100.0))
        geom["norm_growth"].append(float(nh.mean().item() / max(x0_norm_val, 1e-7)))
        geom["dilution_ratios"].append(float((nu / nh).mean().item()))
        cur_flat = h_next.detach().float().reshape(-1, self.config.dim)
        n_rows = min(512, cur_flat.size(0))
        geom["depth_renewal"].append(1.0 - _linear_cka(prev_h_flat[:n_rows], cur_flat[:n_rows]))


def clone_transformer_backbone(source: FromScratchTransformerLM, target: FromScratchTransformerLM) -> None:
    with torch.no_grad():
        target.tok_embeddings.weight.copy_(source.tok_embeddings.weight)
        target.layers.load_state_dict(source.layers.state_dict())
        target.final_norm.load_state_dict(source.final_norm.state_dict())
