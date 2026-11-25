import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


ALL_LM_RESIDUALS = (
    "pre_norm",
    "gor_v2",
    "gor_bounded",
    "gor_antidilute",
    "std_dual_bank",
    "gor_v3_subspace",
    "attn_res_block",
)

RESIDUAL_LABELS = {
    "pre_norm": "Pre-Norm (x + u)",
    "gor_v2": "GOR-v2 (phi<=2, 0-Param)",
    "gor_bounded": "GOR-Bounded (phi<=1, 0-Param)",
    "gor_antidilute": "GOR-AntiDilute (0-Param)",
    "std_dual_bank": "Std Dual-Bank Control",
    "gor_v3_subspace": "GOR-v3 Subspace Contrast",
    "attn_res_block": "Block AttnRes (N=8)",
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        norm = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * norm).to(orig_dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = max(64, ((hidden_dim + 63) // 64) * 64)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _compute_gor_components(
    x: torch.Tensor,
    u: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xf = x.float()
    uf = u.float()
    dot = (xf * uf).sum(dim=-1, keepdim=True)
    norm_sq_x = (xf * xf).sum(dim=-1, keepdim=True)
    norm_sq_u = (uf * uf).sum(dim=-1, keepdim=True)
    norm_x = torch.sqrt(norm_sq_x + eps)
    norm_u = torch.sqrt(norm_sq_u + eps)
    rho = dot / (norm_x * norm_u + eps)
    u_parallel = (dot / (norm_sq_x + eps)) * xf
    u_perp = uf - u_parallel
    return xf, uf, u_parallel, u_perp, rho, norm_x


class PreNormResidual(nn.Module):
    def forward(self, x: torch.Tensor, u: torch.Tensor, x0_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x + u


class GORv2Residual(nn.Module):
    def __init__(self, k: float = 3.0, eps: float = 1e-7):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, x: torch.Tensor, u: torch.Tensor, x0_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        xf, _, u_par, u_perp, rho, _ = _compute_gor_components(x, u, self.eps)
        phi = 2.0 * torch.sigmoid(self.k * rho)
        return (xf + u_perp + phi * u_par).to(dtype=x.dtype)


class GORBoundedResidual(nn.Module):
    def __init__(self, k: float = 3.0, max_phi: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.k = k
        self.max_phi = max_phi
        self.eps = eps

    def forward(self, x: torch.Tensor, u: torch.Tensor, x0_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        xf, _, u_par, u_perp, rho, _ = _compute_gor_components(x, u, self.eps)
        phi = torch.clamp(2.0 * torch.sigmoid(self.k * rho), max=self.max_phi)
        return (xf + u_perp + phi * u_par).to(dtype=x.dtype)


class GORAntiDiluteResidual(nn.Module):
    def __init__(self, k: float = 3.0, max_phi: float = 1.0, max_boost: float = 1.25, eps: float = 1e-7):
        super().__init__()
        self.k = k
        self.max_phi = max_phi
        self.max_boost = max_boost
        self.eps = eps

    def forward(self, x: torch.Tensor, u: torch.Tensor, x0_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        xf, _, u_par, u_perp, rho, norm_x = _compute_gor_components(x, u, self.eps)
        phi = torch.clamp(2.0 * torch.sigmoid(self.k * rho), max=self.max_phi)
        if x0_norm is None:
            psi = 1.0
        else:
            ratio = torch.clamp(norm_x / (x0_norm.float() + self.eps), min=1.0)
            psi = torch.clamp(ratio.pow(0.25), min=1.0, max=self.max_boost)
        return (xf + psi * u_perp + phi * u_par).to(dtype=x.dtype)


@dataclass
class ResidualBankState:
    x0: torch.Tensor
    x0_unit: Optional[torch.Tensor] = None
    x0_norm: Optional[torch.Tensor] = None
    s_fast: Optional[torch.Tensor] = None
    s_slow: Optional[torch.Tensor] = None
    completed: List[torch.Tensor] = field(default_factory=list)
    completed_keys: List[torch.Tensor] = field(default_factory=list)
    partial: Optional[torch.Tensor] = None


class StdDualBankConnector(nn.Module):
    def __init__(self, dim: int, num_sublayers: int):
        super().__init__()
        self.inv_sqrt_l = 1.0 / math.sqrt(max(num_sublayers, 1))
        self.q_read = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(num_sublayers)])
        self.k_norm = RMSNorm(dim)

    def init_state(self, x0: torch.Tensor) -> ResidualBankState:
        return ResidualBankState(x0=x0, s_fast=x0, s_slow=torch.zeros_like(x0))

    def read(self, state: ResidualBankState, sublayer_idx: int) -> torch.Tensor:
        if sublayer_idx == 0:
            return state.x0
        q = self.q_read[sublayer_idx]
        kf = self.k_norm(state.s_fast)
        ks = self.k_norm(state.s_slow)
        k0 = self.k_norm(state.x0)
        logits = torch.stack(
            [
                torch.einsum("d,...d->...", q, kf),
                torch.einsum("d,...d->...", q, ks),
                torch.einsum("d,...d->...", q, k0),
            ],
            dim=0,
        )
        w = F.softmax(logits.float(), dim=0).to(state.s_fast.dtype)
        return w[0].unsqueeze(-1) * state.s_fast + w[1].unsqueeze(-1) * state.s_slow + w[2].unsqueeze(-1) * state.x0

    def update(self, state: ResidualBankState, u: torch.Tensor, sublayer_idx: int) -> None:
        state.s_fast = state.s_fast + u
        state.s_slow = state.s_slow + u * self.inv_sqrt_l

    def finalize(self, state: ResidualBankState) -> torch.Tensor:
        return state.s_fast + state.s_slow


class GORv3SubspaceConnector(nn.Module):
    def __init__(self, dim: int, num_sublayers: int, k: float = 3.0, eps: float = 1e-7):
        super().__init__()
        self.k = k
        self.eps = eps
        self.inv_sqrt_l = 1.0 / math.sqrt(max(num_sublayers, 1))
        self.q_read = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(num_sublayers)])
        self.k_norm = RMSNorm(dim)
        self.gamma_ortho = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(num_sublayers)])
        self.gamma_x0 = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(num_sublayers)])

    def init_state(self, x0: torch.Tensor) -> ResidualBankState:
        x0_unit = F.normalize(x0.float(), p=2, dim=-1, eps=1e-6)
        return ResidualBankState(x0=x0, x0_unit=x0_unit, s_fast=x0, s_slow=torch.zeros_like(x0))

    def read(self, state: ResidualBankState, sublayer_idx: int) -> torch.Tensor:
        if sublayer_idx == 0:
            return state.x0
        q = self.q_read[sublayer_idx]
        kf = self.k_norm(state.s_fast)
        ks = self.k_norm(state.s_slow)
        k0 = self.k_norm(state.x0)
        logits = torch.stack(
            [
                torch.einsum("d,...d->...", q, kf),
                torch.einsum("d,...d->...", q, ks),
                torch.einsum("d,...d->...", q, k0),
            ],
            dim=0,
        )
        w = F.softmax(logits.float(), dim=0).to(state.s_fast.dtype)
        base = w[0].unsqueeze(-1) * state.s_fast + w[1].unsqueeze(-1) * state.s_slow + w[2].unsqueeze(-1) * state.x0
        contrast = (
            torch.tanh(self.gamma_ortho[sublayer_idx]).to(state.s_fast.dtype) * state.s_slow
            + torch.tanh(self.gamma_x0[sublayer_idx]).to(state.s_fast.dtype) * state.x0
        )
        return base + contrast

    def update(self, state: ResidualBankState, u: torch.Tensor, sublayer_idx: int) -> None:
        sf, uf, u_par, u_perp_fast, rho, _ = _compute_gor_components(state.s_fast, u, self.eps)
        phi = torch.clamp(2.0 * torch.sigmoid(self.k * rho), max=1.0)
        state.s_fast = (sf + u_perp_fast + phi * u_par).to(dtype=u.dtype)

        e0 = state.x0_unit
        u_rem0 = uf - (uf * e0).sum(dim=-1, keepdim=True) * e0
        sf_orth0 = sf - (sf * e0).sum(dim=-1, keepdim=True) * e0
        e_fast = F.normalize(sf_orth0, p=2, dim=-1, eps=1e-6)
        u_subspace_perp = u_rem0 - (u_rem0 * e_fast).sum(dim=-1, keepdim=True) * e_fast
        state.s_slow = state.s_slow + u_subspace_perp.to(dtype=u.dtype) * self.inv_sqrt_l

    def finalize(self, state: ResidualBankState) -> torch.Tensor:
        return state.s_fast + state.s_slow


class BlockAttnResConnector(nn.Module):
    def __init__(self, dim: int, num_sublayers: int, num_blocks: int = 8):
        super().__init__()
        self.num_sublayers = num_sublayers
        self.eff_blocks = max(1, min(num_blocks, num_sublayers))
        self.block_size = math.ceil(num_sublayers / self.eff_blocks)
        self.depth_queries = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(num_sublayers)])
        self.key_norms = nn.ModuleList([RMSNorm(dim) for _ in range(num_sublayers + 1)])

    def _attend(self, states: List[torch.Tensor], keys: List[torch.Tensor], query: torch.Tensor) -> torch.Tensor:
        if len(states) == 1:
            return states[0]
        logits = torch.stack([torch.einsum("d,...d->...", query, k) for k in keys], dim=0)
        w = F.softmax(logits.float(), dim=0).to(states[0].dtype)
        out = w[0].unsqueeze(-1) * states[0]
        for idx in range(1, len(states)):
            out = out + w[idx].unsqueeze(-1) * states[idx]
        return out

    def init_state(self, x0: torch.Tensor) -> ResidualBankState:
        return ResidualBankState(
            x0=x0,
            completed=[x0],
            completed_keys=[self.key_norms[0](x0)],
            partial=None,
        )

    def read(self, state: ResidualBankState, sublayer_idx: int) -> torch.Tensor:
        if sublayer_idx == 0:
            return state.x0
        if state.partial is None:
            cands = state.completed
            ckeys = state.completed_keys
        else:
            cands = state.completed + [state.partial]
            ckeys = state.completed_keys + [self.key_norms[sublayer_idx](state.partial)]
        return self._attend(cands, ckeys, self.depth_queries[sublayer_idx - 1])

    def update(self, state: ResidualBankState, u: torch.Tensor, sublayer_idx: int) -> None:
        state.partial = u if state.partial is None else (state.partial + u)
        is_end = ((sublayer_idx + 1) % self.block_size == 0) or (sublayer_idx + 1 == self.num_sublayers)
        if is_end:
            state.completed.append(state.partial)
            state.completed_keys.append(self.key_norms[sublayer_idx + 1](state.partial))
            state.partial = None

    def finalize(self, state: ResidualBankState) -> torch.Tensor:
        return self._attend(state.completed, state.completed_keys, self.depth_queries[self.num_sublayers - 1])
