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
    "dual_bank_gor_v2",
    "gor_v3_subspace",
    "mhc",
    "attn_res_block",
    "attn_res_gor_v2",
)

RESIDUAL_LABELS = {
    "pre_norm": "Pre-Norm (x + u)",
    "gor_v2": "GOR-v2 (phi<=2, 0-Param)",
    "gor_bounded": "GOR-Bounded (phi<=1, 0-Param)",
    "gor_antidilute": "GOR-AntiDilute (0-Param)",
    "std_dual_bank": "Std Dual-Bank Control",
    "dual_bank_gor_v2": "Dual-Bank + GOR-v2 (phi<=2)",
    "gor_v3_subspace": "GOR-v3 Subspace Contrast",
    "mhc": "mHC (Sinkhorn n=4)",
    "attn_res_block": "Block AttnRes (N=8)",
    "attn_res_gor_v2": "Block AttnRes + Attn-GOR-v2",
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


def _gor_v2_coeff(
    ref: torch.Tensor,
    u: torch.Tensor,
    k: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rf = ref.float()
    uf = u.float()
    dot = (rf * uf).sum(dim=-1, keepdim=True)
    norm_sq_r = (rf * rf).sum(dim=-1, keepdim=True) + eps
    norm_sq_u = (uf * uf).sum(dim=-1, keepdim=True) + eps
    rho = dot * torch.rsqrt(norm_sq_r * norm_sq_u)
    phi = 2.0 * torch.sigmoid(k * rho)
    coeff = (phi - 1.0) * (dot / norm_sq_r)
    return rf, uf, coeff


def _gor_v2_residual_add(x: torch.Tensor, u: torch.Tensor, k: float = 3.0, eps: float = 1e-7) -> torch.Tensor:
    xf, uf, coeff = _gor_v2_coeff(x, u, k=k, eps=eps)
    return ((1.0 + coeff) * xf + uf).to(dtype=x.dtype)


def _gor_v2_transform_u(ref: torch.Tensor, u: torch.Tensor, k: float = 3.0, eps: float = 1e-7) -> torch.Tensor:
    rf, uf, coeff = _gor_v2_coeff(ref, u, k=k, eps=eps)
    return (uf + coeff * rf).to(dtype=u.dtype)


class PreNormResidual(nn.Module):
    def forward(self, x: torch.Tensor, u: torch.Tensor, x0_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x + u


class GORv2Residual(nn.Module):
    def __init__(self, k: float = 3.0, eps: float = 1e-7):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, x: torch.Tensor, u: torch.Tensor, x0_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _gor_v2_residual_add(x, u, k=self.k, eps=self.eps)


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
    last_read: Optional[torch.Tensor] = None
    streams: Optional[torch.Tensor] = None
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


class DualBankGORv2Connector(StdDualBankConnector):
    def __init__(self, dim: int, num_sublayers: int, k: float = 3.0, eps: float = 1e-7):
        super().__init__(dim, num_sublayers=num_sublayers)
        self.k = k
        self.eps = eps

    def update(self, state: ResidualBankState, u: torch.Tensor, sublayer_idx: int) -> None:
        state.s_fast = _gor_v2_residual_add(state.s_fast, u, k=self.k, eps=self.eps)
        state.s_slow = state.s_slow + u * self.inv_sqrt_l


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


class DualBankGORv2Connector(StdDualBankConnector):
    def __init__(self, dim: int, num_sublayers: int, k: float = 3.0, eps: float = 1e-7):
        super().__init__(dim, num_sublayers=num_sublayers)
        self.k = k
        self.eps = eps

    def update(self, state: ResidualBankState, u: torch.Tensor, sublayer_idx: int) -> None:
        state.s_fast = _gor_v2_residual_add(state.s_fast, u, k=self.k, eps=self.eps)
        state.s_slow = state.s_slow + u * self.inv_sqrt_l


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
        states_t = torch.stack(states, dim=0)
        keys_t = torch.stack(keys, dim=0)
        logits = (keys_t.float() * query.float()).sum(dim=-1)
        w = F.softmax(logits, dim=0).to(dtype=states_t.dtype)
        return (w.unsqueeze(-1) * states_t).sum(dim=0)

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


class BlockAttnResGORv2Connector(BlockAttnResConnector):
    def __init__(self, dim: int, num_sublayers: int, num_blocks: int = 8, k: float = 3.0, eps: float = 1e-7):
        super().__init__(dim, num_sublayers=num_sublayers, num_blocks=num_blocks)
        self.k = k
        self.eps = eps

    def read(self, state: ResidualBankState, sublayer_idx: int) -> torch.Tensor:
        out = super().read(state, sublayer_idx)
        state.last_read = out
        return out

    def update(self, state: ResidualBankState, u: torch.Tensor, sublayer_idx: int) -> None:
        u_eff = u
        if sublayer_idx % 2 == 0 and state.last_read is not None:
            u_eff = _gor_v2_transform_u(state.last_read, u, k=self.k, eps=self.eps)
        super().update(state, u_eff, sublayer_idx)


def sinkhorn_knopp(logits: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    z = logits.float()
    z = z - z.amax(dim=(-2, -1), keepdim=True)
    m = torch.exp(z).clamp_min(eps)
    for _ in range(iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


class MHCConnector(nn.Module):
    def __init__(
        self,
        dim: int,
        num_sublayers: int,
        n_streams: int = 4,
        sinkhorn_iters: int = 10,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_sublayers = num_sublayers
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

        init_res = torch.full((num_sublayers, n_streams, n_streams), -2.0)
        diag_idx = torch.arange(n_streams)
        init_res[:, diag_idx, diag_idx] = 4.0
        self.h_res_bias = nn.Parameter(init_res)
        self.h_pre_bias = nn.Parameter(torch.zeros(num_sublayers, n_streams))
        self.h_post_bias = nn.Parameter(torch.zeros(num_sublayers, n_streams))

        self.dyn_norm = RMSNorm(dim)
        self.alpha_res = nn.Parameter(torch.zeros(num_sublayers))
        self.alpha_pre = nn.Parameter(torch.zeros(num_sublayers))
        self.alpha_post = nn.Parameter(torch.zeros(num_sublayers))
        self.dyn_res = nn.Linear(dim, num_sublayers * n_streams * n_streams, bias=False)
        self.dyn_pre = nn.Linear(dim, num_sublayers * n_streams, bias=False)
        self.dyn_post = nn.Linear(dim, num_sublayers * n_streams, bias=False)
        nn.init.zeros_(self.dyn_res.weight)
        nn.init.zeros_(self.dyn_pre.weight)
        nn.init.zeros_(self.dyn_post.weight)

    def init_state(self, x0: torch.Tensor) -> ResidualBankState:
        streams = x0.unsqueeze(-2).repeat(*([1] * (x0.ndim - 1)), self.n_streams, 1).contiguous()
        return ResidualBankState(x0=x0, streams=streams)

    def _dyn_weights(self, streams: torch.Tensor, sublayer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.n_streams
        pooled = self.dyn_norm(streams.mean(dim=-2))
        s_off = sublayer_idx * n
        r_off = sublayer_idx * n * n

        pre_w = F.linear(pooled, self.dyn_pre.weight[s_off : s_off + n])
        post_w = F.linear(pooled, self.dyn_post.weight[s_off : s_off + n])
        res_w = F.linear(pooled, self.dyn_res.weight[r_off : r_off + n * n]).view(*pooled.shape[:-1], n, n)

        h_pre_logits = self.h_pre_bias[sublayer_idx] + self.alpha_pre[sublayer_idx] * torch.tanh(pre_w.float())
        h_post_logits = self.h_post_bias[sublayer_idx] + self.alpha_post[sublayer_idx] * torch.tanh(post_w.float())
        h_res_logits = self.h_res_bias[sublayer_idx] + self.alpha_res[sublayer_idx] * torch.tanh(res_w.float())

        h_pre = F.softmax(h_pre_logits.float(), dim=-1)
        h_post = 2.0 * torch.sigmoid(h_post_logits.float())
        h_res = sinkhorn_knopp(h_res_logits, iters=self.sinkhorn_iters, eps=self.eps)
        return h_pre, h_post, h_res

    def read(self, state: ResidualBankState, sublayer_idx: int) -> torch.Tensor:
        streams = state.streams
        h_pre, h_post, h_res = self._dyn_weights(streams, sublayer_idx)
        state.last_read = (h_post, h_res)
        h_in = (h_pre.to(dtype=streams.dtype).unsqueeze(-1) * streams).sum(dim=-2)
        return h_in

    def update(self, state: ResidualBankState, u: torch.Tensor, sublayer_idx: int) -> None:
        streams = state.streams
        if state.last_read is None:
            _, h_post, h_res = self._dyn_weights(streams, sublayer_idx)
        else:
            h_post, h_res = state.last_read
            state.last_read = None
        mixed = torch.matmul(h_res.to(dtype=streams.dtype), streams)
        injected = h_post.to(dtype=u.dtype).unsqueeze(-1) * u.unsqueeze(-2)
        state.streams = mixed + injected

    def finalize(self, state: ResidualBankState) -> torch.Tensor:
        return state.streams.mean(dim=-2)
