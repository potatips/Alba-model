import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


ALL_ARCHITECTURES = (
    "pre_norm",
    "gor",
    "gor_bounded",
    "mhc",
    "attn_res_block",
    "attn_res_full",
    "gated_res",
)


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


class SwiGLULayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = max(64, ((hidden_dim + 31) // 32) * 32)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeometricOrthogonalResidual(nn.Module):
    def __init__(self, eps: float = 1e-7, k: float = 3.0, max_phi: Optional[float] = None):
        super().__init__()
        self.eps = eps
        self.k = k
        self.max_phi = max_phi

    def _compute_phi(self, rho: torch.Tensor) -> torch.Tensor:
        phi = 2.0 * torch.sigmoid(self.k * rho)
        if self.max_phi is not None:
            phi = torch.clamp(phi, max=self.max_phi)
        return phi

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        uf = u.float()

        dot = (xf * uf).sum(dim=-1, keepdim=True)
        norm_sq_x = (xf * xf).sum(dim=-1, keepdim=True)
        norm_sq_u = (uf * uf).sum(dim=-1, keepdim=True)

        norm_x = torch.sqrt(norm_sq_x + self.eps)
        norm_u = torch.sqrt(norm_sq_u + self.eps)

        rho = dot / (norm_x * norm_u + self.eps)
        u_parallel = (dot / (norm_sq_x + self.eps)) * xf
        u_perp = uf - u_parallel

        phi = self._compute_phi(rho)
        out = xf + u_perp + phi * u_parallel
        return out.to(dtype=orig_dtype)

    def forward_with_metrics(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        orig_dtype = x.dtype
        xf = x.float()
        uf = u.float()

        dot = (xf * uf).sum(dim=-1, keepdim=True)
        norm_sq_x = (xf * xf).sum(dim=-1, keepdim=True)
        norm_sq_u = (uf * uf).sum(dim=-1, keepdim=True)

        norm_x = torch.sqrt(norm_sq_x + self.eps)
        norm_u = torch.sqrt(norm_sq_u + self.eps)

        rho = dot / (norm_x * norm_u + self.eps)
        u_parallel = (dot / (norm_sq_x + self.eps)) * xf
        u_perp = uf - u_parallel

        phi = self._compute_phi(rho)
        out = xf + u_perp + phi * u_parallel
        return out.to(dtype=orig_dtype), float(rho.detach().mean().item()), float(phi.detach().mean().item())


def sinkhorn_knopp(
    logits: torch.Tensor,
    n_iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    orig_dtype = logits.dtype
    lf = logits.float()
    shifted = (lf - lf.amax(dim=(-2, -1), keepdim=True)).clamp_min(-30.0)
    m = torch.exp(shifted)
    for _ in range(n_iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m.to(orig_dtype)


def _tensor_stream_stats(x: torch.Tensor) -> Tuple[float, float, float]:
    xf = x.detach().float()
    rms = float(torch.sqrt(xf.pow(2).mean() + 1e-12).item())
    l2 = float(torch.linalg.vector_norm(xf, dim=-1).mean().item())
    outlier = float(xf.abs().max().item() / max(rms, 1e-12))
    return rms, l2, outlier


def _init_swiglu_module(module: nn.Module, std: Optional[float] = None) -> None:
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(1)
        eff_std = std if std is not None else (0.75 / math.sqrt(float(fan_in)))
        nn.init.normal_(module.weight, mean=0.0, std=eff_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BaseResidualNetwork(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        hidden_dim: Optional[int] = None,
        init_std: Optional[float] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.init_std = init_std
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(num_layers)])
        self.blocks = nn.ModuleList([SwiGLULayer(dim, hidden_dim) for _ in range(num_layers)])
        self.final_norm = RMSNorm(dim)
        self.blocks.apply(lambda m: _init_swiglu_module(m, std=init_std))

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_core_params(self) -> int:
        core = sum(p.numel() for p in self.norms.parameters() if p.requires_grad)
        core += sum(p.numel() for p in self.blocks.parameters() if p.requires_grad)
        core += sum(p.numel() for p in self.final_norm.parameters() if p.requires_grad)
        return core

    def count_residual_overhead_params(self) -> int:
        return self.count_total_params() - self.count_core_params()


class PreNormResidualNetwork(BaseResidualNetwork):
    arch_name = "pre_norm"

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
        apply_final_norm: bool = False,
    ):
        stats: Dict[str, List[float]] = {
            "stream_rms": [],
            "stream_l2": [],
            "update_l2": [],
            "dilution_ratio": [],
            "outlier_ratio": [],
        }
        h = x
        if return_stats:
            rms, l2, out_r = _tensor_stream_stats(h)
            stats["stream_rms"].append(rms)
            stats["stream_l2"].append(l2)
            stats["outlier_ratio"].append(out_r)

        for norm, block in zip(self.norms, self.blocks):
            u = block(norm(h))
            if return_stats:
                u_l2 = float(torch.linalg.vector_norm(u.detach().float(), dim=-1).mean().item())
                h_l2 = stats["stream_l2"][-1]
                stats["update_l2"].append(u_l2)
                stats["dilution_ratio"].append(u_l2 / max(h_l2, 1e-12))
            h = h + u
            if return_stats:
                rms, l2, out_r = _tensor_stream_stats(h)
                stats["stream_rms"].append(rms)
                stats["stream_l2"].append(l2)
                stats["outlier_ratio"].append(out_r)

        out = self.final_norm(h) if apply_final_norm else h
        if return_stats:
            return out, stats
        return out


class GORResidualNetwork(BaseResidualNetwork):
    arch_name = "gor"

    def __init__(
        self,
        dim: int,
        num_layers: int,
        hidden_dim: Optional[int] = None,
        init_std: Optional[float] = None,
        gor_k: float = 3.0,
        gor_eps: float = 1e-7,
        gor_max_phi: Optional[float] = None,
    ):
        super().__init__(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, init_std=init_std)
        self.gor_ops = nn.ModuleList(
            [
                GeometricOrthogonalResidual(eps=gor_eps, k=gor_k, max_phi=gor_max_phi)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
        apply_final_norm: bool = False,
    ):
        stats: Dict[str, List[float]] = {
            "stream_rms": [],
            "stream_l2": [],
            "update_l2": [],
            "dilution_ratio": [],
            "outlier_ratio": [],
            "mean_rho": [],
            "mean_phi": [],
        }
        h = x
        if return_stats:
            rms, l2, out_r = _tensor_stream_stats(h)
            stats["stream_rms"].append(rms)
            stats["stream_l2"].append(l2)
            stats["outlier_ratio"].append(out_r)

        for norm, block, gor in zip(self.norms, self.blocks, self.gor_ops):
            u = block(norm(h))
            if return_stats:
                u_l2 = float(torch.linalg.vector_norm(u.detach().float(), dim=-1).mean().item())
                h_l2 = stats["stream_l2"][-1]
                stats["update_l2"].append(u_l2)
                stats["dilution_ratio"].append(u_l2 / max(h_l2, 1e-12))
                h, mean_rho, mean_phi = gor.forward_with_metrics(h, u)
                stats["mean_rho"].append(mean_rho)
                stats["mean_phi"].append(mean_phi)
                rms, l2, out_r = _tensor_stream_stats(h)
                stats["stream_rms"].append(rms)
                stats["stream_l2"].append(l2)
                stats["outlier_ratio"].append(out_r)
            else:
                h = gor(h, u)

        out = self.final_norm(h) if apply_final_norm else h
        if return_stats:
            return out, stats
        return out


class BoundedGORResidualNetwork(GORResidualNetwork):
    arch_name = "gor_bounded"

    def __init__(
        self,
        dim: int,
        num_layers: int,
        hidden_dim: Optional[int] = None,
        init_std: Optional[float] = None,
        gor_k: float = 3.0,
        gor_eps: float = 1e-7,
        gor_max_phi: float = 1.0,
    ):
        super().__init__(
            dim=dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_std=init_std,
            gor_k=gor_k,
            gor_eps=gor_eps,
            gor_max_phi=gor_max_phi,
        )


class MHCLayerCoupling(nn.Module):
    def __init__(
        self,
        dim: int,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        dynamic_scale: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.stream_norm = RMSNorm(n_streams * dim)
        self.proj = nn.Linear(n_streams * dim, n_streams * n_streams + 2 * n_streams, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

        self.alpha_pre = nn.Parameter(torch.tensor(dynamic_scale))
        self.alpha_post = nn.Parameter(torch.tensor(dynamic_scale))
        self.alpha_res = nn.Parameter(torch.tensor(dynamic_scale))

        self.bias_pre = nn.Parameter(torch.zeros(n_streams))
        self.bias_post = nn.Parameter(torch.zeros(n_streams))
        init_res = torch.full((n_streams, n_streams), -3.0)
        init_res.fill_diagonal_(0.0)
        self.bias_res = nn.Parameter(init_res)

    def compute_matrices(
        self, streams: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lead_shape = streams.shape[:-2]
        flat = streams.reshape(*lead_shape, self.n_streams * self.dim)
        normed = self.stream_norm(flat)
        raw = self.proj(normed)

        n = self.n_streams
        raw_pre = raw[..., :n]
        raw_post = raw[..., n : 2 * n]
        raw_res = raw[..., 2 * n :].reshape(*lead_shape, n, n)

        h_pre = F.softmax(self.alpha_pre * raw_pre + self.bias_pre, dim=-1)
        h_post = 2.0 * torch.sigmoid(self.alpha_post * raw_post + self.bias_post)
        a_res = self.alpha_res * raw_res + self.bias_res
        h_res = sinkhorn_knopp(a_res, n_iters=self.sinkhorn_iters)
        return h_pre, h_post, h_res


class MHCResidualNetwork(BaseResidualNetwork):
    arch_name = "mhc"

    def __init__(
        self,
        dim: int,
        num_layers: int,
        hidden_dim: Optional[int] = None,
        init_std: Optional[float] = None,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
    ):
        super().__init__(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, init_std=init_std)
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.couplings = nn.ModuleList(
            [
                MHCLayerCoupling(
                    dim=dim,
                    n_streams=n_streams,
                    sinkhorn_iters=sinkhorn_iters,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
        apply_final_norm: bool = False,
    ):
        stats: Dict[str, List[float]] = {
            "stream_rms": [],
            "stream_l2": [],
            "update_l2": [],
            "dilution_ratio": [],
            "outlier_ratio": [],
            "sinkhorn_max_err": [],
        }
        streams = x.unsqueeze(-2).expand(*x.shape[:-1], self.n_streams, self.dim).clone()
        if return_stats:
            rms, l2, out_r = _tensor_stream_stats(streams.mean(dim=-2))
            stats["stream_rms"].append(rms)
            stats["stream_l2"].append(l2)
            stats["outlier_ratio"].append(out_r)

        for norm, block, coupling in zip(self.norms, self.blocks, self.couplings):
            h_pre, h_post, h_res = coupling.compute_matrices(streams)
            h_in = torch.einsum("...n,...nd->...d", h_pre, streams)
            u = block(norm(h_in))
            mixed = torch.einsum("...ij,...jd->...id", h_res, streams)
            injected = h_post.unsqueeze(-1) * u.unsqueeze(-2)
            streams = mixed + injected

            if return_stats:
                u_l2 = float(torch.linalg.vector_norm(u.detach().float(), dim=-1).mean().item())
                prev_l2 = stats["stream_l2"][-1]
                stats["update_l2"].append(u_l2)
                stats["dilution_ratio"].append(u_l2 / max(prev_l2, 1e-12))

                hr_f = h_res.detach().float()
                row_err = (hr_f.sum(dim=-1) - 1.0).abs().max().item()
                col_err = (hr_f.sum(dim=-2) - 1.0).abs().max().item()
                stats["sinkhorn_max_err"].append(float(max(row_err, col_err)))

                rms, l2, out_r = _tensor_stream_stats(streams.mean(dim=-2))
                stats["stream_rms"].append(rms)
                stats["stream_l2"].append(l2)
                stats["outlier_ratio"].append(out_r)

        h = streams.mean(dim=-2)
        out = self.final_norm(h) if apply_final_norm else h
        if return_stats:
            return out, stats
        return out


class AttnResNetwork(BaseResidualNetwork):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        hidden_dim: Optional[int] = None,
        init_std: Optional[float] = None,
        num_blocks: Optional[int] = 8,
    ):
        super().__init__(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, init_std=init_std)
        self.num_blocks = num_blocks
        self.arch_name = "attn_res_full" if num_blocks is None else "attn_res_block"
        self.depth_queries = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(num_layers)]
        )
        self.key_norms = nn.ModuleList([RMSNorm(dim) for _ in range(num_layers + 1)])

    def _depth_attend(
        self,
        states: List[torch.Tensor],
        keys: List[torch.Tensor],
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        if len(states) == 1:
            return states[0], 0.0
        logit_list = [torch.einsum("d,...d->...", query, k) for k in keys]
        logits = torch.stack(logit_list, dim=0)
        weights = F.softmax(logits.float(), dim=0).to(states[0].dtype)
        attended = weights[0].unsqueeze(-1) * states[0]
        for idx in range(1, len(states)):
            attended = attended + weights[idx].unsqueeze(-1) * states[idx]
        wf = weights.detach().float()
        entropy = float(-(wf * torch.log(wf.clamp_min(1e-12))).sum(dim=0).mean().item())
        return attended, entropy

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
        apply_final_norm: bool = False,
    ):
        stats: Dict[str, List[float]] = {
            "stream_rms": [],
            "stream_l2": [],
            "update_l2": [],
            "dilution_ratio": [],
            "outlier_ratio": [],
            "attn_entropy": [],
            "memory_slots": [],
        }
        if return_stats:
            rms, l2, out_r = _tensor_stream_stats(x)
            stats["stream_rms"].append(rms)
            stats["stream_l2"].append(l2)
            stats["outlier_ratio"].append(out_r)

        if self.num_blocks is None:
            values: List[torch.Tensor] = [x]
            keys: List[torch.Tensor] = [self.key_norms[0](x)]
            for layer_idx, (norm, block) in enumerate(zip(self.norms, self.blocks)):
                if layer_idx == 0:
                    h_l, entropy = x, 0.0
                else:
                    h_l, entropy = self._depth_attend(
                        values, keys, self.depth_queries[layer_idx - 1]
                    )
                u = block(norm(h_l))
                values.append(u)
                keys.append(self.key_norms[layer_idx + 1](u))
                if return_stats:
                    h_l2 = float(torch.linalg.vector_norm(h_l.detach().float(), dim=-1).mean().item())
                    u_l2 = float(torch.linalg.vector_norm(u.detach().float(), dim=-1).mean().item())
                    stats["update_l2"].append(u_l2)
                    stats["dilution_ratio"].append(u_l2 / max(h_l2, 1e-12))
                    stats["attn_entropy"].append(entropy)
                    stats["memory_slots"].append(float(len(values)))
                    if layer_idx > 0:
                        rms, l2, out_r = _tensor_stream_stats(h_l)
                        stats["stream_rms"].append(rms)
                        stats["stream_l2"].append(l2)
                        stats["outlier_ratio"].append(out_r)
            final_h, final_ent = self._depth_attend(
                values, keys, self.depth_queries[self.num_layers - 1]
            )
            if return_stats:
                stats["attn_entropy"].append(final_ent)
                rms, l2, out_r = _tensor_stream_stats(final_h)
                stats["stream_rms"].append(rms)
                stats["stream_l2"].append(l2)
                stats["outlier_ratio"].append(out_r)
        else:
            effective_blocks = max(1, min(self.num_blocks, self.num_layers))
            block_size = math.ceil(self.num_layers / effective_blocks)
            completed: List[torch.Tensor] = [x]
            completed_keys: List[torch.Tensor] = [self.key_norms[0](x)]
            partial: Optional[torch.Tensor] = None

            for layer_idx, (norm, block) in enumerate(zip(self.norms, self.blocks)):
                if layer_idx == 0:
                    candidates = completed
                    h_l, entropy = x, 0.0
                else:
                    if partial is None:
                        candidates = completed
                        cand_keys = completed_keys
                    else:
                        candidates = completed + [partial]
                        cand_keys = completed_keys + [self.key_norms[layer_idx](partial)]
                    h_l, entropy = self._depth_attend(
                        candidates, cand_keys, self.depth_queries[layer_idx - 1]
                    )

                u = block(norm(h_l))
                partial = u if partial is None else (partial + u)

                is_block_end = ((layer_idx + 1) % block_size == 0) or (
                    layer_idx + 1 == self.num_layers
                )
                if is_block_end:
                    completed.append(partial)
                    completed_keys.append(self.key_norms[layer_idx + 1](partial))
                    partial = None

                if return_stats:
                    h_l2 = float(torch.linalg.vector_norm(h_l.detach().float(), dim=-1).mean().item())
                    u_l2 = float(torch.linalg.vector_norm(u.detach().float(), dim=-1).mean().item())
                    stats["update_l2"].append(u_l2)
                    stats["dilution_ratio"].append(u_l2 / max(h_l2, 1e-12))
                    stats["attn_entropy"].append(entropy)
                    stats["memory_slots"].append(float(len(candidates)))
                    if layer_idx > 0:
                        rms, l2, out_r = _tensor_stream_stats(h_l)
                        stats["stream_rms"].append(rms)
                        stats["stream_l2"].append(l2)
                        stats["outlier_ratio"].append(out_r)

            final_h, final_ent = self._depth_attend(
                completed, completed_keys, self.depth_queries[self.num_layers - 1]
            )
            if return_stats:
                stats["attn_entropy"].append(final_ent)
                rms, l2, out_r = _tensor_stream_stats(final_h)
                stats["stream_rms"].append(rms)
                stats["stream_l2"].append(l2)
                stats["outlier_ratio"].append(out_r)

        out = self.final_norm(final_h) if apply_final_norm else final_h
        if return_stats:
            return out, stats
        return out


class GatedResidualLayerGate(nn.Module):
    def __init__(
        self,
        dim: int,
        n_streams: int = 4,
        num_layers: int = 16,
        bottleneck_rank: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        rank = bottleneck_rank if bottleneck_rank is not None else max(8, dim // 16)
        self.stream_norm = RMSNorm(n_streams * dim)
        self.gate_down = nn.Linear(n_streams * dim, rank, bias=False)
        self.gate_read = nn.Linear(rank, n_streams, bias=True)
        self.gate_up_res = nn.Linear(rank, n_streams * dim, bias=True)
        self.gate_up_out = nn.Linear(rank, n_streams * dim, bias=True)
        nn.init.normal_(self.gate_down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.gate_read.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_read.bias)
        nn.init.normal_(self.gate_up_res.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.gate_up_out.weight, mean=0.0, std=0.01)

        nn.init.zeros_(self.gate_up_res.bias)
        nn.init.zeros_(self.gate_up_out.bias)

    def compute_read_and_write_gates(
        self, streams: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lead_shape = streams.shape[:-2]
        flat = streams.reshape(*lead_shape, self.n_streams * self.dim)
        z = F.silu(self.gate_down(self.stream_norm(flat)))
        read_weights = F.softmax(self.gate_read(z), dim=-1)
        h_in = torch.einsum("...n,...nd->...d", read_weights, streams)
        raw_res = self.gate_up_res(z).reshape(*lead_shape, self.n_streams, self.dim)
        raw_out = self.gate_up_out(z).reshape(*lead_shape, self.n_streams, self.dim)
        g_res = 2.0 * torch.sigmoid(raw_res)
        g_out = 2.0 * torch.sigmoid(raw_out)
        return h_in, g_res, g_out


class GatedResidualNetwork(BaseResidualNetwork):
    arch_name = "gated_res"

    def __init__(
        self,
        dim: int,
        num_layers: int,
        hidden_dim: Optional[int] = None,
        init_std: Optional[float] = None,
        n_streams: int = 4,
    ):
        super().__init__(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, init_std=init_std)
        self.n_streams = n_streams
        self.gates = nn.ModuleList(
            [
                GatedResidualLayerGate(
                    dim=dim,
                    n_streams=n_streams,
                    num_layers=num_layers,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
        apply_final_norm: bool = False,
    ):
        stats: Dict[str, List[float]] = {
            "stream_rms": [],
            "stream_l2": [],
            "update_l2": [],
            "dilution_ratio": [],
            "outlier_ratio": [],
            "mean_g_res": [],
            "mean_g_out": [],
        }
        streams = x.unsqueeze(-2).expand(*x.shape[:-1], self.n_streams, self.dim).clone()
        if return_stats:
            rms, l2, out_r = _tensor_stream_stats(streams.mean(dim=-2))
            stats["stream_rms"].append(rms)
            stats["stream_l2"].append(l2)
            stats["outlier_ratio"].append(out_r)

        for norm, block, gate in zip(self.norms, self.blocks, self.gates):
            h_in, g_res, g_out = gate.compute_read_and_write_gates(streams)
            u = block(norm(h_in))
            streams = g_res * streams + g_out * u.unsqueeze(-2)

            if return_stats:
                u_l2 = float(torch.linalg.vector_norm(u.detach().float(), dim=-1).mean().item())
                prev_l2 = stats["stream_l2"][-1]
                stats["update_l2"].append(u_l2)
                stats["dilution_ratio"].append(u_l2 / max(prev_l2, 1e-12))
                stats["mean_g_res"].append(float(g_res.detach().float().mean().item()))
                stats["mean_g_out"].append(float(g_out.detach().float().mean().item()))
                rms, l2, out_r = _tensor_stream_stats(streams.mean(dim=-2))
                stats["stream_rms"].append(rms)
                stats["stream_l2"].append(l2)
                stats["outlier_ratio"].append(out_r)

        h = streams.mean(dim=-2)
        out = self.final_norm(h) if apply_final_norm else h
        if return_stats:
            return out, stats
        return out


def build_residual_network(
    arch: str,
    dim: int,
    num_layers: int,
    hidden_dim: Optional[int] = None,
    init_std: Optional[float] = None,
    n_streams: int = 4,
    attn_num_blocks: int = 8,
    sinkhorn_iters: int = 20,
    gor_k: float = 3.0,
) -> BaseResidualNetwork:
    if arch == "pre_norm":
        return PreNormResidualNetwork(
            dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, init_std=init_std
        )
    if arch == "gor":
        return GORResidualNetwork(
            dim=dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_std=init_std,
            gor_k=gor_k,
        )
    if arch == "gor_bounded":
        return BoundedGORResidualNetwork(
            dim=dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_std=init_std,
            gor_k=gor_k,
            gor_max_phi=1.0,
        )
    if arch == "mhc":
        return MHCResidualNetwork(
            dim=dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_std=init_std,
            n_streams=n_streams,
            sinkhorn_iters=sinkhorn_iters,
        )
    if arch == "attn_res_block":
        return AttnResNetwork(
            dim=dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_std=init_std,
            num_blocks=attn_num_blocks,
        )
    if arch == "attn_res_full":
        return AttnResNetwork(
            dim=dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_std=init_std,
            num_blocks=None,
        )
    if arch == "gated_res":
        return GatedResidualNetwork(
            dim=dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_std=init_std,
            n_streams=n_streams,
        )
    raise ValueError(f"Unknown residual architecture: {arch}")


def clone_shared_block_weights(
    source: BaseResidualNetwork, target: BaseResidualNetwork
) -> None:
    if source.num_layers != target.num_layers or source.dim != target.dim:
        raise ValueError("Source and target networks must have identical depth and dimension.")
    with torch.no_grad():
        target.norms.load_state_dict(source.norms.state_dict())
        target.blocks.load_state_dict(source.blocks.state_dict())
        target.final_norm.load_state_dict(source.final_norm.state_dict())
