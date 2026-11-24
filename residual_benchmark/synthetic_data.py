import math
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


@dataclass
class SyntheticMatrixDataset:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    metadata: Dict[str, float]

    def slice_train(self, n_train: int) -> "SyntheticMatrixDataset":
        if n_train > self.x_train.size(0):
            raise ValueError(f"Requested n_train={n_train} exceeds available {self.x_train.size(0)}")
        meta = dict(self.metadata)
        meta["n_train"] = float(n_train)
        return SyntheticMatrixDataset(
            name=self.name,
            x_train=self.x_train[:n_train].contiguous(),
            y_train=self.y_train[:n_train].contiguous(),
            x_val=self.x_val,
            y_val=self.y_val,
            metadata=meta,
        )


def _make_orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(raw)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    return q * diag_sign.unsqueeze(0)


def _apply_markov_deep_teacher(
    x: torch.Tensor,
    dim: int,
    teacher_depth: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + 7919)
    h = x.clone()
    for _ in range(teacher_depth):
        w_a = _make_orthogonal_matrix(dim, gen)
        w_b = _make_orthogonal_matrix(dim, gen)
        w_out = _make_orthogonal_matrix(dim, gen)
        norm_h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        u = (F.silu(norm_h @ w_a) * (norm_h @ w_b)) @ w_out
        h = h + 0.35 * u
    y = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return y


def generate_markov_deep_matrix_dataset(
    n_train: int = 8192,
    n_val: int = 1024,
    seq_len: int = 8,
    dim: int = 128,
    teacher_depth: int = 10,
    noise_std: float = 0.05,
    seed: int = 42,
) -> SyntheticMatrixDataset:
    gen = torch.Generator().manual_seed(seed)
    n_total = n_train + n_val

    basis = _make_orthogonal_matrix(dim, gen)
    spectrum = torch.linspace(1.6, 0.4, dim)
    cov_factor = basis * spectrum.unsqueeze(0)

    z = torch.randn(n_total, seq_len, dim, generator=gen)
    x = z @ cov_factor.t()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    y_clean = _apply_markov_deep_teacher(x, dim=dim, teacher_depth=teacher_depth, seed=seed)
    noise = torch.randn(y_clean.shape, generator=gen) * noise_std
    y = y_clean + noise

    return SyntheticMatrixDataset(
        name="markov_deep_regression",
        x_train=x[:n_train].contiguous(),
        y_train=y[:n_train].contiguous(),
        x_val=x[n_train:].contiguous(),
        y_val=y_clean[n_train:].contiguous(),
        metadata={
            "n_train": float(n_train),
            "n_val": float(n_val),
            "seq_len": float(seq_len),
            "dim": float(dim),
            "teacher_depth": float(teacher_depth),
            "noise_std": float(noise_std),
            "input_max_to_rms": float(
                x.abs().max().item() / max(math.sqrt(float(x.pow(2).mean().item())), 1e-12)
            ),
        },
    )


def _apply_deep_matrix_teacher(
    x: torch.Tensor,
    dim: int,
    teacher_depth: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + 9973)
    h = x.clone()
    early_proj = _make_orthogonal_matrix(dim, gen)
    mid_proj = _make_orthogonal_matrix(dim, gen)
    final_proj = _make_orthogonal_matrix(dim, gen)
    mid_state = None
    mid_step = max(1, teacher_depth // 2)

    for step in range(teacher_depth):
        w_a = _make_orthogonal_matrix(dim, gen)
        w_b = _make_orthogonal_matrix(dim, gen)
        w_out = _make_orthogonal_matrix(dim, gen)
        norm_h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        gated = F.silu(norm_h @ w_a) * torch.tanh(norm_h @ w_b)
        nonlin = torch.sin((norm_h @ w_a) * 1.2)
        delta = ((0.75 * gated + 0.25 * nonlin) @ w_out) * 0.9
        h = 0.65 * (norm_h @ w_out) + delta
        if step + 1 == mid_step:
            mid_state = h.clone()

    early_branch = torch.tanh((x @ early_proj) * 1.1)
    mid_branch = mid_state if mid_state is not None else h
    mid_norm = (mid_branch @ mid_proj) * torch.rsqrt(
        mid_branch.pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )
    final_norm = (h @ final_proj) * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    y = 0.45 * final_norm + 0.30 * mid_norm + 0.25 * early_branch
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return y


def generate_structured_deep_matrix_dataset(
    n_train: int = 8192,
    n_val: int = 1024,
    seq_len: int = 8,
    dim: int = 128,
    teacher_depth: int = 8,
    noise_std: float = 0.05,
    seed: int = 42,
) -> SyntheticMatrixDataset:
    gen = torch.Generator().manual_seed(seed)
    n_total = n_train + n_val

    basis = _make_orthogonal_matrix(dim, gen)
    spectrum = torch.linspace(1.6, 0.4, dim)
    cov_factor = basis * spectrum.unsqueeze(0)

    z = torch.randn(n_total, seq_len, dim, generator=gen)
    x = z @ cov_factor.t()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    y_clean = _apply_deep_matrix_teacher(x, dim=dim, teacher_depth=teacher_depth, seed=seed)
    noise = torch.randn(y_clean.shape, generator=gen) * noise_std
    y = y_clean + noise

    return SyntheticMatrixDataset(
        name="structured_deep_regression",
        x_train=x[:n_train].contiguous(),
        y_train=y[:n_train].contiguous(),
        x_val=x[n_train:].contiguous(),
        y_val=y_clean[n_train:].contiguous(),
        metadata={
            "n_train": float(n_train),
            "n_val": float(n_val),
            "seq_len": float(seq_len),
            "dim": float(dim),
            "teacher_depth": float(teacher_depth),
            "noise_std": float(noise_std),
            "input_max_to_rms": float(
                x.abs().max().item() / max(math.sqrt(float(x.pow(2).mean().item())), 1e-12)
            ),
        },
    )


def generate_outlier_stress_matrix_dataset(
    n_train: int = 8192,
    n_val: int = 1024,
    seq_len: int = 8,
    dim: int = 128,
    teacher_depth: int = 8,
    outlier_prob: float = 0.02,
    outlier_scale: float = 12.0,
    noise_std: float = 0.08,
    seed: int = 123,
) -> SyntheticMatrixDataset:
    gen = torch.Generator().manual_seed(seed)
    n_total = n_train + n_val

    basis = _make_orthogonal_matrix(dim, gen)
    z = torch.randn(n_total, seq_len, dim, generator=gen)
    x_clean = z @ basis.t()
    x_clean = x_clean * torch.rsqrt(x_clean.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    y_clean = _apply_deep_matrix_teacher(x_clean, dim=dim, teacher_depth=teacher_depth, seed=seed)

    mask = (torch.rand(x_clean.shape, generator=gen) < outlier_prob).float()
    signs = torch.where(
        torch.rand(x_clean.shape, generator=gen) < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    magnitudes = (0.5 + torch.rand(x_clean.shape, generator=gen)) * outlier_scale
    x_corrupted = x_clean + mask * signs * magnitudes

    y_noisy = y_clean + torch.randn(y_clean.shape, generator=gen) * noise_std

    return SyntheticMatrixDataset(
        name="noisy_outlier_regression",
        x_train=x_corrupted[:n_train].contiguous(),
        y_train=y_noisy[:n_train].contiguous(),
        x_val=x_corrupted[n_train:].contiguous(),
        y_val=y_clean[n_train:].contiguous(),
        metadata={
            "n_train": float(n_train),
            "n_val": float(n_val),
            "seq_len": float(seq_len),
            "dim": float(dim),
            "outlier_prob": float(outlier_prob),
            "outlier_scale": float(outlier_scale),
            "input_max_to_rms": float(
                x_corrupted.abs().max().item()
                / max(math.sqrt(float(x_corrupted.pow(2).mean().item())), 1e-12)
            ),
        },
    )


def generate_controlled_alignment_step(
    x: torch.Tensor,
    target_rho: float,
    update_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    eps = 1e-8
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    x_hat = x / x_norm

    if generator.device == x.device:
        rand_vec = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    else:
        rand_vec = torch.randn(x.shape, generator=generator, device=generator.device, dtype=x.dtype).to(x.device)
    proj = (rand_vec * x_hat).sum(dim=-1, keepdim=True) * x_hat
    orth = rand_vec - proj
    orth_norm = torch.linalg.vector_norm(orth, dim=-1, keepdim=True).clamp_min(eps)
    orth_hat = orth / orth_norm

    rho_clamped = max(-0.999, min(0.999, target_rho))
    perp_coeff = math.sqrt(max(0.0, 1.0 - rho_clamped * rho_clamped))
    u_dir = rho_clamped * x_hat + perp_coeff * orth_hat
    return u_dir * (update_scale * x_norm)


def make_random_matrix_batch(
    batch_size: int,
    seq_len: int,
    dim: int,
    distribution: str = "gaussian",
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, seq_len, dim, generator=gen)
    if distribution == "heavy_tail":
        mask = (torch.rand(x.shape, generator=gen) < 0.02).float()
        spikes = (torch.randn(x.shape, generator=gen).sign()) * 12.0
        x = x + mask * spikes
    elif distribution == "correlated":
        basis = _make_orthogonal_matrix(dim, gen)
        spectrum = torch.linspace(2.0, 0.25, dim)
        x = x @ (basis * spectrum.unsqueeze(0)).t()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    elif distribution != "gaussian":
        raise ValueError(f"Unknown distribution: {distribution}")
    return x.to(device)
