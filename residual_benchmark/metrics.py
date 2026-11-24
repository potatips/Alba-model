import math
import time
from typing import Dict, List
import torch
import torch.nn.functional as F
from .residual_architectures import BaseResidualNetwork


def normalized_erank(x: torch.Tensor, max_rows: int = 2048) -> float:
    flat = x.detach().float().reshape(-1, x.size(-1))
    if flat.size(0) > max_rows:
        step = max(1, flat.size(0) // max_rows)
        flat = flat[::step][:max_rows]
    normed = F.normalize(flat, p=2, dim=-1, eps=1e-8)
    centered = normed - normed.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered.cpu())
    s_pos = s[s > 1e-7]
    if s_pos.numel() == 0:
        return 0.0
    p = s_pos / s_pos.sum()
    entropy = -(p * torch.log(p + 1e-12)).sum()
    erank = math.exp(float(entropy.item()))
    denom = float(min(flat.size(0), flat.size(1)))
    return float(erank / max(denom, 1.0))


def activation_outlier_ratio(x: torch.Tensor) -> float:
    xf = x.detach().float()
    rms = float(torch.sqrt(xf.pow(2).mean() + 1e-12).item())
    return float(xf.abs().max().item() / max(rms, 1e-12))


def excess_kurtosis(x: torch.Tensor) -> float:
    xf = x.detach().float().reshape(-1)
    mean = xf.mean()
    centered = xf - mean
    var = centered.pow(2).mean().clamp_min(1e-12)
    fourth = centered.pow(4).mean()
    return float((fourth / (var * var) - 3.0).item())


def check_doubly_stochastic(matrix: torch.Tensor) -> Dict[str, float]:
    mf = matrix.detach().float()
    row_err = float((mf.sum(dim=-1) - 1.0).abs().max().item())
    col_err = float((mf.sum(dim=-2) - 1.0).abs().max().item())
    min_val = float(mf.min().item())
    sample = mf.reshape(-1, mf.size(-2), mf.size(-1))[:16]
    sv = torch.linalg.svdvals(sample.cpu())
    max_spectral = float(sv[:, 0].max().item())
    return {
        "max_row_err": row_err,
        "max_col_err": col_err,
        "min_entry": min_val,
        "max_spectral_norm": max_spectral,
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_signal_and_gradient_flow(
    model: BaseResidualNetwork,
    x: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, object]:
    model.zero_grad(set_to_none=True)
    x_req = x.detach().clone().requires_grad_(True)
    out, stats = model(x_req, return_stats=True, apply_final_norm=False)
    out.retain_grad()

    loss = F.mse_loss(out, target)
    loss.backward()

    input_grad_norm = float(x_req.grad.detach().float().norm().item()) if x_req.grad is not None else 0.0
    output_grad_norm = float(out.grad.detach().float().norm().item()) if out.grad is not None else 0.0
    grad_flow_ratio = input_grad_norm / max(output_grad_norm, 1e-12)

    layer_grad_norms: List[float] = []
    for block in model.blocks:
        sq_sum = 0.0
        for p in block.parameters():
            if p.grad is not None:
                sq_sum += float(p.grad.detach().float().pow(2).sum().item())
        layer_grad_norms.append(math.sqrt(sq_sum))

    mean_g = sum(layer_grad_norms) / max(len(layer_grad_norms), 1)
    var_g = sum((g - mean_g) ** 2 for g in layer_grad_norms) / max(len(layer_grad_norms), 1)
    grad_cv = math.sqrt(var_g) / max(mean_g, 1e-12)
    grad_first_to_last = layer_grad_norms[0] / max(layer_grad_norms[-1], 1e-12)

    stream_rms = stats["stream_rms"]
    norm_growth = stream_rms[-1] / max(stream_rms[0], 1e-12)
    dilution_first = stats["dilution_ratio"][0]
    dilution_last = stats["dilution_ratio"][-1]
    dilution_retention = dilution_last / max(dilution_first, 1e-12)

    result: Dict[str, object] = {
        "norm_growth_ratio": norm_growth,
        "final_stream_rms": stream_rms[-1],
        "dilution_first": dilution_first,
        "dilution_last": dilution_last,
        "dilution_retention": dilution_retention,
        "final_outlier_ratio": activation_outlier_ratio(out),
        "final_kurtosis": excess_kurtosis(out),
        "final_erank": normalized_erank(out),
        "input_to_output_grad_ratio": grad_flow_ratio,
        "layer_grad_cv": grad_cv,
        "layer_grad_first_to_last": grad_first_to_last,
        "layer_grad_min": min(layer_grad_norms),
        "layer_grad_max": max(layer_grad_norms),
        "has_nan_or_inf": bool(
            (not torch.isfinite(out).all().item())
            or (x_req.grad is None)
            or (not torch.isfinite(x_req.grad).all().item())
        ),
        "stream_rms_curve": stream_rms,
        "dilution_curve": stats["dilution_ratio"],
        "outlier_curve": stats["outlier_ratio"],
        "layer_grad_norms": layer_grad_norms,
    }
    if "mean_rho" in stats:
        result["mean_rho"] = float(sum(stats["mean_rho"]) / max(len(stats["mean_rho"]), 1))
        result["mean_phi"] = float(sum(stats["mean_phi"]) / max(len(stats["mean_phi"]), 1))
    if "sinkhorn_max_err" in stats:
        result["sinkhorn_max_err"] = float(max(stats["sinkhorn_max_err"]))
    if "memory_slots" in stats:
        result["max_memory_slots"] = float(max(stats["memory_slots"]))
    if "mean_g_res" in stats:
        result["mean_g_res"] = float(sum(stats["mean_g_res"]) / max(len(stats["mean_g_res"]), 1))
        result["mean_g_out"] = float(sum(stats["mean_g_out"]) / max(len(stats["mean_g_out"]), 1))
    return result


def measure_activation_memory_mb(
    model: BaseResidualNetwork,
    x: torch.Tensor,
) -> float:
    param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen_ptrs = set()
    total_bytes = 0

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total_bytes
        storage = tensor.untyped_storage()
        ptr = storage.data_ptr()
        if ptr not in param_ptrs and ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            total_bytes += storage.nbytes()
        return tensor

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    model.train()
    x_req = x.detach().clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        out = model(x_req)
        loss = out.pow(2).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
    return float(total_bytes / (1024.0 * 1024.0))


def benchmark_speed_and_memory(
    model: BaseResidualNetwork,
    x: torch.Tensor,
    target: torch.Tensor,
    warmup: int = 2,
    repeats: int = 6,
) -> Dict[str, float]:
    device = x.device
    model.train()

    activation_mem_mb = measure_activation_memory_mb(model, x)

    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    _sync_device(device)

    peak_device_mem_mb = activation_mem_mb
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    _sync_device(device)
    elapsed_sec = time.perf_counter() - t0

    if device.type == "cuda":
        peak_device_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    elif device.type == "mps":
        peak_device_mem_mb = float(torch.mps.current_allocated_memory() / (1024.0 * 1024.0))

    ms_per_step = (elapsed_sec / max(repeats, 1)) * 1000.0
    samples_per_sec = float(x.size(0) * repeats) / max(elapsed_sec, 1e-12)
    tokens_per_sec = float(x.size(0) * x.size(1) * repeats) / max(elapsed_sec, 1e-12)
    total_params = model.count_total_params()
    overhead_params = model.count_residual_overhead_params()
    overhead_pct = (100.0 * overhead_params) / max(model.count_core_params(), 1)

    return {
        "ms_per_step": ms_per_step,
        "samples_per_sec": samples_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "activation_mem_mb": activation_mem_mb,
        "peak_mem_mb": peak_device_mem_mb,
        "total_params": float(total_params),
        "overhead_params": float(overhead_params),
        "overhead_pct": overhead_pct,
    }
