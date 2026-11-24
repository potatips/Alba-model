import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import (
    activation_outlier_ratio,
    benchmark_speed_and_memory,
    measure_signal_and_gradient_flow,
    normalized_erank,
)
from .residual_architectures import (
    ALL_ARCHITECTURES,
    GeometricOrthogonalResidual,
    build_residual_network,
    clone_shared_block_weights,
)
from .synthetic_data import (
    SyntheticMatrixDataset,
    generate_controlled_alignment_step,
    generate_markov_deep_matrix_dataset,
    generate_outlier_stress_matrix_dataset,
    generate_structured_deep_matrix_dataset,
    make_random_matrix_batch,
)


ARCH_LABELS = {
    "pre_norm": "Pre-Norm (Baseline)",
    "gor": "GOR (0-Param)",
    "gor_bounded": "Bounded GOR (phi<=1, 0-Param)",
    "mhc": "mHC (Birkhoff n=4)",
    "attn_res_block": "Block AttnRes (N=8)",
    "attn_res_full": "Full AttnRes",
    "gated_res": "Gated Residual (n=4)",
}

ARCH_COLORS = {
    "pre_norm": (0.28, 0.34, 0.42),
    "gor": (0.85, 0.22, 0.18),
    "gor_bounded": (0.00, 0.62, 0.58),
    "mhc": (0.12, 0.47, 0.71),
    "attn_res_block": (0.17, 0.63, 0.17),
    "attn_res_full": (0.58, 0.40, 0.74),
    "gated_res": (0.95, 0.55, 0.10),
}


class ResidualRegressionWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.head.weight)
            self.head.weight.add_(torch.randn_like(self.head.weight) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x, return_stats=False, apply_final_norm=True)
        return self.head(h)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_adamw_param_groups(
    model: nn.Module,
    weight_decay: float = 1e-2,
) -> List[Dict[str, object]]:
    decay_params: List[torch.Tensor] = []
    no_decay_params: List[torch.Tensor] = []
    no_decay_suffixes = ("bias", "bias_res", "bias_pre", "bias_post", "bias_out")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(no_decay_suffixes) or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def run_controlled_alignment_experiment(
    depths: List[int],
    dim: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    rhos = [-0.8, -0.4, 0.0, 0.4, 0.8]
    update_scale = 0.15
    gor_op = GeometricOrthogonalResidual(eps=1e-7, k=3.0).to(device)
    gor_bounded_op = GeometricOrthogonalResidual(eps=1e-7, k=3.0, max_phi=1.0).to(device)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for depth in depths:
        depth_key = str(depth)
        results[depth_key] = {}
        for rho in rhos:
            gen_init = torch.Generator().manual_seed(seed)
            x0 = torch.randn(batch_size, seq_len, dim, generator=gen_init).to(device)
            x0_norm = torch.linalg.vector_norm(x0, dim=-1).mean().item()

            x_pre = x0.clone()
            x_gor = x0.clone()
            x_gor_b = x0.clone()
            gen_pre = torch.Generator(device=device).manual_seed(seed + 1000)
            gen_gor = torch.Generator(device=device).manual_seed(seed + 1000)
            gen_gor_b = torch.Generator(device=device).manual_seed(seed + 1000)

            with torch.no_grad():
                for _ in range(depth):
                    u_pre = generate_controlled_alignment_step(
                        x_pre, target_rho=rho, update_scale=update_scale, generator=gen_pre
                    )
                    u_gor = generate_controlled_alignment_step(
                        x_gor, target_rho=rho, update_scale=update_scale, generator=gen_gor
                    )
                    u_gor_b = generate_controlled_alignment_step(
                        x_gor_b, target_rho=rho, update_scale=update_scale, generator=gen_gor_b
                    )
                    x_pre = x_pre + u_pre
                    x_gor = gor_op(x_gor, u_gor)
                    x_gor_b = gor_bounded_op(x_gor_b, u_gor_b)

            pre_ratio = float(torch.linalg.vector_norm(x_pre, dim=-1).mean().item() / max(x0_norm, 1e-12))
            gor_ratio = float(torch.linalg.vector_norm(x_gor, dim=-1).mean().item() / max(x0_norm, 1e-12))
            gor_b_ratio = float(torch.linalg.vector_norm(x_gor_b, dim=-1).mean().item() / max(x0_norm, 1e-12))
            pre_cos = float(F.cosine_similarity(x_pre, x0, dim=-1).mean().item())
            gor_cos = float(F.cosine_similarity(x_gor, x0, dim=-1).mean().item())
            gor_b_cos = float(F.cosine_similarity(x_gor_b, x0, dim=-1).mean().item())
            pre_erank = normalized_erank(x_pre)
            gor_erank = normalized_erank(x_gor)
            gor_b_erank = normalized_erank(x_gor_b)

            results[depth_key][f"{rho:+.1f}"] = {
                "pre_norm_ratio": pre_ratio,
                "gor_norm_ratio": gor_ratio,
                "gor_bounded_norm_ratio": gor_b_ratio,
                "pre_cos_to_x0": pre_cos,
                "gor_cos_to_x0": gor_cos,
                "gor_bounded_cos_to_x0": gor_b_cos,
                "pre_erank": pre_erank,
                "gor_erank": gor_erank,
                "gor_bounded_erank": gor_b_erank,
            }
    return {"update_scale": update_scale, "depths": results}


def run_depth_scaling_experiment(
    depths: List[int],
    dim: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
    speed_repeats: int = 5,
) -> Dict[str, object]:
    x_gauss = make_random_matrix_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        dim=dim,
        distribution="correlated",
        seed=seed,
        device=str(device),
    )
    target = make_random_matrix_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        dim=dim,
        distribution="gaussian",
        seed=seed + 17,
        device=str(device),
    )

    records: Dict[str, Dict[str, Dict[str, object]]] = {}
    for depth in depths:
        depth_key = str(depth)
        records[depth_key] = {}
        torch.manual_seed(seed + depth)
        base_ref = build_residual_network("pre_norm", dim=dim, num_layers=depth).to(device)

        for arch in ALL_ARCHITECTURES:
            torch.manual_seed(seed + depth)
            model = build_residual_network(arch, dim=dim, num_layers=depth).to(device)
            clone_shared_block_weights(base_ref, model)

            flow = measure_signal_and_gradient_flow(model, x_gauss, target)
            perf = benchmark_speed_and_memory(
                model, x_gauss, target, warmup=1, repeats=speed_repeats
            )
            combined: Dict[str, object] = {}
            for k, v in flow.items():
                if not isinstance(v, list):
                    combined[k] = v
            combined.update(perf)
            records[depth_key][arch] = combined
    return records


def train_single_model(
    arch: str,
    base_ref: nn.Module,
    head_init_weight: torch.Tensor,
    dataset: SyntheticMatrixDataset,
    dim: int,
    num_layers: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    eval_every: int = 25,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    backbone = build_residual_network(arch, dim=dim, num_layers=num_layers).to(device)
    clone_shared_block_weights(base_ref, backbone)
    model = ResidualRegressionWrapper(backbone=backbone, dim=dim).to(device)
    with torch.no_grad():
        model.head.weight.copy_(head_init_weight.to(device))

    optimizer = torch.optim.AdamW(
        build_adamw_param_groups(model, weight_decay=1e-2),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))

    x_train = dataset.x_train.to(device)
    y_train = dataset.y_train.to(device)
    x_val = dataset.x_val.to(device)
    y_val = dataset.y_val.to(device)
    n_train = x_train.size(0)

    gen = torch.Generator(device="cpu").manual_seed(seed + 555)
    history_steps: List[int] = []
    val_mse_hist: List[float] = []
    train_mse_hist: List[float] = []
    grad_norms: List[float] = []

    def eval_validation(compute_geometry: bool = False) -> Tuple[float, float, float, float]:
        model.eval()
        with torch.no_grad():
            pred = model(x_val)
            mse = float(F.mse_loss(pred, y_val).item())
            cos = float(F.cosine_similarity(pred, y_val, dim=-1).mean().item())
            if compute_geometry:
                raw_h = model.backbone(x_val, return_stats=False, apply_final_norm=False)
                erank = normalized_erank(raw_h)
                outlier = activation_outlier_ratio(raw_h)
            else:
                erank = 0.0
                outlier = 0.0
        model.train()
        return mse, cos, erank, outlier

    init_val_mse, init_val_cos, _, _ = eval_validation(compute_geometry=False)
    history_steps.append(0)
    val_mse_hist.append(init_val_mse)
    train_mse_hist.append(init_val_mse)

    best_val_mse = init_val_mse
    best_val_cos = init_val_cos
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        idx = torch.randint(0, n_train, (batch_size,), generator=gen)
        xb = x_train[idx]
        yb = y_train[idx]

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        scheduler.step()
        grad_norms.append(gnorm)

        if step % eval_every == 0 or step == steps:
            v_mse, v_cos, _, _ = eval_validation(compute_geometry=False)
            history_steps.append(step)
            val_mse_hist.append(v_mse)
            train_mse_hist.append(float(loss.item()))
            if v_mse < best_val_mse:
                best_val_mse = v_mse
                best_val_cos = v_cos

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed_sec = time.perf_counter() - t0

    final_val_mse, final_val_cos, final_erank, final_outlier = eval_validation(compute_geometry=True)
    mean_grad = sum(grad_norms) / max(len(grad_norms), 1)
    max_grad = max(grad_norms) if grad_norms else 0.0
    total_samples = steps * batch_size

    return {
        "arch": arch,
        "init_val_mse": init_val_mse,
        "final_val_mse": final_val_mse,
        "best_val_mse": best_val_mse,
        "final_val_cos": final_val_cos,
        "best_val_cos": best_val_cos,
        "final_erank": final_erank,
        "final_outlier_ratio": final_outlier,
        "mean_grad_norm": mean_grad,
        "max_grad_norm": max_grad,
        "train_time_sec": elapsed_sec,
        "samples_per_sec": float(total_samples) / max(elapsed_sec, 1e-12),
        "total_params": float(backbone.count_total_params()),
        "overhead_params": float(backbone.count_residual_overhead_params()),
        "history_steps": history_steps,
        "val_mse_hist": val_mse_hist,
        "train_mse_hist": train_mse_hist,
    }


def run_training_regime(
    dataset: SyntheticMatrixDataset,
    dim: int,
    num_layers: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    eval_every: int = 25,
) -> Dict[str, Dict[str, object]]:
    torch.manual_seed(seed)
    base_ref = build_residual_network("pre_norm", dim=dim, num_layers=num_layers).to(device)
    head_init = torch.eye(dim) + torch.randn(dim, dim) * 0.01

    results: Dict[str, Dict[str, object]] = {}
    for arch in ALL_ARCHITECTURES:
        res = train_single_model(
            arch=arch,
            base_ref=base_ref,
            head_init_weight=head_init,
            dataset=dataset,
            dim=dim,
            num_layers=num_layers,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=device,
            eval_every=eval_every,
        )
        results[arch] = res
        print(
            f"  [{dataset.name} L={num_layers} N={int(dataset.metadata['n_train'])}] "
            f"{arch:14s} best_mse={res['best_val_mse']:.5f} cos={res['best_val_cos']:.4f} "
            f"time={res['train_time_sec']:.1f}s",
            flush=True,
        )
    return results


def run_multiseed_markov_regime(
    n_train: int,
    n_val: int,
    seq_len: int,
    dim: int,
    num_layers: int,
    steps: int,
    batch_size: int,
    lr: float,
    seeds: List[int],
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, float]]]:
    per_seed_runs: List[Dict[str, Dict[str, object]]] = []
    for seed_val in seeds:
        ds = generate_markov_deep_matrix_dataset(
            n_train=n_train,
            n_val=n_val,
            seq_len=seq_len,
            dim=dim,
            teacher_depth=10,
            noise_std=0.05,
            seed=seed_val,
        )
        run = run_training_regime(
            dataset=ds,
            dim=dim,
            num_layers=num_layers,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            seed=seed_val,
            device=device,
            eval_every=max(10, steps // 10),
        )
        per_seed_runs.append(run)

    summary: Dict[str, Dict[str, float]] = {}
    for arch in ALL_ARCHITECTURES:
        mses = [float(r[arch]["best_val_mse"]) for r in per_seed_runs]
        coss = [float(r[arch]["best_val_cos"]) for r in per_seed_runs]
        outs = [float(r[arch]["final_outlier_ratio"]) for r in per_seed_runs]
        eranks = [float(r[arch]["final_erank"]) for r in per_seed_runs]
        mean_mse = sum(mses) / len(mses)
        std_mse = math.sqrt(sum((v - mean_mse) ** 2 for v in mses) / len(mses))
        mean_cos = sum(coss) / len(coss)
        std_cos = math.sqrt(sum((v - mean_cos) ** 2 for v in coss) / len(coss))
        summary[arch] = {
            "mean_best_val_mse": mean_mse,
            "std_best_val_mse": std_mse,
            "mean_best_val_cos": mean_cos,
            "std_best_val_cos": std_cos,
            "mean_final_outlier": sum(outs) / len(outs),
            "mean_final_erank": sum(eranks) / len(eranks),
        }
    return per_seed_runs[0], summary


def run_depth_data_scaling_grid(
    dim: int,
    seq_len: int,
    grid_depths: List[int],
    grid_data_sizes: List[int],
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    max_train = max(grid_data_sizes)
    shared_dataset = generate_markov_deep_matrix_dataset(
        n_train=max_train,
        n_val=1024,
        seq_len=seq_len,
        dim=dim,
        teacher_depth=10,
        noise_std=0.05,
        seed=seed,
    )
    grid_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for n_train in grid_data_sizes:
        dataset = shared_dataset.slice_train(n_train)
        for depth in grid_depths:
            key = f"L{depth}_N{n_train}"
            regime = run_training_regime(
                dataset=dataset,
                dim=dim,
                num_layers=depth,
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                seed=seed + depth,
                device=device,
                eval_every=max(20, steps // 5),
            )
            grid_results[key] = {}
            for arch, res in regime.items():
                grid_results[key][arch] = {
                    "best_val_mse": float(res["best_val_mse"]),
                    "final_val_mse": float(res["final_val_mse"]),
                    "best_val_cos": float(res["best_val_cos"]),
                    "train_time_sec": float(res["train_time_sec"]),
                    "final_erank": float(res["final_erank"]),
                }
    return grid_results


def render_benchmark_figure(results: Dict[str, object], output_png: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle(
        "Synthetic Random Matrix Benchmark: Pre-Norm vs GOR vs Bounded GOR vs mHC vs AttnRes vs Gated Residual (2026)",
        fontsize=14,
        fontweight="bold",
    )

    depth_data = results["depth_scaling"]
    depths = sorted(int(k) for k in depth_data.keys())

    ax = axes[0, 0]
    for arch in ALL_ARCHITECTURES:
        ys = [depth_data[str(d)][arch]["norm_growth_ratio"] for d in depths]
        ax.plot(
            depths,
            ys,
            marker="o",
            linewidth=2.0,
            color=ARCH_COLORS[arch],
            label=ARCH_LABELS[arch],
        )
    ax.set_title("1. Residual Norm Growth (||x_L|| / ||x_0||)")
    ax.set_xlabel("Depth (Layers L)")
    ax.set_ylabel("Norm Growth Ratio")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[0, 1]
    for arch in ALL_ARCHITECTURES:
        ys = [depth_data[str(d)][arch]["activation_mem_mb"] for d in depths]
        ax.plot(
            depths,
            ys,
            marker="s",
            linewidth=2.0,
            color=ARCH_COLORS[arch],
            label=ARCH_LABELS[arch],
        )
    ax.set_title("2. Saved Activation Memory (MB) vs Depth")
    ax.set_xlabel("Depth (Layers L)")
    ax.set_ylabel("Saved Activations (MB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[0, 2]
    for arch in ALL_ARCHITECTURES:
        ys = [depth_data[str(d)][arch]["ms_per_step"] for d in depths]
        ax.plot(
            depths,
            ys,
            marker="^",
            linewidth=2.0,
            color=ARCH_COLORS[arch],
            label=ARCH_LABELS[arch],
        )
    ax.set_title("3. Forward+Backward Latency (ms/step)")
    ax.set_xlabel("Depth (Layers L)")
    ax.set_ylabel("Latency (ms / step)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[0, 3]
    align_depths = results["controlled_alignment"]["depths"]
    max_align_d = str(max(int(k) for k in align_depths.keys()))
    rows = align_depths[max_align_d]
    rho_keys = list(rows.keys())
    rho_vals = [float(k) for k in rho_keys]
    pre_norms = [max(rows[k]["pre_norm_ratio"], 1e-6) for k in rho_keys]
    gor_norms = [max(rows[k]["gor_norm_ratio"], 1e-6) for k in rho_keys]
    gor_b_norms = [max(rows[k]["gor_bounded_norm_ratio"], 1e-6) for k in rho_keys]
    ax.plot(
        rho_vals,
        pre_norms,
        marker="o",
        linewidth=2.6,
        color=ARCH_COLORS["pre_norm"],
        label="Pre-Norm",
    )
    ax.plot(
        rho_vals,
        gor_norms,
        marker="D",
        linewidth=2.2,
        color=ARCH_COLORS["gor"],
        label="GOR (0-Param)",
    )
    ax.plot(
        rho_vals,
        gor_b_norms,
        marker="^",
        linestyle="--",
        linewidth=2.4,
        color=ARCH_COLORS["gor_bounded"],
        label="Bounded GOR (phi<=1)",
    )
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.0, alpha=0.7)
    ax.set_yscale("log")
    ax.set_title(f"4. Controlled Alignment L={max_align_d} (||x_L||/||x_0||)")
    ax.set_xlabel("Target Update Cosine (rho)")
    ax.set_ylabel("Norm Ratio (Log Scale)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    reg_m = results["training_markov"]
    for arch in ALL_ARCHITECTURES:
        steps = reg_m[arch]["history_steps"]
        vals = reg_m[arch]["val_mse_hist"]
        ax.plot(
            steps,
            vals,
            linewidth=2.0,
            color=ARCH_COLORS[arch],
            label=f"{ARCH_LABELS[arch]} ({reg_m[arch]['best_val_mse']:.4f})",
        )
    ax.set_title("5. Regime A: Neutral Markovian Deep Task")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Validation MSE (Lower is Better)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.2)

    ax = axes[1, 1]
    reg_s = results["training_structured"]
    for arch in ALL_ARCHITECTURES:
        steps = reg_s[arch]["history_steps"]
        vals = reg_s[arch]["val_mse_hist"]
        ax.plot(
            steps,
            vals,
            linewidth=2.0,
            color=ARCH_COLORS[arch],
            label=f"{ARCH_LABELS[arch]} ({reg_s[arch]['best_val_mse']:.4f})",
        )
    ax.set_title("6. Regime B: Multi-Scale Skip Matrix Task")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Validation MSE (Lower is Better)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.2)

    ax = axes[1, 2]
    reg_o = results["training_outlier"]
    for arch in ALL_ARCHITECTURES:
        steps = reg_o[arch]["history_steps"]
        vals = reg_o[arch]["val_mse_hist"]
        ax.plot(
            steps,
            vals,
            linewidth=2.0,
            color=ARCH_COLORS[arch],
            label=f"{ARCH_LABELS[arch]} ({reg_o[arch]['best_val_mse']:.4f})",
        )
    ax.set_title("7. Regime C: Heavy-Tail Outlier Stress Task")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Validation MSE (Lower is Better)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.2)

    ax = axes[1, 3]
    grid = results["scaling_grid"]
    grid_keys = list(grid.keys())
    x_pos = list(range(len(grid_keys)))
    n_arch = len(ALL_ARCHITECTURES)
    width = 0.80 / max(n_arch, 1)
    max_h = 0.0
    for idx, arch in enumerate(ALL_ARCHITECTURES):
        offsets = [p + (idx - (n_arch - 1) / 2.0) * width for p in x_pos]
        heights = [grid[k][arch]["best_val_mse"] for k in grid_keys]
        max_h = max(max_h, max(heights))
        ax.bar(
            offsets,
            heights,
            width=width,
            color=ARCH_COLORS[arch],
            label=ARCH_LABELS[arch],
        )
    ax.set_title("8. Shared-Teacher Depth x Data Grid (Markov)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grid_keys, fontsize=8.5)
    ax.set_ylabel("Best Validation MSE (Lower is Better)")
    ax.set_ylim(0.0, max_h * 1.40)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2, loc="upper center")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.close(fig)


def write_markdown_report(results: Dict[str, object], report_path: str) -> None:
    lines: List[str] = []
    meta = results["config"]
    lines.append("# Benchmark Thực Nghiệm: Pre-Norm vs GOR vs Bounded GOR vs mHC vs AttnRes vs Gated Residual (2026)")
    lines.append("")
    lines.append(
        f"- Device: `{meta['device']}` | Matrix dim `d={meta['dim']}` | Seq len `T={meta['seq_len']}` | "
        f"Seeds: `{meta['eval_seeds']}`"
    )
    lines.append("")

    lines.append("## 1. Controlled Cosine-Alignment Stress Test (Pre-Norm vs GOR vs Bounded GOR)")
    lines.append("")
    align = results["controlled_alignment"]["depths"]
    for depth_key, rows in align.items():
        lines.append(f"### Depth L = {depth_key}")
        lines.append(
            "| Target rho | Pre-Norm ||x_L||/||x_0|| | GOR ||x_L||/||x_0|| | Bounded GOR ||x_L||/||x_0|| | Pre-Norm cos(x_L,x_0) | GOR cos(x_L,x_0) | Bounded GOR cos(x_L,x_0) | Pre-Norm eRank | GOR eRank | Bounded GOR eRank |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for rho_key, vals in rows.items():
            lines.append(
                f"| `{rho_key}` | {vals['pre_norm_ratio']:.4f} | {vals['gor_norm_ratio']:.4f} | "
                f"{vals['gor_bounded_norm_ratio']:.4f} | {vals['pre_cos_to_x0']:.4f} | "
                f"{vals['gor_cos_to_x0']:.4f} | {vals['gor_bounded_cos_to_x0']:.4f} | "
                f"{vals['pre_erank']:.4f} | {vals['gor_erank']:.4f} | {vals['gor_bounded_erank']:.4f} |"
            )
        lines.append("")

    lines.append("## 2. Depth Scaling: Signal Propagation, Memory & Speed (L = 8 -> 128)")
    lines.append("")
    depth_data = results["depth_scaling"]
    max_depth = str(max(int(k) for k in depth_data.keys()))
    lines.append(f"### Snapshot at Maximum Depth L = {max_depth}")
    lines.append(
        "| Architecture | Norm Growth | Dilution Retention | eRank | Outlier Ratio | Grad Flow (in/out) | Grad CV | Act Mem (MB) | Latency (ms) | Extra Params (%) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arch in ALL_ARCHITECTURES:
        r = depth_data[max_depth][arch]
        lines.append(
            f"| {ARCH_LABELS[arch]} | {r['norm_growth_ratio']:.3f} | {r['dilution_retention']:.3f} | "
            f"{r['final_erank']:.3f} | {r['final_outlier_ratio']:.2f} | "
            f"{r['input_to_output_grad_ratio']:.3f} | {r['layer_grad_cv']:.3f} | "
            f"{r['activation_mem_mb']:.2f} | {r['ms_per_step']:.2f} | {r['overhead_pct']:.2f}% |"
        )
    lines.append("")

    lines.append("## 3. Supervised Deep Training on Synthetic Random Matrix Datasets")
    lines.append("")

    ms = results["training_markov_multiseed"]
    base_ms = ms["pre_norm"]["mean_best_val_mse"]
    lines.append(
        f"### Regime A (Multi-Seed, n={len(meta['eval_seeds'])}): Neutral Sequential Markovian Deep Regression"
    )
    lines.append(
        "| Architecture | Mean Best Val MSE ± Std | Delta vs Pre-Norm | Mean Val Cosine ± Std | Mean eRank | Mean Outlier |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arch in ALL_ARCHITECTURES:
        r = ms[arch]
        delta_pct = 100.0 * (r["mean_best_val_mse"] - base_ms) / max(base_ms, 1e-12)
        lines.append(
            f"| {ARCH_LABELS[arch]} | {r['mean_best_val_mse']:.5f} ± {r['std_best_val_mse']:.5f} | "
            f"{delta_pct:+.2f}% | {r['mean_best_val_cos']:.4f} ± {r['std_best_val_cos']:.4f} | "
            f"{r['mean_final_erank']:.3f} | {r['mean_final_outlier']:.2f} |"
        )
    lines.append("")

    for key, title in [
        ("training_markov", "Regime A (Seed 0 Detail): Neutral Sequential Markovian Deep Regression"),
        ("training_structured", "Regime B: Multi-Scale Skip Matrix Regression"),
        ("training_outlier", "Regime C: Heavy-Tail Outlier Stress Regression"),
    ]:
        lines.append(f"### {title}")
        reg = results[key]
        base_best = reg["pre_norm"]["best_val_mse"]
        lines.append(
            "| Architecture | Best Val MSE | Delta vs Pre-Norm | Best Val Cosine | Post-Train eRank | Post-Train Outlier | Mean Grad Norm | Train Time (s) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arch in ALL_ARCHITECTURES:
            r = reg[arch]
            delta_pct = 100.0 * (r["best_val_mse"] - base_best) / max(base_best, 1e-12)
            lines.append(
                f"| {ARCH_LABELS[arch]} | {r['best_val_mse']:.5f} | {delta_pct:+.2f}% | "
                f"{r['best_val_cos']:.4f} | {r['final_erank']:.3f} | {r['final_outlier_ratio']:.2f} | "
                f"{r['mean_grad_norm']:.3f} | {r['train_time_sec']:.2f}s |"
            )
        lines.append("")

    lines.append("## 4. Shared-Teacher Depth x Data Scale Matrix (Neutral Markovian Task, Best Val MSE)")
    lines.append("")
    grid = results["scaling_grid"]
    grid_keys = list(grid.keys())
    header = "| Architecture | " + " | ".join(grid_keys) + " |"
    sep = "|---|" + "|".join(["---:" for _ in grid_keys]) + "|"
    lines.append(header)
    lines.append(sep)
    for arch in ALL_ARCHITECTURES:
        row_vals = [f"{grid[k][arch]['best_val_mse']:.5f}" for k in grid_keys]
        lines.append(f"| {ARCH_LABELS[arch]} | " + " | ".join(row_vals) + " |")
    lines.append("")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic Random Matrix Benchmark for Residual Architectures (Pre-Norm, GOR, Bounded GOR, mHC, AttnRes, Gated Residual)"
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--train-layers", type=int, default=24)
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--train-steps", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.quick:
        depths_prop = [8, 16, 32]
        align_depths = [32]
        train_samples = 1024
        train_steps = 40
        train_layers = 8
        grid_depths = [8, 16]
        grid_data = [512, 2048]
        grid_steps = 30
        speed_repeats = 2
        eval_seeds = [args.seed]
    else:
        depths_prop = [8, 16, 32, 64, 128]
        align_depths = [64, 128]
        train_samples = args.train_samples
        train_steps = args.train_steps
        train_layers = args.train_layers
        grid_depths = [12, 32]
        grid_data = [2048, 16384]
        grid_steps = 120
        speed_repeats = 4
        eval_seeds = [args.seed + 1000 * i for i in range(max(1, args.num_seeds))]

    print(f"Running Residual Architecture Benchmark on device={device} (dim={args.dim}, seq_len={args.seq_len})")

    t_all = time.perf_counter()

    print("[1/4] Controlled Cosine-Alignment Stress Test...")
    controlled_alignment = run_controlled_alignment_experiment(
        depths=align_depths,
        dim=args.dim,
        batch_size=32,
        seq_len=args.seq_len,
        seed=args.seed,
        device=device,
    )

    print(f"[2/4] Deep Signal Propagation & Speed/Memory Sweep (L={depths_prop})...")
    depth_scaling = run_depth_scaling_experiment(
        depths=depths_prop,
        dim=args.dim,
        batch_size=32,
        seq_len=args.seq_len,
        seed=args.seed,
        device=device,
        speed_repeats=speed_repeats,
    )

    print(
        f"[3/4] Deep Supervised Training (L={train_layers}, N_train={train_samples}, steps={train_steps}, seeds={eval_seeds})..."
    )
    training_markov, training_markov_multiseed = run_multiseed_markov_regime(
        n_train=train_samples,
        n_val=1024,
        seq_len=args.seq_len,
        dim=args.dim,
        num_layers=train_layers,
        steps=train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seeds=eval_seeds,
        device=device,
    )

    ds_structured = generate_structured_deep_matrix_dataset(
        n_train=train_samples,
        n_val=1024,
        seq_len=args.seq_len,
        dim=args.dim,
        teacher_depth=8,
        noise_std=0.05,
        seed=args.seed,
    )
    training_structured = run_training_regime(
        dataset=ds_structured,
        dim=args.dim,
        num_layers=train_layers,
        steps=train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        eval_every=max(10, train_steps // 10),
    )

    ds_outlier = generate_outlier_stress_matrix_dataset(
        n_train=train_samples,
        n_val=1024,
        seq_len=args.seq_len,
        dim=args.dim,
        teacher_depth=8,
        outlier_prob=0.02,
        outlier_scale=12.0,
        noise_std=0.08,
        seed=args.seed + 101,
    )
    training_outlier = run_training_regime(
        dataset=ds_outlier,
        dim=args.dim,
        num_layers=train_layers,
        steps=train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 101,
        device=device,
        eval_every=max(10, train_steps // 10),
    )

    print(f"[4/4] Shared-Teacher Multi-Depth x Multi-Data Scaling Grid (L={grid_depths} x N={grid_data})...")
    scaling_grid = run_depth_data_scaling_grid(
        dim=args.dim,
        seq_len=args.seq_len,
        grid_depths=grid_depths,
        grid_data_sizes=grid_data,
        steps=grid_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 303,
        device=device,
    )

    total_sec = time.perf_counter() - t_all

    payload = {
        "config": {
            "device": str(device),
            "dim": args.dim,
            "seq_len": args.seq_len,
            "train_layers": train_layers,
            "train_samples": train_samples,
            "train_steps": train_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "eval_seeds": eval_seeds,
            "depths_prop": depths_prop,
            "grid_depths": grid_depths,
            "grid_data": grid_data,
            "total_runtime_sec": total_sec,
        },
        "controlled_alignment": controlled_alignment,
        "depth_scaling": depth_scaling,
        "training_markov": training_markov,
        "training_markov_multiseed": training_markov_multiseed,
        "training_structured": training_structured,
        "training_outlier": training_outlier,
        "scaling_grid": scaling_grid,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    png_path = os.path.join(args.output_dir, "residual_benchmark_2026.png")
    md_path = os.path.join(args.output_dir, "BENCHMARK_REPORT.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    render_benchmark_figure(payload, png_path)
    write_markdown_report(payload, md_path)

    print(f"Benchmark completed in {total_sec:.1f}s")
    print(f"JSON:   {json_path}")
    print(f"Report: {md_path}")
    print(f"Figure: {png_path}")


if __name__ == "__main__":
    main()
