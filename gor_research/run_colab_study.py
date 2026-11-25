import argparse
import base64
import contextlib
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_pipeline import TokenDatasetSplit, build_pretraining_dataset, build_synthetic_lm_split
from .residuals import (
    ALL_LM_RESIDUALS,
    RESIDUAL_LABELS,
    BlockAttnResConnector,
    BlockAttnResGORv2Connector,
    DualBankGORv2Connector,
    GORAntiDiluteResidual,
    GORBoundedResidual,
    GORv2Residual,
    GORv3SubspaceConnector,
    PreNormResidual,
    RMSNorm,
    StdDualBankConnector,
    SwiGLU,
)
from .transformer_lm import (
    FromScratchTransformerLM,
    TransformerConfig,
    clone_transformer_backbone,
)


ARCH_COLORS = {
    "pre_norm": "#475569",
    "gor_v2": "#dc2626",
    "gor_bounded": "#ea580c",
    "gor_antidilute": "#0d9488",
    "std_dual_bank": "#2563eb",
    "dual_bank_gor_v2": "#0284c7",
    "gor_v3_subspace": "#7c3aed",
    "attn_res_block": "#16a34a",
    "attn_res_gor_v2": "#be185d",
}


class MatrixResidualBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int, residual_type: str, gor_k: float = 3.0):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.residual_type = residual_type
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(num_layers)])
        hidden_dim = max(64, ((int(2 * (4 * dim) / 3) + 31) // 32) * 32)
        self.blocks = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(num_layers)])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, dim, bias=False)

        if residual_type == "pre_norm":
            self.res_op = PreNormResidual()
            self.connector = None
        elif residual_type == "gor_v2":
            self.res_op = GORv2Residual(k=gor_k)
            self.connector = None
        elif residual_type == "gor_bounded":
            self.res_op = GORBoundedResidual(k=gor_k, max_phi=1.0)
            self.connector = None
        elif residual_type == "gor_antidilute":
            self.res_op = GORAntiDiluteResidual(k=gor_k, max_phi=1.0, max_boost=2.0)
            self.connector = None
        elif residual_type == "std_dual_bank":
            self.res_op = None
            self.connector = StdDualBankConnector(dim, num_sublayers=num_layers)
        elif residual_type == "dual_bank_gor_v2":
            self.res_op = None
            self.connector = DualBankGORv2Connector(dim, num_sublayers=num_layers, k=gor_k)
        elif residual_type == "gor_v3_subspace":
            self.res_op = None
            self.connector = GORv3SubspaceConnector(dim, num_sublayers=num_layers, k=gor_k)
        elif residual_type == "attn_res_block":
            self.res_op = None
            self.connector = BlockAttnResConnector(dim, num_sublayers=num_layers, num_blocks=8)
        elif residual_type == "attn_res_gor_v2":
            self.res_op = None
            self.connector = BlockAttnResGORv2Connector(dim, num_sublayers=num_layers, num_blocks=8, k=gor_k)
        else:
            raise ValueError(f"Unknown residual_type: {residual_type}")

        for norm, block in zip(self.norms, self.blocks):
            for mod in block.modules():
                if isinstance(mod, nn.Linear):
                    eff_std = 0.75 / math.sqrt(float(mod.weight.size(1)))
                    nn.init.normal_(mod.weight, mean=0.0, std=eff_std)
        with torch.no_grad():
            nn.init.eye_(self.head.weight)
            self.head.weight.add_(torch.randn_like(self.head.weight) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.connector is None:
            h = x
            x0_norm = torch.sqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) + 1e-7)
            for norm, block in zip(self.norms, self.blocks):
                u = block(norm(h))
                h = self.res_op(h, u, x0_norm=x0_norm)
        else:
            state = self.connector.init_state(x)
            for idx, (norm, block) in enumerate(zip(self.norms, self.blocks)):
                h_in = self.connector.read(state, idx)
                u = block(norm(h_in))
                self.connector.update(state, u, idx)
            h = self.connector.finalize(state)
        return self.head(self.final_norm(h))

    def count_overhead_params(self) -> int:
        if self.connector is None:
            return 0
        return sum(p.numel() for p in self.connector.parameters() if p.requires_grad)


def _make_orthogonal_matrix(dim: int, gen: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=gen))
    sign = torch.sign(torch.diag(r))
    sign[sign == 0] = 1.0
    return q * sign.unsqueeze(0)


def build_matrix_dataset(
    n_train: int,
    n_val: int,
    seq_len: int,
    dim: int,
    outlier_prob: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    total = n_train + n_val
    basis = _make_orthogonal_matrix(dim, gen)
    spec = torch.linspace(1.6, 0.4, dim)
    z = torch.randn(total, seq_len, dim, generator=gen)
    x = z @ (basis * spec.unsqueeze(0)).t()
    if outlier_prob > 0.0:
        mask = (torch.rand(total, seq_len, dim, generator=gen) < outlier_prob).float()
        x = x + mask * torch.randn(total, seq_len, dim, generator=gen) * 12.0
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    t_gen = torch.Generator().manual_seed(seed + 7919)
    h = x.clone()
    for _ in range(8):
        wa = _make_orthogonal_matrix(dim, t_gen)
        wb = _make_orthogonal_matrix(dim, t_gen)
        wo = _make_orthogonal_matrix(dim, t_gen)
        nh = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        u = (F.silu(nh @ wa) * (nh @ wb)) @ wo
        h = h + 0.35 * u
    y_clean = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    y_noisy = y_clean + torch.randn(y_clean.shape, generator=gen) * 0.05
    return (
        x[:n_train].contiguous(),
        y_noisy[:n_train].contiguous(),
        x[n_train:].contiguous(),
        y_clean[n_train:].contiguous(),
    )


def build_adamw_groups(model: nn.Module, weight_decay: float = 0.01) -> List[Dict[str, object]]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "gamma" in name or "q_read" in name or "depth_queries" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def run_stage1_matrix_benchmark(
    variants: List[str],
    device: torch.device,
    dim: int = 128,
    num_layers: int = 24,
    n_train: int = 4096,
    steps: int = 200,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    regimes = {
        "structured_l24": build_matrix_dataset(n_train, 1024, 8, dim, outlier_prob=0.0, seed=seed),
        "outlier_l24": build_matrix_dataset(n_train, 1024, 8, dim, outlier_prob=0.02, seed=seed + 101),
    }
    output: Dict[str, Dict[str, Dict[str, float]]] = {}

    for regime_name, (xt, yt, xv, yv) in regimes.items():
        xt_d, yt_d = xt.to(device), yt.to(device)
        xv_d, yv_d = xv.to(device), yv.to(device)

        torch.manual_seed(seed)
        ref = MatrixResidualBlock(dim, num_layers, "pre_norm").to(device)
        output[regime_name] = {}
        print(f"[Stage 1] Matrix Benchmark: {regime_name} (L={num_layers}, d={dim}, steps={steps})", flush=True)

        for arch in variants:
            torch.manual_seed(seed)
            model = MatrixResidualBlock(dim, num_layers, arch).to(device)
            with torch.no_grad():
                model.norms.load_state_dict(ref.norms.state_dict())
                model.blocks.load_state_dict(ref.blocks.state_dict())
                model.final_norm.load_state_dict(ref.final_norm.state_dict())
                model.head.weight.copy_(ref.head.weight)

            opt = torch.optim.AdamW(build_adamw_groups(model, 1e-2), lr=1.5e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
            b_gen = torch.Generator(device="cpu").manual_seed(seed + 555)
            best_mse, best_cos = 1e9, -1.0
            t0 = time.perf_counter()

            for step in range(1, steps + 1):
                idx = torch.randint(0, xt_d.size(0), (batch_size,), generator=b_gen)
                opt.zero_grad(set_to_none=True)
                pred = model(xt_d[idx])
                loss = F.mse_loss(pred, yt_d[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                if step % max(10, steps // 10) == 0 or step == steps:
                    model.eval()
                    with torch.no_grad():
                        vp = model(xv_d)
                        vmse = float(F.mse_loss(vp, yv_d).item())
                        vcos = float(F.cosine_similarity(vp, yv_d, dim=-1).mean().item())
                        if vmse < best_mse:
                            best_mse, best_cos = vmse, vcos
                    model.train()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - t0
            output[regime_name][arch] = {
                "best_val_mse": best_mse,
                "best_val_cos": best_cos,
                "overhead_params": int(model.count_overhead_params()),
                "train_time_sec": elapsed,
            }
            print(
                f"  {arch:16s} | best_mse={best_mse:.5f} | cos={best_cos:.4f} | oh={model.count_overhead_params():6d} | {elapsed:.1f}s",
                flush=True,
            )

    return output


@torch.no_grad()
def evaluate_lm_split(
    model: FromScratchTransformerLM,
    blocks: torch.Tensor,
    bytes_per_token: float,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 40,
    collect_geom: bool = False,
) -> Tuple[float, float, float, Optional[Dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    geom_accum: Optional[Dict[str, List[float]]] = None

    n_blocks = blocks.size(0)
    step = max(1, (n_blocks + batch_size - 1) // batch_size)
    eval_batches = min(step, max_batches)

    for b_idx in range(eval_batches):
        start = b_idx * batch_size
        end = min(n_blocks, start + batch_size)
        batch = blocks[start:end].to(device, non_blocking=True)
        x = batch[:, :-1]
        y = batch[:, 1:]
        want_geom = collect_geom and (b_idx == 0)
        ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
        with ctx:
            _, loss, geom = model(x, targets=y, collect_geometry=want_geom)
        ntok = int(y.numel())
        total_loss += float(loss.item()) * ntok
        total_tokens += ntok
        if want_geom and geom is not None:
            geom_accum = {k: list(v) for k, v in geom.items()}

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    bpb = avg_loss / (math.log(2.0) * max(bytes_per_token, 1e-6))
    model.train()
    return avg_loss, ppl, bpb, geom_accum


def run_stage2_lm_pretraining(
    variants: List[str],
    dataset: TokenDatasetSplit,
    device: torch.device,
    dim: int = 384,
    n_layers: int = 16,
    n_heads: int = 6,
    seq_len: int = 256,
    steps: int = 700,
    batch_size: int = 32,
    grad_accum: int = 2,
    lr: float = 6e-4,
    seed: int = 42,
) -> Dict[str, Dict[str, object]]:
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = amp_enabled and (amp_dtype == torch.float16)

    ref_cfg = TransformerConfig(
        vocab_size=dataset.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=seq_len,
        residual_type="pre_norm",
    )
    torch.manual_seed(seed)
    ref_model = FromScratchTransformerLM(ref_cfg)

    results: Dict[str, Dict[str, object]] = {}
    eval_every = max(25, steps // 10)
    warmup_steps = max(20, int(0.08 * steps))

    print(
        f"\n[Stage 2] From-Scratch Causal LM Pretraining: L={n_layers}, d={dim}, H={n_heads}, T={seq_len}, "
        f"steps={steps}, eff_bs={batch_size * grad_accum}, amp={amp_dtype if amp_enabled else 'fp32'}",
        flush=True,
    )

    for arch in variants:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        cfg = TransformerConfig(
            vocab_size=dataset.vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=seq_len,
            residual_type=arch,
        )
        torch.manual_seed(seed)
        model = FromScratchTransformerLM(cfg).to(device)
        clone_transformer_backbone(ref_model, model)

        opt = torch.optim.AdamW(build_adamw_groups(model, 0.1), lr=lr, betas=(0.9, 0.95), eps=1e-8)
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        def lr_lambda(step_idx: int) -> float:
            if step_idx < warmup_steps:
                return float(step_idx + 1) / float(max(warmup_steps, 1))
            progress = float(step_idx - warmup_steps) / float(max(steps - warmup_steps, 1))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        b_gen = torch.Generator(device="cpu").manual_seed(seed + 2026)
        n_train_blocks = dataset.train_blocks.size(0)
        hist_steps: List[int] = []
        hist_val_loss: List[float] = []
        hist_val_bpb: List[float] = []
        best_val_loss = 1e9
        best_val_bpb = 1e9

        init_vloss, _, init_vbpb, _ = evaluate_lm_split(
            model,
            dataset.val_blocks,
            dataset.val_bytes_per_token,
            device,
            batch_size=batch_size,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_batches=24,
        )
        hist_steps.append(0)
        hist_val_loss.append(init_vloss)
        hist_val_bpb.append(init_vbpb)

        t0 = time.perf_counter()
        total_train_tokens = 0
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                idx = torch.randint(0, n_train_blocks, (batch_size,), generator=b_gen)
                batch = dataset.train_blocks[idx].to(device, non_blocking=True)
                xb = batch[:, :-1]
                yb = batch[:, 1:]
                total_train_tokens += int(yb.numel())
                ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
                with ctx:
                    _, loss, _ = model(xb, targets=yb)
                    scaled_loss = loss / float(grad_accum)
                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if use_scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()

            if step % eval_every == 0 or step == steps:
                vloss, _, vbpb, _ = evaluate_lm_split(
                    model,
                    dataset.val_blocks,
                    dataset.val_bytes_per_token,
                    device,
                    batch_size=batch_size,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    max_batches=32,
                )
                hist_steps.append(step)
                hist_val_loss.append(vloss)
                hist_val_bpb.append(vbpb)
                if vloss < best_val_loss:
                    best_val_loss = vloss
                    best_val_bpb = vbpb

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_vram_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        else:
            peak_vram_gb = 0.0
        elapsed = time.perf_counter() - t0
        tok_per_sec = float(total_train_tokens) / max(elapsed, 1e-6)

        final_vloss, final_vppl, final_vbpb, geom = evaluate_lm_split(
            model,
            dataset.val_blocks,
            dataset.val_bytes_per_token,
            device,
            batch_size=batch_size,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_batches=48,
            collect_geom=True,
        )
        ood_loss, ood_ppl, ood_bpb, _ = evaluate_lm_split(
            model,
            dataset.ood_blocks,
            dataset.ood_bytes_per_token,
            device,
            batch_size=batch_size,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_batches=48,
        )

        conflict_list = geom["conflict_rates"] if geom else [0.0]
        growth_list = geom["norm_growth"] if geom else [1.0]
        dilution_list = geom["dilution_ratios"] if geom else [1.0]
        renewal_list = geom["depth_renewal"] if geom else [0.0]

        avg_conflict = sum(conflict_list) / max(len(conflict_list), 1)
        final_norm_growth = growth_list[-1]
        dilution_retention = dilution_list[-1] / max(dilution_list[0], 1e-7)
        avg_renewal = sum(renewal_list) / max(len(renewal_list), 1)

        results[arch] = {
            "label": RESIDUAL_LABELS.get(arch, arch),
            "total_params": int(model.count_total_params()),
            "overhead_params": int(model.count_residual_overhead_params()),
            "best_val_loss": min(best_val_loss, final_vloss),
            "final_val_loss": final_vloss,
            "final_val_ppl": final_vppl,
            "final_val_bpb": final_vbpb,
            "best_val_bpb": min(best_val_bpb, final_vbpb),
            "ood_loss": ood_loss,
            "ood_ppl": ood_ppl,
            "ood_bpb": ood_bpb,
            "avg_conflict_pct": avg_conflict,
            "norm_growth": final_norm_growth,
            "dilution_retention": dilution_retention,
            "avg_depth_renewal": avg_renewal,
            "tok_per_sec": tok_per_sec,
            "peak_vram_gb": peak_vram_gb,
            "train_time_sec": elapsed,
            "hist_steps": hist_steps,
            "hist_val_loss": hist_val_loss,
            "hist_val_bpb": hist_val_bpb,
            "layer_conflict": conflict_list,
            "layer_norm_growth": growth_list,
            "layer_renewal": renewal_list,
        }
        print(
            f"  {arch:16s} | Val Loss={final_vloss:.4f} | PPL={final_vppl:.2f} | BPB={final_vbpb:.4f} | "
            f"OOD BPB={ood_bpb:.4f} | Renewal={avg_renewal:.4f} | {tok_per_sec:,.0f} tok/s | {elapsed:.1f}s",
            flush=True,
        )

    return results


def render_report_assets(
    payload: Dict[str, object],
    png_path: str,
    html_path: str,
) -> None:
    stage1 = payload["stage1_matrix"]
    stage2 = payload["stage2_lm"]
    variants = list(stage2.keys())

    fig, axs = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "GOR From-Scratch Research Study: Matrix Depth Ablations & Causal Transformer LM Pretraining",
        fontsize=15,
        fontweight="bold",
    )

    ax = axs[0, 0]
    struct_vals = [stage1["structured_l24"][v]["best_val_mse"] for v in variants]
    out_vals = [stage1["outlier_l24"][v]["best_val_mse"] for v in variants]
    x_idx = range(len(variants))
    ax.bar([i - 0.18 for i in x_idx], struct_vals, width=0.36, color="#2563eb", label="Structured L=24 MSE")
    ax.bar([i + 0.18 for i in x_idx], out_vals, width=0.36, color="#f97316", label="Outlier L=24 MSE")
    ax.set_xticks(list(x_idx))
    ax.set_xticklabels(variants, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Best Val MSE (Lower Better)")
    ax.set_title("Panel 1: Stage 1 Deep Matrix Benchmark")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    for v in variants:
        r = stage2[v]
        ax.plot(r["hist_steps"], r["hist_val_bpb"], marker="o", markersize=3, linewidth=2, color=ARCH_COLORS.get(v, "#333333"), label=v)
    ax.set_xlabel("Pretraining Step")
    ax.set_ylabel("Validation BPB (Lower Better)")
    ax.set_title("Panel 2: From-Scratch LM Validation BPB Curve")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    ax = axs[0, 2]
    in_bpb = [stage2[v]["final_val_bpb"] for v in variants]
    ood_bpb = [stage2[v]["ood_bpb"] for v in variants]
    ax.bar([i - 0.18 for i in x_idx], in_bpb, width=0.36, color="#0d9488", label="WikiText Val BPB")
    ax.bar([i + 0.18 for i in x_idx], ood_bpb, width=0.36, color="#7c3aed", label="External OOD BPB")
    ax.set_xticks(list(x_idx))
    ax.set_xticklabels(variants, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Bits-Per-Byte (Lower Better)")
    ax.set_title("Panel 3: Final In-Dist vs OOD BPB")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    for v in variants:
        g = stage2[v]["layer_norm_growth"]
        ax.plot(range(len(g)), g, linewidth=1.8, color=ARCH_COLORS.get(v, "#333333"), label=v)
    ax.set_xlabel("Sublayer Index (Attn + FFN)")
    ax.set_ylabel("||h_l|| / ||h_0||")
    ax.set_title("Panel 4: Residual Norm Growth Across Depth")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    for v in variants:
        ren = stage2[v]["layer_renewal"]
        ax.plot(range(len(ren)), ren, linewidth=1.8, color=ARCH_COLORS.get(v, "#333333"), label=v)
    ax.set_xlabel("Sublayer Index")
    ax.set_ylabel("1 - CKA(h_{l-1}, h_l)")
    ax.set_title("Panel 5: Depth Representation Renewal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    ax = axs[1, 2]
    for v in variants:
        r = stage2[v]
        ax.scatter(r["tok_per_sec"], r["final_val_bpb"], s=130, color=ARCH_COLORS.get(v, "#333333"), edgecolor="black", zorder=5)
        ax.annotate(v, (r["tok_per_sec"], r["final_val_bpb"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Training Throughput (tokens/sec)")
    ax.set_ylabel("Final Val BPB (Lower Better)")
    ax.set_title("Panel 6: Speed vs Quality Pareto")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    with open(png_path, "rb") as img_f:
        img_b64 = base64.b64encode(img_f.read()).decode("ascii")

    base_arch = "pre_norm" if "pre_norm" in stage2 else variants[0]
    base_mse = stage1["structured_l24"][base_arch]["best_val_mse"]
    base_bpb = stage2[base_arch]["final_val_bpb"]

    rows_html = []
    for v in variants:
        s1 = stage1["structured_l24"][v]
        s1_out = stage1["outlier_l24"][v]
        s2 = stage2[v]
        d_mse = 100.0 * (s1["best_val_mse"] - base_mse) / max(base_mse, 1e-12)
        d_bpb = 100.0 * (s2["final_val_bpb"] - base_bpb) / max(base_bpb, 1e-12)
        rows_html.append(
            f"<tr><td><b>{s2['label']}</b></td>"
            f"<td>{s2['overhead_params']:,}</td>"
            f"<td>{s1['best_val_mse']:.5f} ({d_mse:+.2f}%)</td>"
            f"<td>{s1_out['best_val_mse']:.5f}</td>"
            f"<td>{s2['final_val_loss']:.4f}</td>"
            f"<td>{s2['final_val_ppl']:.2f}</td>"
            f"<td>{s2['final_val_bpb']:.4f} ({d_bpb:+.2f}%)</td>"
            f"<td>{s2['ood_bpb']:.4f}</td>"
            f"<td>{s2['norm_growth']:.2f}x</td>"
            f"<td>{s2['avg_depth_renewal']:.4f}</td>"
            f"<td>{s2['tok_per_sec']:,.0f}</td></tr>"
        )

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>GOR From-Scratch Research Report</title>"
        "<style>body{font-family:system-ui,-apple-system,sans-serif;max-width:1280px;margin:24px auto;padding:0 16px;color:#0f172a;}"
        "table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13px;}"
        "th,td{border:1px solid #cbd5e1;padding:8px 10px;text-align:right;}th:first-child,td:first-child{text-align:left;}"
        "th{background:#f1f5f9;font-weight:600;}img{max-width:100%;border:1px solid #e2e8f0;border-radius:8px;}</style></head><body>"
        "<h1>GOR From-Scratch Research Report</h1>"
        f"<p>Device: <b>{payload['meta']['device_name']}</b> | LM Config: L={payload['meta']['lm_layers']}, d={payload['meta']['lm_dim']}, steps={payload['meta']['lm_steps']}</p>"
        "<table><thead><tr><th>Architecture</th><th>Overhead Params</th><th>Matrix Struct MSE</th><th>Matrix Outlier MSE</th>"
        "<th>LM Val Loss</th><th>LM Val PPL</th><th>LM Val BPB</th><th>OOD BPB</th><th>Norm Growth</th><th>Depth Renewal</th><th>tok/s</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
        f"<h2>6-Panel Experimental Diagnostics</h2><img src='data:image/png;base64,{img_b64}'/></body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def auto_tune_profile(device: torch.device, quick: bool) -> Dict[str, int]:
    if quick or device.type == "cpu":
        return {
            "matrix_steps": 30,
            "matrix_samples": 512,
            "lm_dim": 64,
            "lm_layers": 4,
            "lm_heads": 4,
            "lm_seq": 64,
            "lm_steps": 20,
            "lm_bs": 16,
            "lm_ga": 1,
        }
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if vram_gb >= 35.0:
            return {
                "matrix_steps": 200,
                "matrix_samples": 4096,
                "lm_dim": 384,
                "lm_layers": 16,
                "lm_heads": 6,
                "lm_seq": 256,
                "lm_steps": 1000,
                "lm_bs": 32,
                "lm_ga": 2,
            }
        if vram_gb >= 20.0:
            return {
                "matrix_steps": 200,
                "matrix_samples": 4096,
                "lm_dim": 384,
                "lm_layers": 16,
                "lm_heads": 6,
                "lm_seq": 256,
                "lm_steps": 750,
                "lm_bs": 24,
                "lm_ga": 2,
            }
        return {
            "matrix_steps": 180,
            "matrix_samples": 4096,
            "lm_dim": 320,
            "lm_layers": 16,
            "lm_heads": 5,
            "lm_seq": 256,
            "lm_steps": 500,
            "lm_bs": 16,
            "lm_ga": 2,
        }
    return {
        "matrix_steps": 120,
        "matrix_samples": 2048,
        "lm_dim": 256,
        "lm_layers": 12,
        "lm_heads": 4,
        "lm_seq": 128,
        "lm_steps": 250,
        "lm_bs": 16,
        "lm_ga": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GOR-v3 From-Scratch Research Study (Matrix + Causal LM)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--synthetic-lm", action="store_true")
    parser.add_argument("--variants", type=str, default=",".join(ALL_LM_RESIDUALS))
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    prof = auto_tune_profile(device, quick=args.quick)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    print(f"Device: {dev_name} | Variants: {variants}", flush=True)
    stage1 = run_stage1_matrix_benchmark(
        variants=variants,
        device=device,
        dim=128,
        num_layers=24,
        n_train=prof["matrix_samples"],
        steps=prof["matrix_steps"],
        batch_size=64,
        seed=args.seed,
    )

    if args.synthetic_lm or args.quick:
        dataset = build_synthetic_lm_split(seq_len=prof["lm_seq"], seed=args.seed)
    else:
        dataset = build_pretraining_dataset(seq_len=prof["lm_seq"])

    stage2 = run_stage2_lm_pretraining(
        variants=variants,
        dataset=dataset,
        device=device,
        dim=prof["lm_dim"],
        n_layers=prof["lm_layers"],
        n_heads=prof["lm_heads"],
        seq_len=prof["lm_seq"],
        steps=prof["lm_steps"],
        batch_size=prof["lm_bs"],
        grad_accum=prof["lm_ga"],
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "gor_from_scratch_results.json")
    png_path = os.path.join(args.output_dir, "gor_from_scratch_report.png")
    html_path = os.path.join(args.output_dir, "gor_from_scratch_report.html")

    payload = {
        "meta": {
            "device_name": dev_name,
            "lm_dim": prof["lm_dim"],
            "lm_layers": prof["lm_layers"],
            "lm_heads": prof["lm_heads"],
            "lm_seq": prof["lm_seq"],
            "lm_steps": prof["lm_steps"],
            "vocab_size": dataset.vocab_size,
        },
        "stage1_matrix": stage1,
        "stage2_lm": stage2,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    render_report_assets(payload, png_path=png_path, html_path=html_path)
    print(f"Saved JSON: {json_path}\nSaved PNG: {png_path}\nSaved HTML: {html_path}", flush=True)


if __name__ == "__main__":
    main()
