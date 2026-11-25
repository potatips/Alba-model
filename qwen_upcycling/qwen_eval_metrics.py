import base64
import math
import time
from io import BytesIO
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qwen_moefication import QwenOSMoEBlock, _locate_decoder_layers, count_model_params


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    dot_X = torch.matmul(Xc, Xc.t())
    dot_Y = torch.matmul(Yc, Yc.t())
    hsic_xy = torch.sum(dot_X * dot_Y)
    hsic_xx = torch.sum(dot_X * dot_X)
    hsic_yy = torch.sum(dot_Y * dot_Y)
    cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-12)
    return float(torch.clamp(cka, 0.0, 1.0).item())


def _token_norm_erank(hs: torch.Tensor) -> float:
    flat = hs.reshape(-1, hs.size(-1)).float()
    normed = F.normalize(flat, p=2, dim=-1, eps=1e-8)
    centered = normed - normed.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered)
    s_pos = s[s > 1e-7]
    if s_pos.numel() == 0:
        return 0.0
    p = s_pos / s_pos.sum()
    entropy = -(p * torch.log(p + 1e-12)).sum()
    erank = torch.exp(entropy).item()
    denom = float(min(flat.size(0), flat.size(1)))
    return float(erank / max(denom, 1.0))


def _fast_spectral_norm(W: torch.Tensor, iters: int = 8) -> float:
    Wf = W.float()
    if Wf.size(0) < Wf.size(1):
        Wf = Wf.t()
    v = torch.randn(Wf.size(1), device=Wf.device, dtype=torch.float32)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        u = torch.mv(Wf, v)
        u = u / (u.norm() + 1e-12)
        v = torch.mv(Wf.t(), u)
        v = v / (v.norm() + 1e-12)
    sigma = torch.dot(u, torch.mv(Wf, v)).abs().item()
    return max(sigma, 1e-8)


def evaluate_domain_bpb(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 4,
    device: str = "cuda",
) -> Dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            valid_toks = int((labels[:, 1:] != -100).sum().item())
            if valid_toks > 0:
                total_nll += float(outputs.loss.item()) * valid_toks
                total_tokens += valid_toks

            for b_idx in range(input_ids.size(0)):
                valid_ids = input_ids[b_idx][attention_mask[b_idx] == 1]
                decoded = tokenizer.decode(valid_ids, skip_special_tokens=True)
                total_bytes += max(len(decoded.encode("utf-8")), 1)

    avg_loss = total_nll / max(total_tokens, 1)
    ppl = float(math.exp(min(avg_loss, 20.0)))
    bpb = float((total_nll / max(total_bytes, 1)) / math.log(2.0))
    return {
        "loss": round(avg_loss, 4),
        "ppl": round(ppl, 3),
        "bpb": round(bpb, 4),
        "tokens": total_tokens,
    }


def evaluate_residual_and_moe_geometry(
    model,
    tokenizer,
    probe_texts: List[str],
    max_length: int = 256,
    device: str = "cuda",
) -> Dict:
    model.eval()
    layers = _locate_decoder_layers(model)
    last_raw = {}

    def _capture_last(_, __, out):
        t = out[0] if isinstance(out, tuple) else out
        last_raw["h"] = t.detach()

    hook_last = layers[-1].register_forward_hook(_capture_last)

    joined = "\n\n".join(probe_texts[:8])
    enc = tokenizer(joined, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        _ = model(input_ids=input_ids[:, : min(32, input_ids.size(1))], use_cache=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = max(time.perf_counter() - t0, 1e-6)

    hook_last.remove()
    throughput = float(input_ids.numel() / dt)

    hs_list = list(outputs.hidden_states)
    if "h" in last_raw:
        hs_list[-1] = last_raw["h"]

    conflict_curve = []
    ortho_energy_curve = []
    erank_curve = []
    bi_curve = []
    cka_renewal_curve = []

    for l in range(len(hs_list) - 1):
        x_l = hs_list[l].squeeze(0).float()
        x_next = hs_list[l + 1].squeeze(0).float()
        u_l = x_next - x_l

        dot = (x_l * u_l).sum(dim=-1)
        denom = x_l.norm(dim=-1) * u_l.norm(dim=-1) + 1e-8
        rho = torch.clamp(dot / denom, -1.0, 1.0)

        conflict_curve.append(round(float((rho < 0).float().mean().item() * 100.0), 3))
        ortho_energy_curve.append(round(float((1.0 - rho.pow(2)).mean().item() * 100.0), 3))
        erank_curve.append(round(_token_norm_erank(x_next), 4))

        cos_seq = F.cosine_similarity(x_l, x_next, dim=-1).mean().item()
        bi_curve.append(round(float(1.0 - cos_seq), 4))
        cka_renewal_curve.append(round(float(1.0 - linear_cka(x_l, x_next)), 4))

    moe_entropy_curve = []
    moe_ortho_curve = []
    max_vio_global = 0.0
    max_vio_batch = 0.0
    max_lse = 0.0
    has_moe = False

    for m in model.modules():
        if isinstance(m, QwenOSMoEBlock):
            has_moe = True
            with torch.no_grad():
                gf = F.normalize(m.router.gate.float(), p=2, dim=-1, eps=1e-7)
                gram = torch.matmul(gf, gf.t())
                eye = torch.eye(m.num_experts, device=gf.device, dtype=torch.float32)
                defect = torch.sqrt(torch.sum((gram - eye) ** 2) / (m.num_experts * (m.num_experts - 1))).item()
                moe_ortho_curve.append(round(float(defect), 4))

                if m.last_router_logits is not None:
                    logits = m.last_router_logits.float().reshape(-1, m.num_experts)
                    lse = float(torch.logsumexp(logits, dim=-1).max().item())
                    max_lse = max(max_lse, lse)

                    probs = F.softmax(logits, dim=-1)
                    mean_p = probs.mean(dim=0)
                    h_marg = float(-(mean_p * torch.log(mean_p + 1e-12)).sum().item())
                    h_pct = (h_marg / math.log(m.num_experts)) * 100.0
                    moe_entropy_curve.append(round(h_pct, 2))

                    _, topk_idx = torch.topk(logits, m.top_k, dim=-1)
                    counts = torch.bincount(topk_idx.reshape(-1), minlength=m.num_experts).float()
                    avg_load = counts.mean().item()
                    if avg_load > 0:
                        vio = float((counts.max().item() - avg_load) / avg_load)
                        max_vio_global = max(max_vio_global, vio)
                        max_vio_batch = max(max_vio_batch, vio)

    srank_curve = []
    for layer in layers:
        layer_sranks = []
        for name, param in layer.named_parameters():
            if param.dim() == 2 and min(param.shape) >= 64:
                if any(k in name for k in ("q_proj", "v_proj", "o_proj", "up_proj", "down_proj")):
                    wf = param.detach().float()
                    frob_sq = float(torch.sum(wf * wf).item())
                    spec = _fast_spectral_norm(wf, iters=6)
                    srank_ratio = (frob_sq / (spec * spec + 1e-12)) / float(min(wf.shape))
                    layer_sranks.append(srank_ratio)
        if layer_sranks:
            srank_curve.append(round(float(np.mean(layer_sranks)), 4))

    return {
        "throughput_tok_s": round(throughput, 1),
        "avg_conflict_pct": round(float(np.mean(conflict_curve)), 2),
        "avg_ortho_energy_pct": round(float(np.mean(ortho_energy_curve)), 2),
        "avg_erank_norm": round(float(np.mean(erank_curve)), 4),
        "avg_bi": round(float(np.mean(bi_curve)), 4),
        "avg_cka_renewal": round(float(np.mean(cka_renewal_curve)), 4),
        "effective_depth_ratio_pct": round(float(np.mean(cka_renewal_curve) * 100.0), 2),
        "avg_srank_w": round(float(np.mean(srank_curve)), 4) if srank_curve else 0.0,
        "is_moe": has_moe,
        "avg_moe_entropy_pct": round(float(np.mean(moe_entropy_curve)), 2) if moe_entropy_curve else 0.0,
        "avg_moe_ortho_defect": round(float(np.mean(moe_ortho_curve)), 4) if moe_ortho_curve else 0.0,
        "max_vio_global": round(max_vio_global, 3),
        "max_vio_batch": round(max_vio_batch, 3),
        "max_lse": round(max_lse, 2),
        "conflict_curve": conflict_curve,
        "erank_curve": erank_curve,
        "cka_renewal_curve": cka_renewal_curve,
        "srank_curve": srank_curve,
        "moe_entropy_curve": moe_entropy_curve,
        "moe_ortho_curve": moe_ortho_curve,
    }


def evaluate_qwen_variant_full(
    model,
    tokenizer,
    eval_logic_texts: List[str],
    eval_code_texts: List[str],
    ood_eval_texts: Optional[Dict[str, List[str]]] = None,
    max_length: int = 512,
    eval_batch_size: int = 4,
    device: str = "cuda",
) -> Dict:
    params_info = count_model_params(model)
    logic_res = evaluate_domain_bpb(model, tokenizer, eval_logic_texts, max_length=max_length, batch_size=eval_batch_size, device=device)
    code_res = evaluate_domain_bpb(model, tokenizer, eval_code_texts, max_length=max_length, batch_size=eval_batch_size, device=device)
    overall_bpb = round(0.5 * (logic_res["bpb"] + code_res["bpb"]), 4)
    overall_loss = round(0.5 * (logic_res["loss"] + code_res["loss"]), 4)
    overall_ppl = round(float(math.exp(min(overall_loss, 20.0))), 3)

    ood_metrics = {}
    if ood_eval_texts:
        ood_bpbs = []
        ood_losses = []
        for key, texts in ood_eval_texts.items():
            if texts:
                res = evaluate_domain_bpb(model, tokenizer, texts, max_length=max_length, batch_size=eval_batch_size, device=device)
                ood_metrics[f"{key}_bpb"] = res["bpb"]
                ood_metrics[f"{key}_ppl"] = res["ppl"]
                ood_metrics[f"{key}_loss"] = res["loss"]
                ood_bpbs.append(res["bpb"])
                ood_losses.append(res["loss"])
        if ood_bpbs:
            ood_mean_loss = round(float(np.mean(ood_losses)), 4)
            ood_metrics["ood_bpb"] = round(float(np.mean(ood_bpbs)), 4)
            ood_metrics["ood_loss"] = ood_mean_loss
            ood_metrics["ood_ppl"] = round(float(math.exp(min(ood_mean_loss, 20.0))), 3)

    geom = evaluate_residual_and_moe_geometry(
        model,
        tokenizer,
        eval_logic_texts[:4] + eval_code_texts[:4],
        max_length=256,
        device=device,
    )

    return {
        "total_params_b": params_info["total_params_b"],
        "active_params_b": params_info["active_params_b"],
        "overall_bpb": overall_bpb,
        "overall_ppl": overall_ppl,
        "overall_loss": overall_loss,
        "logic_bpb": logic_res["bpb"],
        "logic_ppl": logic_res["ppl"],
        "code_bpb": code_res["bpb"],
        "code_ppl": code_res["ppl"],
        **ood_metrics,
        **geom,
    }


def generate_ablation_html_report(
    results: Dict[str, Dict],
    output_html_path: str = "qwen3b_ablation_report.html",
    output_png_path: str = "qwen3b_ablation_grid.png",
) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(22, 12.5), dpi=150)
    plt.subplots_adjust(hspace=0.40, wspace=0.26)

    names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(names), 1)))
    has_ood = all("ood_bpb" in results[n] for n in names) if names else False
    active_vals = [results[n]["active_params_b"] for n in names] if names else [0.0]
    dense_only = (max(active_vals) - min(active_vals)) < 0.05

    ax1 = axes[0, 0]
    x_pos = np.arange(len(names))
    if has_ood:
        w = 0.20
        ax1.bar(x_pos - 1.5 * w, [results[n]["logic_bpb"] for n in names], width=w, label="In-Dist Logic BPB", color="#2563eb")
        ax1.bar(x_pos - 0.5 * w, [results[n]["code_bpb"] for n in names], width=w, label="In-Dist Coding BPB", color="#16a34a")
        ax1.bar(x_pos + 0.5 * w, [results[n]["overall_bpb"] for n in names], width=w, label="In-Dist Overall BPB", color="#dc2626")
        ax1.bar(x_pos + 1.5 * w, [results[n]["ood_bpb"] for n in names], width=w, label="OOD External BPB", color="#9333ea")
        ax1.set_title("Panel 1: In-Dist vs External OOD BPB (Lower is Better)", fontsize=11, fontweight="bold")
    else:
        w = 0.27
        ax1.bar(x_pos - w, [results[n]["logic_bpb"] for n in names], width=w, label="Logic BPB", color="#2563eb")
        ax1.bar(x_pos, [results[n]["code_bpb"] for n in names], width=w, label="Coding BPB", color="#16a34a")
        ax1.bar(x_pos + w, [results[n]["overall_bpb"] for n in names], width=w, label="Overall BPB", color="#dc2626")
        ax1.set_title("Panel 1: Held-Out Logic & Coding BPB Comparison", fontsize=11, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=24, ha="right", fontsize=8)
    ax1.set_ylabel("Bits-Per-Byte (Lower is Better)")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8)

    ax2 = axes[0, 1]
    for n, c in zip(names, colors):
        hist = results[n].get("train_loss_curve", [])
        if hist:
            steps = [pt[0] for pt in hist]
            vals = [pt[1] for pt in hist]
            ax2.plot(steps, vals, label=n, color=c, linewidth=2.0, alpha=0.9)
    ax2.set_title("Panel 2: Training Convergence (Baseline vs GOR)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Training LM Loss")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=7.5, loc="best")

    ax3 = axes[0, 2]
    for n, c in zip(names, colors):
        curve = results[n].get("conflict_curve", [])
        if curve:
            ax3.plot(np.linspace(0, 100, len(curve)), curve, label=n, color=c, linewidth=1.8, alpha=0.85)
    ax3.set_title("Panel 3 (GOR): Residual Conflict Rate R_conflict(l)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Normalized Depth (%)")
    ax3.set_ylabel("Destructive Conflict (%)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=7, loc="best")

    ax4 = axes[1, 0]
    for n, c in zip(names, colors):
        curve = results[n].get("erank_curve", [])
        if curve:
            ax4.plot(np.linspace(0, 100, len(curve)), curve, label=n, color=c, linewidth=1.8, alpha=0.85)
    ax4.set_title("Panel 4 (GOR): Token-Normalized Effective Rank Ratio", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Normalized Depth (%)")
    ax4.set_ylabel("erank_norm / min(T, d)")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(fontsize=7, loc="best")

    ax5 = axes[1, 1]
    for n, c in zip(names, colors):
        curve = results[n].get("srank_curve", [])
        if curve:
            ax5.plot(np.linspace(0, 100, len(curve)), curve, label=n, color=c, linewidth=1.8, alpha=0.85)
    ax5.set_title("Panel 5 (GEO-Muon): Weight Stable Rank Ratio srank(W)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Normalized Depth (%)")
    ax5.set_ylabel("srank(W_l)")
    ax5.grid(True, linestyle=":", alpha=0.6)
    ax5.legend(fontsize=7, loc="best")

    ax6 = axes[1, 2]
    if has_ood and dense_only:
        for idx, (n, c) in enumerate(zip(names, colors)):
            r = results[n]
            ax6.scatter(r["overall_bpb"], r["ood_bpb"], color=c, s=130, marker="o", edgecolors="black", zorder=5)
            dx = 8 if idx % 2 == 0 else -12
            dy = 10 if idx % 3 == 0 else -14
            ax6.annotate(
                f"{n}\n({r['overall_bpb']:.4f}, {r['ood_bpb']:.4f})",
                (r["overall_bpb"], r["ood_bpb"]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=7.2,
                fontweight="bold",
                color=c,
                arrowprops=dict(arrowstyle="-", color=c, lw=0.7, alpha=0.6),
            )
        ax6.set_title("Panel 6: In-Dist vs OOD Generalization (Bottom-Left Best)", fontsize=11, fontweight="bold")
        ax6.set_xlabel("In-Distribution Overall BPB (Lower is Better)")
        ax6.set_ylabel("External OOD Overall BPB (Lower is Better)")
    else:
        for idx, (n, c) in enumerate(zip(names, colors)):
            r = results[n]
            marker = "s" if r.get("is_moe") else "o"
            ax6.scatter(r["active_params_b"], r["overall_bpb"], color=c, s=130, marker=marker, edgecolors="black", zorder=5)
            dy = 12 if idx % 2 == 0 else -18
            ax6.annotate(
                f"{n}\n({r['overall_bpb']:.4f} BPB)",
                (r["active_params_b"], r["overall_bpb"]),
                textcoords="offset points",
                xytext=(8, dy),
                fontsize=7.5,
                fontweight="bold",
                color=c,
                arrowprops=dict(arrowstyle="-", color=c, lw=0.7, alpha=0.6),
            )
        ax6.set_title("Panel 6 (Pareto): Active Params vs Overall Logic+Code BPB", fontsize=11, fontweight="bold")
        ax6.set_xlabel("Active Parameters per Token (Billions)")
        ax6.set_ylabel("Held-Out Overall BPB (Lower is Better)")
    ax6.grid(True, linestyle=":", alpha=0.6)

    fig.savefig(output_png_path, format="png", bbox_inches="tight")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    plot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["overall_bpb"])
    winner_name, winner_res = ranked[0]

    rows_html = []
    for rank, (m_name, r) in enumerate(ranked, 1):
        bg = "#f0fdf4" if rank == 1 else "#ffffff"
        moe_str = f"{r['avg_moe_entropy_pct']}% / {r['avg_moe_ortho_defect']}" if r["is_moe"] else "—"
        vio_str = f"{r['max_vio_global']} / {r['max_lse']}" if r["is_moe"] else "—"
        ood_cell = ""
        if has_ood:
            ood_cell = (
                f"<td style='padding:8px; font-weight:bold; color:#7e22ce;'>{r['ood_bpb']}</td>"
                f"<td style='padding:8px;'>{r.get('ood_logic_bpb', '—')} / {r.get('ood_code_bpb', '—')} / {r.get('ood_general_bpb', '—')}</td>"
            )
        rows_html.append(
            f"<tr style='background:{bg}; border-bottom:1px solid #e2e8f0;'>"
            f"<td style='padding:8px; font-weight:bold;'>#{rank}</td>"
            f"<td style='padding:8px; text-align:left; font-weight:bold;'>{m_name}</td>"
            f"<td style='padding:8px;'>{r['total_params_b']}B / {r['active_params_b']}B</td>"
            f"<td style='padding:8px; font-weight:bold; color:#0f766e;'>{r['overall_bpb']}</td>"
            f"<td style='padding:8px;'>{r['logic_bpb']} ({r['logic_ppl']})</td>"
            f"<td style='padding:8px;'>{r['code_bpb']} ({r['code_ppl']})</td>"
            f"{ood_cell}"
            f"<td style='padding:8px;'>{r['avg_conflict_pct']}% / {r['avg_ortho_energy_pct']}%</td>"
            f"<td style='padding:8px;'>{r['avg_erank_norm']}</td>"
            f"<td style='padding:8px;'>{r['avg_bi']} / {r['avg_cka_renewal']}</td>"
            f"<td style='padding:8px;'>{r['avg_srank_w']}</td>"
            f"<td style='padding:8px;'>{moe_str}</td>"
            f"<td style='padding:8px;'>{vio_str}</td>"
            f"<td style='padding:8px;'>{r.get('peak_vram_gb', 0.0)} GB</td>"
            f"<td style='padding:8px;'>{r['throughput_tok_s']}</td>"
            f"</tr>"
        )

    ood_headers = ""
    winner_ood_note = ""
    if has_ood:
        ood_headers = "<th>OOD BPB</th><th>OOD (MMLU / HumanEval / Wiki)</th>"
        winner_ood_note = f", External OOD BPB: <strong>{winner_res['ood_bpb']}</strong>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Qwen-2.5-3B Alba Ablation Report (GOR, OS-MoE, GEO-Muon vs AdamW)</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background:#f8fafc; color:#0f172a; padding:24px; }}
.container {{ max-width:1460px; margin:0 auto; background:#fff; padding:28px; border-radius:12px; border:1px solid #e2e8f0; }}
h1 {{ border-bottom:2px solid #2563eb; padding-bottom:10px; font-size:24px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; text-align:center; margin-top:16px; }}
th {{ background:#f1f5f9; padding:10px 6px; border-bottom:2px solid #cbd5e1; }}
.winner {{ background:#f0fdf4; border-left:4px solid #16a34a; padding:14px; margin:16px 0; border-radius:0 8px 8px 0; }}
.fig {{ text-align:center; margin-top:24px; }}
.fig img {{ max-width:100%; border:1px solid #cbd5e1; border-radius:8px; }}
</style>
</head>
<body>
<div class="container">
<h1>Qwen-2.5-3B Alba Ablation Study: Baseline vs GOR × (AdamW vs GEO-Muon)</h1>
<div class="winner">
<strong>Top Ranked Model on Held-Out Logic + Coding Evaluation:</strong> {winner_name} — Overall BPB: <strong>{winner_res['overall_bpb']}</strong> (Logic BPB: {winner_res['logic_bpb']}, Coding BPB: {winner_res['code_bpb']}{winner_ood_note}, Active Params: {winner_res['active_params_b']}B).
</div>
<table>
<thead>
<tr>
<th>Rank</th>
<th style="text-align:left;">Variant</th>
<th>Total / Active</th>
<th>Overall BPB</th>
<th>Logic BPB (PPL)</th>
<th>Coding BPB (PPL)</th>
{ood_headers}
<th>Conflict / Ortho</th>
<th>Avg erank_norm</th>
<th>Avg BI / 1-CKA</th>
<th>Avg srank(W)</th>
<th>MoE H_route / D_ortho</th>
<th>MaxVio / Max LSE</th>
<th>Peak VRAM</th>
<th>Tok/s</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
<div class="fig">
<img src="data:image/png;base64,{plot_b64}" alt="Qwen-2.5-3B Ablation 6-Panel Grid">
</div>
</div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_html_path
