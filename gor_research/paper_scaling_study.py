import argparse
import base64
import contextlib
import json
import math
import os
import statistics
import time
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.optimizer import GEOMuon
from .paper_data import (
    MemmapTokenDataset,
    PaperTokenCorpus,
    build_or_load_fineweb_edu_corpus,
    build_synthetic_paper_corpus,
)
from .residuals import RESIDUAL_LABELS
from .transformer_lm import (
    FromScratchTransformerLM,
    TransformerConfig,
    clone_transformer_backbone,
)


SCALE_PRESETS: Dict[str, Dict[str, object]] = {
    "smoke": {
        "dim": 64,
        "n_layers": 4,
        "n_heads": 4,
        "seq_len": 64,
        "batch_size": 8,
        "grad_accum": 1,
        "steps": 24,
        "lr": 1e-3,
        "corpus_tokens": 131072,
        "val_tokens": 16384,
        "grad_ckpt": False,
    },
    "ablation_40m": {
        "dim": 384,
        "n_layers": 12,
        "n_heads": 6,
        "seq_len": 512,
        "batch_size": 16,
        "grad_accum": 4,
        "steps": 1500,
        "lr": 6e-4,
        "corpus_tokens": 60_000_000,
        "val_tokens": 1_000_000,
        "grad_ckpt": False,
    },
    "scale_125m": {
        "dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "seq_len": 1024,
        "batch_size": 8,
        "grad_accum": 8,
        "steps": 4500,
        "lr": 5e-4,
        "corpus_tokens": 300_000_000,
        "val_tokens": 2_000_000,
        "grad_ckpt": False,
    },
    "scale_220m": {
        "dim": 896,
        "n_layers": 18,
        "n_heads": 14,
        "seq_len": 1024,
        "batch_size": 6,
        "grad_accum": 10,
        "steps": 6000,
        "lr": 4e-4,
        "corpus_tokens": 400_000_000,
        "val_tokens": 2_000_000,
        "grad_ckpt": True,
    },
    "scale_300m": {
        "dim": 1024,
        "n_layers": 20,
        "n_heads": 16,
        "seq_len": 1024,
        "batch_size": 4,
        "grad_accum": 16,
        "steps": 7500,
        "lr": 3e-4,
        "corpus_tokens": 500_000_000,
        "val_tokens": 2_000_000,
        "grad_ckpt": True,
    },
}

DEFAULT_PAPER_VARIANTS = [
    "pre_norm",
    "pre_norm_isotime",
    "mhc",
    "gor_v2",
    "attn_res_block",
    "attn_res_gor_v2",
]

PAPER_LABELS = {
    **RESIDUAL_LABELS,
    "pre_norm_isotime": "Pre-Norm (Iso-Time Deeper)",
}

PAPER_COLORS = {
    "pre_norm": "#475569",
    "pre_norm_isotime": "#94a3b8",
    "mhc": "#ca8a04",
    "gor_v2": "#dc2626",
    "attn_res_block": "#16a34a",
    "attn_res_gor_v2": "#be185d",
    "dual_bank_gor_v2": "#0284c7",
    "std_dual_bank": "#2563eb",
    "gor_v3_subspace": "#7c3aed",
}


def _resolve_variant_spec(variant: str, base_layers: int, iso_extra_layers: int) -> Tuple[str, int]:
    if variant == "pre_norm_isotime":
        extra = iso_extra_layers if iso_extra_layers > 0 else max(2, round(0.25 * base_layers))
        return "pre_norm", base_layers + extra
    return variant, base_layers


def _build_paper_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    opt_key = optimizer_name.lower()
    if opt_key == "geomuon":
        muon_params, adamw_decay, adamw_nodecay = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_connector = name.startswith("connector.")
            is_embed = "tok_embeddings" in name or "lm_head" in name
            is_norm = "norm" in name or p.ndim < 2
            if (p.ndim >= 2) and (not is_connector) and (not is_embed) and (not is_norm):
                muon_params.append(p)
            elif is_norm or is_connector or p.ndim < 2:
                adamw_nodecay.append(p)
            else:
                adamw_decay.append(p)
        groups = []
        if muon_params:
            groups.append({"params": muon_params, "use_muon": True, "lr": lr * 1.5, "momentum": 0.95, "weight_decay": weight_decay})
        if adamw_decay:
            groups.append({"params": adamw_decay, "use_muon": False, "lr": lr, "betas": (0.9, 0.95), "weight_decay": weight_decay})
        if adamw_nodecay:
            groups.append({"params": adamw_nodecay, "use_muon": False, "lr": lr, "betas": (0.9, 0.95), "weight_decay": 0.0})
        return GEOMuon(groups)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "depth_queries" in name or "alpha_" in name or "h_res_bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = device.type == "cuda"
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)


@torch.no_grad()
def evaluate_blocks(
    model: FromScratchTransformerLM,
    blocks: torch.Tensor,
    bytes_per_token: float,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 32,
    collect_geom: bool = False,
) -> Tuple[float, float, float, Optional[Dict[str, List[float]]]]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    geom_accum: Optional[Dict[str, List[float]]] = None

    n_blocks = blocks.size(0)
    n_avail = (n_blocks + batch_size - 1) // batch_size
    n_eval = min(n_avail, max_batches)

    for b_idx in range(n_eval):
        s = b_idx * batch_size
        e = min(n_blocks, s + batch_size)
        batch = blocks[s:e].to(device, non_blocking=True)
        xb = batch[:, :-1]
        yb = batch[:, 1:]
        want_geom = collect_geom and (b_idx == 0)
        ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
        with ctx:
            _, loss, geom = model(xb, targets=yb, collect_geometry=want_geom)
        ntok = int(yb.numel())
        total_loss += float(loss.item()) * ntok
        total_tokens += ntok
        if want_geom and geom is not None:
            geom_accum = {k: list(v) for k, v in geom.items()}

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    bpb = avg_loss / (math.log(2.0) * max(bytes_per_token, 1e-6))
    model.train()
    return avg_loss, ppl, bpb, geom_accum


@torch.no_grad()
def evaluate_downstream(
    model: FromScratchTransformerLM,
    downstream_tasks: Dict[str, List[Dict[str, object]]],
    tokenizer_name: str,
    max_seq_len: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, Dict[str, float]]:
    if not downstream_tasks:
        return {}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.eval()
    metrics: Dict[str, Dict[str, float]] = {}

    for task_name, items in downstream_tasks.items():
        if not items:
            continue
        if task_name == "lambada":
            correct = 0
            total_nll = 0.0
            total_tok = 0
            for item in items:
                ctx_ids = tokenizer.encode(str(item["ctx"]), add_special_tokens=False)
                tgt_ids = tokenizer.encode(str(item["target"]), add_special_tokens=False)
                if not tgt_ids:
                    continue
                keep_ctx = max(1, max_seq_len - len(tgt_ids))
                full_ids = ctx_ids[-keep_ctx:] + tgt_ids
                ctx_len = len(full_ids) - len(tgt_ids)
                inp = torch.tensor([full_ids], dtype=torch.long, device=device)
                ctx_mgr = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
                with ctx_mgr:
                    logits, _, _ = model(inp, targets=None)
                pred_slices = logits[0, ctx_len - 1 : len(full_ids) - 1, :]
                log_probs = F.log_softmax(pred_slices.float(), dim=-1)
                tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long, device=device)
                tok_ll = log_probs.gather(-1, tgt_tensor.unsqueeze(-1)).squeeze(-1)
                total_nll -= float(tok_ll.sum().item())
                total_tok += len(tgt_ids)
                greedy = pred_slices.argmax(dim=-1)
                if torch.equal(greedy, tgt_tensor):
                    correct += 1
            n_eval = max(len(items), 1)
            avg_nll = total_nll / max(total_tok, 1)
            metrics["lambada"] = {
                "acc_pct": 100.0 * float(correct) / float(n_eval),
                "ppl": math.exp(min(avg_nll, 20.0)),
                "n": float(n_eval),
            }
            continue

        correct = 0
        for item in items:
            ctx_ids = tokenizer.encode(str(item["ctx"]), add_special_tokens=False)
            choices: List[str] = list(item["choices"])
            label = int(item["label"])
            scores: List[float] = []
            for ch in choices:
                ch_ids = tokenizer.encode(" " + ch, add_special_tokens=False)
                if not ch_ids:
                    scores.append(-1e9)
                    continue
                keep_ctx = max(1, max_seq_len - len(ch_ids))
                full_ids = ctx_ids[-keep_ctx:] + ch_ids
                ctx_len = len(full_ids) - len(ch_ids)
                inp = torch.tensor([full_ids], dtype=torch.long, device=device)
                ctx_mgr = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
                with ctx_mgr:
                    logits, _, _ = model(inp, targets=None)
                pred_slices = logits[0, ctx_len - 1 : len(full_ids) - 1, :]
                log_probs = F.log_softmax(pred_slices.float(), dim=-1)
                ch_tensor = torch.tensor(ch_ids, dtype=torch.long, device=device)
                ll = float(log_probs.gather(-1, ch_tensor.unsqueeze(-1)).squeeze(-1).sum().item())
                scores.append(ll / float(max(len(ch.encode("utf-8")), 1)))
            pred = max(range(len(scores)), key=lambda idx: scores[idx])
            if pred == label:
                correct += 1
        n_eval = max(len(items), 1)
        metrics[task_name] = {
            "acc_norm_pct": 100.0 * float(correct) / float(n_eval),
            "n": float(n_eval),
        }

    model.train()
    return metrics


def _save_step_checkpoint(ckpt_path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    tmp_path = ckpt_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)


def _train_single_run(
    variant: str,
    seed: int,
    corpus: PaperTokenCorpus,
    preset_cfg: Dict[str, object],
    tokenizer_name: str,
    optimizer_name: str,
    iso_extra_layers: int,
    use_compile: bool,
    save_ckpt: bool,
    resume: bool,
    ckpt_every: int,
    ckpt_mins: float,
    max_session_mins: float,
    max_steps_per_session: int,
    keep_step_ckpt: bool,
    session_t0: float,
    output_dir: str,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    steps = int(preset_cfg["steps"])
    run_file = os.path.join(output_dir, "runs", f"{variant}__seed{seed}.json")
    step_ckpt_file = os.path.join(output_dir, "step_checkpoints", f"{variant}__seed{seed}_latest.pt")
    if resume and os.path.exists(run_file):
        with open(run_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if (
            int(cached.get("steps", -1)) == steps
            and int(cached.get("dim", -1)) == int(preset_cfg["dim"])
            and int(cached.get("base_layers", -1)) == int(preset_cfg["n_layers"])
        ):
            print(f"[Resume] Loaded completed run: {variant} (seed={seed})", flush=True)
            return cached

    dim = int(preset_cfg["dim"])
    base_layers = int(preset_cfg["n_layers"])
    n_heads = int(preset_cfg["n_heads"])
    seq_len = int(preset_cfg["seq_len"])
    batch_size = int(preset_cfg["batch_size"])
    grad_accum = int(preset_cfg["grad_accum"])
    lr = float(preset_cfg["lr"])
    grad_ckpt = bool(preset_cfg["grad_ckpt"])

    res_type, actual_layers = _resolve_variant_spec(variant, base_layers, iso_extra_layers)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = amp_enabled and (amp_dtype == torch.float16)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(seed)
    ref_cfg = TransformerConfig(
        vocab_size=corpus.vocab_size,
        dim=dim,
        n_layers=actual_layers,
        n_heads=n_heads,
        max_seq_len=seq_len,
        residual_type="pre_norm",
    )
    ref_model = FromScratchTransformerLM(ref_cfg)

    cfg = TransformerConfig(
        vocab_size=corpus.vocab_size,
        dim=dim,
        n_layers=actual_layers,
        n_heads=n_heads,
        max_seq_len=seq_len,
        residual_type=res_type,
        gradient_checkpointing=grad_ckpt,
    )
    torch.manual_seed(seed)
    base_model = FromScratchTransformerLM(cfg).to(device)
    clone_transformer_backbone(ref_model, base_model)
    del ref_model

    model_fwd: nn.Module = base_model
    if use_compile and hasattr(torch, "compile") and device.type == "cuda":
        model_fwd = torch.compile(base_model, mode="default")

    train_ds = MemmapTokenDataset(corpus.train_path, seq_len=seq_len)
    val_ds = MemmapTokenDataset(corpus.val_path, seq_len=seq_len, stride=seq_len)
    val_blocks = val_ds.get_eval_blocks(max_blocks=64)

    opt = _build_paper_optimizer(base_model, optimizer_name, lr=lr, weight_decay=0.1, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    warmup_steps = max(10, int(0.06 * steps))

    def lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(max(warmup_steps, 1))
        progress = float(step_idx - warmup_steps) / float(max(steps - warmup_steps, 1))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    b_gen = torch.Generator(device="cpu").manual_seed(seed + 9001)

    eval_every = max(10, steps // 10)
    hist_steps: List[int] = []
    hist_tokens: List[int] = []
    hist_val_bpb: List[float] = []
    hist_val_loss: List[float] = []
    hist_grad_norm: List[float] = []
    start_step = 1
    total_tokens_seen = 0
    elapsed_prev = 0.0

    total_params = int(base_model.count_total_params())
    emb_params = int(base_model.tok_embeddings.weight.numel())
    non_emb_params = total_params - emb_params
    overhead_params = int(base_model.count_residual_overhead_params())
    tokens_per_step = batch_size * grad_accum * seq_len

    resumed_from_ckpt = False
    if resume and os.path.exists(step_ckpt_file):
        ckpt = torch.load(step_ckpt_file, map_location="cpu", weights_only=False)
        if (
            int(ckpt.get("dim", -1)) == dim
            and int(ckpt.get("base_layers", -1)) == base_layers
            and int(ckpt.get("steps", -1)) == steps
            and str(ckpt.get("variant", "")) == variant
        ):
            base_model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            if use_scaler and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            b_gen.set_state(ckpt["b_gen_state"])
            if ckpt.get("cpu_rng_state") is not None:
                torch.set_rng_state(ckpt["cpu_rng_state"])
            if device.type == "cuda" and ckpt.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"], device)
            start_step = int(ckpt["step"]) + 1
            total_tokens_seen = int(ckpt["total_tokens_seen"])
            elapsed_prev = float(ckpt["elapsed_prev"])
            best_val_loss = float(ckpt["best_val_loss"])
            best_val_bpb = float(ckpt["best_val_bpb"])
            hist_steps = list(ckpt["hist_steps"])
            hist_tokens = list(ckpt["hist_tokens"])
            hist_val_loss = list(ckpt["hist_val_loss"])
            hist_val_bpb = list(ckpt["hist_val_bpb"])
            hist_grad_norm = list(ckpt["hist_grad_norm"])
            resumed_from_ckpt = True
            print(
                f"[Resume] Restored {variant} (seed={seed}) at step {start_step - 1}/{steps} "
                f"({total_tokens_seen / 1e6:.1f}M tok, elapsed={elapsed_prev / 60.0:.1f}m)",
                flush=True,
            )

    if not resumed_from_ckpt:
        init_loss, _, init_bpb, _ = evaluate_blocks(
            base_model, val_blocks, corpus.val_bytes_per_token, device, batch_size, amp_enabled, amp_dtype, max_batches=16
        )
        hist_steps.append(0)
        hist_tokens.append(0)
        hist_val_loss.append(init_loss)
        hist_val_bpb.append(init_bpb)
        best_val_loss = init_loss
        best_val_bpb = init_bpb

    print(
        f"[Train] {variant:18s} | seed={seed} | L={actual_layers} | d={dim} | "
        f"total={total_params / 1e6:.2f}M (non-emb={non_emb_params / 1e6:.2f}M, res-oh={overhead_params:,}) | "
        f"steps={start_step}..{steps} (~{steps * tokens_per_step / 1e6:.1f}M tok)",
        flush=True,
    )

    t0 = time.perf_counter()
    last_ckpt_time = t0
    steps_this_session = 0

    def _build_ckpt_state(cur_step: int) -> Dict[str, object]:
        return {
            "variant": variant,
            "seed": seed,
            "dim": dim,
            "base_layers": base_layers,
            "steps": steps,
            "step": cur_step,
            "model": base_model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if use_scaler else None,
            "b_gen_state": b_gen.get_state(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "total_tokens_seen": total_tokens_seen,
            "elapsed_prev": elapsed_prev + (time.perf_counter() - t0),
            "best_val_loss": best_val_loss,
            "best_val_bpb": best_val_bpb,
            "hist_steps": hist_steps,
            "hist_tokens": hist_tokens,
            "hist_val_loss": hist_val_loss,
            "hist_val_bpb": hist_val_bpb,
            "hist_grad_norm": hist_grad_norm,
        }

    for step in range(start_step, steps + 1):
        try:
            opt.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                batch = train_ds.sample_batch(batch_size, b_gen).to(device, non_blocking=True)
                xb = batch[:, :-1]
                yb = batch[:, 1:]
                total_tokens_seen += int(yb.numel())
                ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
                with ctx:
                    _, loss, _ = model_fwd(xb, targets=yb)
                    scaled_loss = loss / float(grad_accum)
                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if use_scaler:
                scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0).item())
            if use_scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            steps_this_session += 1
        except KeyboardInterrupt:
            _save_step_checkpoint(step_ckpt_file, _build_ckpt_state(step - 1))
            print(f"[Interrupt] Saved checkpoint at step {step - 1}/{steps} -> {step_ckpt_file}", flush=True)
            return None

        if step % eval_every == 0 or step == steps:
            hist_grad_norm.append(gnorm)
            vloss, _, vbpb, _ = evaluate_blocks(
                base_model, val_blocks, corpus.val_bytes_per_token, device, batch_size, amp_enabled, amp_dtype, max_batches=24
            )
            hist_steps.append(step)
            hist_tokens.append(total_tokens_seen)
            hist_val_loss.append(vloss)
            hist_val_bpb.append(vbpb)
            if vloss < best_val_loss:
                best_val_loss = vloss
                best_val_bpb = vbpb
            elapsed_now = elapsed_prev + (time.perf_counter() - t0)
            print(
                f"  step {step:5d}/{steps} | tok={total_tokens_seen / 1e6:6.1f}M | "
                f"val_bpb={vbpb:.4f} | gnorm={gnorm:.2f} | {total_tokens_seen / max(elapsed_now, 1e-6):,.0f} tok/s",
                flush=True,
            )

        now = time.perf_counter()
        hit_step_budget = max_steps_per_session > 0 and steps_this_session >= max_steps_per_session and step < steps
        hit_time_budget = max_session_mins > 0.0 and (now - session_t0) >= (max_session_mins * 60.0) and step < steps
        need_periodic_ckpt = (
            (ckpt_every > 0 and step % ckpt_every == 0)
            or (ckpt_mins > 0.0 and (now - last_ckpt_time) >= (ckpt_mins * 60.0))
        ) and step < steps

        if hit_step_budget or hit_time_budget:
            _save_step_checkpoint(step_ckpt_file, _build_ckpt_state(step))
            print(
                f"[SessionPause] Saved exact state for {variant} (seed={seed}) at step {step}/{steps} -> {step_ckpt_file}",
                flush=True,
            )
            return None

        if need_periodic_ckpt:
            _save_step_checkpoint(step_ckpt_file, _build_ckpt_state(step))
            last_ckpt_time = now
            print(f"  [Ckpt] Saved step {step}/{steps} -> {step_ckpt_file}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    else:
        peak_vram_gb = 0.0
    elapsed = elapsed_prev + (time.perf_counter() - t0)
    tok_per_sec = float(total_tokens_seen) / max(elapsed, 1e-6)

    final_loss, final_ppl, final_bpb, geom = evaluate_blocks(
        base_model,
        val_blocks,
        corpus.val_bytes_per_token,
        device,
        batch_size,
        amp_enabled,
        amp_dtype,
        max_batches=40,
        collect_geom=True,
    )

    ood_metrics: Dict[str, Dict[str, float]] = {}
    for ood_name, (ood_blk, ood_bpt) in corpus.ood_splits.items():
        oloss, oppl, obpb, _ = evaluate_blocks(
            base_model, ood_blk, ood_bpt, device, batch_size, amp_enabled, amp_dtype, max_batches=24
        )
        ood_metrics[ood_name] = {"loss": oloss, "ppl": oppl, "bpb": obpb}

    ood_avg_bpb = (
        sum(v["bpb"] for v in ood_metrics.values()) / float(len(ood_metrics))
        if ood_metrics
        else final_bpb
    )

    downstream = evaluate_downstream(
        base_model,
        corpus.downstream_tasks,
        tokenizer_name=tokenizer_name,
        max_seq_len=seq_len,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    conflict_list = geom["conflict_rates"] if geom else [0.0]
    growth_list = geom["norm_growth"] if geom else [1.0]
    renewal_list = geom["depth_renewal"] if geom else [0.0]
    erank_list = geom["erank_norm"] if geom else [0.0]

    approx_total_flops = 6.0 * float(non_emb_params) * float(total_tokens_seen)

    result: Dict[str, object] = {
        "variant": variant,
        "residual_type": res_type,
        "label": PAPER_LABELS.get(variant, variant),
        "seed": seed,
        "dim": dim,
        "base_layers": base_layers,
        "actual_layers": actual_layers,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "steps": steps,
        "tokens_seen": total_tokens_seen,
        "total_params": total_params,
        "non_emb_params": non_emb_params,
        "overhead_params": overhead_params,
        "approx_flops": approx_total_flops,
        "final_val_loss": final_loss,
        "final_val_ppl": final_ppl,
        "final_val_bpb": final_bpb,
        "best_val_bpb": min(best_val_bpb, final_bpb),
        "ood_avg_bpb": ood_avg_bpb,
        "ood_metrics": ood_metrics,
        "downstream": downstream,
        "avg_conflict_pct": sum(conflict_list) / max(len(conflict_list), 1),
        "final_norm_growth": growth_list[-1],
        "avg_depth_renewal": sum(renewal_list) / max(len(renewal_list), 1),
        "avg_erank_norm": sum(erank_list) / max(len(erank_list), 1),
        "tok_per_sec": tok_per_sec,
        "peak_vram_gb": peak_vram_gb,
        "train_time_sec": elapsed,
        "hist_steps": hist_steps,
        "hist_tokens": hist_tokens,
        "hist_val_loss": hist_val_loss,
        "hist_val_bpb": hist_val_bpb,
        "hist_grad_norm": hist_grad_norm,
        "layer_conflict": conflict_list,
        "layer_norm_growth": growth_list,
        "layer_renewal": renewal_list,
        "layer_erank": erank_list,
    }

    os.makedirs(os.path.dirname(run_file), exist_ok=True)
    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if save_ckpt:
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(base_model.state_dict(), os.path.join(ckpt_dir, f"{variant}__seed{seed}.pt"))

    if (not keep_step_ckpt) and os.path.exists(step_ckpt_file):
        os.remove(step_ckpt_file)

    return result


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def aggregate_seed_runs(variants: List[str], runs: Dict[str, List[Dict[str, object]]]) -> Dict[str, Dict[str, object]]:
    agg: Dict[str, Dict[str, object]] = {}
    for v in variants:
        v_runs = runs[v]
        first = v_runs[0]

        val_bpb_m, val_bpb_s = _mean_std([float(r["final_val_bpb"]) for r in v_runs])
        val_ppl_m, val_ppl_s = _mean_std([float(r["final_val_ppl"]) for r in v_runs])
        ood_bpb_m, ood_bpb_s = _mean_std([float(r["ood_avg_bpb"]) for r in v_runs])
        conf_m, conf_s = _mean_std([float(r["avg_conflict_pct"]) for r in v_runs])
        erank_m, erank_s = _mean_std([float(r["avg_erank_norm"]) for r in v_runs])
        ren_m, ren_s = _mean_std([float(r["avg_depth_renewal"]) for r in v_runs])
        tps_m, tps_s = _mean_std([float(r["tok_per_sec"]) for r in v_runs])
        vram_m, _ = _mean_std([float(r["peak_vram_gb"]) for r in v_runs])
        time_m, _ = _mean_std([float(r["train_time_sec"]) for r in v_runs])

        ood_keys = list(first["ood_metrics"].keys())
        ood_summary: Dict[str, Dict[str, float]] = {}
        for ok in ood_keys:
            m, s = _mean_std([float(r["ood_metrics"][ok]["bpb"]) for r in v_runs if ok in r["ood_metrics"]])
            ood_summary[ok] = {"mean": m, "std": s}

        ds_summary: Dict[str, Dict[str, float]] = {}
        for dk in first["downstream"].keys():
            metric_key = "acc_pct" if dk == "lambada" else "acc_norm_pct"
            m, s = _mean_std([float(r["downstream"][dk][metric_key]) for r in v_runs if dk in r["downstream"]])
            ds_summary[dk] = {"mean": m, "std": s, "metric": metric_key}

        n_pts = len(first["hist_tokens"])
        curve_mean: List[float] = []
        curve_std: List[float] = []
        for idx in range(n_pts):
            pts = [float(r["hist_val_bpb"][idx]) for r in v_runs if idx < len(r["hist_val_bpb"])]
            pm, ps = _mean_std(pts)
            curve_mean.append(pm)
            curve_std.append(ps)

        n_sub = len(first["layer_conflict"])
        depth_conf_m = [ _mean_std([float(r["layer_conflict"][i]) for r in v_runs])[0] for i in range(n_sub) ]
        depth_erank_m = [ _mean_std([float(r["layer_erank"][i]) for r in v_runs])[0] for i in range(n_sub) ]
        depth_ren_m = [ _mean_std([float(r["layer_renewal"][i]) for r in v_runs])[0] for i in range(n_sub) ]

        agg[v] = {
            "label": first["label"],
            "n_seeds": len(v_runs),
            "total_params": first["total_params"],
            "non_emb_params": first["non_emb_params"],
            "overhead_params": first["overhead_params"],
            "actual_layers": first["actual_layers"],
            "tokens_seen": first["tokens_seen"],
            "approx_flops": first["approx_flops"],
            "val_bpb": {"mean": val_bpb_m, "std": val_bpb_s},
            "val_ppl": {"mean": val_ppl_m, "std": val_ppl_s},
            "ood_avg_bpb": {"mean": ood_bpb_m, "std": ood_bpb_s},
            "ood_breakdown": ood_summary,
            "downstream": ds_summary,
            "conflict_pct": {"mean": conf_m, "std": conf_s},
            "erank_norm": {"mean": erank_m, "std": erank_s},
            "depth_renewal": {"mean": ren_m, "std": ren_s},
            "tok_per_sec": {"mean": tps_m, "std": tps_s},
            "peak_vram_gb": vram_m,
            "train_time_sec": time_m,
            "hist_tokens": first["hist_tokens"],
            "curve_val_bpb_mean": curve_mean,
            "curve_val_bpb_std": curve_std,
            "depth_conflict_mean": depth_conf_m,
            "depth_erank_mean": depth_erank_m,
            "depth_renewal_mean": depth_ren_m,
        }
    return agg


def generate_latex_tables(summary: Dict[str, Dict[str, object]], out_path: str) -> str:
    variants = list(summary.keys())
    best_val = min(summary[v]["val_bpb"]["mean"] for v in variants)
    best_ood = min(summary[v]["ood_avg_bpb"]["mean"] for v in variants)
    best_conf = min(summary[v]["conflict_pct"]["mean"] for v in variants)
    best_tps = max(summary[v]["tok_per_sec"]["mean"] for v in variants)

    ds_names = sorted({k for v in variants for k in summary[v]["downstream"].keys()})
    best_ds: Dict[str, float] = {}
    for dk in ds_names:
        vals = [summary[v]["downstream"][dk]["mean"] for v in variants if dk in summary[v]["downstream"]]
        if vals:
            best_ds[dk] = max(vals)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{From-scratch pretraining scaling results across residual architectures (mean $\pm$ std across seeds). Lower is better for BPB/PPL/Conflict; higher is better for Throughput and Zero-Shot Accuracy.}",
        r"\begin{tabular}{lccccccc" + ("c" * len(ds_names)) + r"}",
        r"\toprule",
    ]

    ds_hdr = "".join(f" & {dk.replace('_', '-').upper()} (\\%)" for dk in ds_names)
    lines.append(
        r"Architecture & Params & Layers & Val BPB $\downarrow$ & Val PPL $\downarrow$ & OOD BPB $\downarrow$ & Conflict (\%) $\downarrow$ & Tok/s $\uparrow$"
        + ds_hdr
        + r" \\"
    )
    lines.append(r"\midrule")

    def _fmt(entry: Dict[str, float], is_best: bool, prec: int = 4) -> str:
        m = entry["mean"]
        s = entry["std"]
        txt = f"{m:.{prec}f} \\pm {s:.{prec}f}" if s > 0 else f"{m:.{prec}f}"
        return f"$\\mathbf{{{txt}}}$" if is_best else f"${txt}$"

    for v in variants:
        r = summary[v]
        lbl = str(r["label"]).replace("_", r"\_")
        p_str = f"{float(r['total_params']) / 1e6:.1f}M"
        l_str = str(r["actual_layers"])
        val_s = _fmt(r["val_bpb"], abs(r["val_bpb"]["mean"] - best_val) < 1e-9, 4)
        ppl_m = r["val_ppl"]["mean"]
        ppl_s = f"{ppl_m:.2f}"
        ood_s = _fmt(r["ood_avg_bpb"], abs(r["ood_avg_bpb"]["mean"] - best_ood) < 1e-9, 4)
        conf_s = _fmt(r["conflict_pct"], abs(r["conflict_pct"]["mean"] - best_conf) < 1e-9, 2)
        tps_m = r["tok_per_sec"]["mean"]
        tps_s = f"\\textbf{{{tps_m:,.0f}}}" if abs(tps_m - best_tps) < 1e-6 else f"{tps_m:,.0f}"
        ds_cells = []
        for dk in ds_names:
            if dk in r["downstream"]:
                cell = _fmt(r["downstream"][dk], abs(r["downstream"][dk]["mean"] - best_ds.get(dk, 1e9)) < 1e-6, 1)
            else:
                cell = "--"
            ds_cells.append(cell)
        ds_str = (" & " + " & ".join(ds_cells)) if ds_cells else ""
        lines.append(f"{lbl} & {p_str} & {l_str} & {val_s} & {ppl_s} & {ood_s} & {conf_s} & {tps_s}{ds_str} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    tex_src = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_src)
    return tex_src


def render_paper_assets(payload: Dict[str, object], png_path: str, pdf_path: str, html_path: str, tex_src: str) -> None:
    summary = payload["summary"]
    meta = payload["meta"]
    variants = list(summary.keys())
    x_idx = list(range(len(variants)))

    fig, axs = plt.subplots(2, 3, figsize=(21, 11.5))
    fig.suptitle(
        f"Alba Residual Architecture Paper Study ({meta['preset']} | ~{summary[variants[0]]['total_params']/1e6:.0f}M Params | {meta['dataset']})",
        fontsize=14,
        fontweight="bold",
    )

    ax = axs[0, 0]
    for v in variants:
        r = summary[v]
        xs = [t / 1e6 for t in r["hist_tokens"]]
        ym = r["curve_val_bpb_mean"]
        ys = r["curve_val_bpb_std"]
        col = PAPER_COLORS.get(v, "#334155")
        ax.plot(xs, ym, marker="o", markersize=3, linewidth=2.0, color=col, label=r["label"])
        if any(s > 0 for s in ys):
            lo = [m - s for m, s in zip(ym, ys)]
            hi = [m + s for m, s in zip(ym, ys)]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.14)
    ax.set_xlabel("Training Tokens Seen (Millions)")
    ax.set_ylabel("Validation BPB (Lower Better)")
    ax.set_title("Panel 1: Pretraining Scaling Curve (Mean ± Std)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.5)

    ax = axs[0, 1]
    in_m = [summary[v]["val_bpb"]["mean"] for v in variants]
    in_s = [summary[v]["val_bpb"]["std"] for v in variants]
    ood_m = [summary[v]["ood_avg_bpb"]["mean"] for v in variants]
    ood_s = [summary[v]["ood_avg_bpb"]["std"] for v in variants]
    ax.bar([i - 0.18 for i in x_idx], in_m, yerr=in_s, capsize=3, width=0.36, color="#0d9488", label="In-Dist Val BPB")
    ax.bar([i + 0.18 for i in x_idx], ood_m, yerr=ood_s, capsize=3, width=0.36, color="#7c3aed", label="External OOD BPB")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(variants, rotation=24, ha="right", fontsize=8)
    ax.set_ylabel("Bits-Per-Byte (Lower Better)")
    ax.set_title("Panel 2: Final In-Dist vs OOD BPB")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    ax = axs[0, 2]
    for v in variants:
        c = summary[v]["depth_conflict_mean"]
        depths = [100.0 * (i + 1) / max(len(c), 1) for i in range(len(c))]
        ax.plot(depths, c, linewidth=1.8, color=PAPER_COLORS.get(v, "#334155"), label=v)
    ax.set_xlabel("Relative Depth (%)")
    ax.set_ylabel("Destructive Conflict Rate (%)")
    ax.set_title("Panel 3: Residual Conflict Across Depth")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.5)

    ax = axs[1, 0]
    for v in variants:
        er = summary[v]["depth_erank_mean"]
        depths = [100.0 * (i + 1) / max(len(er), 1) for i in range(len(er))]
        ax.plot(depths, er, linewidth=1.8, color=PAPER_COLORS.get(v, "#334155"), label=v)
    ax.set_xlabel("Relative Depth (%)")
    ax.set_ylabel("Normalized Effective Rank")
    ax.set_title("Panel 4: Representation Effective Rank Across Depth")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.5)

    ax = axs[1, 1]
    ds_keys = sorted({k for v in variants for k in summary[v]["downstream"].keys()})
    if ds_keys:
        width = 0.72 / max(len(ds_keys), 1)
        ds_cols = ["#2563eb", "#16a34a", "#f97316", "#be185d"]
        for d_idx, dk in enumerate(ds_keys):
            offs = [i + (d_idx - (len(ds_keys) - 1) / 2.0) * width for i in x_idx]
            ys = [summary[v]["downstream"][dk]["mean"] if dk in summary[v]["downstream"] else 0.0 for v in variants]
            ax.bar(offs, ys, width=width, color=ds_cols[d_idx % len(ds_cols)], label=dk.upper())
        ax.set_xticks(x_idx)
        ax.set_xticklabels(variants, rotation=24, ha="right", fontsize=8)
        ax.set_ylabel("Zero-Shot Accuracy (%)")
        ax.set_title("Panel 5: Zero-Shot Downstream Accuracy")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)
    else:
        for v in variants:
            rn = summary[v]["depth_renewal_mean"]
            depths = [100.0 * (i + 1) / max(len(rn), 1) for i in range(len(rn))]
            ax.plot(depths, rn, linewidth=1.8, color=PAPER_COLORS.get(v, "#334155"), label=v)
        ax.set_xlabel("Relative Depth (%)")
        ax.set_ylabel("1 - CKA(h_{l-1}, h_l)")
        ax.set_title("Panel 5: Depth Representation Renewal")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7.5)

    ax = axs[1, 2]
    for v in variants:
        r = summary[v]
        ax.scatter(
            r["tok_per_sec"]["mean"],
            r["val_bpb"]["mean"],
            s=140,
            color=PAPER_COLORS.get(v, "#334155"),
            edgecolor="black",
            zorder=5,
        )
        ax.annotate(
            v,
            (r["tok_per_sec"]["mean"], r["val_bpb"]["mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("Training Throughput (tokens/sec)")
    ax.set_ylabel("Final Val BPB (Lower Better)")
    ax.set_title("Panel 6: Speed vs Quality Pareto (Bottom-Right Best)")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    rows = []
    for v in variants:
        r = summary[v]
        rows.append(
            f"<tr><td><strong>{r['label']}</strong></td>"
            f"<td>{float(r['total_params'])/1e6:.1f}M</td>"
            f"<td>{r['actual_layers']}</td>"
            f"<td>{r['val_bpb']['mean']:.4f} ± {r['val_bpb']['std']:.4f}</td>"
            f"<td>{r['ood_avg_bpb']['mean']:.4f} ± {r['ood_avg_bpb']['std']:.4f}</td>"
            f"<td>{r['conflict_pct']['mean']:.2f}%</td>"
            f"<td>{r['erank_norm']['mean']:.4f}</td>"
            f"<td>{r['tok_per_sec']['mean']:,.0f}</td>"
            f"<td>{r['peak_vram_gb']:.2f} GB</td></tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Alba Residual Paper Scaling Report</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
  .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 18px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; }}
  th {{ background: #f1f5f9; }}
  pre {{ background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; }}
  img {{ max-width: 100%; border-radius: 8px; border: 1px solid #cbd5e1; }}
</style>
</head>
<body>
  <div class="card">
    <h1>Alba Residual Architecture Paper Scaling Report</h1>
    <p>Preset: <strong>{meta['preset']}</strong> | Device: <strong>{meta['device_name']}</strong> | Seeds: <strong>{meta['seeds']}</strong> | Dataset: <strong>{meta['dataset']}</strong> | Optimizer: <strong>{meta['optimizer']}</strong></p>
  </div>
  <div class="card">
    <h2>1. Aggregated Benchmark Results</h2>
    <table>
      <thead><tr><th>Variant</th><th>Params</th><th>Layers</th><th>Val BPB</th><th>OOD BPB</th><th>Conflict</th><th>eRank</th><th>Tok/s</th><th>VRAM</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>2. Publication Figures</h2>
    <img src="data:image/png;base64,{b64}" alt="Paper Scaling Figures" />
  </div>
  <div class="card">
    <h2>3. LaTeX Table (Ready to Paste)</h2>
    <pre>{tex_src}</pre>
  </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-scale 100M+ paper benchmark suite for Alba residual architectures.")
    parser.add_argument("--preset", type=str, default="scale_125m", choices=list(SCALE_PRESETS.keys()))
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_PAPER_VARIANTS))
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "geomuon"])
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", type=str, default="sample-10BT")
    parser.add_argument("--synthetic-lm", action="store_true")
    parser.add_argument("--dim", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=0)
    parser.add_argument("--n-heads", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--corpus-tokens", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--iso-extra-layers", type=int, default=0)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--ckpt-every", type=int, default=250)
    parser.add_argument("--ckpt-mins", type=float, default=15.0)
    parser.add_argument("--max-session-mins", type=float, default=0.0)
    parser.add_argument("--max-steps-per-session", type=int, default=0)
    parser.add_argument("--keep-step-checkpoints", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="gor_research/paper_cache")
    parser.add_argument("--output-dir", type=str, default="gor_research/paper_results")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    preset_cfg = dict(SCALE_PRESETS[args.preset])
    if args.dim > 0:
        preset_cfg["dim"] = args.dim
    if args.n_layers > 0:
        preset_cfg["n_layers"] = args.n_layers
    if args.n_heads > 0:
        preset_cfg["n_heads"] = args.n_heads
    if args.seq_len > 0:
        preset_cfg["seq_len"] = args.seq_len
    if args.batch_size > 0:
        preset_cfg["batch_size"] = args.batch_size
    if args.grad_accum > 0:
        preset_cfg["grad_accum"] = args.grad_accum
    if args.steps > 0:
        preset_cfg["steps"] = args.steps
    if args.corpus_tokens > 0:
        preset_cfg["corpus_tokens"] = args.corpus_tokens
    if args.lr > 0.0:
        preset_cfg["lr"] = args.lr
    if args.grad_ckpt:
        preset_cfg["grad_ckpt"] = True

    os.makedirs(args.output_dir, exist_ok=True)
    if args.synthetic_lm or args.preset == "smoke":
        corpus = build_synthetic_paper_corpus(
            cache_dir=args.cache_dir,
            seq_len=int(preset_cfg["seq_len"]),
            train_tokens=int(preset_cfg["corpus_tokens"]),
            val_tokens=int(preset_cfg["val_tokens"]),
        )
    else:
        corpus = build_or_load_fineweb_edu_corpus(
            cache_dir=args.cache_dir,
            seq_len=int(preset_cfg["seq_len"]),
            train_tokens=int(preset_cfg["corpus_tokens"]),
            val_tokens=int(preset_cfg["val_tokens"]),
            tokenizer_name=args.tokenizer,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
        )

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device).upper()

    session_t0 = time.perf_counter()
    runs: Dict[str, List[Dict[str, object]]] = {v: [] for v in variants}
    for seed in seeds:
        for v in variants:
            res = _train_single_run(
                variant=v,
                seed=seed,
                corpus=corpus,
                preset_cfg=preset_cfg,
                tokenizer_name=args.tokenizer,
                optimizer_name=args.optimizer,
                iso_extra_layers=args.iso_extra_layers,
                use_compile=args.compile,
                save_ckpt=args.save_checkpoints,
                resume=not args.no_resume,
                ckpt_every=args.ckpt_every,
                ckpt_mins=args.ckpt_mins,
                max_session_mins=args.max_session_mins,
                max_steps_per_session=args.max_steps_per_session,
                keep_step_ckpt=args.keep_step_checkpoints,
                session_t0=session_t0,
                output_dir=args.output_dir,
                device=device,
            )
            if res is None:
                print(
                    "[SessionPaused] Training paused after saving exact checkpoint. Re-run the exact same command to resume.",
                    flush=True,
                )
                return
            runs[v].append(res)

    summary = aggregate_seed_runs(variants, runs)
    tex_path = os.path.join(args.output_dir, "paper_tables.tex")
    tex_src = generate_latex_tables(summary, tex_path)

    payload: Dict[str, object] = {
        "meta": {
            "preset": args.preset,
            "device_name": dev_name,
            "optimizer": args.optimizer,
            "seeds": seeds,
            "dataset": corpus.provenance.get("dataset", "synthetic"),
            "provenance": corpus.provenance,
            "preset_cfg": preset_cfg,
        },
        "summary": summary,
        "runs": runs,
    }

    json_path = os.path.join(args.output_dir, "paper_results.json")
    png_path = os.path.join(args.output_dir, "paper_figures.png")
    pdf_path = os.path.join(args.output_dir, "paper_figures.pdf")
    html_path = os.path.join(args.output_dir, "paper_report.html")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    render_paper_assets(payload, png_path=png_path, pdf_path=pdf_path, html_path=html_path, tex_src=tex_src)

    print(f"Saved JSON: {json_path}", flush=True)
    print(f"Saved PNG:  {png_path}", flush=True)
    print(f"Saved PDF:  {pdf_path}", flush=True)
    print(f"Saved TEX:  {tex_path}", flush=True)
    print(f"Saved HTML: {html_path}", flush=True)


if __name__ == "__main__":
    main()
