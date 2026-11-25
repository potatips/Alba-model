import gc
import json
import math
import os
import random
import time
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_dataset import (
    CausalTextDataset,
    build_or_load_logic_code_dataset,
    build_or_load_ood_eval_dataset,
    get_eval_texts,
    get_ood_eval_texts,
)
from qwen_eval_metrics import evaluate_qwen_variant_full, generate_ablation_html_report
from qwen_geo_muon import create_qwen_adamw_optimizer, create_qwen_geomuon_optimizer
from qwen_moefication import build_qwen_variant, collect_ortho_loss, count_model_params


GOR_VS_BASELINE_VARIANTS = [
    {
        "name": "Qwen-2.5-3B (Base Untouched)",
        "enable_gor": False,
        "enable_moe": False,
        "optimizer": "none",
    },
    {
        "name": "Qwen-2.5-3B (Baseline AdamW)",
        "enable_gor": False,
        "enable_moe": False,
        "optimizer": "adamw",
    },
    {
        "name": "Qwen-2.5-3B (Baseline GEO-Muon)",
        "enable_gor": False,
        "enable_moe": False,
        "optimizer": "geo_muon",
    },
    {
        "name": "Qwen-2.5-3B + GOR (AdamW)",
        "enable_gor": True,
        "enable_moe": False,
        "optimizer": "adamw",
    },
    {
        "name": "Qwen-2.5-3B + GOR (GEO-Muon)",
        "enable_gor": True,
        "enable_moe": False,
        "optimizer": "geo_muon",
    },
]


GOR_ONLY_VARIANTS = [
    {
        "name": "Qwen-2.5-3B (Base Untouched)",
        "enable_gor": False,
        "enable_moe": False,
        "optimizer": "none",
    },
    {
        "name": "Qwen-2.5-3B + GOR (AdamW)",
        "enable_gor": True,
        "enable_moe": False,
        "optimizer": "adamw",
    },
    {
        "name": "Qwen-2.5-3B + GOR (GEO-Muon)",
        "enable_gor": True,
        "enable_moe": False,
        "optimizer": "geo_muon",
    },
]

ABLATION_VARIANTS = [
    {
        "name": "Qwen-2.5-3B (Base Untouched)",
        "enable_gor": False,
        "enable_moe": False,
        "optimizer": "none",
    },
    {
        "name": "Qwen-2.5-3B + GOR (AdamW)",
        "enable_gor": True,
        "enable_moe": False,
        "optimizer": "adamw",
    },
    {
        "name": "Qwen-2.5-3B + GOR (GEO-Muon)",
        "enable_gor": True,
        "enable_moe": False,
        "optimizer": "geo_muon",
    },
    {
        "name": "Qwen-2.5-3B + OS-MoE (AdamW)",
        "enable_gor": False,
        "enable_moe": True,
        "optimizer": "adamw",
    },
    {
        "name": "Qwen-2.5-3B + OS-MoE (GEO-Muon)",
        "enable_gor": False,
        "enable_moe": True,
        "optimizer": "geo_muon",
    },
    {
        "name": "Qwen-2.5-3B + OS-MoE + GOR (AdamW)",
        "enable_gor": True,
        "enable_moe": True,
        "optimizer": "adamw",
    },
    {
        "name": "Qwen-2.5-3B + OS-MoE + GOR (GEO-Muon)",
        "enable_gor": True,
        "enable_moe": True,
        "optimizer": "geo_muon",
    },
]


def _seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.1) -> float:
    if step <= warmup_steps:
        return max(step / max(warmup_steps, 1), 0.05)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _load_base_model(model_name: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model.to(device)
    return model


def train_single_variant(
    variant_cfg: Dict,
    model_name: str,
    tokenizer,
    train_dataset: CausalTextDataset,
    eval_logic_texts: List[str],
    eval_code_texts: List[str],
    ood_eval_texts: Optional[Dict[str, List[str]]] = None,
    total_steps: int = 200,
    micro_batch_size: int = 2,
    grad_accum_steps: int = 8,
    adamw_lr: float = 5e-5,
    muon_lr: float = 1e-4,
    ortho_weight: float = 0.01,
    moe_mode: str = "lite",
    num_experts: int = 8,
    top_k: int = 2,
    seed: int = 42,
    save_checkpoint_dir: Optional[str] = None,
) -> Dict:
    _seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    v_name = variant_cfg["name"]
    print(f"\n============================================================")
    print(f"Running Variant: {v_name}")
    print(f"============================================================")

    t_start = time.perf_counter()
    model = _load_base_model(model_name, device=device, dtype=dtype)
    if variant_cfg["enable_gor"] or variant_cfg["enable_moe"]:
        model = build_qwen_variant(
            model=model,
            enable_gor=variant_cfg["enable_gor"],
            enable_moe=variant_cfg["enable_moe"],
            moe_mode=moe_mode,
            num_experts=num_experts,
            top_k=top_k,
        )

    p_info = count_model_params(model)
    print(f"Params -> Total: {p_info['total_params_b']}B | Active: {p_info['active_params_b']}B")

    train_loss_curve = []
    opt_type = variant_cfg["optimizer"]

    if opt_type != "none" and total_steps > 0:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        if opt_type == "geo_muon":
            optimizer = create_qwen_geomuon_optimizer(model, lr=muon_lr, weight_decay=0.01)
        else:
            optimizer = create_qwen_adamw_optimizer(model, lr=adamw_lr, weight_decay=0.01)

        base_lrs = [g["lr"] for g in optimizer.param_groups]
        warmup_steps = max(1, int(total_steps * 0.1))

        g_loader = torch.Generator()
        g_loader.manual_seed(seed)
        loader = DataLoader(
            train_dataset,
            batch_size=micro_batch_size,
            shuffle=True,
            drop_last=True,
            generator=g_loader,
        )
        data_iter = iter(loader)

        model.train()
        for step in range(1, total_steps + 1):
            lr_mult = _cosine_lr_multiplier(step, total_steps, warmup_steps)
            for grp, base_lr in zip(optimizer.param_groups, base_lrs):
                grp["lr"] = base_lr * lr_mult

            optimizer.zero_grad(set_to_none=True)
            accum_lm = 0.0
            accum_ortho = 0.0

            for _ in range(grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                lm_loss = outputs.loss
                ortho_loss = collect_ortho_loss(model, device=device)
                total_loss = (lm_loss + ortho_weight * ortho_loss) / grad_accum_steps
                total_loss.backward()

                accum_lm += float(lm_loss.item()) / grad_accum_steps
                accum_ortho += float(ortho_loss.item()) / grad_accum_steps

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step == 1 or step % 10 == 0 or step == total_steps:
                train_loss_curve.append([step, round(accum_lm, 4)])
                print(
                    f"  Step {step:04d}/{total_steps} | LM Loss: {accum_lm:.4f} | "
                    f"Ortho Loss: {accum_ortho:.4f} | LR mult: {lr_mult:.3f}"
                )

        if save_checkpoint_dir:
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            safe_name = v_name.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
            ckpt_path = os.path.join(save_checkpoint_dir, f"{safe_name}.pt")
            torch.save(model.state_dict(), ckpt_path)

        del optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    eval_metrics = evaluate_qwen_variant_full(
        model=model,
        tokenizer=tokenizer,
        eval_logic_texts=eval_logic_texts,
        eval_code_texts=eval_code_texts,
        ood_eval_texts=ood_eval_texts,
        max_length=512,
        eval_batch_size=4,
        device=device,
    )

    peak_vram = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else 0.0
    elapsed_sec = round(time.perf_counter() - t_start, 1)

    eval_metrics["train_loss_curve"] = train_loss_curve
    eval_metrics["peak_vram_gb"] = peak_vram
    eval_metrics["elapsed_sec"] = elapsed_sec
    eval_metrics["enable_gor"] = variant_cfg["enable_gor"]
    eval_metrics["enable_moe"] = variant_cfg["enable_moe"]
    eval_metrics["optimizer"] = opt_type

    ood_summary = f" | OOD BPB: {eval_metrics['ood_bpb']}" if "ood_bpb" in eval_metrics else ""
    print(
        f"Finished {v_name} in {elapsed_sec}s | Overall BPB: {eval_metrics['overall_bpb']} "
        f"(Logic: {eval_metrics['logic_bpb']}, Code: {eval_metrics['code_bpb']}){ood_summary} | "
        f"Conflict: {eval_metrics['avg_conflict_pct']}% | Peak VRAM: {peak_vram} GB"
    )

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return eval_metrics


def run_qwen3b_ablation_suite(
    model_name: str = "Qwen/Qwen2.5-3B",
    total_steps: int = 200,
    seq_len: int = 512,
    micro_batch_size: int = 2,
    grad_accum_steps: int = 8,
    n_train_logic: int = 2000,
    n_train_code: int = 2000,
    n_eval_logic: int = 100,
    n_eval_code: int = 100,
    adamw_lr: float = 5e-5,
    muon_lr: float = 1e-4,
    ortho_weight: float = 0.01,
    moe_mode: str = "lite",
    dataset_cache_path: str = "qwen_logic_code_dataset.json",
    ood_cache_path: str = "qwen_ood_eval_dataset.json",
    n_ood_per_domain: int = 80,
    eval_ood: bool = True,
    results_cache_path: str = "qwen3b_ablation_results.json",
    output_html_path: str = "qwen3b_ablation_report.html",
    output_png_path: str = "qwen3b_ablation_grid.png",
    save_checkpoint_dir: Optional[str] = None,
    variants: Optional[List[Dict]] = None,
    seed: int = 42,
) -> Dict[str, Dict]:
    active_variants = variants if variants is not None else ABLATION_VARIANTS
    dataset_payload = build_or_load_logic_code_dataset(
        cache_path=dataset_cache_path,
        n_train_logic=n_train_logic,
        n_train_code=n_train_code,
        n_eval_logic=n_eval_logic,
        n_eval_code=n_eval_code,
        seed=seed,
    )
    print(f"Dataset ready: {dataset_payload['meta']}")

    ood_eval_texts = None
    if eval_ood:
        ood_payload = build_or_load_ood_eval_dataset(
            cache_path=ood_cache_path,
            n_ood_logic=n_ood_per_domain,
            n_ood_code=n_ood_per_domain,
            n_ood_general=n_ood_per_domain,
            seed=seed,
        )
        ood_eval_texts = get_ood_eval_texts(ood_payload)
        print(f"OOD Eval Dataset ready: {ood_payload['meta']}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CausalTextDataset(dataset_payload["train"], tokenizer, max_length=seq_len)
    eval_logic_texts, eval_code_texts = get_eval_texts(dataset_payload)

    results = {}
    if os.path.exists(results_cache_path):
        try:
            with open(results_cache_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Resumed {len(results)} completed variants from {results_cache_path}")
        except Exception:
            results = {}

    for cfg in active_variants:
        v_name = cfg["name"]
        has_required = v_name in results and "overall_bpb" in results[v_name]
        if eval_ood:
            has_required = has_required and "ood_bpb" in results[v_name]
        if has_required:
            print(f"Skipping already completed variant: {v_name}")
            continue

        metrics = train_single_variant(
            variant_cfg=cfg,
            model_name=model_name,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_logic_texts=eval_logic_texts,
            eval_code_texts=eval_code_texts,
            ood_eval_texts=ood_eval_texts,
            total_steps=total_steps,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            adamw_lr=adamw_lr,
            muon_lr=muon_lr,
            ortho_weight=ortho_weight,
            moe_mode=moe_mode,
            seed=seed,
            save_checkpoint_dir=save_checkpoint_dir,
        )
        results[v_name] = metrics
        with open(results_cache_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    report_path = generate_ablation_html_report(
        results=results,
        output_html_path=output_html_path,
        output_png_path=output_png_path,
    )
    print(f"Ablation HTML report exported to: {report_path}")
    return results


if __name__ == "__main__":
    run_qwen3b_ablation_suite()
