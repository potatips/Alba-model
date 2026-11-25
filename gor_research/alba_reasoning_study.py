import argparse
import base64
import contextlib
from dataclasses import dataclass
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.config import ModelArgs
from src.model import MiniLLM
from .data_pipeline import _pack_texts_to_blocks, build_synthetic_lm_split
from .run_colab_study import ARCH_COLORS, build_adamw_groups


VARIANT_SPECS: Dict[str, Tuple[str, str, str]] = {
    "pre_norm": ("pre_norm", "swiglu", "Alba MiniLLM Pre-Norm (SwiGLU)"),
    "gor_v2": ("gor_v2", "swiglu", "Alba MiniLLM GOR-v2 (0-Param)"),
    "dual_bank_gor_v2": ("dual_bank_gor_v2", "swiglu", "Alba MiniLLM Dual-Bank + GOR-v2"),
    "attn_res_block": ("attn_res_block", "swiglu", "Alba MiniLLM Block AttnRes"),
    "attn_res_gor_v2": ("attn_res_gor_v2", "swiglu", "Alba MiniLLM AttnRes + GOR-v2"),
    "attn_res_gor_v2_moe": ("attn_res_gor_v2", "os_moe", "Alba MiniLLM AttnRes + GOR-v2 + OS-MoE"),
}

EXTENDED_COLORS = {
    **ARCH_COLORS,
    "attn_res_gor_v2_moe": "#7c3aed",
}


@dataclass
class MultiSkillSplit:
    train_blocks: torch.Tensor
    val_blocks: torch.Tensor
    eval_logic_blocks: torch.Tensor
    eval_code_blocks: torch.Tensor
    eval_tool_blocks: torch.Tensor
    eval_ood_blocks: torch.Tensor
    val_bpt: float
    logic_bpt: float
    code_bpt: float
    tool_bpt: float
    ood_bpt: float
    vocab_size: int
    sources_meta: Dict[str, int]


def _format_gsm8k(row: Dict) -> str:
    return (
        "### Logical Math Reasoning\n"
        f"Question: {str(row['question']).strip()}\n"
        "<think>\n"
        f"{str(row['answer']).strip()}\n"
        "</think>"
    )


def _format_arc(row: Dict) -> str:
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    opts = "\n".join(f"{lb}. {tx}" for lb, tx in zip(labels, texts))
    ans_key = row["answerKey"]
    ans_text = next((tx for lb, tx in zip(labels, texts) if lb == ans_key), "")
    return (
        "### Deductive Science Reasoning\n"
        f"Question: {str(row['question']).strip()}\n"
        f"Options:\n{opts}\n"
        f"<think>\nEvaluate each option against physical laws and select {ans_key}.\n</think>\n"
        f"Answer: {ans_key}. {ans_text}"
    )


def _format_code_alpaca(row: Dict) -> str:
    instr = str(row.get("instruction") or "").strip()
    inp = str(row.get("input") or "").strip()
    out = str(row.get("output") or "").strip()
    inp_block = f"\nInput:\n{inp}" if inp else ""
    return (
        "### Coding Task\n"
        f"{instr}{inp_block}\n"
        f"```python\n{out}\n```"
    )


def _format_mbpp(row: Dict) -> str:
    prompt = str(row.get("prompt") or row.get("text") or "").strip()
    code = str(row.get("code") or "").strip()
    tests = "\n".join((row.get("test_list") or [])[:2])
    return (
        "### Verified Python Synthesis\n"
        f"{prompt}\nTests:\n{tests}\n"
        f"```python\n{code}\n```"
    )


def _format_hf_tool_row(row: Dict) -> Optional[str]:
    if "query" in row and "tools" in row and "answers" in row:
        query = str(row["query"]).strip()
        tools = str(row["tools"]).strip()[:500]
        answers = str(row["answers"]).strip()
        if query and answers:
            return (
                "### Tool Calling Task\n"
                f"<tools>\n{tools}\n</tools>\n"
                f"User: {query}\n"
                f"<think>\nSelect matching tool and bind arguments strictly to schema.\n</think>\n"
                f"<tool_call>\n{answers}\n</tool_call>"
            )
    if "conversations" in row and isinstance(row["conversations"], list):
        turns = []
        for turn in row["conversations"][:4]:
            role = str(turn.get("from") or turn.get("role") or "user")
            val = str(turn.get("value") or turn.get("content") or "").strip()
            if val:
                turns.append(f"{role}: {val}")
        joined = "\n".join(turns)
        if joined and ("tool_call" in joined or "function" in joined or "{" in joined):
            return f"### Tool Calling Task\n{joined[:900]}"
    if "chat" in row and "system" in row:
        text = f"{str(row['system']).strip()[:400]}\n{str(row['chat']).strip()[:600]}"
        if text.strip():
            return f"### Tool Calling Task\n{text}"
    return None


def _build_synthetic_tool_traces(count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    schemas = [
        (
            "run_python",
            '{"name": "run_python", "parameters": {"code": "string", "timeout_sec": "integer"}}',
            lambda i: (
                f"Verify whether matrix size {2 + (i % 7)}x{2 + (i % 7)} yields positive determinant.",
                {"code": f"import numpy as np\nm=np.eye({2 + (i % 7)})*{1 + (i % 5)}\nprint(np.linalg.det(m) > 0)", "timeout_sec": 5},
                "True",
                f"Confirmed: scaled identity matrix ({2 + (i % 7)}x{2 + (i % 7)}) has positive determinant.",
            ),
        ),
        (
            "search_code",
            '{"name": "search_code", "parameters": {"pattern": "string", "path": "string", "case_sensitive": "boolean"}}',
            lambda i: (
                f"Locate definition of `ForwardBlock_{i % 19}` inside `src/layers` with exact casing.",
                {"pattern": f"class ForwardBlock_{i % 19}", "path": "src/layers", "case_sensitive": True},
                f"src/layers/block_{i % 4}.py:42: class ForwardBlock_{i % 19}(nn.Module):",
                f"Found `ForwardBlock_{i % 19}` in `src/layers/block_{i % 4}.py` at line 42.",
            ),
        ),
        (
            "run_pytest",
            '{"name": "run_pytest", "parameters": {"test_path": "string", "keyword": "string", "fail_fast": "boolean"}}',
            lambda i: (
                f"Run unit tests matching `residual_{i % 11}` in `tests/test_arch.py` and stop on first failure.",
                {"test_path": "tests/test_arch.py", "keyword": f"residual_{i % 11}", "fail_fast": True},
                "1 passed in 0.14s",
                f"Test suite `residual_{i % 11}` passed cleanly in 0.14s.",
            ),
        ),
        (
            "query_sql",
            '{"name": "query_sql", "parameters": {"table": "string", "columns": "array", "limit": "integer"}}',
            lambda i: (
                f"Fetch top {5 + (i % 15)} latency and ppl records from `experiments` table.",
                {"table": "experiments", "columns": ["model_id", "val_ppl", "tok_per_sec"], "limit": 5 + (i % 15)},
                f"[{{\"model_id\": \"attn_res_gor_v2_{i % 5}\", \"val_ppl\": {116.2 + (i % 5):.1f}}}]",
                "Retrieved top experiment records sorted by table limit.",
            ),
        ),
        (
            "http_fetch",
            '{"name": "http_fetch", "parameters": {"url": "string", "method": "string", "retries": "integer"}}',
            lambda i: (
                f"Check service health endpoint for worker node {i % 16} with 2 retries.",
                {"url": f"https://api.internal/v1/worker/{i % 16}/health", "method": "GET", "retries": 2},
                "{\"status\": \"ok\", \"queue_depth\": 0}",
                f"Worker node {i % 16} is healthy with empty queue.",
            ),
        ),
    ]
    traces: List[str] = []
    for i in range(count):
        fn_name, schema_json, builder = schemas[rng.randrange(len(schemas))]
        prompt, args_dict, obs, final_ans = builder(i + seed)
        if i % 4 == 0:
            bad_args = dict(args_dict)
            first_key = next(iter(bad_args))
            bad_args[first_key] = None
            trace = (
                "### Multi-Turn Tool Recovery\n"
                f"<tools>\n{schema_json}\n</tools>\n"
                f"User: {prompt}\n"
                "<think>\nInvoke schema-matching tool.\n</think>\n"
                f"<tool_call>\n{json.dumps({'name': fn_name, 'arguments': bad_args})}\n</tool_call>\n"
                f"<tool_response>\n{{\"error\": \"SchemaValidationError: missing non-null '{first_key}'\"}}\n</tool_response>\n"
                f"<think>\nRepair null '{first_key}' argument from user request and retry.\n</think>\n"
                f"<tool_call>\n{json.dumps({'name': fn_name, 'arguments': args_dict})}\n</tool_call>\n"
                f"<tool_response>\n{obs}\n</tool_response>\n"
                f"Assistant: {final_ans}"
            )
        else:
            trace = (
                "### Tool Calling Task\n"
                f"<tools>\n{schema_json}\n</tools>\n"
                f"User: {prompt}\n"
                "<think>\nParse constraints and emit valid JSON tool arguments.\n</think>\n"
                f"<tool_call>\n{json.dumps({'name': fn_name, 'arguments': args_dict})}\n</tool_call>\n"
                f"<tool_response>\n{obs}\n</tool_response>\n"
                f"Assistant: {final_ans}"
            )
        traces.append(trace)
    return traces


def _load_hf_tool_texts(max_samples: int, seed: int) -> Tuple[List[str], int]:
    from datasets import load_dataset

    candidates = [
        ("Salesforce/xlam-function-calling-60k", None, "train"),
        ("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train"),
    ]
    collected: List[str] = []
    for ds_name, cfg_name, split_name in candidates:
        try:
            ds = load_dataset(ds_name, cfg_name, split=split_name) if cfg_name else load_dataset(ds_name, split=split_name)
            rng = random.Random(seed + 77)
            indices = rng.sample(range(len(ds)), min(max_samples, len(ds)))
            for idx in indices:
                formatted = _format_hf_tool_row(ds[idx])
                if formatted:
                    collected.append(formatted)
            if len(collected) >= 120:
                return collected, len(collected)
        except Exception:
            continue
    return [], 0


def build_reasoning_dataset(
    seq_len: int = 256,
    max_train_tokens: int = 4_500_000,
    max_eval_tokens: int = 180_000,
    tokenizer_name: str = "gpt2",
    seed: int = 42,
) -> MultiSkillSplit:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.eos_token_id = tokenizer.pad_token_id

    rng = random.Random(seed)
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    arc_train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    code_alpaca = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    mbpp_train = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    mbpp_test = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    train_logic = [_format_gsm8k(gsm_train[i]) for i in range(min(3000, len(gsm_train)))] + [
        _format_arc(arc_train[i]) for i in range(min(len(arc_train), 1100))
    ]
    eval_logic = [_format_gsm8k(gsm_test[i]) for i in range(min(400, len(gsm_test)))] + [
        _format_arc(arc_test[i]) for i in range(min(400, len(arc_test)))
    ]

    train_code = [_format_mbpp(mbpp_train[i]) for i in range(len(mbpp_train))] + [
        _format_code_alpaca(code_alpaca[i]) for i in range(min(3500, len(code_alpaca) - 1000))
    ]
    alpaca_tail_start = max(len(code_alpaca) - 1000, 0)
    eval_code = [_format_mbpp(mbpp_test[i]) for i in range(len(mbpp_test))] + [
        _format_code_alpaca(code_alpaca[i]) for i in range(alpaca_tail_start, len(code_alpaca))
    ]

    hf_tools, n_hf_tools = _load_hf_tool_texts(max_samples=2200, seed=seed)
    synth_tools_train = _build_synthetic_tool_traces(1400, seed=seed + 11)
    synth_tools_eval = _build_synthetic_tool_traces(450, seed=seed + 909)
    if hf_tools:
        split_at = max(1, int(0.8 * len(hf_tools)))
        train_tool = hf_tools[:split_at] + synth_tools_train
        eval_tool = hf_tools[split_at:] + synth_tools_eval
    else:
        train_tool = synth_tools_train
        eval_tool = synth_tools_eval

    ood_texts: List[str] = []
    try:
        mmlu_fl = load_dataset("cais/mmlu", "formal_logic", split="test")
        mmlu_math = load_dataset("cais/mmlu", "high_school_mathematics", split="test")
        humaneval = load_dataset("openai/openai_humaneval", split="test")
        for row in mmlu_fl:
            opts = "\n".join(f"({c}) {t}" for c, t in zip(["A", "B", "C", "D"], row["choices"]))
            ood_texts.append(f"Question: {row['question']}\n{opts}\nAnswer: ({['A','B','C','D'][row['answer']]})")
        for row in mmlu_math:
            opts = "\n".join(f"({c}) {t}" for c, t in zip(["A", "B", "C", "D"], row["choices"]))
            ood_texts.append(f"Question: {row['question']}\n{opts}\nAnswer: ({['A','B','C','D'][row['answer']]})")
        for row in humaneval:
            ood_texts.append(f"{row['prompt']}{row['canonical_solution']}")
    except Exception:
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        ood_texts.extend(r["text"] for r in wikitext if len(r["text"].strip()) >= 120)

    train_all = train_logic + train_code + train_tool
    rng.shuffle(train_all)
    val_all = eval_logic[:150] + eval_code[:150] + eval_tool[:150]
    rng.shuffle(val_all)

    train_blocks, _ = _pack_texts_to_blocks(train_all, tokenizer, seq_len=seq_len, max_tokens=max_train_tokens)
    val_blocks, val_bpt = _pack_texts_to_blocks(val_all, tokenizer, seq_len=seq_len, max_tokens=max_eval_tokens)
    logic_blocks, logic_bpt = _pack_texts_to_blocks(eval_logic, tokenizer, seq_len=seq_len, max_tokens=max_eval_tokens)
    code_blocks, code_bpt = _pack_texts_to_blocks(eval_code, tokenizer, seq_len=seq_len, max_tokens=max_eval_tokens)
    tool_blocks, tool_bpt = _pack_texts_to_blocks(eval_tool, tokenizer, seq_len=seq_len, max_tokens=max_eval_tokens)
    ood_blocks, ood_bpt = _pack_texts_to_blocks(ood_texts, tokenizer, seq_len=seq_len, max_tokens=max_eval_tokens)

    return MultiSkillSplit(
        train_blocks=train_blocks,
        val_blocks=val_blocks,
        eval_logic_blocks=logic_blocks,
        eval_code_blocks=code_blocks,
        eval_tool_blocks=tool_blocks,
        eval_ood_blocks=ood_blocks,
        val_bpt=val_bpt,
        logic_bpt=logic_bpt,
        code_bpt=code_bpt,
        tool_bpt=tool_bpt,
        ood_bpt=ood_bpt,
        vocab_size=int(len(tokenizer)),
        sources_meta={
            "train_logic": len(train_logic),
            "train_code": len(train_code),
            "train_tool": len(train_tool),
            "hf_tool_rows": n_hf_tools,
            "ood_items": len(ood_texts),
        },
    )


def clone_alba_shared_weights(source: MiniLLM, target: MiniLLM) -> None:
    with torch.no_grad():
        target.tok_embeddings.weight.copy_(source.tok_embeddings.weight)
        target.norm.load_state_dict(source.norm.state_dict())
        for s_layer, t_layer in zip(source.layers, target.layers):
            t_layer.attention_norm.load_state_dict(s_layer.attention_norm.state_dict())
            t_layer.attention.load_state_dict(s_layer.attention.state_dict())
            t_layer.ffn_norm.load_state_dict(s_layer.ffn_norm.state_dict())
            if s_layer.ffn_type == t_layer.ffn_type:
                t_layer.feed_forward.load_state_dict(s_layer.feed_forward.state_dict())


def auto_tune_alba_profile(device: torch.device, quick: bool) -> Dict[str, int]:
    if quick:
        return {
            "dim": 64,
            "layers": 4,
            "heads": 4,
            "seq": 64,
            "steps": 6,
            "bs": 4,
            "ga": 1,
        }
    if device.type != "cuda":
        return {
            "dim": 128,
            "layers": 6,
            "heads": 4,
            "seq": 128,
            "steps": 60,
            "bs": 8,
            "ga": 1,
        }
    vram_gb = float(torch.cuda.get_device_properties(device).total_memory / (1024 ** 3))
    if vram_gb >= 35.0:
        return {
            "dim": 384,
            "layers": 12,
            "heads": 6,
            "seq": 384,
            "steps": 600,
            "bs": 24,
            "ga": 2,
        }
    if vram_gb >= 15.0:
        return {
            "dim": 384,
            "layers": 12,
            "heads": 6,
            "seq": 256,
            "steps": 450,
            "bs": 12,
            "ga": 2,
        }
    return {
        "dim": 256,
        "layers": 8,
        "heads": 4,
        "seq": 256,
        "steps": 350,
        "bs": 12,
        "ga": 2,
    }


@torch.no_grad()
def evaluate_split(
    model: MiniLLM,
    blocks: torch.Tensor,
    bytes_per_token: float,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 36,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_blocks = blocks.size(0)
    n_batches = min(max(1, (n_blocks + batch_size - 1) // batch_size), max_batches)
    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end = min(n_blocks, start + batch_size)
        batch = blocks[start:end].to(device, non_blocking=True)
        xb = batch[:, :-1]
        yb = batch[:, 1:]
        ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
        with ctx:
            _, loss = model(xb, targets=yb)
        ntok = int(yb.numel())
        total_loss += float(loss.item()) * ntok
        total_tokens += ntok
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    bpb = avg_loss / (math.log(2.0) * max(bytes_per_token, 1e-6))
    model.train()
    return avg_loss, ppl, bpb


def run_benchmark(
    variant_names: List[str],
    split: MultiSkillSplit,
    device: torch.device,
    profile: Dict[str, int],
    lr: float = 6e-4,
    seed: int = 42,
) -> Dict[str, Dict[str, object]]:
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = amp_enabled and (amp_dtype == torch.float16)

    torch.manual_seed(seed)
    ref_args = ModelArgs(
        vocab_size=split.vocab_size,
        dim=profile["dim"],
        n_layers=profile["layers"],
        n_heads=profile["heads"],
        max_seq_len=profile["seq"],
        residual_type="pre_norm",
        ffn_type="swiglu",
    )
    ref_model = MiniLLM(ref_args)

    results: Dict[str, Dict[str, object]] = {}
    steps = profile["steps"]
    batch_size = profile["bs"]
    grad_accum = profile["ga"]
    eval_every = max(2, steps // 10)
    warmup_steps = max(2, int(0.08 * steps))

    print(
        f"[Alba MiniLLM Reasoning Study] L={profile['layers']}, d={profile['dim']}, H={profile['heads']}, "
        f"T={profile['seq']}, steps={steps}, eff_bs={batch_size * grad_accum}",
        flush=True,
    )

    for name in variant_names:
        res_type, ffn_type, label = VARIANT_SPECS[name]
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        args = ModelArgs(
            vocab_size=split.vocab_size,
            dim=profile["dim"],
            n_layers=profile["layers"],
            n_heads=profile["heads"],
            max_seq_len=profile["seq"],
            residual_type=res_type,
            ffn_type=ffn_type,
        )
        torch.manual_seed(seed)
        model = MiniLLM(args).to(device)
        clone_alba_shared_weights(ref_model, model)

        opt = torch.optim.AdamW(build_adamw_groups(model, 0.1), lr=lr, betas=(0.9, 0.95), eps=1e-8)
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        def lr_lambda(step_idx: int) -> float:
            if step_idx < warmup_steps:
                return float(step_idx + 1) / float(max(warmup_steps, 1))
            progress = float(step_idx - warmup_steps) / float(max(steps - warmup_steps, 1))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        gen = torch.Generator(device="cpu").manual_seed(seed + 409)
        n_train_blocks = split.train_blocks.size(0)

        hist_steps: List[int] = [0]
        init_loss, _, init_bpb = evaluate_split(
            model, split.val_blocks, split.val_bpt, device, batch_size, amp_enabled, amp_dtype, max_batches=16
        )
        hist_val_bpb: List[float] = [init_bpb]

        t0 = time.perf_counter()
        total_tok = 0
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                idx = torch.randint(0, n_train_blocks, (batch_size,), generator=gen)
                batch = split.train_blocks[idx].to(device, non_blocking=True)
                xb = batch[:, :-1]
                yb = batch[:, 1:]
                total_tok += int(yb.numel())
                ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
                with ctx:
                    _, loss = model(xb, targets=yb)
                    scaled = loss / float(grad_accum)
                if use_scaler:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

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
                _, _, vbpb = evaluate_split(
                    model, split.val_blocks, split.val_bpt, device, batch_size, amp_enabled, amp_dtype, max_batches=20
                )
                hist_steps.append(step)
                hist_val_bpb.append(vbpb)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_vram_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        else:
            peak_vram_gb = 0.0
        elapsed = time.perf_counter() - t0
        tok_s = float(total_tok) / max(elapsed, 1e-6)

        val_loss, val_ppl, val_bpb = evaluate_split(
            model, split.val_blocks, split.val_bpt, device, batch_size, amp_enabled, amp_dtype
        )
        logic_loss, logic_ppl, logic_bpb = evaluate_split(
            model, split.eval_logic_blocks, split.logic_bpt, device, batch_size, amp_enabled, amp_dtype
        )
        code_loss, code_ppl, code_bpb = evaluate_split(
            model, split.eval_code_blocks, split.code_bpt, device, batch_size, amp_enabled, amp_dtype
        )
        tool_loss, tool_ppl, tool_bpb = evaluate_split(
            model, split.eval_tool_blocks, split.tool_bpt, device, batch_size, amp_enabled, amp_dtype
        )
        ood_loss, ood_ppl, ood_bpb = evaluate_split(
            model, split.eval_ood_blocks, split.ood_bpt, device, batch_size, amp_enabled, amp_dtype
        )

        results[name] = {
            "label": label,
            "residual_type": res_type,
            "ffn_type": ffn_type,
            "total_params": int(model.count_parameters()),
            "active_params": int(model.count_active_parameters()),
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "val_bpb": val_bpb,
            "logic_loss": logic_loss,
            "logic_ppl": logic_ppl,
            "logic_bpb": logic_bpb,
            "code_loss": code_loss,
            "code_ppl": code_ppl,
            "code_bpb": code_bpb,
            "tool_loss": tool_loss,
            "tool_ppl": tool_ppl,
            "tool_bpb": tool_bpb,
            "ood_loss": ood_loss,
            "ood_ppl": ood_ppl,
            "ood_bpb": ood_bpb,
            "tok_per_sec": tok_s,
            "peak_vram_gb": peak_vram_gb,
            "train_time_sec": elapsed,
            "hist_steps": hist_steps,
            "hist_val_bpb": hist_val_bpb,
        }
        print(
            f"  {name:20s} | ValBPB={val_bpb:.4f} | Logic={logic_bpb:.4f} | Code={code_bpb:.4f} | "
            f"Tool={tool_bpb:.4f} | OOD={ood_bpb:.4f} | {tok_s:,.0f} tok/s",
            flush=True,
        )

    return results


def render_alba_report(payload: Dict[str, object], png_path: str, html_path: str) -> None:
    res = payload["results"]
    names = list(res.keys())

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Alba MiniLLM Integration Study: Logic + Coding + Tool Calling + OOD",
        fontsize=14,
        fontweight="bold",
    )

    ax = axs[0, 0]
    x_idx = range(len(names))
    w = 0.2
    ax.bar([i - 1.5 * w for i in x_idx], [res[n]["logic_bpb"] for n in names], width=w, color="#2563eb", label="Logic BPB")
    ax.bar([i - 0.5 * w for i in x_idx], [res[n]["code_bpb"] for n in names], width=w, color="#16a34a", label="Code BPB")
    ax.bar([i + 0.5 * w for i in x_idx], [res[n]["tool_bpb"] for n in names], width=w, color="#f59e0b", label="Tool BPB")
    ax.bar([i + 1.5 * w for i in x_idx], [res[n]["ood_bpb"] for n in names], width=w, color="#7c3aed", label="OOD BPB")
    ax.set_xticks(list(x_idx))
    ax.set_xticklabels(names, rotation=24, ha="right", fontsize=8)
    ax.set_ylabel("Bits-Per-Byte (Lower Better)")
    ax.set_title("Panel 1: Per-Domain Logic / Code / Tool / OOD BPB")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    for n in names:
        r = res[n]
        ax.plot(r["hist_steps"], r["hist_val_bpb"], marker="o", markersize=3, linewidth=2, color=EXTENDED_COLORS.get(n, "#333333"), label=n)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Validation BPB (Lower Better)")
    ax.set_title("Panel 2: Multi-Skill Training Convergence")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    for n in names:
        r = res[n]
        ax.scatter(r["tool_bpb"], r["ood_bpb"], s=110, color=EXTENDED_COLORS.get(n, "#333333"))
        ax.annotate(n, (r["tool_bpb"], r["ood_bpb"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Tool Calling BPB (Lower Better)")
    ax.set_ylabel("OOD BPB (Lower Better)")
    ax.set_title("Panel 3: Tool Fidelity vs OOD Generalization (Bottom-Left Best)")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax = axs[1, 1]
    for n in names:
        r = res[n]
        ax.scatter(r["tok_per_sec"], r["val_bpb"], s=110, color=EXTENDED_COLORS.get(n, "#333333"))
        ax.annotate(n, (r["tok_per_sec"], r["val_bpb"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Throughput (tokens/sec)")
    ax.set_ylabel("Overall Val BPB (Lower Better)")
    ax.set_title("Panel 4: Speed vs Quality Pareto")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(png_path, dpi=180)
    plt.close(fig)

    with open(png_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")

    rows = []
    for n in names:
        r = res[n]
        rows.append(
            f"<tr><td>{n}</td><td>{r['active_params']:,}</td><td>{r['val_bpb']:.4f}</td>"
            f"<td>{r['logic_bpb']:.4f}</td><td>{r['code_bpb']:.4f}</td><td>{r['tool_bpb']:.4f}</td>"
            f"<td>{r['ood_bpb']:.4f}</td><td>{r['tok_per_sec']:,.0f}</td><td>{r['peak_vram_gb']:.2f} GB</td></tr>"
        )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Alba MiniLLM Logic/Code/Tool Study</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#0f172a}"
        "table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}"
        "th,td{border:1px solid #cbd5e1;padding:7px 9px;text-align:right}th:first-child,td:first-child{text-align:left}"
        "th{background:#f1f5f9}img{max-width:100%;border:1px solid #e2e8f0;border-radius:8px}</style></head><body>"
        f"<h1>Alba MiniLLM: Logic + Coding + Tool Calling Report</h1>"
        f"<p>Meta: {json.dumps(payload['meta'])}</p>"
        "<table><thead><tr><th>Variant</th><th>Active Params</th><th>Val BPB</th><th>Logic BPB</th>"
        "<th>Code BPB</th><th>Tool BPB</th><th>OOD BPB</th><th>Tok/s</th><th>VRAM</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<img src='data:image/png;base64,{img_b64}'/></body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alba MiniLLM Logic + Code + Tool Study")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--variants",
        type=str,
        default="pre_norm,gor_v2,dual_bank_gor_v2,attn_res_block,attn_res_gor_v2,attn_res_gor_v2_moe",
    )
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_alba"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    profile = auto_tune_alba_profile(device, quick=args.quick)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.quick:
        syn = build_synthetic_lm_split(seq_len=profile["seq"], seed=args.seed)
        split = MultiSkillSplit(
            train_blocks=syn.train_blocks,
            val_blocks=syn.val_blocks,
            eval_logic_blocks=syn.val_blocks,
            eval_code_blocks=syn.val_blocks,
            eval_tool_blocks=syn.val_blocks,
            eval_ood_blocks=syn.ood_blocks,
            val_bpt=syn.val_bytes_per_token,
            logic_bpt=syn.val_bytes_per_token,
            code_bpt=syn.val_bytes_per_token,
            tool_bpt=syn.val_bytes_per_token,
            ood_bpt=syn.ood_bytes_per_token,
            vocab_size=syn.vocab_size,
            sources_meta={"synthetic": 1},
        )
    else:
        split = build_reasoning_dataset(seq_len=profile["seq"], seed=args.seed)

    results = run_benchmark(variants, split=split, device=device, profile=profile, seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "alba_reasoning_results.json")
    png_path = os.path.join(args.output_dir, "alba_reasoning_report.png")
    html_path = os.path.join(args.output_dir, "alba_reasoning_report.html")

    payload = {
        "meta": {
            "device_name": dev_name,
            **profile,
            "vocab_size": split.vocab_size,
            "sources": split.sources_meta,
        },
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    render_alba_report(payload, png_path=png_path, html_path=html_path)
    print(f"Saved JSON: {json_path}\nSaved PNG: {png_path}\nSaved HTML: {html_path}", flush=True)


if __name__ == "__main__":
    main()
