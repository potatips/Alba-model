import json
import os
import random
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def _format_gsm8k(row: Dict) -> str:
    return (
        "### Logical Math Problem\n"
        f"{row['question'].strip()}\n\n"
        "### Step-by-Step Reasoning & Answer\n"
        f"{row['answer'].strip()}"
    )


def _format_arc(row: Dict) -> str:
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    opts = "\n".join(f"{lb}. {tx}" for lb, tx in zip(labels, texts))
    ans_key = row["answerKey"]
    ans_text = ""
    for lb, tx in zip(labels, texts):
        if lb == ans_key:
            ans_text = tx
            break
    return (
        "### Logical Science Question\n"
        f"{row['question'].strip()}\n"
        f"Options:\n{opts}\n\n"
        "### Correct Logical Answer\n"
        f"{ans_key}. {ans_text}"
    )


def _format_code_alpaca(row: Dict) -> str:
    instr = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    out = (row.get("output") or "").strip()
    if inp:
        return (
            "### Coding Task\n"
            f"{instr}\nInput:\n{inp}\n\n"
            "### Python Solution\n"
            f"{out}"
        )
    return (
        "### Coding Task\n"
        f"{instr}\n\n"
        "### Python Solution\n"
        f"{out}"
    )


def _format_mbpp(row: Dict) -> str:
    prompt = (row.get("prompt") or "").strip()
    code = (row.get("code") or "").strip()
    tests = row.get("test_list") or []
    test_str = "\n".join(tests[:2])
    return (
        "### Python Programming Problem\n"
        f"{prompt}\n"
        f"Tests:\n{test_str}\n\n"
        "### Verified Python Implementation\n"
        f"```python\n{code}\n```"
    )


def _format_hf_tool_row(row: Dict) -> str:
    if "query" in row and "tools" in row and "answers" in row:
        query = str(row["query"]).strip()
        tools = str(row["tools"]).strip()[:500]
        answers = str(row["answers"]).strip()
        if query and answers:
            return (
                "### Tool Calling Task\n"
                f"<tools>\n{tools}\n</tools>\n"
                f"User: {query}\n"
                "<think>\nSelect matching tool and bind arguments strictly to schema.\n</think>\n"
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
    return ""


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


def _load_hf_tool_texts(max_samples: int, seed: int) -> List[str]:
    candidates = [
        ("Salesforce/xlam-function-calling-60k", None, "train"),
        ("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train"),
    ]
    collected: List[str] = []
    for ds_name, cfg_name, split_name in candidates:
        try:
            ds = load_dataset(ds_name, cfg_name, split=split_name) if cfg_name else load_dataset(ds_name, split=split_name)
            rng = random.Random(seed + 77)
            for idx in rng.sample(range(len(ds)), min(max_samples, len(ds))):
                formatted = _format_hf_tool_row(ds[idx])
                if formatted:
                    collected.append(formatted)
            if len(collected) >= 120:
                return collected
        except Exception:
            continue
    return []


def build_or_load_logic_code_dataset(
    cache_path: str = "qwen_logic_code_dataset.json",
    n_train_logic: int = 2000,
    n_train_code: int = 2000,
    n_eval_logic: int = 100,
    n_eval_code: int = 100,
    seed: int = 42,
    n_train_tool: int = 0,
    n_eval_tool: int = 0,
) -> Dict[str, List[Dict[str, str]]]:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        has_tool = len(data.get("eval_tool", [])) >= n_eval_tool if n_eval_tool > 0 else True
        if (
            len(data.get("train", [])) >= (n_train_logic + n_train_code + n_train_tool) * 0.9
            and len(data.get("eval_logic", [])) >= n_eval_logic
            and len(data.get("eval_code", [])) >= n_eval_code
            and has_tool
        ):
            return data

    rng = random.Random(seed)

    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    arc_train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    code_alpaca = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    mbpp_train = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    mbpp_test = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    n_arc_train = min(int(n_train_logic * 0.30), len(arc_train))
    n_gsm_train = min(n_train_logic - n_arc_train, len(gsm_train))

    gsm_train_idx = rng.sample(range(len(gsm_train)), n_gsm_train)
    arc_train_idx = rng.sample(range(len(arc_train)), n_arc_train)

    train_logic = [
        {"domain": "logic", "source": "gsm8k", "text": _format_gsm8k(gsm_train[i])}
        for i in gsm_train_idx
    ] + [
        {"domain": "logic", "source": "arc_challenge", "text": _format_arc(arc_train[i])}
        for i in arc_train_idx
    ]

    n_mbpp_train = min(len(mbpp_train), int(n_train_code * 0.10))
    n_alpaca_train = min(n_train_code - n_mbpp_train, 15000)

    mbpp_train_idx = rng.sample(range(len(mbpp_train)), n_mbpp_train)
    alpaca_train_idx = rng.sample(range(15000), n_alpaca_train)

    train_code = [
        {"domain": "code", "source": "mbpp", "text": _format_mbpp(mbpp_train[i])}
        for i in mbpp_train_idx
    ] + [
        {"domain": "code", "source": "python_alpaca", "text": _format_code_alpaca(code_alpaca[i])}
        for i in alpaca_train_idx
    ]

    train_tool = []
    eval_tool = []
    if n_train_tool > 0 or n_eval_tool > 0:
        n_hf_target = int((n_train_tool + n_eval_tool) * 0.70)
        hf_tools = _load_hf_tool_texts(max_samples=n_hf_target, seed=seed)
        tool_texts = []
        for txt in hf_tools:
            tool_texts.append({"domain": "tool", "source": "hf_tool", "text": txt})
        n_synth = max(0, (n_train_tool + n_eval_tool) - len(tool_texts))
        if n_synth > 0:
            for txt in _build_synthetic_tool_traces(n_synth, seed=seed + 55):
                tool_texts.append({"domain": "tool", "source": "synth_tool", "text": txt})
        rng.shuffle(tool_texts)
        train_tool = tool_texts[:n_train_tool]
        eval_tool = tool_texts[n_train_tool : n_train_tool + n_eval_tool]

    train_records = train_logic + train_code + train_tool
    rng.shuffle(train_records)

    n_arc_eval = min(int(n_eval_logic * 0.30), len(arc_test))
    n_gsm_eval = min(n_eval_logic - n_arc_eval, len(gsm_test))
    eval_logic = [
        {"domain": "logic", "source": "gsm8k", "text": _format_gsm8k(gsm_test[i])}
        for i in rng.sample(range(len(gsm_test)), n_gsm_eval)
    ] + [
        {"domain": "logic", "source": "arc_challenge", "text": _format_arc(arc_test[i])}
        for i in rng.sample(range(len(arc_test)), n_arc_eval)
    ]

    n_mbpp_eval = min(int(n_eval_code * 0.50), len(mbpp_test))
    n_alpaca_eval = min(n_eval_code - n_mbpp_eval, len(code_alpaca) - 15000)
    eval_code = [
        {"domain": "code", "source": "mbpp", "text": _format_mbpp(mbpp_test[i])}
        for i in rng.sample(range(len(mbpp_test)), n_mbpp_eval)
    ] + [
        {"domain": "code", "source": "python_alpaca", "text": _format_code_alpaca(code_alpaca[15000 + i])}
        for i in rng.sample(range(len(code_alpaca) - 15000), n_alpaca_eval)
    ]

    payload = {
        "meta": {
            "seed": seed,
            "n_train": len(train_records),
            "n_train_logic": len(train_logic),
            "n_train_code": len(train_code),
            "n_train_tool": len(train_tool),
            "n_eval_logic": len(eval_logic),
            "n_eval_code": len(eval_code),
            "n_eval_tool": len(eval_tool),
        },
        "train": train_records,
        "eval_logic": eval_logic,
        "eval_code": eval_code,
        "eval_tool": eval_tool,
    }

    parent = os.path.dirname(os.path.abspath(cache_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


class CausalTextDataset(Dataset):
    def __init__(self, records: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.records[idx]["text"]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_eval_texts(dataset_payload: Dict) -> Tuple[List[str], List[str]]:
    logic_texts = [r["text"] for r in dataset_payload["eval_logic"]]
    code_texts = [r["text"] for r in dataset_payload["eval_code"]]
    return logic_texts, code_texts


def get_tool_eval_texts(dataset_payload: Dict) -> List[str]:
    return [r["text"] for r in dataset_payload.get("eval_tool", [])]


def _format_mmlu_ood(row: Dict) -> str:
    letters = ["A", "B", "C", "D"]
    choices = row.get("choices") or []
    opts = "\n".join(f"({letters[idx]}) {txt}" for idx, txt in enumerate(choices[:4]))
    ans_idx = int(row.get("answer", 0))
    ans_letter = letters[ans_idx] if 0 <= ans_idx < len(letters) else "A"
    ans_text = choices[ans_idx] if 0 <= ans_idx < len(choices) else ""
    subj = str(row.get("subject") or "logic").replace("_", " ")
    return (
        f"Subject: {subj}\n"
        f"Question: {row['question'].strip()}\n"
        f"{opts}\n"
        f"Answer: ({ans_letter}) {ans_text}"
    )


def build_or_load_ood_eval_dataset(
    cache_path: str = "qwen_ood_eval_dataset.json",
    n_ood_logic: int = 80,
    n_ood_code: int = 80,
    n_ood_general: int = 80,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if (
            len(data.get("ood_logic", [])) >= n_ood_logic
            and len(data.get("ood_code", [])) >= n_ood_code
            and len(data.get("ood_general", [])) >= n_ood_general
        ):
            return data

    rng = random.Random(seed + 1000)

    mmlu_logic = load_dataset("cais/mmlu", "formal_logic", split="test")
    mmlu_math = load_dataset("cais/mmlu", "high_school_mathematics", split="test")
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    n_fl = min(n_ood_logic // 2, len(mmlu_logic))
    n_hm = min(n_ood_logic - n_fl, len(mmlu_math))
    ood_logic = [
        {"domain": "ood_logic", "source": "mmlu_formal_logic", "text": _format_mmlu_ood(mmlu_logic[i])}
        for i in rng.sample(range(len(mmlu_logic)), n_fl)
    ] + [
        {"domain": "ood_logic", "source": "mmlu_hs_math", "text": _format_mmlu_ood(mmlu_math[i])}
        for i in rng.sample(range(len(mmlu_math)), n_hm)
    ]

    n_he = min(n_ood_code, len(humaneval))
    ood_code = [
        {
            "domain": "ood_code",
            "source": "humaneval",
            "text": f"{humaneval[i]['prompt']}{humaneval[i]['canonical_solution']}",
        }
        for i in rng.sample(range(len(humaneval)), n_he)
    ]

    wiki_candidates = [
        r["text"].strip()
        for r in wikitext
        if len(r["text"].strip()) >= 160 and not r["text"].strip().startswith("=")
    ]
    n_wk = min(n_ood_general, len(wiki_candidates))
    ood_general = [
        {"domain": "ood_general", "source": "wikitext2", "text": wiki_candidates[i]}
        for i in rng.sample(range(len(wiki_candidates)), n_wk)
    ]

    payload = {
        "meta": {
            "seed": seed + 1000,
            "n_ood_logic": len(ood_logic),
            "n_ood_code": len(ood_code),
            "n_ood_general": len(ood_general),
        },
        "ood_logic": ood_logic,
        "ood_code": ood_code,
        "ood_general": ood_general,
    }

    parent = os.path.dirname(os.path.abspath(cache_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def get_ood_eval_texts(ood_payload: Dict) -> Dict[str, List[str]]:
    return {
        "ood_logic": [r["text"] for r in ood_payload.get("ood_logic", [])],
        "ood_code": [r["text"] for r in ood_payload.get("ood_code", [])],
        "ood_general": [r["text"] for r in ood_payload.get("ood_general", [])],
    }
