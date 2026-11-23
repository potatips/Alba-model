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


def build_or_load_logic_code_dataset(
    cache_path: str = "qwen_logic_code_dataset.json",
    n_train_logic: int = 2000,
    n_train_code: int = 2000,
    n_eval_logic: int = 100,
    n_eval_code: int = 100,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if (
            len(data.get("train", [])) >= (n_train_logic + n_train_code) * 0.9
            and len(data.get("eval_logic", [])) >= n_eval_logic
            and len(data.get("eval_code", [])) >= n_eval_code
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

    train_records = train_logic + train_code
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
            "n_eval_logic": len(eval_logic),
            "n_eval_code": len(eval_code),
        },
        "train": train_records,
        "eval_logic": eval_logic,
        "eval_code": eval_code,
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
