from dataclasses import dataclass
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


@dataclass
class PaperTokenCorpus:
    train_path: str
    val_path: str
    dtype_name: str
    vocab_size: int
    train_tokens: int
    val_tokens: int
    val_bytes_per_token: float
    ood_splits: Dict[str, Tuple[torch.Tensor, float]]
    downstream_tasks: Dict[str, List[Dict[str, object]]]
    provenance: Dict[str, object]


class MemmapTokenDataset:
    def __init__(self, path: str, seq_len: int, stride: Optional[int] = None):
        self.arr = np.load(path, mmap_mode="r")
        self.seq_len = int(seq_len)
        self.block_len = self.seq_len + 1
        self.stride = int(stride) if stride is not None else self.seq_len
        usable = len(self.arr) - self.block_len
        if usable < 0:
            raise ValueError(f"Token file {path} length {len(self.arr)} < block_len {self.block_len}")
        self.num_windows = usable // self.stride + 1

    def sample_batch(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        max_start = len(self.arr) - self.block_len
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator).tolist()
        batch = np.stack([self.arr[s : s + self.block_len] for s in starts], axis=0)
        return torch.from_numpy(batch.astype(np.int64, copy=False))

    def get_eval_blocks(self, max_blocks: int = 64) -> torch.Tensor:
        n = min(self.num_windows, max_blocks)
        rows = []
        for idx in range(n):
            s = idx * self.stride
            rows.append(self.arr[s : s + self.block_len])
        return torch.from_numpy(np.stack(rows, axis=0).astype(np.int64, copy=False))


def _pack_texts_to_blocks(
    texts: List[str],
    tokenizer,
    seq_len: int,
    max_tokens: int,
) -> Tuple[torch.Tensor, float]:
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    all_ids: List[int] = []
    total_bytes = 0
    for txt in texts:
        cleaned = str(txt).strip()
        if len(cleaned) < 24:
            continue
        ids = tokenizer.encode(cleaned, add_special_tokens=False)
        if not ids:
            continue
        all_ids.extend(ids)
        all_ids.append(eos_id)
        total_bytes += len(cleaned.encode("utf-8")) + 1
        if len(all_ids) >= max_tokens:
            break

    block_len = seq_len + 1
    usable = (len(all_ids) // block_len) * block_len
    if usable == 0:
        raise ValueError("Not enough tokens to build OOD evaluation block.")
    tensor = torch.tensor(all_ids[:usable], dtype=torch.long).view(-1, block_len)
    bpt = float(total_bytes) / float(max(len(all_ids), 1))
    return tensor, bpt


def build_synthetic_paper_corpus(
    cache_dir: str,
    vocab_size: int = 256,
    seq_len: int = 64,
    train_tokens: int = 131072,
    val_tokens: int = 16384,
    seed: int = 42,
) -> PaperTokenCorpus:
    os.makedirs(cache_dir, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    total_train = max(train_tokens, seq_len * 16)
    total_val = max(val_tokens, seq_len * 4)

    def _gen_stream(n: int) -> np.ndarray:
        arr = torch.randint(0, vocab_size, (n,), generator=gen, dtype=torch.int64)
        for idx in range(2, n):
            if idx % 7 != 0:
                arr[idx] = (arr[idx - 1] * 3 + arr[idx - 2] + 5) % vocab_size
        return arr.numpy().astype(np.uint16)

    train_arr = _gen_stream(total_train)
    val_arr = _gen_stream(total_val)
    ood_arr = _gen_stream(total_val)

    train_path = os.path.join(cache_dir, "train_tokens.npy")
    val_path = os.path.join(cache_dir, "val_tokens.npy")
    np.save(train_path, train_arr)
    np.save(val_path, val_arr)

    block_len = seq_len + 1
    usable_ood = (len(ood_arr) // block_len) * block_len
    ood_tensor = torch.from_numpy(ood_arr[:usable_ood].reshape(-1, block_len).astype(np.int64))

    return PaperTokenCorpus(
        train_path=train_path,
        val_path=val_path,
        dtype_name="uint16",
        vocab_size=vocab_size,
        train_tokens=int(len(train_arr)),
        val_tokens=int(len(val_arr)),
        val_bytes_per_token=1.0,
        ood_splits={
            "ood_wiki": (ood_tensor, 1.0),
        },
        downstream_tasks={},
        provenance={
            "dataset": "synthetic_markov",
            "tokenizer": "byte_synthetic",
            "train_tokens": int(len(train_arr)),
            "val_tokens": int(len(val_arr)),
        },
    )


def _save_token_chunk(path: str, chunks: List[np.ndarray], np_dtype) -> int:
    merged = np.concatenate(chunks, axis=0).astype(np_dtype, copy=False)
    np.save(path, merged)
    return int(len(merged))


def _load_ood_splits(tokenizer, seq_len: int, max_ood_tokens: int = 120_000) -> Tuple[Dict[str, Tuple[torch.Tensor, float]], Dict[str, int]]:
    from datasets import load_dataset

    splits: Dict[str, Tuple[torch.Tensor, float]] = {}
    counts: Dict[str, int] = {}

    try:
        wiki_test = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        wiki_texts = [row["text"] for row in wiki_test]
        w_blk, w_bpt = _pack_texts_to_blocks(wiki_texts, tokenizer, seq_len, max_ood_tokens)
        splits["ood_wiki"] = (w_blk, w_bpt)
        counts["ood_wiki_blocks"] = int(w_blk.size(0))
    except Exception:
        pass

    try:
        gsm = load_dataset("openai/gsm8k", "main", split="test[:400]")
        math_texts = [f"Question: {r['question']}\nSolution: {r['answer']}" for r in gsm]
        m_blk, m_bpt = _pack_texts_to_blocks(math_texts, tokenizer, seq_len, max_ood_tokens)
        splits["ood_math"] = (m_blk, m_bpt)
        counts["ood_math_blocks"] = int(m_blk.size(0))
    except Exception:
        pass

    code_texts: List[str] = []
    try:
        he = load_dataset("openai/openai_humaneval", split="test")
        code_texts = [f"{r['prompt']}{r['canonical_solution']}" for r in he]
    except Exception:
        try:
            mbpp = load_dataset("google-research-datasets/mbpp", split="test[:300]")
            code_texts = [f"# {r['text']}\n{r['code']}" for r in mbpp]
        except Exception:
            code_texts = []
    if code_texts:
        try:
            c_blk, c_bpt = _pack_texts_to_blocks(code_texts, tokenizer, seq_len, max_ood_tokens)
            splits["ood_code"] = (c_blk, c_bpt)
            counts["ood_code_blocks"] = int(c_blk.size(0))
        except Exception:
            pass

    return splits, counts


def _load_downstream_tasks(max_per_task: int = 160) -> Dict[str, List[Dict[str, object]]]:
    from datasets import load_dataset

    tasks: Dict[str, List[Dict[str, object]]] = {}

    try:
        ds = load_dataset("Rowan/hellaswag", split=f"validation[:{max_per_task}]")
        items = []
        for row in ds:
            ctx = str(row.get("ctx", "")).strip()
            endings = [str(e).strip() for e in row.get("endings", [])]
            label_raw = row.get("label", "")
            if ctx and len(endings) == 4 and str(label_raw).isdigit():
                items.append({"ctx": ctx, "choices": endings, "label": int(label_raw)})
        if items:
            tasks["hellaswag"] = items
    except Exception:
        pass

    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"validation[:{max_per_task}]")
        items = []
        for row in ds:
            q = str(row.get("question", "")).strip()
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            ans = row.get("answerKey", "")
            if ans in labels and len(texts) >= 2:
                items.append(
                    {
                        "ctx": f"Question: {q}\nAnswer:",
                        "choices": [str(t).strip() for t in texts],
                        "label": int(labels.index(ans)),
                    }
                )
        if items:
            tasks["arc_easy"] = items
    except Exception:
        pass

    try:
        ds = load_dataset("EleutherAI/lambada_openai", "en", split=f"test[:{max_per_task}]")
        items = []
        for row in ds:
            text = str(row.get("text", "")).strip()
            parts = text.rsplit(" ", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                items.append({"ctx": parts[0].strip(), "target": " " + parts[1].strip()})
        if items:
            tasks["lambada"] = items
    except Exception:
        pass

    return tasks


def build_or_load_fineweb_edu_corpus(
    cache_dir: str,
    seq_len: int = 1024,
    train_tokens: int = 300_000_000,
    val_tokens: int = 2_000_000,
    tokenizer_name: str = "gpt2",
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    max_ood_tokens: int = 120_000,
    downstream_items: int = 160,
) -> PaperTokenCorpus:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "train_tokens.npy")
    val_path = os.path.join(cache_dir, "val_tokens.npy")
    meta_path = os.path.join(cache_dir, "corpus_meta.json")
    ood_path = os.path.join(cache_dir, "ood_eval.pt")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    vocab_size = int(len(tokenizer))
    np_dtype = np.uint16 if vocab_size < 65535 else np.uint32
    dtype_name = "uint16" if vocab_size < 65535 else "uint32"

    cached_ok = False
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("train_tokens", 0) >= int(0.98 * train_tokens)
            and meta.get("val_tokens", 0) >= int(0.95 * val_tokens)
            and meta.get("tokenizer") == tokenizer_name
            and meta.get("dataset") == dataset_name
        ):
            cached_ok = True

    if not cached_ok:
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        try:
            stream = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
        except Exception:
            dataset_name = "Salesforce/wikitext"
            dataset_config = "wikitext-103-raw-v1"
            stream = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)

        val_chunks: List[np.ndarray] = []
        val_buf: List[int] = []
        val_count = 0
        val_bytes = 0

        train_chunks: List[np.ndarray] = []
        train_buf: List[int] = []
        train_count = 0
        flush_threshold = 2_000_000

        print(
            f"[PaperData] Streaming {dataset_name} ({dataset_config}) -> "
            f"val={val_tokens:,} tok, train={train_tokens:,} tok",
            flush=True,
        )

        for row in stream:
            txt = str(row.get("text", "")).strip()
            if len(txt) < 64:
                continue
            ids = tokenizer.encode(txt, add_special_tokens=False)
            if not ids:
                continue
            ids.append(eos_id)

            if val_count < val_tokens:
                val_buf.extend(ids)
                val_count += len(ids)
                val_bytes += len(txt.encode("utf-8")) + 1
                if len(val_buf) >= 250_000:
                    val_chunks.append(np.array(val_buf, dtype=np_dtype))
                    val_buf = []
                continue

            train_buf.extend(ids)
            train_count += len(ids)
            if len(train_buf) >= flush_threshold:
                train_chunks.append(np.array(train_buf, dtype=np_dtype))
                train_buf = []
                if train_count % 20_000_000 < flush_threshold:
                    print(f"  packed train tokens: {train_count:,} / {train_tokens:,}", flush=True)

            if train_count >= train_tokens:
                break

        if val_buf:
            val_chunks.append(np.array(val_buf, dtype=np_dtype))
        if train_buf:
            train_chunks.append(np.array(train_buf, dtype=np_dtype))

        actual_val = _save_token_chunk(val_path, val_chunks, np_dtype)
        actual_train = _save_token_chunk(train_path, train_chunks, np_dtype)
        val_bpt = float(val_bytes) / float(max(actual_val, 1))
        meta = {
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "tokenizer": tokenizer_name,
            "vocab_size": vocab_size,
            "dtype_name": dtype_name,
            "train_tokens": actual_train,
            "val_tokens": actual_val,
            "val_bytes_per_token": val_bpt,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    if os.path.exists(ood_path):
        saved_ood = torch.load(ood_path, map_location="cpu", weights_only=False)
        if saved_ood.get("seq_len") == seq_len and saved_ood.get("tokenizer") == tokenizer_name:
            ood_splits = saved_ood["ood_splits"]
            downstream_tasks = saved_ood["downstream_tasks"]
            ood_counts = saved_ood["ood_counts"]
        else:
            ood_splits, ood_counts = _load_ood_splits(tokenizer, seq_len, max_ood_tokens=max_ood_tokens)
            downstream_tasks = _load_downstream_tasks(max_per_task=downstream_items)
            torch.save(
                {
                    "seq_len": seq_len,
                    "tokenizer": tokenizer_name,
                    "ood_splits": ood_splits,
                    "ood_counts": ood_counts,
                    "downstream_tasks": downstream_tasks,
                },
                ood_path,
            )
    else:
        ood_splits, ood_counts = _load_ood_splits(tokenizer, seq_len, max_ood_tokens=max_ood_tokens)
        downstream_tasks = _load_downstream_tasks(max_per_task=downstream_items)
        torch.save(
            {
                "seq_len": seq_len,
                "tokenizer": tokenizer_name,
                "ood_splits": ood_splits,
                "ood_counts": ood_counts,
                "downstream_tasks": downstream_tasks,
            },
            ood_path,
        )

    return PaperTokenCorpus(
        train_path=train_path,
        val_path=val_path,
        dtype_name=str(meta["dtype_name"]),
        vocab_size=int(meta["vocab_size"]),
        train_tokens=int(meta["train_tokens"]),
        val_tokens=int(meta["val_tokens"]),
        val_bytes_per_token=float(meta["val_bytes_per_token"]),
        ood_splits=ood_splits,
        downstream_tasks=downstream_tasks,
        provenance={
            **meta,
            "ood_counts": ood_counts,
            "downstream_counts": {k: len(v) for k, v in downstream_tasks.items()},
        },
    )
