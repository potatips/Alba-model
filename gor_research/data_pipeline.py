from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch


@dataclass
class TokenDatasetSplit:
    train_blocks: torch.Tensor
    val_blocks: torch.Tensor
    ood_blocks: torch.Tensor
    val_bytes_per_token: float
    ood_bytes_per_token: float
    vocab_size: int


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
        cleaned = txt.strip()
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
        raise ValueError("Not enough tokens to build even a single block.")
    tensor = torch.tensor(all_ids[:usable], dtype=torch.long).view(-1, block_len)
    bpt = float(total_bytes) / float(max(len(all_ids), 1))
    return tensor, bpt


def build_pretraining_dataset(
    seq_len: int = 256,
    max_train_tokens: int = 12_000_000,
    max_val_tokens: int = 300_000,
    max_ood_tokens: int = 300_000,
    tokenizer_name: str = "gpt2",
) -> TokenDatasetSplit:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.eos_token_id = tokenizer.pad_token_id

    wiki_train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    wiki_val = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")

    train_texts = [row["text"] for row in wiki_train]
    val_texts = [row["text"] for row in wiki_val]

    ood_texts: List[str] = []
    try:
        stories = load_dataset("roneneldan/TinyStories", split="validation[:1200]")
        ood_texts.extend(row["text"] for row in stories)
    except Exception:
        wiki_test = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        ood_texts.extend(row["text"] for row in wiki_test)

    train_blocks, _ = _pack_texts_to_blocks(train_texts, tokenizer, seq_len=seq_len, max_tokens=max_train_tokens)
    val_blocks, val_bpt = _pack_texts_to_blocks(val_texts, tokenizer, seq_len=seq_len, max_tokens=max_val_tokens)
    ood_blocks, ood_bpt = _pack_texts_to_blocks(ood_texts, tokenizer, seq_len=seq_len, max_tokens=max_ood_tokens)

    return TokenDatasetSplit(
        train_blocks=train_blocks,
        val_blocks=val_blocks,
        ood_blocks=ood_blocks,
        val_bytes_per_token=val_bpt,
        ood_bytes_per_token=ood_bpt,
        vocab_size=int(len(tokenizer)),
    )


def build_synthetic_lm_split(
    vocab_size: int = 256,
    seq_len: int = 64,
    n_train: int = 256,
    n_val: int = 64,
    seed: int = 42,
) -> TokenDatasetSplit:
    gen = torch.Generator().manual_seed(seed)
    total = n_train + n_val
    data = torch.randint(0, vocab_size, (total, seq_len + 1), generator=gen)
    for t in range(2, seq_len + 1):
        mask = torch.rand(total, generator=gen) < 0.85
        data[mask, t] = (data[mask, t - 1] * 3 + data[mask, t - 2] + 5) % vocab_size
    return TokenDatasetSplit(
        train_blocks=data[:n_train].contiguous(),
        val_blocks=data[n_train:].contiguous(),
        ood_blocks=data[n_train:].contiguous(),
        val_bytes_per_token=1.0,
        ood_bytes_per_token=1.0,
        vocab_size=vocab_size,
    )
