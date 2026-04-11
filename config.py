"""
Data loading and tokenisation for CT-MoE experiments.

Streams CodeParrot-Clean (Python code) and arXiv-Summarization (scientific text),
trains a SentencePiece BPE tokeniser, and returns TokenDataset instances.

Falls back to WikiText-2 if HuggingFace streams are unavailable.
"""

import os
from torch.utils.data import Dataset
import torch


def load_domain_texts(max_chars: int = 25_000_000):
    """
    Returns (train_text, val_code_text, val_arxiv_text) as raw strings.
    Each domain is split 90/10 train/val. The returned train_text is
    the interleaved 90% from both domains.
    """
    from datasets import load_dataset

    def stream(name, cfg_name, split, field, mc, label):
        print(f"  [{label}] streaming …")
        ds = load_dataset(name, cfg_name, split=split, streaming=True,
                          trust_remote_code=True)
        texts, chars = [], 0
        for item in ds:
            t = (item.get(field) or "").strip()
            if not t:
                continue
            texts.append(t)
            chars += len(t)
            if chars >= mc:
                break
        print(f"    → {chars:,} chars, {len(texts):,} docs")
        return texts

    try:
        code  = stream("codeparrot/codeparrot-clean", None,
                       "train", "content", max_chars, "code")
        arxiv = stream("ccdv/arxiv-summarization", "document",
                       "train", "article", max_chars, "arxiv")
    except Exception as e:
        print(f"  Domain streams failed ({e}) — falling back to WikiText-2")
        ds   = load_dataset("wikitext", "wikitext-2-raw-v1")
        join = lambda s: "\n".join(t for t in ds[s]["text"] if t.strip())
        full = join("train")
        val  = join("validation")
        mid  = len(val) // 2
        return full, val[:mid], val[mid:]

    def split90(lst):
        n = max(1, len(lst) // 10)
        return lst[n:], lst[:n]

    ct, cv = split90(code)
    at, av = split90(arxiv)
    join   = lambda lst: "\n\n".join(lst)
    return join(ct) + "\n\n" + join(at), join(cv), join(av)


def train_spm(text: str, vocab_size: int, prefix: str = "spm_ct_moe"):
    """
    Trains a SentencePiece BPE tokeniser on the first 4M characters of text.
    Skips training if the model file already exists (idempotent).
    """
    import sentencepiece as spm

    if not os.path.exists(f"{prefix}.model"):
        tmp = f"{prefix}_raw.txt"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text[:4_000_000])
        spm.SentencePieceTrainer.train(
            input=tmp,
            model_prefix=prefix,
            vocab_size=vocab_size,
            character_coverage=0.9995,
            model_type="bpe",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            input_sentence_size=5_000_000,
            shuffle_input_sentence=True,
        )
        os.remove(tmp)

    sp = spm.SentencePieceProcessor()
    sp.load(f"{prefix}.model")
    return sp


class TokenDataset(Dataset):
    """
    Fixed-length token sequence dataset.
    Text is tokenised in 500k-character chunks to avoid memory spikes.
    Each item is (input_ids, target_ids) of length seq_len.
    """

    def __init__(self, text: str, sp, seq_len: int):
        self.L = seq_len
        ids = []
        for s in range(0, len(text), 500_000):
            ids.extend(sp.encode(text[s:s + 500_000], out_type=int))
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return (len(self.data) - 1) // self.L

    def __getitem__(self, i):
        s = i * self.L
        c = self.data[s:s + self.L + 1]
        return c[:-1], c[1:]
