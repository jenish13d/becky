"""Corpus loading, byte-level tokenisation, splits and batching.

The tokenizer is raw UTF-8 bytes (vocab 256): no training step, nothing unknown,
and every model in a lineage shares it, so their losses are directly comparable
in *bits per byte* (bpb).

The corpus is split contiguously into train / val / test. Successors are selected
on val only; test is touched once, at the end, to report the final champion.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids) -> str:
    return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


def load_corpus(path: Path | None = None) -> bytes:
    """Read a text corpus; with no path, download Tiny Shakespeare once and cache it."""
    if path is not None:
        return Path(path).read_bytes()
    if not DEFAULT_CORPUS.exists():
        DEFAULT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=60) as r:
            DEFAULT_CORPUS.write_bytes(r.read())
    return DEFAULT_CORPUS.read_bytes()


@dataclass
class Splits:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


def split(data: bytes, val_frac: float = 0.05, test_frac: float = 0.05) -> Splits:
    t = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    n = len(t)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) < 2:
        raise ValueError(f"corpus of {n} bytes is too small to split")
    return Splits(train=t[:n_train], val=t[n_train : n_train + n_val], test=t[n_train + n_val :])


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = min(block_size, len(data) - 1)
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


@torch.no_grad()
def bits_per_byte(model, data: torch.Tensor, max_bytes: int | None = None, batch_size: int = 64) -> float:
    """Average next-byte loss over ``data`` in bits, every byte scored exactly once.

    The data is cut into non-overlapping chunks of the model's own context length,
    so each candidate is scored on the same bytes regardless of its block size.
    """
    if max_bytes is not None:
        data = data[: max_bytes + 1]
    bs = model.cfg.block_size
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    starts = list(range(0, len(data) - 1, bs))
    for b in range(0, len(starts), batch_size):
        chunk = starts[b : b + batch_size]
        n = min(bs, len(data) - 1 - chunk[-1])  # the last chunk may be short
        full = [s for s in chunk if s + bs + 1 <= len(data)]
        groups = [(full, bs)] if full else []
        if len(full) < len(chunk):
            groups.append(([chunk[-1]], n))
        for group, length in groups:
            x = torch.stack([data[s : s + length] for s in group]).to(device)
            y = torch.stack([data[s + 1 : s + 1 + length] for s in group]).to(device)
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total += loss.item()
            count += y.numel()
    model.train(was_training)
    return total / count / math.log(2)
