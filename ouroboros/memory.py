"""Becky's memory: searchable, and free of repeats.

Her notes live in ``memory.md`` in her workspace, one ``- note`` per line. This module adds:

* **search**: find the notes most relevant to a request;
* **de-duplication**: a note that is (nearly) a repeat of one she already has is not stored twice.

How "similar" is decided: each note is turned into a vector of hashed character 3-grams and
whole words, normalised, and compared by cosine similarity. No outside model is involved.
This is *fuzzy text matching*, not understanding of meaning: ``buy milk`` matches ``buy some
milk`` and ``Buy MILK!`` but not ``purchase dairy``. That is a deliberate, honest limit.

Notes are written by a model reading untrusted text, so they are treated as untrusted
information everywhere they are used.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import torch

DIM = 1024
DUPLICATE_AT = 0.90  # cosine similarity at or above which a note counts as a repeat
RELEVANT_AT = 0.20  # minimum similarity for a search hit
MAX_NOTE_CHARS = 2000


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def embed(text: str) -> torch.Tensor:
    """A unit vector for ``text``: hashed character 3-grams plus whole words."""
    norm = _normalise(text)
    v = torch.zeros(DIM)
    padded = f" {norm} "
    for i in range(len(padded) - 2):
        v[zlib.crc32(padded[i : i + 3].encode()) % DIM] += 1.0
    for w in norm.split():
        v[zlib.crc32(("w:" + w).encode()) % DIM] += 2.0
    n = v.norm()
    return v / n if n > 0 else v


def similarity(a: str, b: str) -> float:
    return float(embed(a) @ embed(b))


class Memory:
    """The notes in one ``memory.md``."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def notes(self) -> list[str]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("- ") and line[2:].strip():
                out.append(line[2:].strip())
        return out

    def find_duplicate(self, note: str) -> str | None:
        """An existing note this one repeats, or None."""
        norm, vec = _normalise(note), embed(note)
        for existing in self.notes():
            if _normalise(existing) == norm or float(vec @ embed(existing)) >= DUPLICATE_AT:
                return existing
        return None

    def add(self, note: str) -> tuple[bool, str | None]:
        """Store ``note`` unless it repeats one. Returns (stored, the note it duplicates)."""
        note = " ".join(note.split())[:MAX_NOTE_CHARS]
        if not note:
            raise ValueError("note must be non-empty text")
        dup = self.find_duplicate(note)
        if dup is not None:
            return False, dup
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write("- " + note + "\n")
        return True, None

    def search(self, query: str, k: int = 5, threshold: float = RELEVANT_AT) -> list[tuple[float, str]]:
        notes = self.notes()
        if not notes:
            return []
        q = embed(query)
        scored = sorted(((float(q @ embed(n)), n) for n in notes), key=lambda x: -x[0])
        return [(round(s, 3), n) for s, n in scored[:k] if s >= threshold]

    def dedupe(self) -> list[str]:
        """Rewrite the file without repeats (keeping the first of each). Returns what was removed."""
        keep: list[str] = []
        removed: list[str] = []
        for note in self.notes():
            vec = embed(note)
            if any(_normalise(note) == _normalise(k) or float(vec @ embed(k)) >= DUPLICATE_AT for k in keep):
                removed.append(note)
            else:
                keep.append(note)
        if removed:
            self.path.write_text("".join(f"- {n}\n" for n in keep), encoding="utf-8")
        return removed
