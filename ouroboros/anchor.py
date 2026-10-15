"""An integrity anchor: fingerprints of Becky's brain, rules and code, signed by you.

``anchor`` records the SHA-256 of every code file in the package, the brain weights, and the
signed rules, and signs that list with your owner key (so it needs your passphrase). From
then on, ``becky`` and ``hub`` recompute the fingerprints at start-up and refuse to run if
anything differs, naming exactly what changed. After a change you intended (a ``git pull``,
an adopted brain), you review it and run ``anchor`` again.

What this does and does not do
------------------------------
It catches accidental damage and casual tampering: a corrupted brain file, a hand-edited
source file, a swapped weights file. It does **not** stop someone who controls your account
and can replace the checker itself as well as the anchor; against that, no check that lives on
the same machine can help. The agent cannot touch the anchor: the owner directory is
protected from her tools.

This is plain hashing and an ordinary digital signature. Nothing about it is distributed or
quantum.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .owner import sign_bytes, verify_signature

PACKAGE = Path(__file__).resolve().parent
ANCHOR_FILE, SIG_FILE = "anchor.json", "anchor.sig"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints(owner_dir: Path, brain: Path | None = None) -> dict[str, str]:
    """SHA-256 of everything Becky is built from, keyed by a stable name."""
    brain = Path(brain) if brain else PACKAGE / "brain.pt"
    out = {f"code/{p.name}": _sha256(p) for p in sorted(PACKAGE.glob("*.py"))}
    out["rules/default-constitution.toml"] = _sha256(PACKAGE / "constitution.toml")
    signed = Path(owner_dir) / "constitution.toml"
    if signed.exists():
        out["rules/signed-constitution.toml"] = _sha256(signed)
    if brain.exists():
        out["brain/brain.pt"] = _sha256(brain)
    return out


def _canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()


def create_anchor(owner_dir: Path, passphrase: str, brain: Path | None = None) -> dict:
    owner_dir = Path(owner_dir)
    manifest = {"created": int(time.time()), "files": fingerprints(owner_dir, brain)}
    signature = sign_bytes(owner_dir, passphrase, _canonical(manifest))  # a wrong passphrase stops here
    (owner_dir / ANCHOR_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (owner_dir / SIG_FILE).write_text(signature, encoding="utf-8")
    return manifest


@dataclass
class AnchorReport:
    present: bool
    ok: bool = True
    signature_ok: bool = True
    changed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.present:
            return "no integrity anchor set"
        if self.ok:
            return "integrity anchor verified"
        parts = []
        if not self.signature_ok:
            parts.append("the anchor itself was not signed by the owner")
        for label, names in (("changed", self.changed), ("missing", self.missing), ("new", self.added)):
            if names:
                parts.append(f"{label}: " + ", ".join(names))
        return "integrity check FAILED: " + "; ".join(parts)


def verify_anchor(owner_dir: Path, brain: Path | None = None) -> AnchorReport:
    owner_dir = Path(owner_dir)
    af, sf = owner_dir / ANCHOR_FILE, owner_dir / SIG_FILE
    if not af.exists() and not sf.exists():
        return AnchorReport(present=False)
    try:
        manifest = json.loads(af.read_text(encoding="utf-8"))
        recorded = manifest["files"]
        signature_ok = verify_signature(owner_dir, _canonical(manifest), sf.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return AnchorReport(present=True, ok=False, signature_ok=False)
    now = fingerprints(owner_dir, brain)
    changed = sorted(k for k in recorded if k in now and now[k] != recorded[k])
    missing = sorted(k for k in recorded if k not in now)
    added = sorted(k for k in now if k not in recorded)
    return AnchorReport(present=True, ok=signature_ok and not (changed or missing or added),
                        signature_ok=signature_ok, changed=changed, missing=missing, added=added)
