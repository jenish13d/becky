"""Owner authority: only the holder of one passphrase can change Becky's rules.

How it works
------------
``init_owner`` generates an Ed25519 signing key and encrypts its private half with a key
derived from your passphrase (scrypt + AES-GCM). It stores the *public* key, the encrypted
private key, and a signed copy of the constitution in the owner directory. The passphrase
itself is never stored, logged, or passed on a command line.

Becky only starts, and only keeps running, while the constitution's signature verifies
against that public key. Changing the rules means producing a new signature, which needs
the private key, which needs the passphrase. A hand-edited rules file simply won't run.

What this protects, and what it doesn't
---------------------------------------
It stops the agent and anything that merely edits the rules file. It does *not* stop
someone who can replace the guardian's own code or the stored public key; if they control
your account, they control everything. Protect the owner directory with normal OS file
permissions, and choose a passphrase that is long and not a keyboard pattern.

The limits that matter most (workspace confinement, the fixed tool set, the size caps)
live in code. No passphrase and no signature can lift them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DEFAULT_OWNER_DIR = Path.home() / ".becky"
KDF_N, KDF_R, KDF_P = 2**15, 8, 1  # scrypt cost; stored in the key file so it can change later
FAIL_DELAY = 1.5  # seconds to wait after a wrong passphrase
MIN_LEN = 12
_AAD = b"becky-owner-v1"
_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm", "1234567890", "abcdefghijklmnopqrstuvwxyz")


class OwnerError(Exception):
    """Bad passphrase, missing/invalid signature, or an invalid amendment."""


# ------------------------------------------------------------------------ passphrase
def check_passphrase_strength(p: str) -> None:
    if len(p) < MIN_LEN:
        raise OwnerError(f"passphrase must be at least {MIN_LEN} characters")
    low = p.lower()
    if len(set(low)) < 6:
        raise OwnerError("passphrase is too repetitive")
    for row in _ROWS:
        for seq in (row, row[::-1]):
            if any(seq[i : i + 5] in low for i in range(len(seq) - 4)):
                raise OwnerError("passphrase contains a keyboard or alphabet run (like 'asdfg', '12345' or 'abcde')")


def _derive(passphrase: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(passphrase.encode(), salt=salt, n=n, r=r, p=p, dklen=32, maxmem=512 * 1024 * 1024)


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _paths(owner_dir: Path) -> dict[str, Path]:
    d = Path(owner_dir)
    return {
        "public": d / "owner_public.key",
        "private": d / "owner_private.json",
        "constitution": d / "constitution.toml",
        "sig": d / "constitution.sig",
    }


def is_initialised(owner_dir: Path = DEFAULT_OWNER_DIR) -> bool:
    return all(p.exists() for p in _paths(owner_dir).values())


# --------------------------------------------------------------------------- signing
def _load_private(owner_dir: Path, passphrase: str) -> Ed25519PrivateKey:
    f = _paths(owner_dir)["private"]
    try:
        blob = json.loads(f.read_text())
        salt, nonce, ct = (base64.b64decode(blob[k]) for k in ("salt", "nonce", "ct"))
        key = _derive(passphrase, salt, blob["n"], blob["r"], blob["p"])
        raw = AESGCM(key).decrypt(nonce, ct, _AAD)
    except InvalidTag:
        time.sleep(FAIL_DELAY)
        raise OwnerError("wrong passphrase") from None
    except (OSError, KeyError, ValueError) as exc:
        raise OwnerError(f"owner key file is unreadable: {type(exc).__name__}") from None
    return Ed25519PrivateKey.from_private_bytes(raw)


def verify_bytes(owner_dir: Path, data: bytes) -> None:
    """Raise OwnerError unless ``data`` carries a valid owner signature."""
    paths = _paths(owner_dir)
    try:
        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(paths["public"].read_text()))
        sig = base64.b64decode(paths["sig"].read_text())
        pub.verify(sig, data)
    except InvalidSignature:
        raise OwnerError("constitution signature is invalid: it was not signed by the owner") from None
    except (OSError, ValueError) as exc:
        raise OwnerError(f"owner signature material is missing or unreadable: {type(exc).__name__}") from None


def _sign_and_write(owner_dir: Path, priv: Ed25519PrivateKey, text: str) -> None:
    paths = _paths(owner_dir)
    tmp = paths["constitution"].with_suffix(".toml.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, paths["constitution"])
    paths["sig"].write_text(_b64(priv.sign(paths["constitution"].read_bytes())))


def init_owner(owner_dir: Path, passphrase: str, constitution_src: Path | None = None, overwrite: bool = False) -> Path:
    from .guardian import DEFAULT_CONSTITUTION, Constitution

    check_passphrase_strength(passphrase)
    paths = _paths(owner_dir)
    Path(owner_dir).mkdir(parents=True, exist_ok=True)
    if paths["public"].exists() and not overwrite:
        raise OwnerError(f"{owner_dir} is already initialised; refusing to replace the owner key")
    src = Path(constitution_src or DEFAULT_CONSTITUTION)
    Constitution.from_bytes(src.read_bytes())  # never sign something that wouldn't load

    priv = Ed25519PrivateKey.generate()
    raw = priv.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    salt, nonce = os.urandom(16), os.urandom(12)
    ct = AESGCM(_derive(passphrase, salt, KDF_N, KDF_R, KDF_P)).encrypt(nonce, raw, _AAD)
    paths["private"].write_text(json.dumps({"salt": _b64(salt), "nonce": _b64(nonce), "ct": _b64(ct),
                                            "n": KDF_N, "r": KDF_R, "p": KDF_P}))
    pub = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    paths["public"].write_text(_b64(pub))
    for f in (paths["private"],):
        try:
            os.chmod(f, 0o600)
        except OSError:
            pass
    _sign_and_write(owner_dir, priv, src.read_text(encoding="utf-8"))
    return Path(owner_dir)


# ------------------------------------------------------------------------- amending
def _set_key(text: str, section: str, key: str, literal: str) -> str:
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == f"[{section}]"), None)
    if start is None:
        raise OwnerError(f"constitution has no [{section}] section")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
    for i in range(start + 1, end):
        if re.match(rf"\s*{re.escape(key)}\s*=", lines[i]):
            lines[i] = f"{key} = {literal}"
            break
    else:
        insert_at = end
        while insert_at > start + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines.insert(insert_at, f"{key} = {literal}")
    return "\n".join(lines) + "\n"


def amend(owner_dir: Path, passphrase: str, approvals: dict[str, str] | None = None,
          limits: dict[str, int] | None = None, audit=None) -> list[str]:
    """Change tunable rules and re-sign. Needs the passphrase; nothing changes if it's wrong
    or if the result wouldn't be a valid constitution. Returns a list of what changed."""
    from .agent import TOOLS
    from .guardian import MAX_STEPS_CAP, MAX_WRITE_CAP, VERDICTS, Constitution

    approvals, limits = approvals or {}, limits or {}
    for tool, verdict in approvals.items():
        if tool not in TOOLS:
            raise OwnerError(f"{tool!r} is not a tool Becky has; tools are fixed in code")
        if verdict not in VERDICTS:
            raise OwnerError(f"verdict must be one of {VERDICTS}")
    caps = {"max_steps": MAX_STEPS_CAP, "max_write_bytes": MAX_WRITE_CAP}
    for k, v in limits.items():
        if k not in caps:
            raise OwnerError(f"{k!r} is not an adjustable limit")
        if not isinstance(v, int) or not 1 <= v <= caps[k]:
            raise OwnerError(f"{k} must be between 1 and {caps[k]} (a cap fixed in code)")

    priv = _load_private(owner_dir, passphrase)  # wrong passphrase stops here
    paths = _paths(owner_dir)
    old_text = paths["constitution"].read_text(encoding="utf-8")
    before = Constitution.from_bytes(old_text.encode())
    text = old_text
    changes = []
    for tool, verdict in approvals.items():
        text = _set_key(text, "approval", tool, f'"{verdict}"')
        changes.append(f"{tool}: {before.approval.get(tool, '(unset)')} -> {verdict}")
    for k, v in limits.items():
        text = _set_key(text, "limits", k, str(v))
        changes.append(f"{k}: {getattr(before, k)} -> {v}")
    try:
        Constitution.from_bytes(text.encode())
    except ValueError as exc:
        raise OwnerError(f"amendment would produce an invalid constitution: {exc}") from None
    _sign_and_write(owner_dir, priv, text)
    if audit is not None:
        audit.append(kind="amend", changes=changes)
    return changes


def reset_constitution(owner_dir: Path, passphrase: str) -> None:
    """Restore the shipped default rules (still needs the passphrase)."""
    from .guardian import DEFAULT_CONSTITUTION

    priv = _load_private(owner_dir, passphrase)
    _sign_and_write(owner_dir, priv, Path(DEFAULT_CONSTITUTION).read_text(encoding="utf-8"))


# --------------------------------------------------------------- signing arbitrary data
def sign_bytes(owner_dir: Path, passphrase: str, data: bytes) -> str:
    """Sign ``data`` with the owner key (needs the passphrase). Returns a base64 signature."""
    return _b64(_load_private(owner_dir, passphrase).sign(data))


def verify_signature(owner_dir: Path, data: bytes, signature_b64: str) -> bool:
    try:
        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(_paths(owner_dir)["public"].read_text()))
        pub.verify(base64.b64decode(signature_b64), data)
        return True
    except (InvalidSignature, OSError, ValueError):
        return False
