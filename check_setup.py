"""Checks that this PC has everything Becky needs, and says exactly what is missing.

Uses only the Python standard library, so it still runs when the real install failed.

    python check_setup.py            show every check
    python check_setup.py --quiet    show only problems (used by start.bat)

Also writes check-report.txt next to this file, so you can send it instead of copying text
out of a window.
"""

from __future__ import annotations

import hashlib
import importlib
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_PYTHON = (3, 11)


def _try_import(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        return True, str(getattr(module, "__version__", "ok"))
    except Exception as exc:  # any failure to import is a failed check
        return False, f"{type(exc).__name__}: {exc}"


def checks() -> list[tuple[str, bool, str, str]]:
    """(what, passed, detail, how to fix) for each requirement."""
    out: list[tuple[str, bool, str, str]] = []
    out.append(("Python 3.11 or newer", sys.version_info >= MIN_PYTHON, platform.python_version(),
                "Install Python 3.12 or 3.13 from python.org and tick 'Add python.exe to PATH', then run setup.bat."))
    for name in ("torch", "cryptography"):
        ok, detail = _try_import(name)
        out.append((f"{name} is installed", ok, detail, "Double-click setup.bat and read its messages to the end."))
    out.append(("scrypt (for your passphrase)", hasattr(hashlib, "scrypt"), "available" if hasattr(hashlib, "scrypt") else "missing",
                "Your Python was built without OpenSSL scrypt. Reinstall Python from python.org."))
    brain = HERE / "ouroboros" / "brain.pt"
    size = brain.stat().st_size if brain.is_file() else 0
    out.append(("her brain file is present", size > 500_000, f"{size:,} bytes",
                "ouroboros\\brain.pt is missing or damaged. Unzip becky.zip again."))
    sys.path.insert(0, str(HERE))
    ok, detail = _try_import("ouroboros.brain")
    out.append(("Becky's code loads", ok, detail, "Run setup.bat again; if it still fails, send me check-report.txt."))
    try:
        home = Path.home() / ".becky"
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        out.append(("can write to " + str(home), True, "yes", ""))
    except Exception as exc:
        out.append(("can write to your home folder", False, f"{type(exc).__name__}: {exc}",
                    "Run from a normal (not read-only or synced) folder and user account."))
    return out


def render(results: list[tuple[str, bool, str, str]], quiet: bool = False) -> str:
    lines = [f"Becky setup check  (Python {platform.python_version()} on {platform.platform()})", ""]
    for what, ok, detail, fix in results:
        if ok and quiet:
            continue
        lines.append(f"  [{'OK' if ok else 'FAIL'}] {what}: {detail}")
        if not ok:
            lines.append(f"         fix: {fix}")
    failed = sum(1 for r in results if not r[1])
    lines += ["", "Everything is ready." if not failed else f"{failed} problem(s) found. Fix the lines marked FAIL."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    quiet = "--quiet" in (argv if argv is not None else sys.argv[1:])
    results = checks()
    text = render(results)
    try:
        (HERE / "check-report.txt").write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass
    failed = any(not r[1] for r in results)
    if failed or not quiet:
        print(render(results, quiet=quiet))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
