"""Tests for the setup check and the crash report that keep a failure from vanishing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def check():
    spec = importlib.util.spec_from_file_location("check_setup", ROOT / "check_setup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_everything_passes_on_a_working_install(check, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    results = check.checks()
    assert all(ok for _, ok, _, _ in results), [r for r in results if not r[1]]
    assert check.main(["--quiet"]) == 0


def test_a_missing_package_is_reported_with_a_fix_and_a_failing_exit_code(check, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    monkeypatch.setattr(check, "HERE", tmp_path)  # so the report file goes somewhere harmless
    assert check.main(["--quiet"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] torch is installed" in out and "setup.bat" in out and "problem(s) found" in out
    assert "[OK]" not in out  # --quiet shows only problems
    assert "FAIL" in (tmp_path / "check-report.txt").read_text()


def test_an_old_python_or_missing_brain_is_caught(check, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(check, "MIN_PYTHON", (99, 0))
    monkeypatch.setattr(check, "HERE", tmp_path)
    failed = {what for what, ok, _, _ in check.checks() if not ok}
    assert "Python 3.11 or newer" in failed and "her brain file is present" in failed


def test_the_full_report_lists_passes_when_not_quiet(check, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    check.main([])
    out = capsys.readouterr().out
    assert "[OK] Python 3.11 or newer" in out and "Everything is ready." in out


def test_an_unexpected_crash_is_saved_to_a_file_and_exits_cleanly(tmp_path, monkeypatch, capsys):
    from ouroboros import __main__ as cli

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["ouroboros", "audit"])
    monkeypatch.setattr(cli, "cmd_audit", lambda a: (_ for _ in ()).throw(RuntimeError("something odd")))
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1
    report = (tmp_path / ".becky" / "last-error.txt").read_text()
    assert "RuntimeError: something odd" in report and "command: ouroboros audit" in report
    assert "last-error.txt" in capsys.readouterr().out


def test_deliberate_exits_are_not_treated_as_crashes(tmp_path, monkeypatch):
    from ouroboros import __main__ as cli

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["ouroboros", "audit"])
    monkeypatch.setattr(cli, "cmd_audit", lambda a: (_ for _ in ()).throw(SystemExit("Refused: wrong passphrase")))
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert str(e.value) == "Refused: wrong passphrase"
    assert not (tmp_path / ".becky" / "last-error.txt").exists()


def test_the_launchers_run_the_check_first_and_never_close_silently():
    start = (ROOT / "start.bat").read_text()
    setup = (ROOT / "setup.bat").read_text()
    assert start.index("check_setup.py --quiet") < start.index("-m ouroboros chat")
    assert "last-error.txt" in start and start.rstrip().endswith("pause")
    assert "check_setup.py" in setup and setup.count("pause") >= 3
    serve = (ROOT / "serve.bat").read_text()
    assert serve.index("check_setup.py --quiet") < serve.index("-m ouroboros serve") and serve.rstrip().endswith("pause")
    for name in ("start.bat", "setup.bat", "serve.bat"):
        raw = (ROOT / name).read_bytes()
        assert raw.count(b"\r\n") == raw.count(b"\n")  # every line ends CRLF, as Windows batch files need
