"""Tests for the chat window: command dispatch, failure handling, and a full scripted session."""

from __future__ import annotations

import argparse

import pytest

from ouroboros import owner as owner_mod
from ouroboros.__main__ import cmd_chat
from ouroboros.chat import HELP, Chat

PW = "correct horse battery staple"


def make_chat(**over):
    said, calls = [], []

    def rec(name, result="ok"):
        def fn(*a):
            calls.append((name, a))
            if isinstance(result, Exception):
                raise result
            return result
        return fn

    parts = dict(run_request=rec("request", "did it"), improve=rec("improve", "improved"),
                 review=rec("review", "reviewed"), compress=rec("compress", "compressed"),
                 search=rec("search", "found"))
    parts.update(over)
    chat = Chat(parts["run_request"], said.append, lambda p: "", parts["improve"], parts["review"],
                parts["compress"], parts["search"])
    return chat, said, calls


def test_requests_commands_help_and_quit_are_routed_correctly():
    chat, said, calls = make_chat()
    assert chat.handle("list the files") is True
    assert chat.handle("improve") is True and chat.handle("REVIEW") is True and chat.handle("compress") is True
    assert chat.handle("memory the dentist") is True
    assert chat.handle("help") is True and chat.handle("   ") is True
    assert [c[0] for c in calls] == ["request", "improve", "review", "compress", "search"]
    assert calls[0][1] == ("list the files",) and calls[4][1] == ("the dentist",)
    assert said == ["did it", "improved", "reviewed", "compressed", "found", HELP]
    for q in ("quit", "exit", "Q"):
        assert chat.handle(q) is False


def test_a_request_that_merely_contains_a_command_word_is_still_a_request():
    chat, said, calls = make_chat()
    chat.handle("please improve my notes")
    chat.handle("memory")  # no query: falls through as an ordinary request
    assert [c[0] for c in calls] == ["request", "request"]


@pytest.mark.parametrize("error,expected", [
    (RuntimeError("boom"), "That did not work: RuntimeError: boom"),
    (SystemExit("Refusing to start: x"), "Refusing to start: x"),
    (KeyboardInterrupt(), "(stopped)"),
])
def test_a_failure_is_shown_and_the_conversation_continues(error, expected):
    chat, said, _ = make_chat(improve=lambda: (_ for _ in ()).throw(error))
    assert chat.handle("improve") is True
    assert said == [expected]
    assert chat.handle("list the files") is True and said[-1] == "did it"


def test_the_loop_ends_on_quit_and_on_end_of_input():
    lines = iter(["list the files", "quit"])
    chat, said, calls = make_chat()
    chat.ask = lambda p: next(lines)
    chat.loop()
    assert said[-1] == "Goodbye." and len(calls) == 1

    def eof(prompt):
        raise EOFError

    chat2, said2, _ = make_chat()
    chat2.ask = eof
    chat2.loop()  # must return, not raise


def test_a_full_session_with_the_real_brain_guardian_and_tools(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(owner_mod, "KDF_N", 2**10)
    monkeypatch.setattr(owner_mod, "FAIL_DELAY", 0)
    owner = tmp_path / "owner"
    owner_mod.init_owner(owner, PW)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.txt").write_text("meeting at noon")
    script = iter(["list the files", "read notes.txt", "write hello world to greeting.txt",
                   "y", "remember call mom", "memory call", "hello there", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(script))
    cmd_chat(argparse.Namespace(owner_dir=str(owner), workspace=str(ws), audit=str(tmp_path / "audit.jsonl"),
                                runs=str(tmp_path / "runs")))
    out = capsys.readouterr().out
    assert "Files: notes.txt" in out
    assert "notes.txt says: meeting at noon" in out
    assert (ws / "greeting.txt").read_text() == "hello world"  # she asked, the 'y' allowed it
    assert "call mom" in (ws / "memory.md").read_text()
    assert "I can only list files" in out  # 'hello there' is declined, not guessed at
    assert "Goodbye." in out
    from ouroboros.guardian import AuditLog
    assert AuditLog(tmp_path / "audit.jsonl").verify()[0]


def test_a_declined_write_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(owner_mod, "KDF_N", 2**10)
    monkeypatch.setattr(owner_mod, "FAIL_DELAY", 0)
    owner = tmp_path / "owner"
    owner_mod.init_owner(owner, PW)
    script = iter(["write hello world to greeting.txt", "n", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(script))
    cmd_chat(argparse.Namespace(owner_dir=str(owner), workspace=str(tmp_path / "ws"),
                                audit=str(tmp_path / "a.jsonl"), runs=str(tmp_path / "runs")))
    assert not (tmp_path / "ws" / "greeting.txt").exists()
    assert "not allowed" in capsys.readouterr().out


def _session(tmp_path, monkeypatch, lines):
    monkeypatch.setattr(owner_mod, "KDF_N", 2**10)
    monkeypatch.setattr(owner_mod, "FAIL_DELAY", 0)
    owner = tmp_path / "owner"
    owner_mod.init_owner(owner, PW)
    script = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(script))
    return argparse.Namespace(owner_dir=str(owner), workspace=str(tmp_path / "ws"),
                              audit=str(tmp_path / "a.jsonl"), runs=str(tmp_path / "runs"))


def test_improve_adopts_a_better_brain_then_reanchors_with_the_passphrase(tmp_path, monkeypatch, capsys):
    from ouroboros import anchor as anchor_mod
    from ouroboros import grow as grow_mod

    args = _session(tmp_path, monkeypatch, ["improve", "y", "quit"])
    made = {}

    def fake_grow(champion, cfg, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / "champion.pt").write_bytes(b"x")

    def fake_adopt(cand, cur, confirm):
        made["asked"] = confirm("current brain 90.0% -> candidate 95.0% on 80 fresh held-out examples")
        return made["asked"]

    states = iter([anchor_mod.AnchorReport(present=False), anchor_mod.AnchorReport(present=True, ok=False)])
    monkeypatch.setattr(grow_mod, "grow", fake_grow)
    monkeypatch.setattr(grow_mod, "adopt", fake_adopt)
    monkeypatch.setattr(anchor_mod, "verify_anchor", lambda od: next(states))
    monkeypatch.setattr(anchor_mod, "create_anchor", lambda od, pw: made.setdefault("anchored_with", pw))
    monkeypatch.setattr("getpass.getpass", lambda prompt="": PW)
    cmd_chat(args)
    out = capsys.readouterr().out
    assert made["asked"] is True and made["anchored_with"] == PW
    assert "updated and re-anchored" in out


def test_improve_leaves_her_brain_alone_when_you_say_no_or_it_is_not_better(tmp_path, monkeypatch, capsys):
    from ouroboros import grow as grow_mod

    args = _session(tmp_path, monkeypatch, ["improve", "n", "improve", "quit"])
    monkeypatch.setattr(grow_mod, "grow", lambda c, cfg, out: (out.mkdir(parents=True, exist_ok=True),
                                                              (out / "champion.pt").write_bytes(b"x")))
    results = iter([False, ValueError("refusing: the candidate is not better")])

    def fake_adopt(cand, cur, confirm):
        r = next(results)
        if isinstance(r, Exception):
            raise r
        confirm("summary")
        return r

    monkeypatch.setattr(grow_mod, "adopt", fake_adopt)
    cmd_chat(args)
    out = capsys.readouterr().out
    assert out.count("Left her brain unchanged") == 2 and "not better" in out


def test_first_run_creates_a_workspace_with_a_readable_sample(tmp_path, monkeypatch, capsys):
    args = _session(tmp_path, monkeypatch, ["list the files", "read notes.txt", "quit"])
    cmd_chat(args)
    out = capsys.readouterr().out
    assert "Files: notes.txt" in out and "notes.txt says: welcome to becky" in out
