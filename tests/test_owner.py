"""Tests for owner authority: only the passphrase holder can change Becky's rules."""

from __future__ import annotations

import json

import pytest

from ouroboros import owner as owner_mod
from ouroboros.agent import Agent
from ouroboros.guardian import Action, AuditLog, Constitution, Guardian, Halted
from ouroboros.owner import (
    OwnerError, amend, check_passphrase_strength, init_owner, is_initialised, reset_constitution,
    verify_bytes,
)

PW = "correct horse battery staple"


@pytest.fixture(autouse=True)
def fast_kdf(monkeypatch):
    monkeypatch.setattr(owner_mod, "KDF_N", 2**10)
    monkeypatch.setattr(owner_mod, "FAIL_DELAY", 0)


@pytest.fixture
def owner(tmp_path):
    d = tmp_path / "owner"
    init_owner(d, PW)
    return d


def guardian(tmp_path, owner_dir):
    return Guardian(owner_dir / "constitution.toml", tmp_path / "ws", AuditLog(tmp_path / "state" / "audit.jsonl"),
                    stop_file=tmp_path / "STOP", owner_dir=owner_dir)


# ------------------------------------------------------------------------ passphrase
@pytest.mark.parametrize("pw", [
    "short", "aaaaaaaaaaaaaaaa", "asdfghjkl13M", "qwertyuiop1234", "1234567890abcdefgh",
    "zyxwvutsrqponm", "aaaabbbbaaaabbbb",
])
def test_weak_passphrases_are_rejected(pw):
    with pytest.raises(OwnerError):
        check_passphrase_strength(pw)


def test_reasonable_passphrases_pass():
    for pw in (PW, "Tr0ub4dor&3-plum-lantern", "river-moss-copper-velvet"):
        check_passphrase_strength(pw)


# ------------------------------------------------------------------------------ init
def test_init_creates_signed_constitution_and_never_stores_the_passphrase(owner):
    assert is_initialised(owner)
    verify_bytes(owner, (owner / "constitution.toml").read_bytes())
    for f in owner.iterdir():
        assert PW.encode() not in f.read_bytes()
    blob = json.loads((owner / "owner_private.json").read_text())
    assert set(blob) == {"salt", "nonce", "ct", "n", "r", "p"}


def test_init_refuses_to_replace_an_existing_owner(owner):
    with pytest.raises(OwnerError, match="already initialised"):
        init_owner(owner, "another long passphrase here")


def test_init_refuses_a_weak_passphrase_and_leaves_nothing_behind(tmp_path):
    with pytest.raises(OwnerError):
        init_owner(tmp_path / "o", "asdfghjkl13M")
    assert not is_initialised(tmp_path / "o")


# ------------------------------------------------------------------- the signature
def test_hand_edited_rules_do_not_verify_and_will_not_start(tmp_path, owner):
    guardian(tmp_path, owner)  # starts fine while signed
    c = owner / "constitution.toml"
    c.write_text(c.read_text().replace('write_file = "ask"', 'write_file = "allow"'))
    with pytest.raises(OwnerError, match="not signed by the owner"):
        guardian(tmp_path, owner)


def test_editing_the_rules_mid_run_halts(tmp_path, owner):
    g = guardian(tmp_path, owner)
    assert g.check(Action("list_dir", {})).verdict == "allow"
    c = owner / "constitution.toml"
    c.write_text(c.read_text().replace('write_file = "ask"', 'write_file = "allow"'))
    with pytest.raises(Halted):
        g.check(Action("list_dir", {}))


def test_someone_elses_signature_is_rejected(tmp_path, owner):
    other = tmp_path / "other"
    init_owner(other, "a totally different passphrase")
    forged = (other / "constitution.sig").read_text()
    (owner / "constitution.sig").write_text(forged)
    with pytest.raises(OwnerError):
        guardian(tmp_path, owner)


def test_missing_signature_refuses_to_start(tmp_path, owner):
    (owner / "constitution.sig").unlink()
    with pytest.raises(OwnerError, match="missing"):
        guardian(tmp_path, owner)


def test_owner_directory_is_unreachable_to_the_agent(tmp_path, owner):
    g = Guardian(owner / "constitution.toml", tmp_path / "ws", AuditLog(tmp_path / "state" / "a.jsonl"),
                 stop_file=tmp_path / "STOP", owner_dir=owner)
    assert g.check(Action("read_file", {"path": f"../owner/owner_private.json"})).verdict == "deny"
    assert g.is_protected(owner / "owner_private.json")


# --------------------------------------------------------------------------- amend
def test_amend_with_the_right_passphrase_changes_rules_and_resigns(tmp_path, owner):
    audit = AuditLog(tmp_path / "state" / "audit.jsonl")
    changes = amend(owner, PW, approvals={"write_file": "allow"}, limits={"max_steps": 40}, audit=audit)
    assert "write_file: ask -> allow" in changes and "max_steps: 25 -> 40" in changes
    g = guardian(tmp_path, owner)  # still verifies after re-signing
    assert g.constitution.max_steps == 40
    assert g.check(Action("write_file", {"path": "a.txt", "content": "x"})).verdict == "allow"
    entry = json.loads(audit.path.read_text().splitlines()[-1])
    assert entry["kind"] == "amend" and PW not in json.dumps(entry)


def test_amend_with_the_wrong_passphrase_changes_nothing(tmp_path, owner):
    before = (owner / "constitution.toml").read_bytes(), (owner / "constitution.sig").read_text()
    with pytest.raises(OwnerError, match="wrong passphrase"):
        amend(owner, "not the passphrase at all", approvals={"write_file": "allow"})
    assert ((owner / "constitution.toml").read_bytes(), (owner / "constitution.sig").read_text()) == before


@pytest.mark.parametrize("kwargs", [
    {"approvals": {"shell": "allow"}},                 # tools are fixed in code
    {"approvals": {"write_file": "sure"}},             # not a verdict
    {"limits": {"max_steps": 10_000}},                 # above the cap fixed in code
    {"limits": {"max_steps": 0}},
    {"limits": {"max_write_bytes": 10**9}},
    {"limits": {"workspace": 1}},                      # not an adjustable limit
])
def test_amend_cannot_cross_the_limits_fixed_in_code(owner, kwargs):
    before = (owner / "constitution.toml").read_bytes()
    with pytest.raises(OwnerError):
        amend(owner, PW, **kwargs)
    assert (owner / "constitution.toml").read_bytes() == before


def test_a_signed_constitution_still_cannot_exceed_the_caps():
    with pytest.raises(ValueError, match="max_steps"):
        Constitution.from_bytes(b"[limits]\nmax_steps = 99999\n")


def test_a_signed_allow_everything_constitution_still_cannot_grant_a_shell_or_escape(tmp_path, owner):
    amend(owner, PW, approvals={"write_file": "allow", "read_file": "allow", "list_dir": "allow"})
    g = guardian(tmp_path, owner)
    assert g.check(Action("shell", {"cmd": "ls"})).verdict == "deny"
    assert g.check(Action("write_file", {"path": "../../x", "content": "x"})).verdict == "deny"
    assert g.check(Action("read_file", {"path": "/etc/passwd"})).verdict == "deny"


def test_reset_restores_defaults_with_the_passphrase_only(tmp_path, owner):
    amend(owner, PW, approvals={"write_file": "allow"})
    with pytest.raises(OwnerError):
        reset_constitution(owner, "wrong wrong wrong wrong")
    reset_constitution(owner, PW)
    assert guardian(tmp_path, owner).constitution.approval["write_file"] == "ask"


def test_the_agent_runs_normally_under_a_signed_constitution(tmp_path, owner):
    replies = iter([json.dumps({"tool": "write_file", "args": {"path": "n.txt", "content": "hi"}}),
                    json.dumps({"final": "done"})])
    g = guardian(tmp_path, owner)
    assert Agent(lambda m: next(replies), g, lambda a, d: True).run("x") == "done"
    assert (tmp_path / "ws" / "n.txt").read_text() == "hi"
