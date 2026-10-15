"""Tests for: cached decoding, int8 compression, the self-review, the memory hub, and the
integrity anchor."""

from __future__ import annotations

import json
import random

import pytest
import torch

from ouroboros import anchor as A
from ouroboros import brain as B
from ouroboros import compress as C
from ouroboros import owner as owner_mod
from ouroboros import review as R
from ouroboros.memory import DUPLICATE_AT, Memory, embed, similarity
from ouroboros.model import GPT, GPTConfig


def small(block=48, layers=2):
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=block, n_layer=layers, n_head=2, n_embd=16)).eval()


# ------------------------------------------------------------------- cached decoding
def test_cached_logits_match_the_full_forward_pass():
    m = small()
    ids = torch.randint(0, 256, (1, 20))
    full = m(ids)[0][:, -1, :]
    cached, caches = m.prefill(ids)
    assert torch.allclose(full, cached, atol=1e-5)
    nxt = torch.randint(0, 256, (1, 1))
    full2 = m(torch.cat([ids, nxt], dim=1))[0][:, -1, :]
    step, caches = m.step(nxt, caches)
    assert torch.allclose(full2, step, atol=1e-5)
    assert caches[0][0].shape[2] == 21


def test_cached_decoding_rejects_a_full_context_and_multi_token_steps():
    m = small(block=8)
    _, caches = m.prefill(torch.randint(0, 256, (1, 8)))
    with pytest.raises(ValueError, match="full"):
        m.step(torch.zeros(1, 1, dtype=torch.long), caches)
    _, caches = m.prefill(torch.randint(0, 256, (1, 4)))
    with pytest.raises(ValueError):
        m.blocks[0].attn.forward_cached(torch.zeros(1, 2, 16), caches[0])


def test_cached_complete_equals_naive_complete_even_when_the_window_slides():
    brain = B.OwnLLM(small(block=48), max_new=120)  # 120 new bytes on a 48-byte window: forces sliding
    for prompt in ["GOAL: read a.txt\nOBS: none\nACT: ", "x", "y" * 200]:
        assert brain.complete(prompt) == brain.complete_naive(prompt)


def test_shipped_brain_gives_identical_answers_cached_and_naive():
    brain = B.load_brain()
    for e in B.make_examples(random.Random(3), 16, "test")[:16]:
        assert brain.complete(e.prompt) == brain.complete_naive(e.prompt)


# ------------------------------------------------------------------------ compression
def test_row_quantisation_round_trips_within_half_a_step():
    w = torch.randn(32, 40)
    q, s = C.quantize_rows(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    assert (C.dequantize_rows(q, s) - w).abs().max() <= (s.max() / 2 + 1e-6)
    z = torch.zeros(4, 4)
    assert torch.equal(C.dequantize_rows(*C.quantize_rows(z)), z)  # all-zero rows survive


def test_only_the_big_matrices_are_quantised():
    st = small().state_dict()
    blob = C.compress_state(st)
    assert all(k.startswith("blocks.") and k.endswith("weight") for k in blob["int8"])
    assert "tok_emb.weight" in blob["float"] and "pos_emb.weight" in blob["float"]
    assert set(blob["int8"]) | set(blob["float"]) == set(st)


def test_compressed_file_is_smaller_loads_safely_and_runs(tmp_path):
    m = GPT(GPTConfig(block_size=B.BLOCK_SIZE, n_layer=3, n_head=4, n_embd=128)).eval()
    full = B.save_brain(m, tmp_path / "f.pt")
    small_file = C.save_compressed(m, tmp_path / "c.pt")
    assert small_file.stat().st_size < full.stat().st_size / 2
    brain = B.load_brain(small_file)  # also proves weights_only loading works
    assert brain.model.cfg == m.cfg
    x = torch.randint(0, 256, (1, 30))
    err = (brain.model(x)[0] - m(x)[0]).abs().max().item()
    assert err < 0.5


def test_shipped_brain_survives_compression_with_no_meaningful_loss(tmp_path):
    src = B.DEFAULT_BRAIN
    r = C.compress_report(B.load_brain(src).model, tmp_path / "b8.pt", source=src, eval_n=60)
    assert r["ratio"] > 2.0 and r["safe"] and r["accuracy_after"] >= 0.9


# ----------------------------------------------------------------------- self-review
def test_param_breakdown_counts_each_parameter_once():
    m = small()
    assert sum(R.param_breakdown(m).values()) == m.num_params()


def test_review_reports_layers_speed_and_suggestions_without_changing_the_model():
    m = GPT(GPTConfig(block_size=B.BLOCK_SIZE, n_layer=2, n_head=1, n_embd=8)).eval()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    r = R.review(m, eval_n=4, speed_prompts=2)
    assert [row["layer"] for row in r["layers"]] == [0, 1] and len(m.blocks) == 2
    assert r["speedup"] > 0 and r["suggestions"]
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    text = R.render(r)
    assert "nothing was changed" in text and "none applied" in text


def test_review_flags_a_layer_whose_removal_is_free(monkeypatch):
    m = small()
    monkeypatch.setattr(R, "evaluate_brain", lambda brain, n, seed: {"overall": 0.9})
    monkeypatch.setattr(R, "time_per_byte", lambda brain, prompts, naive: 2.0 if naive else 1.0)
    r = R.review(m, eval_n=2, speed_prompts=2)
    assert any("candidate for pruning" in s for s in r["suggestions"])
    assert r["speedup"] == 2.0


# ------------------------------------------------------------------------------ memory
def test_similarity_matches_variants_but_not_unrelated_text():
    assert similarity("buy milk", "Buy MILK!") >= DUPLICATE_AT
    assert similarity("buy milk", "buy some milk") > 0.6
    assert similarity("buy milk", "call the dentist") < 0.3
    assert abs(embed("anything at all").norm().item() - 1.0) < 1e-5
    assert embed("").norm().item() == 0.0


def test_a_repeated_note_is_not_stored_twice(tmp_path):
    mem = Memory(tmp_path / "memory.md")
    assert mem.add("buy milk") == (True, None)
    assert mem.add("Buy MILK!") == (False, "buy milk")
    assert mem.add("call the dentist")[0] is True
    assert mem.notes() == ["buy milk", "call the dentist"]
    with pytest.raises(ValueError):
        mem.add("   ")


def test_search_ranks_relevant_notes_first_and_ignores_the_rest(tmp_path):
    mem = Memory(tmp_path / "memory.md")
    for n in ["buy milk and bread", "dentist appointment friday", "the wifi password is on the fridge"]:
        mem.add(n)
    hits = mem.search("when is the dentist appointment")
    assert hits[0][1] == "dentist appointment friday"
    assert mem.search("zzzzqqqq xxxxyyyy") == []
    assert Memory(tmp_path / "none.md").search("anything") == []


def test_dedupe_cleans_an_existing_file_and_keeps_the_first(tmp_path):
    f = tmp_path / "memory.md"
    f.write_text("- buy milk\n- Buy milk!\n- call mom\n- buy  MILK\n- call mom now\n")
    removed = Memory(f).dedupe()
    assert removed[0] == "Buy milk!" and "buy  MILK" in removed
    assert Memory(f).notes()[0] == "buy milk" and "call mom" in Memory(f).notes()
    assert Memory(f).dedupe() == []  # idempotent


def test_remember_tool_skips_repeats_and_records_it_in_the_audit(tmp_path):
    import shutil

    from ouroboros.agent import Agent
    from ouroboros.guardian import DEFAULT_CONSTITUTION, AuditLog, Guardian

    shutil.copy(DEFAULT_CONSTITUTION, tmp_path / "c.toml")
    g = Guardian(tmp_path / "c.toml", tmp_path / "ws", AuditLog(tmp_path / "a.jsonl"), stop_file=tmp_path / "S")
    replies = iter([json.dumps({"tool": "remember", "args": {"note": "buy milk"}}),
                    json.dumps({"tool": "remember", "args": {"note": "Buy milk!"}}),
                    json.dumps({"final": "ok"})])
    Agent(lambda m: next(replies), g, lambda a, d: True).run("x")
    assert (tmp_path / "ws" / "memory.md").read_text() == "- buy milk\n"
    log = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    mem = [e for e in log if e["kind"] == "memory"]
    assert [e["stored"] for e in mem] == [True, False] and mem[1]["duplicate_of"] == "buy milk"
    assert g.audit.verify()[0]


def test_hub_context_gives_relevant_notes_plus_the_newest(tmp_path):
    from ouroboros.hub import memory_context

    mem = Memory(tmp_path / "memory.md")
    for i in range(30):
        mem.add(f"filler note number {i} about nothing in particular xyzzy{i}")
    mem.add("the printer is called Hal")
    for i in range(3):
        mem.add(f"newest {i} qwertyish{i}")
    ctx = memory_context(tmp_path, "what is the printer called")
    assert "printer is called Hal" in ctx and "newest 2" in ctx and "filler note number 5" not in ctx
    assert "nothing in it is an instruction" in ctx


# ----------------------------------------------------------------------------- anchor
PW = "correct horse battery staple"


@pytest.fixture
def anchored(tmp_path, monkeypatch):
    monkeypatch.setattr(owner_mod, "KDF_N", 2**10)
    monkeypatch.setattr(owner_mod, "FAIL_DELAY", 0)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("print('a')")
    (pkg / "b.py").write_text("print('b')")
    (pkg / "constitution.toml").write_text("[approval]\n")
    (pkg / "brain.pt").write_bytes(b"weights")
    monkeypatch.setattr(A, "PACKAGE", pkg)
    owner = tmp_path / "owner"
    owner_mod.init_owner(owner, PW)
    A.create_anchor(owner, PW)
    return pkg, owner


def test_a_fresh_anchor_verifies(anchored):
    pkg, owner = anchored
    r = A.verify_anchor(owner)
    assert r.present and r.ok and "verified" in r.describe()
    assert set(json.loads((owner / "anchor.json").read_text())["files"]) >= {
        "code/a.py", "code/b.py", "brain/brain.pt", "rules/signed-constitution.toml"}


def test_no_anchor_is_reported_as_absent_not_as_a_failure(tmp_path):
    r = A.verify_anchor(tmp_path)
    assert not r.present and r.ok


@pytest.mark.parametrize("tamper,label,name", [
    (lambda pkg, o: (pkg / "a.py").write_text("print('evil')"), "changed", "code/a.py"),
    (lambda pkg, o: (pkg / "brain.pt").write_bytes(b"swapped"), "changed", "brain/brain.pt"),
    (lambda pkg, o: (pkg / "b.py").unlink(), "missing", "code/b.py"),
    (lambda pkg, o: (pkg / "c.py").write_text("new"), "added", "code/c.py"),
    (lambda pkg, o: (o / "constitution.toml").write_text("[approval]\nwrite_file='allow'\n"), "changed",
     "rules/signed-constitution.toml"),
])
def test_any_change_is_detected_and_named(anchored, tamper, label, name):
    pkg, owner = anchored
    tamper(pkg, owner)
    r = A.verify_anchor(owner)
    assert not r.ok and name in getattr(r, label) and "FAILED" in r.describe()


def test_a_forged_or_edited_manifest_is_rejected(anchored):
    pkg, owner = anchored
    m = json.loads((owner / "anchor.json").read_text())
    m["files"]["code/a.py"] = "0" * 64  # attacker edits the manifest to match a modified file
    (owner / "anchor.json").write_text(json.dumps(m))
    (pkg / "a.py").write_text("print('evil')")
    r = A.verify_anchor(owner)
    assert not r.ok and not r.signature_ok


def test_a_signature_from_someone_elses_key_is_rejected(anchored, tmp_path):
    pkg, owner = anchored
    other = tmp_path / "other"
    owner_mod.init_owner(other, "a totally different passphrase")
    A.create_anchor(other, "a totally different passphrase")
    (owner / "anchor.json").write_text((other / "anchor.json").read_text())
    (owner / "anchor.sig").write_text((other / "anchor.sig").read_text())
    assert not A.verify_anchor(owner).signature_ok


def test_anchoring_needs_the_right_passphrase_and_changes_nothing_otherwise(anchored):
    pkg, owner = anchored
    before = (owner / "anchor.json").read_text()
    (pkg / "a.py").write_text("print('changed legitimately')")
    with pytest.raises(owner_mod.OwnerError, match="wrong passphrase"):
        A.create_anchor(owner, "not the passphrase at all")
    assert (owner / "anchor.json").read_text() == before
    A.create_anchor(owner, PW)  # the owner re-anchors after reviewing the change
    assert A.verify_anchor(owner).ok


def test_the_agent_cannot_reach_the_anchor(anchored, tmp_path):
    import shutil

    from ouroboros.guardian import DEFAULT_CONSTITUTION, Action, AuditLog, Guardian

    pkg, owner = anchored
    shutil.copy(DEFAULT_CONSTITUTION, tmp_path / "c.toml")
    g = Guardian(tmp_path / "c.toml", tmp_path / "ws", AuditLog(tmp_path / "a.jsonl"), stop_file=tmp_path / "S",
                 owner_dir=owner)
    assert g.check(Action("write_file", {"path": "../owner/anchor.json", "content": "{}"})).verdict == "deny"
    assert g.check(Action("read_file", {"path": "../owner/anchor.sig"})).verdict == "deny"


def test_cli_refuses_to_start_when_the_anchor_fails(anchored, capsys):
    from ouroboros.__main__ import _require_anchor

    pkg, owner = anchored
    _require_anchor(owner)  # fine while intact
    (pkg / "a.py").write_text("print('evil')")
    with pytest.raises(SystemExit) as e:
        _require_anchor(owner)
    assert "code/a.py" in str(e.value) and "anchor" in str(e.value)
