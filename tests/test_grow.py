"""Tests for the keep-improving loop: it must climb a ladder of tactics when stuck, reset on
progress, stop honestly at a plateau, never regress, and never touch Becky's brain unasked."""

from __future__ import annotations

import json
import random

import pytest
import torch

from ouroboros import grow as G
from ouroboros.brain import BLOCK_SIZE, OwnLLM, save_brain
from ouroboros.model import GPT, GPTConfig


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=BLOCK_SIZE, n_layer=1, n_head=1, n_embd=8))


class Script:
    """Scripted stand-ins for training and scoring so the control flow is exact."""

    def __init__(self, gains: dict[int, float]):
        self.gains, self.rungs, self.round = gains, [], 0

    def trainer(self, champion, rung, cfg, rng, seed, log):
        self.round += 1
        self.rungs.append(rung)
        return tiny()  # content irrelevant; scorer decides

    def scorer(self, brain, eval_n, seed):
        # called as: champion score, then candidate score, for each round
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls == 1:
            return 0.50  # start
        is_candidate = self.calls % 2 == 1  # after the start call: champ, cand, champ, cand ...
        base = getattr(self, "level", 0.50)
        if not is_candidate:
            return base
        gain = self.gains.get(self.round, 0.0)
        if gain > 0:
            self.level = base + gain
        return base + gain


def run(script, tmp_path, **kw):
    check = kw.pop("regression_check", lambda b: 1.0)
    guard = kw.pop("guard", lambda b, n, s: 1.0)
    cfg = G.GrowConfig(rounds=kw.pop("rounds", 10), eval_n=4, **kw)
    return G.grow(tiny(), cfg, tmp_path, trainer=script.trainer, scorer=script.scorer,
                  regression_check=check, guard=guard, log=lambda *_: None)


def test_ladder_is_cheap_to_expensive():
    assert G.LADDER == ("fresh", "failures", "diversify", "polish", "grow")


def test_when_stuck_it_changes_tactic_each_round_then_stops_at_a_plateau(tmp_path):
    s = Script(gains={})
    j = run(s, tmp_path)
    assert s.rungs == ["fresh", "failures", "diversify", "polish", "grow"]
    assert j.stop_reason.startswith("plateau")
    assert j.champion_score == j.start_score == 0.5
    assert [r["result"] for r in j.rounds] == ["no gain"] * 5


def test_progress_resets_to_the_cheapest_tactic(tmp_path):
    s = Script(gains={1: 0.1, 2: 0.0, 3: 0.1})  # gain, stall, gain
    j = run(s, tmp_path, rounds=8)
    assert s.rungs[:4] == ["fresh", "fresh", "failures", "fresh"]  # reset after each promotion
    assert [r["result"] for r in j.rounds[:3]] == ["promoted", "no gain", "promoted"]
    assert j.champion_score == pytest.approx(0.7)


def test_the_champion_never_gets_worse_and_is_saved_on_promotion(tmp_path):
    s = Script(gains={1: 0.1})
    j = run(s, tmp_path, rounds=3)
    saved = json.loads((tmp_path / "journal.json").read_text())
    assert saved["champion_score"] == j.champion_score
    scores = [r["champion"] for r in j.rounds]
    assert scores == sorted(scores)
    assert (tmp_path / "champion.pt").exists()


def test_a_gain_smaller_than_the_margin_does_not_count(tmp_path):
    s = Script(gains={1: 0.001})
    j = run(s, tmp_path, rounds=1, margin=0.005)
    assert j.rounds[0]["result"] == "no gain"


def test_a_candidate_that_regresses_on_everyday_requests_is_blocked(tmp_path):
    calls = iter([1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # baseline 1.0, then candidate drops to 0.5
    s = Script(gains={1: 0.2})
    j = run(s, tmp_path, rounds=1, regression_check=lambda b: next(calls))
    assert j.rounds[0]["result"].startswith("blocked")
    assert j.regressions_blocked == 1 and j.champion_score == j.start_score


def test_a_gain_elsewhere_cannot_buy_a_regression_on_the_core_phrasings(tmp_path):
    scores = iter([1.0, 0.9])  # champion's core score, then the candidate's: down 10 points
    s = Script(gains={1: 0.2})
    j = run(s, tmp_path, rounds=1, guard=lambda b, n, sd: next(scores))
    assert j.rounds[0]["result"] == "blocked: worse on the standard phrasings"
    assert j.regressions_blocked == 1 and j.champion_score == j.start_score


def test_a_tiny_slip_on_the_core_phrasings_within_tolerance_is_allowed(tmp_path):
    scores = iter([1.0, 0.99])
    s = Script(gains={1: 0.2})
    j = run(s, tmp_path, rounds=1, guard=lambda b, n, sd: next(scores))
    assert j.rounds[0]["result"] == "promoted"


def test_time_budget_stops_the_run(tmp_path):
    s = Script(gains={})
    j = run(s, tmp_path, time_budget_s=-1.0)
    assert j.stop_reason == "time budget" and j.rounds == []


def test_grow_rung_is_skipped_not_forced_when_over_the_parameter_cap(tmp_path):
    def trainer(champion, rung, cfg, rng, seed, log):
        return G.build_candidate(champion, rung, cfg, rng, seed, log) if rung == "grow" else tiny()

    cfg = G.GrowConfig(rounds=5, eval_n=4, max_params=10, steps=1, fresh_examples=60, mine_pool=1)
    j = G.grow(tiny(), cfg, tmp_path, trainer=trainer, scorer=lambda b, n, s: 0.5,
               regression_check=lambda b: 1.0, log=lambda *_: None)
    assert j.rounds[-1]["tactic"] == "grow" and j.rounds[-1]["result"] == "skipped"


# ------------------------------------------------------------------ the tactics
def test_grow_model_adds_a_layer_and_keeps_every_learned_weight():
    parent = tiny()
    child = G.grow_model(parent)
    assert child.cfg.n_layer == parent.cfg.n_layer + 1
    assert torch.equal(child.blocks[0].mlp.fc.weight, parent.blocks[0].mlp.fc.weight)
    assert torch.equal(child.tok_emb.weight, parent.tok_emb.weight)
    x = torch.randint(0, 256, (1, 10))
    assert child.eval()(x)[0].shape == parent.eval()(x)[0].shape


def test_failure_mining_returns_only_real_mistakes():
    class AlwaysWrong(OwnLLM):
        def complete(self, prompt):
            return "nope"

    bad = G.mine_failures(AlwaysWrong(tiny()), random.Random(0), 30)
    assert len(bad) == 30 and all(e.target != "nope" for e in bad)

    class AlwaysRight(OwnLLM):
        def __init__(self):
            pass

        def complete(self, prompt):
            return self._answers[prompt]

    right = AlwaysRight()
    pool = G.make_examples(random.Random(0), 30, "train")[:30]
    right._answers = {e.prompt: e.target for e in pool}
    assert G.mine_failures(right, random.Random(0), 30) == []


def test_each_real_tactic_trains_a_child_without_touching_the_champion(tmp_path):
    champ = tiny()
    before = {k: v.clone() for k, v in champ.state_dict().items()}
    cfg = G.GrowConfig(steps=2, fresh_examples=80, mine_pool=2, batch_size=2, max_params=10**7)
    for rung in G.LADDER:
        child = G.build_candidate(champ, rung, cfg, random.Random(1), 1, lambda *_: None)
        assert child is not None
    assert all(torch.equal(before[k], v) for k, v in champ.state_dict().items())


# ------------------------------------------------------------------- adoption
@pytest.fixture
def brains(tmp_path):
    cur, cand = tmp_path / "brain.pt", tmp_path / "cand.pt"
    save_brain(tiny(), cur)
    save_brain(tiny(), cand)
    return cur, cand


def test_adopt_refuses_a_candidate_that_is_not_better(brains, monkeypatch):
    cur, cand = brains
    monkeypatch.setattr(G, "score", lambda b, n, s: 0.5)
    with pytest.raises(ValueError, match="not better"):
        G.adopt(cand, cur, confirm=lambda s: True)


def test_adopt_does_nothing_without_a_yes(brains, monkeypatch):
    cur, cand = brains
    before = cur.read_bytes()
    scores = iter([0.9, 0.5])
    monkeypatch.setattr(G, "score", lambda b, n, s: next(scores))
    assert G.adopt(cand, cur, confirm=lambda s: False) is False
    assert cur.read_bytes() == before and not cur.with_suffix(".previous.pt").exists()


def test_adopt_swaps_in_the_better_brain_and_keeps_a_backup(brains, monkeypatch):
    cur, cand = brains
    torch.manual_seed(1)
    save_brain(GPT(GPTConfig(block_size=BLOCK_SIZE, n_layer=2, n_head=1, n_embd=8)), cand)
    old, new = cur.read_bytes(), cand.read_bytes()
    scores = iter([0.9, 0.5])
    monkeypatch.setattr(G, "score", lambda b, n, s: next(scores))
    seen = []
    assert G.adopt(cand, cur, confirm=lambda s: seen.append(s) or True) is True
    assert cur.read_bytes() == new and cur.with_suffix(".previous.pt").read_bytes() == old
    assert "90.0%" in seen[0] and "50.0%" in seen[0]


def test_diversify_rewords_requests_without_leaking_held_out_phrasings():
    from ouroboros import brain as B

    plain = B.make_examples(random.Random(0), 400, "train")
    wrapped = B.make_examples(random.Random(0), 400, "train", wrap=0.9)
    plain_goals = {e.prompt.splitlines()[0] for e in plain}
    wrapped_goals = [e.prompt.splitlines()[0] for e in wrapped]
    fluff = ("can you ", "could you ", "hey becky, ", "i want you to ", "would you ", "go ahead and ", "quickly ")
    assert sum(any(w.startswith("GOAL: " + f) for f in fluff) or w.endswith((" for me", " now", ", thanks"))
               for w in wrapped_goals) > 150
    assert not any(g.startswith("GOAL: " + f) for g in plain_goals for f in fluff)
    held = {t.split("{")[0].strip() for v in B.HELD_OUT.values() for t in v}
    assert not any(g.startswith("GOAL: " + h) for g in wrapped_goals for h in held if h)


def test_build_corpus_never_includes_an_example_longer_than_the_context():
    from ouroboros import brain as B

    long_ex = B.Example("x", "GOAL: " + "a" * 300 + "\nOBS: none\nACT: ", '{"final": "x"}')
    short = B.make_examples(random.Random(0), 4, "train")[:2]
    data, mask = B.build_corpus([long_ex, *short])
    assert len(data) == sum(len(B.example_text(e)) for e in short)


def test_score_blends_trained_and_unseen_phrasings(monkeypatch):
    calls = []

    def fake(brain, n=150, seed=123, held_out=False):
        calls.append(held_out)
        return {"overall": 0.2 if held_out else 1.0}

    monkeypatch.setattr(G, "evaluate_brain", fake)
    assert G.score(None, 10, 1) == pytest.approx(0.6) and calls == [False, True]
