"""Tests for the permission-gated self-improving code.

The safety properties are the point, so most of these are adversarial: can a
candidate smuggle in an import, a call, a file write? Can the loop ever modify
the target without the confirm callback saying yes?
"""

from __future__ import annotations

import json
import random

import pytest

from ouroboros import selfedit as se
from ouroboros.selfedit import (
    SelfEditConfig, Task, UnsafeCandidate, apply, evaluate, extract_region,
    improve, mutate, splice_region, validate_region,
)

SEED_FILE = """x = 1
# EVOLVE-REGION
def solve(x):
    return 0.0 * x * x + 0.0 * x + 0.0
# END-EVOLVE-REGION
after = 2
"""


@pytest.fixture
def target(tmp_path):
    p = tmp_path / "organism.py"
    p.write_text(SEED_FILE)
    return p


def test_extract_and_splice_roundtrip():
    before, region, after = extract_region(SEED_FILE)
    assert "def solve" in region and "EVOLVE-REGION" not in region
    assert splice_region(SEED_FILE, region) == SEED_FILE
    spliced = splice_region(SEED_FILE, "def solve(x):\n    return x\n")
    assert "return x" in spliced and before in spliced and after in spliced


def test_rejects_files_without_exactly_one_region():
    for bad in ["no markers", SEED_FILE + SEED_FILE, SEED_FILE.replace("# END-EVOLVE-REGION", "")]:
        with pytest.raises(ValueError):
            extract_region(bad)


@pytest.mark.parametrize("body", [
    "import os\ndef solve(x):\n    return x\n",
    "def solve(x):\n    return os.system('echo hi')\n",
    "def solve(x):\n    return (1).__class__\n",
    "def solve(x):\n    return x.bit_length()\n",
    "def solve(x):\n    return abs(x)\n",            # calls are disallowed, even safe ones
    "def helper(x):\n    return x\ndef solve(x):\n    return x\n",  # extra function
    "def notsolve(x):\n    return x\n",
])
def test_validator_rejects_anything_beyond_arithmetic(body):
    with pytest.raises(UnsafeCandidate):
        validate_region(body)


def test_validator_accepts_plain_arithmetic():
    validate_region("def solve(x):\n    return 2.0 * x * x - 3.0 * x + 1.0\n")


def test_mutation_stays_valid_and_distinct_and_safe():
    rng = random.Random(0)
    body = "def solve(x):\n    return 0.0 * x * x + 0.0 * x + 0.0\n"
    seen = set()
    for _ in range(300):
        m = mutate(body, rng, n_edits=2)
        assert m != body
        validate_region(m)              # never produces anything unsafe
        assert "def solve" in m
        seen.add(m)
    assert len(seen) > 50               # it actually explores


def test_evaluate_scores_seed_and_perfect_solution():
    task = Task()
    ok, seed_score = evaluate("def solve(x):\n    return 0.0*x*x + 0.0*x + 0.0\n", task.holdout())
    assert ok and seed_score > 1.0
    ok, perfect = evaluate(
        f"def solve(x):\n    return {task.a}*x*x + ({task.b})*x + {task.c}\n", task.holdout())
    assert ok and perfect == pytest.approx(0.0, abs=1e-9)
    assert perfect < seed_score


def test_evaluate_refuses_unsafe_body_without_running_it():
    ok, score = evaluate("import os\ndef solve(x):\n    os.remove('x')\n    return x\n", Task().train())
    assert not ok and score == float("inf")


def test_evaluate_survives_a_crashing_candidate():
    ok, score = evaluate("def solve(x):\n    return 1.0 / 0.0\n", Task().train())
    assert not ok and score == float("inf")


def test_improve_reduces_error_and_never_touches_target(target):
    original = target.read_text()
    cfg = SelfEditConfig(generations=40, population=16, seed=1, patience=40)
    out = target.parent / "run"
    j = improve(target, cfg=cfg, out_dir=out, log=lambda *_: None)

    assert j.champion_score < j.baseline_score * 0.5   # at least halved the error
    assert target.read_text() == original              # propose-only: target untouched
    assert j.history and [h["score"] for h in j.history] == sorted(
        (h["score"] for h in j.history), reverse=True)  # monotonically better

    on_disk = json.loads((out / "journal.json").read_text())
    assert on_disk["applied"] is False
    champ = (out / "champion.py").read_text()
    assert "EVOLVE-REGION" in champ
    validate_region(extract_region(champ)[1])


def test_improve_can_reach_the_hidden_target(target):
    cfg = SelfEditConfig(generations=200, population=24, seed=3, patience=200, scale=1.5)
    j = improve(target, cfg=cfg, log=lambda *_: None)
    assert j.champion_score < 0.05   # effectively recovers 2x^2 - 3x + 1


def test_apply_does_nothing_when_confirm_says_no(target):
    original = target.read_text()
    body = "def solve(x):\n    return 2.0 * x * x - 3.0 * x + 1.0\n"
    assert apply(target, body, confirm=lambda diff: False) is False
    assert target.read_text() == original


def test_apply_writes_back_only_when_confirmed(target):
    body = "def solve(x):\n    return 2.0 * x * x - 3.0 * x + 1.0\n"
    seen = {}
    assert apply(target, body, confirm=lambda diff: seen.setdefault("diff", diff) or True) is True
    assert "2.0 * x * x" in target.read_text()
    assert "- 3.0 * x" in seen["diff"]      # the human was shown the real diff
    # and the rest of the file is intact
    assert target.read_text().startswith("x = 1\n")
    assert target.read_text().rstrip().endswith("after = 2")


def test_apply_refuses_unsafe_body_even_if_confirmed(target):
    with pytest.raises(UnsafeCandidate):
        apply(target, "import os\ndef solve(x):\n    return x\n", confirm=lambda d: True)
    assert "import os" not in target.read_text()


def test_apply_refuses_broken_body_even_if_confirmed(target):
    with pytest.raises(ValueError, match="run cleanly"):
        apply(target, "def solve(x):\n    return 1.0 / 0.0\n", confirm=lambda d: True)


def test_shipped_organism_is_valid_and_runs():
    from pathlib import Path
    import ouroboros
    organism = Path(ouroboros.__file__).parent / "organism.py"
    _, region, _ = extract_region(organism.read_text())
    validate_region(region)
    ok, _ = evaluate(region, Task().train())
    assert ok
