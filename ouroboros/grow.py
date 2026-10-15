"""Keep improving, and when stuck, change approach.

A plateau is not the end. When a round fails to beat the champion, the next round tries a
*different tactic*, climbing a ladder from cheap to expensive:

    0 fresh      more training on freshly generated data
    1 failures   mine the champion's own mistakes and train on them harder
    2 diversify  the same requests reworded many ways, so it learns to see through fluff
    3 polish     a much lower learning rate and twice the steps, to settle fine errors
    4 grow       add a layer (keeping every learned weight) and train it

Any improvement drops back to rung 0, the cheapest tactic. The run ends only when every
rung has been tried in a row without a gain: then it reports honestly that it has plateaued
and what it would take to go further (more data variety or a bigger budget).

Bounds, all set by the person running it
----------------------------------------
* rounds, steps per round, a wall-clock limit, and a parameter cap (the ``grow`` rung is
  skipped, not forced, once it would exceed it);
* a candidate replaces the champion only if it beats it on freshly generated held-out
  examples *by a margin*, and does not regress on the end-to-end everyday-request check;
* nothing outside ``out_dir`` is touched. The improved brain is a proposal; it only becomes
  Becky's brain through :func:`adopt`, which shows the scores and waits for your yes.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from .brain import (
    DEFAULT_BRAIN, OwnLLM, build_corpus, evaluate_brain, load_brain, make_examples, real_world_eval,
    save_brain, train_masked,
)
from .model import GPT, GPTConfig, inherit_weights
from .train import TrainRecipe

LADDER = ("fresh", "failures", "diversify", "polish", "grow")


@dataclass
class GrowConfig:
    rounds: int = 8
    steps: int = 600
    batch_size: int = 24
    lr: float = 1e-3
    time_budget_s: float | None = None
    margin: float = 0.02  # a gain smaller than the measurement noise does not count
    guard_tolerance: float = 0.02  # how far the core (standard-phrasing) score may slip for a gain elsewhere
    eval_n: int = 120
    mine_pool: int = 400
    fresh_examples: int = 8000
    max_params: int = 3_000_000
    seed: int = 0


@dataclass
class Journal:
    start_score: float
    champion_score: float
    rounds: list[dict] = field(default_factory=list)
    stop_reason: str = "rounds"
    regressions_blocked: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ the tactics
def mine_failures(brain: OwnLLM, rng: random.Random, pool: int):
    """Fresh training-split examples the brain currently gets wrong."""
    return [e for e in make_examples(rng, pool, "train")[:pool] if brain.complete(e.prompt) != e.target]


def grow_model(parent: GPT) -> GPT:
    cfg = parent.cfg
    child = GPT(GPTConfig(cfg.vocab_size, cfg.block_size, cfg.n_layer + 1, cfg.n_head, cfg.n_embd, cfg.dropout))
    inherit_weights(child, parent)
    return child


def build_candidate(champion: GPT, rung: str, cfg: GrowConfig, rng: random.Random, seed: int,
                    log: Callable[[str], None]) -> GPT | None:
    """Train a child of ``champion`` with the tactic for ``rung``. None means: skip this rung."""
    model = GPT(champion.cfg)
    model.load_state_dict(champion.state_dict())
    lr, steps = cfg.lr, cfg.steps
    examples = make_examples(rng, cfg.fresh_examples, "train")
    if rung == "failures":
        bad = mine_failures(OwnLLM(champion), rng, cfg.mine_pool)
        log(f"    mined {len(bad)} failures from {cfg.mine_pool} fresh examples")
        examples = examples + bad * 6
        rng.shuffle(examples)
    elif rung == "diversify":
        examples = make_examples(rng, cfg.fresh_examples, "train", wrap=0.7)
    elif rung == "polish":
        lr, steps = cfg.lr * 0.3, cfg.steps * 2
    elif rung == "grow":
        model = grow_model(champion)
        if model.num_params() > cfg.max_params:
            log(f"    grow would need {model.num_params():,} parameters (cap {cfg.max_params:,}); skipped")
            return None
        lr = cfg.lr * 1.5
    data, mask = build_corpus(examples)
    recipe = TrainRecipe(steps=steps, batch_size=cfg.batch_size, lr=lr, warmup=max(1, steps // 20))
    train_masked(model, data, mask, recipe, seed=seed, log=log, log_every=max(1, steps // 4))
    return model


def score(brain: OwnLLM, eval_n: int, seed: int) -> float:
    """Half exact-match on phrasings it was trained on, half on phrasings it never saw, all with
    words reserved for testing. Unseen phrasings are where there is room left to improve."""
    seen = evaluate_brain(brain, n=eval_n, seed=seed)["overall"]
    held = evaluate_brain(brain, n=eval_n, seed=seed, held_out=True)["overall"]
    return (seen + held) / 2


def standard_score(brain: OwnLLM, eval_n: int, seed: int) -> float:
    """Exact-match on the phrasings Becky is documented to understand. This is her core job."""
    return evaluate_brain(brain, n=eval_n, seed=seed)["overall"]


def everyday_rate(brain: OwnLLM) -> float:
    r = real_world_eval(brain)
    return r["passed"] / r["total"]


def _holds_core(guard, champion: GPT, cand: GPT, cfg: GrowConfig, seed: int) -> bool:
    """True if the candidate has not slipped on the core phrasings by more than the tolerance."""
    before = guard(OwnLLM(champion), cfg.eval_n, seed)
    after = guard(OwnLLM(cand), cfg.eval_n, seed)
    return after >= before - cfg.guard_tolerance


# --------------------------------------------------------------------- the loop
def grow(
    champion: GPT,
    cfg: GrowConfig,
    out_dir: Path,
    trainer: Callable[..., GPT | None] = build_candidate,
    scorer: Callable[[OwnLLM, int, int], float] = score,
    regression_check: Callable[[OwnLLM], float] | None = None,
    guard: Callable[[OwnLLM, int, int], float] = standard_score,
    log: Callable[[str], None] = print,
) -> Journal:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    t0 = time.monotonic()
    regression_check = regression_check or everyday_rate

    base_everyday = regression_check(OwnLLM(champion))
    j = Journal(start_score=round(scorer(OwnLLM(champion), cfg.eval_n, cfg.seed), 4), champion_score=0.0)
    j.champion_score = j.start_score
    log(f"start: {j.start_score:.1%} on held-out words, everyday requests {base_everyday:.0%}")
    save_brain(champion, out_dir / "champion.pt", meta={"score": j.champion_score})

    stuck = 0
    for rnd in range(1, cfg.rounds + 1):
        if cfg.time_budget_s is not None and time.monotonic() - t0 > cfg.time_budget_s:
            j.stop_reason = "time budget"
            break
        rung = LADDER[stuck]
        log(f"\nround {rnd}: tactic '{rung}'" + ("  (stuck, so changing approach)" if stuck else ""))
        eval_seed = cfg.seed + 1000 + rnd  # fresh held-out examples every round
        champ_score = scorer(OwnLLM(champion), cfg.eval_n, eval_seed)
        cand = trainer(champion, rung, cfg, rng, cfg.seed + rnd, log)
        entry = {"round": rnd, "tactic": rung, "champion": round(champ_score, 4)}
        if cand is None:
            entry["result"] = "skipped"
            outcome = False
        else:
            cand_score = scorer(OwnLLM(cand), cfg.eval_n, eval_seed)
            entry.update(candidate=round(cand_score, 4), params=cand.num_params())
            outcome = cand_score > champ_score + cfg.margin
            cand_everyday = regression_check(OwnLLM(cand)) if outcome else None
            if outcome and cand_everyday < base_everyday - 1e-9:
                outcome = False
                j.regressions_blocked += 1
                entry["result"] = "blocked: worse on everyday requests"
            elif outcome and not _holds_core(guard, champion, cand, cfg, eval_seed):
                outcome = False
                j.regressions_blocked += 1
                entry["result"] = "blocked: worse on the standard phrasings"
            elif outcome:
                entry["result"] = "promoted"
            else:
                entry["result"] = "no gain"
        log(f"  {entry.get('candidate', '-')} vs champion {entry['champion']}: {entry['result']}")
        if outcome:
            champion = cand
            j.champion_score = entry["candidate"]
            base_everyday = max(base_everyday, cand_everyday)
            save_brain(champion, out_dir / "champion.pt", meta={"score": j.champion_score, "round": rnd})
            stuck = 0
        else:
            stuck += 1
        j.rounds.append(entry)
        (out_dir / "journal.json").write_text(json.dumps(j.to_dict(), indent=2))
        if stuck >= len(LADDER):
            j.stop_reason = "plateau: every tactic tried without a gain"
            log("\nplateau: all tactics failed in a row. To go further, add more variety to the data or raise the budget.")
            break

    (out_dir / "journal.json").write_text(json.dumps(j.to_dict(), indent=2))
    log(f"\nfinished ({j.stop_reason}): {j.start_score:.1%} -> {j.champion_score:.1%}")
    return j


# ---------------------------------------------------------------- adoption gate
def adopt(candidate: Path, current: Path, confirm: Callable[[str], bool], eval_n: int = 120, seed: int = 4242) -> bool:
    """Replace Becky's brain with ``candidate`` only if it scores better and ``confirm`` says yes."""
    cand, cur = load_brain(candidate), load_brain(current)
    cs, bs = score(cand, eval_n, seed), score(cur, eval_n, seed)
    summary = f"current brain {bs:.1%} -> candidate {cs:.1%} on {eval_n} fresh held-out examples"
    if cs <= bs:
        raise ValueError(f"refusing: the candidate is not better ({summary})")
    if not confirm(summary):
        return False
    backup = Path(current).with_suffix(".previous.pt")
    Path(backup).write_bytes(Path(current).read_bytes())
    Path(current).write_bytes(Path(candidate).read_bytes())
    return True
