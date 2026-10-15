"""Recursive self-improvement: each champion designs and trains its own successor.

A lineage member is two things:

* a **Genome** — the model's design (depth, width, heads, context) and training
  recipe (lr, batch, regularisation, how much to learn from the parent);
* a **Designer** — the policy that proposes the *next* genome: which knobs to
  touch, how far to move them, and a learned drift direction.

Every generation the champion's designer spawns mutated copies of itself, each
proposes a child genome, and the child is built, initialised from the
champion's weights where shapes allow, and trained by the champion
(distillation). A child replaces the champion only if it beats it on held-out
validation bits-per-byte by a margin. The designer that produced the winner is
inherited along with the model and updates its drift from the step that worked,
so the improver is itself subject to improvement: that is the recursive part.

One child per generation is a *control*: the champion's own design, trained the
same extra steps. A designed child only wins if its design beats "the same model,
trained more".

Everything is bounded and recorded. The search space is declarative data, never
generated code; each candidate must fit a parameter cap and a compute (FLOP)
budget set by the human running it; the run stops at the generation, patience or
wall-clock limit; and every candidate, accepted or not, is written to
``lineage.json``.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import torch

from .data import Splits, bits_per_byte, decode, encode
from .model import GPT, GPTConfig, inherit_weights
from .train import TrainRecipe, train


# --------------------------------------------------------------------- knobs
@dataclass(frozen=True)
class Knob:
    lo: float
    hi: float
    log: bool = False
    step: float | None = None  # rounding grid for integer-valued knobs

    def to_unit(self, v: float) -> float:
        if self.log:
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: float) -> float:
        u = min(1.0, max(0.0, u))
        if u in (0.0, 1.0):
            v = self.lo if u == 0.0 else self.hi
            return int(v) if self.step is not None else float(v)
        if self.log:
            v = math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        else:
            v = self.lo + u * (self.hi - self.lo)
        if self.step is not None:
            v = self.lo + round((v - self.lo) / self.step) * self.step
            return int(min(self.hi, max(self.lo, v)))
        # 4 significant figures: a round trip through unit space must not count as a change.
        return float(f"{min(self.hi, max(self.lo, v)):.4g}")


SPACE: dict[str, Knob] = {
    "n_layer": Knob(1, 8, step=1),
    "n_head": Knob(1, 8, step=1),
    "head_dim": Knob(8, 64, step=8),
    "block_size": Knob(32, 256, step=32),
    "dropout": Knob(0.0, 0.3),
    "lr": Knob(3e-4, 1e-2, log=True),
    "batch_size": Knob(8, 64, step=8),
    "weight_decay": Knob(0.0, 0.3),
    "distill_alpha": Knob(0.0, 0.9),
    "distill_temp": Knob(1.0, 4.0),
}


@dataclass(frozen=True)
class Genome:
    n_layer: int = 2
    n_head: int = 2
    head_dim: int = 32
    block_size: int = 64
    dropout: float = 0.0
    lr: float = 3e-3
    batch_size: int = 32
    weight_decay: float = 0.1
    distill_alpha: float = 0.0
    distill_temp: float = 2.0

    def model_config(self) -> GPTConfig:
        return GPTConfig(
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_head * self.head_dim,
            dropout=self.dropout,
        )

    def recipe(self, steps: int) -> TrainRecipe:
        return TrainRecipe(
            steps=steps,
            batch_size=self.batch_size,
            lr=self.lr,
            warmup=max(1, steps // 10),
            weight_decay=self.weight_decay,
            distill_alpha=self.distill_alpha,
            distill_temp=self.distill_temp,
        )

    def unit(self) -> dict[str, float]:
        return {k: SPACE[k].to_unit(getattr(self, k)) for k in SPACE}

    @classmethod
    def from_unit(cls, u: dict[str, float]) -> "Genome":
        return cls(**{k: SPACE[k].from_unit(v) for k, v in u.items()})


def count_params(g: Genome) -> int:
    d, L, V, T = g.n_head * g.head_dim, g.n_layer, 256, g.block_size
    per_block = 12 * d * d + 13 * d  # qkv + proj + 2-layer MLP + 2 LayerNorms, with biases
    return V * d + T * d + L * per_block + 2 * d


def train_flops(g: Genome, steps: int) -> float:
    """Standard 6 * params * tokens estimate for one training run."""
    return 6.0 * count_params(g) * steps * g.batch_size * g.block_size


# ------------------------------------------------------------------ designer
@dataclass
class Designer:
    """The policy that designs successors, and is itself inherited and mutated.

    ``p``: probability each knob is touched. ``sigma``: step size in the knob's
    normalised [0, 1] space. ``drift``: a direction learned from the steps that
    led to promotions, added to every proposal.
    """

    p: dict[str, float] = field(default_factory=lambda: {k: 0.3 for k in SPACE})
    sigma: dict[str, float] = field(default_factory=lambda: {k: 0.15 for k in SPACE})
    drift: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in SPACE})
    tau: float = 0.3  # meta-mutation rate (fixed: the recursion has to bottom out)
    drift_rate: float = 0.3

    def spawn(self, rng: random.Random) -> "Designer":
        """A mutated copy: self-adaptive step sizes and touch probabilities."""
        child = copy.deepcopy(self)
        for k in SPACE:
            child.sigma[k] = min(0.5, max(0.02, self.sigma[k] * math.exp(self.tau * rng.gauss(0, 1))))
            logit = math.log(self.p[k] / (1 - self.p[k])) + self.tau * rng.gauss(0, 1)
            child.p[k] = min(0.9, max(0.05, 1 / (1 + math.exp(-logit))))
        return child

    def propose(self, parent: Genome, rng: random.Random, attempts: int = 100) -> Genome | None:
        """A genome that differs from ``parent``, or None if none turned up."""
        for _ in range(attempts):
            u = parent.unit()
            touched = [k for k in SPACE if rng.random() < self.p[k]] or [rng.choice(list(SPACE))]
            for k in touched:
                u[k] = u[k] + self.drift[k] + rng.gauss(0, self.sigma[k])
            child = Genome.from_unit(u)
            if child != parent:
                return child
        return None

    def widen(self, factor: float = 1.6) -> "Designer":
        """A bolder copy, for when the search is stuck: bigger steps, more knobs touched."""
        wider = copy.deepcopy(self)
        for k in SPACE:
            wider.sigma[k] = min(0.5, self.sigma[k] * factor)
            wider.p[k] = min(0.9, self.p[k] + 0.2)
        return wider

    def learn(self, parent: Genome, child: Genome) -> None:
        """Remember the step that produced a promotion as a drift direction."""
        pu, cu = parent.unit(), child.unit()
        for k in SPACE:
            delta = cu[k] - pu[k]
            self.drift[k] = (1 - self.drift_rate) * self.drift[k] + self.drift_rate * delta * 0.5

    def to_dict(self) -> dict:
        r = lambda d: {k: round(v, 4) for k, v in d.items()}  # noqa: E731
        return {"p": r(self.p), "sigma": r(self.sigma), "drift": r(self.drift)}


# ---------------------------------------------------------------------- loop
@dataclass
class EvolveConfig:
    generations: int = 6
    children: int = 4  # designed children per generation (plus one control)
    steps: int = 300  # training steps each candidate gets per generation
    seed_steps: int | None = None  # steps for generation 0 (default: steps)
    max_params: int = 2_000_000
    max_flops: float = 5e13
    margin: float = 0.002  # bpb a child must beat the champion by
    patience: int = 3  # generations without promotion before the search counts as stuck
    escalations: int = 2  # when stuck, widen the search this many times before giving up
    time_budget_s: float | None = None
    eval_bytes: int = 50_000
    control: bool = True
    seed: int = 0
    device: str = "cpu"
    sample_prompt: str = "ROMEO:"
    sample_tokens: int = 200


@dataclass
class Member:
    gen: int
    genome: Genome
    designer: Designer
    model: GPT
    val_bpb: float


def within_budget(g: Genome, cfg: EvolveConfig) -> bool:
    return count_params(g) <= cfg.max_params and train_flops(g, cfg.steps) <= cfg.max_flops


def save_checkpoint(path: Path, m: Member, extra: dict | None = None) -> None:
    torch.save(
        {
            "gen": m.gen,
            "genome": asdict(m.genome),
            "model_config": m.model.cfg.to_dict(),
            "designer": m.designer.to_dict(),
            "val_bpb": m.val_bpb,
            "state_dict": m.model.state_dict(),
            **(extra or {}),
        },
        path,
    )


def load_model(path: Path, device: str = "cpu") -> GPT:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def sample(model: GPT, prompt: str, n: int, seed: int = 0, temperature: float = 0.8) -> str:
    device = next(model.parameters()).device
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = model.generate(idx, n, temperature=temperature, top_k=40, generator=gen)
    return decode(out[0].tolist())


def evolve(
    splits: Splits,
    cfg: EvolveConfig,
    out_dir: Path,
    seed_genome: Genome | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    t0 = time.time()
    lineage: dict = {"config": asdict(cfg), "candidates": [], "champions": []}

    def write_lineage() -> None:
        (out_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))

    def build(g: Genome, seed: int) -> GPT:
        torch.manual_seed(seed)
        return GPT(g.model_config()).to(cfg.device)

    def record_champion(m: Member) -> None:
        text = sample(m.model, cfg.sample_prompt, cfg.sample_tokens, seed=cfg.seed)
        lineage["champions"].append({
            "gen": m.gen, "val_bpb": round(m.val_bpb, 4), "params": m.model.num_params(),
            "genome": asdict(m.genome), "designer": m.designer.to_dict(), "sample": text,
        })
        save_checkpoint(out_dir / f"gen{m.gen:02d}.pt", m)
        save_checkpoint(out_dir / "champion.pt", m)
        write_lineage()

    # ---- generation 0: trained from scratch, no teacher -------------------
    g0 = seed_genome or Genome()
    if not within_budget(g0, cfg):
        raise ValueError(f"seed genome exceeds the budget: {count_params(g0)} params, "
                         f"{train_flops(g0, cfg.steps):.2e} FLOPs")
    log(f"gen 0: training seed model ({count_params(g0):,} params)")
    model = build(g0, cfg.seed)
    train(model, splits.train, replace(g0.recipe(cfg.seed_steps or cfg.steps), distill_alpha=0.0),
          seed=cfg.seed, log=log)
    champ = Member(0, g0, Designer(), model, bits_per_byte(model, splits.val, cfg.eval_bytes))
    log(f"gen 0: val {champ.val_bpb:.4f} bpb")
    record_champion(champ)

    stale = 0
    widened = 0
    stop_reason = "generations"
    for gen in range(1, cfg.generations + 1):
        if cfg.time_budget_s is not None and time.time() - t0 > cfg.time_budget_s:
            stop_reason = "time_budget"
            break
        log(f"\ngen {gen}: champion gen {champ.gen} at {champ.val_bpb:.4f} bpb designs "
            f"{cfg.children} children" + (" + control" if cfg.control else ""))
        teacher = champ.model
        proposals: list[tuple[str, Genome, Designer]] = []
        if cfg.control:
            proposals.append(("control", champ.genome, champ.designer))
        for _ in range(cfg.children):
            designer = champ.designer.spawn(rng)
            for _attempt in range(50):
                g = designer.propose(champ.genome, rng)
                if g is not None and within_budget(g, cfg):
                    proposals.append(("designed", g, designer))
                    break
            else:
                log("  designer could not find a proposal inside the budget; skipped")

        best: Member | None = None
        for i, (kind, g, designer) in enumerate(proposals):
            seed = cfg.seed + 1000 * gen + i
            child = build(g, seed)
            inherited = inherit_weights(child, teacher)
            recipe = g.recipe(cfg.steps)
            train(child, splits.train, recipe, teacher=teacher, seed=seed)
            bpb = bits_per_byte(child, splits.val, cfg.eval_bytes)
            changed = {k: [getattr(champ.genome, k), getattr(g, k)]
                       for k in SPACE if getattr(champ.genome, k) != getattr(g, k)}
            lineage["candidates"].append({
                "gen": gen, "kind": kind, "val_bpb": round(bpb, 4), "params": child.num_params(),
                "inherited_frac": round(inherited / child.num_params(), 3),
                "changed": changed, "genome": asdict(g),
            })
            log(f"  [{kind:8s}] val {bpb:.4f} bpb  {child.num_params():>9,} params  "
                f"inherited {inherited / child.num_params():4.0%}  {changed or ''}")
            if best is None or bpb < best.val_bpb:
                best = Member(gen, g, designer, child, bpb)

        if best is not None and best.val_bpb < champ.val_bpb - cfg.margin:
            if best.genome != champ.genome:
                best.designer = copy.deepcopy(best.designer)
                best.designer.learn(champ.genome, best.genome)
            log(f"gen {gen}: PROMOTED {champ.val_bpb:.4f} -> {best.val_bpb:.4f} bpb")
            champ = best
            record_champion(champ)
            stale = 0
            widened = 0
        else:
            stale += 1
            log(f"gen {gen}: no child beat the champion by {cfg.margin} bpb ({stale}/{cfg.patience})")
            write_lineage()
            if stale >= cfg.patience:
                if widened < cfg.escalations:
                    widened += 1
                    stale = 0
                    champ.designer = champ.designer.widen()
                    lineage.setdefault("escalations", []).append(gen)
                    log(f"gen {gen}: stuck, so changing approach: bolder steps, more knobs "
                        f"({widened}/{cfg.escalations})")
                    write_lineage()
                else:
                    stop_reason = "patience"
                    break

    # ---- final report: test split, touched once ---------------------------
    test_bpb = bits_per_byte(champ.model, splits.test)
    lineage["result"] = {
        "champion_gen": champ.gen,
        "val_bpb": round(champ.val_bpb, 4),
        "test_bpb": round(test_bpb, 4),
        "seed_val_bpb": lineage["champions"][0]["val_bpb"],
        "stop_reason": stop_reason,
        "seconds": round(time.time() - t0, 1),
    }
    write_lineage()
    log(f"\nfinal champion: gen {champ.gen}, val {champ.val_bpb:.4f} bpb, test {test_bpb:.4f} bpb "
        f"(stopped: {stop_reason})")
    return lineage
