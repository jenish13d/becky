"""Permission-gated self-improving code.

A program that reads its own source, proposes an upgrade to a clearly bounded
region of itself, proves the upgrade is better on a held-out task, and adopts
*nothing* until a human approves the diff.

The three properties that make this safe rather than a runaway:

1. **Bounded edit surface.** Only the code between ``# EVOLVE-REGION`` and
   ``# END-EVOLVE-REGION`` in the target file is ever changed. One function,
   one file.
2. **Safe by construction.** Mutations happen at the syntax-tree level and only
   tune numeric constants and swap arithmetic operators. Every candidate is then
   re-validated to contain no imports, no attribute access, no function calls,
   and no dunder names — the mutation space cannot express those, and the
   validator rejects them if they somehow appear anyway.
3. **Sandboxed, test-gated, human-gated.** Each candidate runs in a separate,
   resource-limited process with restricted builtins and a timeout. A candidate
   is kept only if it passes correctness checks and beats the incumbent. And the
   improved source is written back into the target file only through
   :func:`apply`, which calls a ``confirm`` callback first — the human switch.
"""

from __future__ import annotations

import ast
import json
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

REGION_START = "# EVOLVE-REGION"
REGION_END = "# END-EVOLVE-REGION"

_SWAPPABLE = (ast.Add, ast.Sub, ast.Mult, ast.Div)


# ------------------------------------------------------------------- region I/O
def extract_region(source: str) -> tuple[str, str, str]:
    """Split a file into (before, region_body, after) around the markers.

    ``region_body`` is the code *between* the marker lines, markers excluded.
    """
    lines = source.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ln.strip() == REGION_START or ln.strip().startswith(REGION_START + " ")]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == REGION_END]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise ValueError("target must contain exactly one EVOLVE-REGION ... END-EVOLVE-REGION block")
    s, e = starts[0], ends[0]
    return "".join(lines[: s + 1]), "".join(lines[s + 1 : e]), "".join(lines[e:])


def splice_region(source: str, new_body: str) -> str:
    before, _, after = extract_region(source)
    if not new_body.endswith("\n"):
        new_body += "\n"
    return before + new_body + after


# ---------------------------------------------------------------- safety checks
class UnsafeCandidate(ValueError):
    """Raised when a candidate region contains anything outside the allowed space."""


def validate_region(body: str) -> ast.Module:
    """Parse a region body and reject anything beyond constant/operator arithmetic."""
    tree = ast.parse(body)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if [f.name for f in funcs] != ["solve"]:
        raise UnsafeCandidate("region must define exactly one function named solve")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call, ast.Attribute,
                             ast.Lambda, ast.Global, ast.Nonlocal, ast.Await,
                             ast.comprehension, ast.With, ast.Try)):
            raise UnsafeCandidate(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeCandidate(f"disallowed name: {node.id}")
    return tree


# --------------------------------------------------------------------- mutation
def _mutable_sites(tree: ast.AST) -> list[ast.AST]:
    sites: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            sites.append(node)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, _SWAPPABLE):
            sites.append(node)
    return sites


def mutate(body: str, rng: random.Random, n_edits: int = 1, scale: float = 1.0) -> str:
    """Return a new region body differing from ``body`` by a few bounded edits."""
    for _ in range(50):
        tree = ast.parse(body)
        sites = _mutable_sites(tree)
        if not sites:
            raise ValueError("no mutable sites in region")
        for node in rng.sample(sites, min(n_edits, len(sites))):
            if isinstance(node, ast.Constant):
                old = float(node.value)
                if rng.random() < 0.5:
                    node.value = round(old + rng.gauss(0, scale), 4)
                else:
                    node.value = round(old * rng.uniform(0.5, 1.5), 4)
            else:  # BinOp
                node.op = rng.choice([op for op in _SWAPPABLE if not isinstance(node.op, op)])()
        ast.fix_missing_locations(tree)
        new_body = ast.unparse(tree) + "\n"
        if new_body != (ast.unparse(ast.parse(body)) + "\n"):
            validate_region(new_body)  # defence in depth; raises if unsafe
            return new_body
    raise ValueError("could not produce a distinct mutation")


# ---------------------------------------------------------------- sandboxed eval
_RUNNER = r"""
import json, sys
src = open(sys.argv[1]).read()
data = json.load(open(sys.argv[2]))
safe = {"abs": abs, "min": min, "max": max, "round": round, "float": float,
        "int": int, "len": len, "range": range, "pow": pow}
ns = {"__builtins__": safe}
try:
    exec(compile(src, "<candidate>", "exec"), ns)
    solve = ns["solve"]
    err = 0.0
    for x, y in data:
        p = solve(x)
        if not isinstance(p, (int, float)) or isinstance(p, bool) or p != p or abs(p) == float("inf"):
            print(json.dumps({"ok": False})); sys.exit(0)
        err += (float(p) - float(y)) ** 2
    print(json.dumps({"ok": True, "score": err / len(data)}))
except Exception as exc:  # a broken candidate simply fails to score
    print(json.dumps({"ok": False, "error": type(exc).__name__}))
"""


def _limits():  # pragma: no cover - POSIX resource caps, exercised in subprocess
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    except Exception:
        pass


def evaluate(body: str, data: list[tuple[float, float]], timeout: float = 5.0) -> tuple[bool, float]:
    """Run a candidate region in an isolated process; return (ok, score)."""
    try:
        validate_region(body)
    except UnsafeCandidate:
        return False, float("inf")
    with tempfile.TemporaryDirectory() as d:
        cand = Path(d) / "cand.py"
        cand.write_text(body)
        data_path = Path(d) / "data.json"
        data_path.write_text(json.dumps(data))
        try:
            out = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER, str(cand), str(data_path)],
                capture_output=True, text=True, timeout=timeout,
                preexec_fn=_limits if sys.platform != "win32" else None,
            )
        except subprocess.TimeoutExpired:
            return False, float("inf")
    if out.returncode != 0 or not out.stdout.strip():
        return False, float("inf")
    try:
        res = json.loads(out.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return False, float("inf")
    return (res.get("ok", False), res.get("score", float("inf")) if res.get("ok") else float("inf"))


# --------------------------------------------------------------------- the task
@dataclass(frozen=True)
class Task:
    """A hidden target the organism must learn to reproduce (symbolic regression).

    The data lives here, in the harness — never inside the evolvable region — so a
    candidate cannot 'cheat' by memorising answers; it can only change its formula.
    """

    a: float = 2.0
    b: float = -3.0
    c: float = 1.0
    lo: float = -2.0
    hi: float = 2.0
    n: int = 41

    def _points(self, offset: float) -> list[tuple[float, float]]:
        step = (self.hi - self.lo) / (self.n - 1)
        xs = [self.lo + offset * step / 2 + i * step for i in range(self.n)]
        return [(x, self.a * x * x + self.b * x + self.c) for x in xs]

    def train(self) -> list[tuple[float, float]]:
        return self._points(0.0)

    def holdout(self) -> list[tuple[float, float]]:
        return self._points(1.0)  # shifted grid, so selection is on unseen points


# --------------------------------------------------------------------- the loop
@dataclass
class SelfEditConfig:
    generations: int = 20
    population: int = 12
    n_edits: int = 2
    scale: float = 2.0
    margin: float = 1e-9
    patience: int = 8
    seed: int = 0


@dataclass
class Journal:
    target: str
    baseline_score: float
    champion_body: str
    champion_score: float
    history: list[dict] = field(default_factory=list)
    candidates: int = 0
    stop_reason: str = "generations"
    applied: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def improve(
    target_path: Path,
    task: Task | None = None,
    cfg: SelfEditConfig | None = None,
    out_dir: Path | None = None,
    log: Callable[[str], None] = print,
) -> Journal:
    """Propose improvements to the EVOLVE-REGION of ``target_path`` (never written back here).

    Returns a :class:`Journal`; the winning source is saved to ``out_dir/champion.py``.
    Call :func:`apply` separately, with human confirmation, to adopt it.
    """
    task = task or Task()
    cfg = cfg or SelfEditConfig()
    target_path = Path(target_path)
    source = target_path.read_text()
    _, region, _ = extract_region(source)
    validate_region(region)

    train, holdout = task.train(), task.holdout()
    rng = random.Random(cfg.seed)

    ok, base = evaluate(region, holdout)
    if not ok:
        raise ValueError("the seed region does not even run — fix the target first")
    best_body, best_score = region, base
    log(f"baseline: {base:.6g} (held-out error)")

    j = Journal(target=str(target_path), baseline_score=round(base, 6),
                champion_body=best_body, champion_score=round(base, 6))
    stale = 0
    for gen in range(1, cfg.generations + 1):
        gen_best_body, gen_best_score = None, best_score
        tried = 0
        for _ in range(cfg.population):
            # Each candidate gets a step size drawn across three orders of
            # magnitude, so every generation mixes coarse exploration with the
            # fine-tuning a fixed step could never reach.
            scale = cfg.scale * (10 ** rng.uniform(-3, 0))
            n_edits = rng.randint(1, max(1, cfg.n_edits))
            try:
                cand = mutate(best_body, rng, n_edits, scale)
            except (ValueError, UnsafeCandidate):
                continue
            tried += 1
            # select on held-out, but a candidate must also run cleanly on train
            ok_tr, _ = evaluate(cand, train)
            ok_ho, score = evaluate(cand, holdout)
            if ok_tr and ok_ho and score < gen_best_score:
                gen_best_body, gen_best_score = cand, score
        j.candidates += tried
        if gen_best_body is not None and gen_best_score < best_score - cfg.margin:
            best_body, best_score = gen_best_body, gen_best_score
            j.history.append({"gen": gen, "score": round(best_score, 6)})
            log(f"gen {gen:3d}: improved -> {best_score:.6g}")
            stale = 0
        else:
            stale += 1
            log(f"gen {gen:3d}: no improvement ({stale}/{cfg.patience})")
            if stale >= cfg.patience:
                j.stop_reason = "patience"
                break

    j.champion_body, j.champion_score = best_body, round(best_score, 6)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "champion.py").write_text(splice_region(source, best_body))
        (out_dir / "journal.json").write_text(json.dumps(j.to_dict(), indent=2))
    log(f"\nproposed champion: {best_score:.6g} (was {base:.6g}); "
        f"improvement {100 * (1 - best_score / base):.1f}%  — NOT applied")
    return j


def diff_region(target_path: Path, new_body: str) -> str:
    import difflib
    _, old, _ = extract_region(Path(target_path).read_text())
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new_body.splitlines(keepends=True),
        fromfile="current", tofile="proposed"))


def apply(
    target_path: Path,
    new_body: str,
    confirm: Callable[[str], bool],
    task: Task | None = None,
) -> bool:
    """Adopt ``new_body`` into the target file — only if ``confirm(diff)`` returns True.

    This is the single place the program is allowed to rewrite itself, and it
    cannot do so without the human-supplied ``confirm`` callback saying yes. It
    re-validates and re-runs the candidate first, so an unsafe or broken body is
    refused even if confirmed.
    """
    target_path = Path(target_path)
    validate_region(new_body)
    ok, _ = evaluate(new_body, (task or Task()).holdout())
    if not ok:
        raise ValueError("refusing to apply: candidate does not run cleanly")
    if not confirm(diff_region(target_path, new_body)):
        return False
    target_path.write_text(splice_region(target_path.read_text(), new_body))
    return True
