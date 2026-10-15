"""Command line.

    python -m ouroboros train  --steps 2000 --out runs/base
    python -m ouroboros evolve --generations 6 --children 4 --steps 300 --out runs/evo
    python -m ouroboros sample --ckpt runs/evo/champion.pt --prompt "ROMEO:" --tokens 400
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .data import bits_per_byte, load_corpus, split
from .evolve import EvolveConfig, Genome, Member, Designer, evolve, load_model, sample, save_checkpoint
from .model import GPT
from .train import train


def _device(arg: str | None) -> str:
    return arg or ("cuda" if torch.cuda.is_available() else "cpu")


def cmd_train(a: argparse.Namespace) -> None:
    splits = split(load_corpus(a.corpus))
    g = Genome(n_layer=a.layers, n_head=a.heads, head_dim=a.head_dim, block_size=a.block_size,
               lr=a.lr, batch_size=a.batch_size)
    torch.manual_seed(a.seed)
    model = GPT(g.model_config()).to(_device(a.device))
    print(f"training {model.num_params():,} params for {a.steps} steps")
    train(model, splits.train, g.recipe(a.steps), seed=a.seed, log=print)
    bpb = bits_per_byte(model, splits.val, 50_000)
    print(f"val {bpb:.4f} bpb\n\n{sample(model, a.prompt, 300)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / "model.pt", Member(0, g, Designer(), model, bpb))
    print(f"\nsaved {out / 'model.pt'}")


def cmd_evolve(a: argparse.Namespace) -> None:
    splits = split(load_corpus(a.corpus))
    cfg = EvolveConfig(
        generations=a.generations, children=a.children, steps=a.steps, seed_steps=a.seed_steps,
        max_params=a.max_params, max_flops=a.max_flops, margin=a.margin, patience=a.patience,
        time_budget_s=a.time_budget, control=not a.no_control, seed=a.seed, escalations=a.escalations,
        device=_device(a.device), sample_prompt=a.prompt,
    )
    lineage = evolve(splits, cfg, Path(a.out))
    print("\ngen  val_bpb  params     design")
    for c in lineage["champions"]:
        g = Genome(**c["genome"])
        print(f"{c['gen']:>3}  {c['val_bpb']:.4f}  {c['params']:>9,}  "
              f"L{g.n_layer} H{g.n_head}x{g.head_dim} ctx{g.block_size} lr{g.lr:.1e} "
              f"distill{g.distill_alpha:.2f}")
    print(f"\nlast champion sample:\n{lineage['champions'][-1]['sample']}")


def cmd_sample(a: argparse.Namespace) -> None:
    model = load_model(Path(a.ckpt), _device(a.device))
    print(sample(model, a.prompt, a.tokens, seed=a.seed, temperature=a.temperature))


def cmd_selfedit(a: argparse.Namespace) -> None:
    from .selfedit import SelfEditConfig, apply, improve

    cfg = SelfEditConfig(generations=a.generations, population=a.population,
                         n_edits=a.n_edits, patience=a.patience, seed=a.seed)
    j = improve(Path(a.target), cfg=cfg, out_dir=Path(a.out))
    if not a.apply:
        print(f"\nproposed source written to {Path(a.out) / 'champion.py'}")
        print("re-run with --apply to let it rewrite the target (you will be asked to confirm).")
        return

    def confirm(diff: str) -> bool:
        print("\nProposed change to the organism's own code:\n")
        print(diff or "(identical — nothing to apply)")
        return input("\nApply this change to the target file? [y/N] ").strip().lower() in ("y", "yes")

    applied = apply(Path(a.target), j.champion_body, confirm=confirm)
    print("applied — the organism rewrote itself." if applied else "left unchanged.")


def _require_anchor(owner_dir: Path) -> None:
    """Refuse to run if the owner's integrity anchor says something was changed."""
    from .anchor import verify_anchor

    report = verify_anchor(owner_dir)
    if not report.present:
        print("note: no integrity anchor yet. `python -m ouroboros anchor` protects her brain, rules and code.")
    elif not report.ok:
        raise SystemExit(f"Refusing to start: {report.describe()}\n"
                         "If you made these changes, review them and run:  python -m ouroboros anchor")


def _brain(a: argparse.Namespace):
    """Becky's model: her own trained brain by default; Ollama only if you ask for it."""
    if a.brain == "ollama":
        from .agent import OllamaLLM

        return OllamaLLM(a.model)
    from .brain import DEFAULT_BRAIN, load_brain

    try:
        return load_brain(Path(a.brain_path) if a.brain_path else DEFAULT_BRAIN)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))


def cmd_brain_train(a: argparse.Namespace) -> None:
    from .brain import DEFAULT_BRAIN, OwnLLM, load_brain, save_brain, train_brain

    init = load_brain(DEFAULT_BRAIN).model.train() if a.resume else None
    model = train_brain(steps=a.steps, n_examples=a.examples, seed=a.seed, lr=a.lr, log=print, init=init)
    out = save_brain(model, Path(a.out) if a.out else DEFAULT_BRAIN, meta={"steps": a.steps, "seed": a.seed})
    print(f"\nsaved {out}")
    _print_brain_scores(OwnLLM(model))


def _print_brain_scores(brain) -> None:
    from .brain import evaluate_brain

    seen = evaluate_brain(brain, n=150)
    held = evaluate_brain(brain, n=100, held_out=True)
    print(f"exact-match on new words, trained phrasings : {seen['overall']:.1%}  {seen['by_kind']}")
    print(f"exact-match on new words, unseen phrasings  : {held['overall']:.1%}  {held['by_kind']}")
    from .brain import real_world_eval

    rw = real_world_eval(brain)
    print(f"everyday requests, end to end              : {rw['passed']}/{rw['total']}")
    for c in rw["cases"]:
        if not c["ok"]:
            print(f"   missed: {c['goal']!r} -> {c['answer']!r}")


def cmd_brain_improve(a: argparse.Namespace) -> None:
    from .brain import DEFAULT_BRAIN, load_brain
    from .grow import GrowConfig, adopt, grow

    cfg = GrowConfig(rounds=a.rounds, steps=a.steps, time_budget_s=a.time_budget, eval_n=a.eval_n, seed=a.seed,
                     max_params=a.max_params)
    out = Path(a.out)
    j = grow(load_brain(DEFAULT_BRAIN).model.train(), cfg, out)
    print(f"\nimproved brain saved to {out / 'champion.pt'} (journal: {out / 'journal.json'})")
    if not a.apply:
        print("Becky's own brain is unchanged. Re-run with --apply to be asked about adopting it.")
        return

    def confirm(summary: str) -> bool:
        print("\n" + summary)
        return input("Make this Becky's brain? The old one is kept as brain.previous.pt  [y/N] ").strip().lower() in ("y", "yes")

    try:
        done = adopt(out / "champion.pt", DEFAULT_BRAIN, confirm)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print("adopted." if done else "left unchanged.")


def cmd_memory_search(a: argparse.Namespace) -> None:
    from .memory import Memory

    hits = Memory(Path(a.workspace) / "memory.md").search(a.query, k=a.k)
    if not hits:
        print("nothing relevant in memory")
    for score, note in hits:
        print(f"{score:.2f}  {note}")


def cmd_memory_dedupe(a: argparse.Namespace) -> None:
    from .memory import Memory

    removed = Memory(Path(a.workspace) / "memory.md").dedupe()
    print(f"removed {len(removed)} repeated note(s)" + ("".join(f"\n  - {n}" for n in removed) if removed else ""))


def cmd_chat(a: argparse.Namespace) -> None:
    import getpass

    from .agent import Agent
    from .anchor import create_anchor, verify_anchor
    from .brain import DEFAULT_BRAIN, load_brain
    from .chat import Chat
    from .compress import compress_report
    from .grow import GrowConfig, adopt, grow
    from .guardian import DEFAULT_CONSTITUTION, AuditLog, Guardian
    from .memory import Memory
    from .owner import OwnerError, is_initialised
    from .review import render, review

    owner_dir, workspace = Path(a.owner_dir), Path(a.workspace)
    if not is_initialised(owner_dir):
        print("First time: choose the passphrase that makes Becky answer only to you.\n")
        cmd_init_owner(argparse.Namespace(owner_dir=str(owner_dir)))
        print()
    _require_anchor(owner_dir)
    if not workspace.exists():  # first run: give her something to read
        workspace.mkdir(parents=True)
        (workspace / "notes.txt").write_text("welcome to becky\n", encoding="utf-8")
        print(f"Created her workspace with a sample notes.txt: {workspace.resolve()}\n")
    audit = AuditLog(Path(a.audit))
    state = {"brain": load_brain(DEFAULT_BRAIN)}

    def confirm(action, decision) -> bool:
        print(f"\nBecky wants to run: {action.tool} {action.args}")
        return input("Allow this? [y/N] ").strip().lower() in ("y", "yes")

    def run_request(goal: str) -> str:
        try:
            g = Guardian(DEFAULT_CONSTITUTION, workspace, audit, owner_dir=owner_dir)
        except OwnerError as exc:
            return f"Refusing to run: {exc}"
        return Agent(state["brain"], g, confirm).run(goal)

    def improve() -> str:
        out = Path(a.runs) / "brain-improve"
        print("Improving her brain. This takes about 30 minutes; press Ctrl+C to stop early.\n")
        cfg = GrowConfig(rounds=4, steps=300, eval_n=80, time_budget_s=1800)
        grow(state["brain"].model.train(), cfg, out)
        if not (out / "champion.pt").exists():
            return "No improved brain was produced."

        def keep(summary: str) -> bool:
            print("\n" + summary)
            return input("Make this her brain? Your current one is kept as brain.previous.pt [y/N] ").strip().lower() in ("y", "yes")

        try:
            adopted = adopt(out / "champion.pt", DEFAULT_BRAIN, keep)
        except ValueError as exc:
            return f"Left her brain unchanged: {exc}"
        if not adopted:
            return "Left her brain unchanged."
        state["brain"] = load_brain(DEFAULT_BRAIN)
        report = verify_anchor(owner_dir)
        if report.present and not report.ok:
            print("\nHer brain changed, so her integrity anchor must be updated or she will refuse to start next time.")
            try:
                create_anchor(owner_dir, getpass.getpass("Owner passphrase to re-anchor: "))
                return "Her brain is updated and re-anchored."
            except OwnerError as exc:
                return f"Her brain is updated, but re-anchoring failed ({exc}). Run:  python -m ouroboros anchor"
        return "Her brain is updated."

    def do_review() -> str:
        return render(review(state["brain"].model, eval_n=30, speed_prompts=8))

    def do_compress() -> str:
        out = Path(a.runs) / "brain-int8.pt"
        r = compress_report(state["brain"].model, out, source=DEFAULT_BRAIN, eval_n=60)
        verdict = "Safe to use" if r["safe"] else "NOT safe: accuracy dropped too much"
        return (f"{r['size_before_bytes']:,} -> {r['size_after_bytes']:,} bytes (x{r['ratio']:.1f} smaller); "
                f"accuracy {r['accuracy_before']:.0%} -> {r['accuracy_after']:.0%}. {verdict}: {out}")

    def search(query: str) -> str:
        hits = Memory(workspace / "memory.md").search(query)
        return "\n".join(f"{s:.2f}  {n}" for s, n in hits) or "nothing relevant in memory"

    print(f"workspace: {workspace.resolve()}   audit log: {audit.path}\nkill switch: create {Path.home() / '.becky-stop'}\n")
    Chat(run_request, print, input, improve, do_review, do_compress, search).loop()


def cmd_anchor(a: argparse.Namespace) -> None:
    import getpass

    from .anchor import create_anchor
    from .owner import OwnerError, is_initialised

    if not is_initialised(Path(a.owner_dir)):
        raise SystemExit("No owner passphrase is set up yet. Run:  python -m ouroboros init-owner")
    try:
        manifest = create_anchor(Path(a.owner_dir), getpass.getpass("Owner passphrase: "))
    except OwnerError as exc:
        raise SystemExit(f"Refused: {exc}")
    print(f"anchored {len(manifest['files'])} files (code, rules, brain) in {a.owner_dir}")


def cmd_verify_anchor(a: argparse.Namespace) -> None:
    from .anchor import verify_anchor

    report = verify_anchor(Path(a.owner_dir))
    print(report.describe())
    raise SystemExit(0 if (report.ok or not report.present) else 1)


def cmd_compress(a: argparse.Namespace) -> None:
    from .brain import DEFAULT_BRAIN, load_brain
    from .compress import compress_report

    src = Path(a.path) if a.path else DEFAULT_BRAIN
    r = compress_report(load_brain(src).model, Path(a.out), source=src, eval_n=a.eval_n)
    print(f"size: {r['size_before_bytes']:,} -> {r['size_after_bytes']:,} bytes  (x{r['ratio']:.1f} smaller)")
    print(f"accuracy on {a.eval_n} fresh examples: {r['accuracy_before']:.1%} -> {r['accuracy_after']:.1%}")
    print(f"largest weight rounding error: {r['max_weight_error']:.4f}")
    print(("Safe to use: " if r["safe"] else "NOT safe, accuracy dropped too much: ") + f"{a.out}")
    print("Use it with:  python -m ouroboros becky \"...\" --brain-path " + str(a.out))


def cmd_review(a: argparse.Namespace) -> None:
    from .brain import DEFAULT_BRAIN, load_brain
    from .review import render, review

    print(render(review(load_brain(Path(a.path) if a.path else DEFAULT_BRAIN).model, eval_n=a.eval_n)))


def cmd_brain_eval(a: argparse.Namespace) -> None:
    from .brain import DEFAULT_BRAIN, load_brain

    _print_brain_scores(load_brain(Path(a.path) if a.path else DEFAULT_BRAIN))


def cmd_task_add(a: argparse.Namespace) -> None:
    import json

    f = Path(a.owner_dir) / "tasks.json"
    tasks = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    if any(t.get("id") == a.id for t in tasks):
        raise SystemExit(f"a task with id {a.id!r} already exists in {f}")
    tasks.append({"id": a.id, "goal": a.goal})
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"added task {a.id!r}; {len(tasks)} task(s) in {f}")


def cmd_becky(a: argparse.Namespace) -> None:
    from .agent import Agent
    from .guardian import DEFAULT_CONSTITUTION, AuditLog, ClaudeReviewer, Guardian
    from .owner import DEFAULT_OWNER_DIR, OwnerError, is_initialised

    owner_dir = Path(a.owner_dir)
    if not is_initialised(owner_dir) and not a.unsigned:
        raise SystemExit("No owner passphrase is set up yet. Run:  python -m ouroboros init-owner\n"
                         "(or pass --unsigned to run with unsigned rules)")
    if not a.unsigned:
        _require_anchor(owner_dir)
    audit = AuditLog(Path(a.audit))
    reviewer = ClaudeReviewer() if a.claude_review else None
    try:
        if a.unsigned:
            g = Guardian(DEFAULT_CONSTITUTION, Path(a.workspace), audit, reviewer=reviewer)
        else:
            g = Guardian(DEFAULT_CONSTITUTION, Path(a.workspace), audit, reviewer=reviewer, owner_dir=owner_dir)
    except OwnerError as exc:
        raise SystemExit(f"Refusing to start: {exc}")

    def confirm(action, decision) -> bool:
        print(f"\nBecky wants to run: {action.tool} {action.args}")
        return input("Allow this? [y/N] ").strip().lower() in ("y", "yes")

    print(f"workspace: {g.workspace}\naudit log: {audit.path}\nkill switch: create {g.stop_file}")
    print("\n" + Agent(_brain(a), g, confirm).run(a.goal))


def cmd_init_owner(a: argparse.Namespace) -> None:
    import getpass

    from .owner import OwnerError, init_owner

    print("Choose a passphrase. It is typed only here, never stored, and I (the assistant) never see it.")
    print("Use 4+ random words or 16+ characters. Avoid keyboard patterns.")
    pw = getpass.getpass("Passphrase: ")
    if pw != getpass.getpass("Again: "):
        raise SystemExit("Passphrases did not match.")
    try:
        d = init_owner(Path(a.owner_dir), pw)
    except OwnerError as exc:
        raise SystemExit(f"Not set: {exc}")
    print(f"\nOwner key created in {d}\nLose the passphrase and no one can change Becky's rules again; "
          "write it down somewhere only you can reach.")


def cmd_amend(a: argparse.Namespace) -> None:
    import getpass

    from .guardian import AuditLog
    from .owner import OwnerError, amend, reset_constitution

    approvals, limits = {}, {}
    for item in a.set or []:
        key, _, val = item.partition("=")
        if key in ("max_steps", "max_write_bytes"):
            limits[key] = int(val)
        else:
            approvals[key] = val
    if not (approvals or limits or a.reset):
        raise SystemExit("Nothing to do. Example: --set write_file=allow --set max_steps=40")
    pw = getpass.getpass("Owner passphrase: ")
    try:
        if a.reset:
            reset_constitution(Path(a.owner_dir), pw)
            print("Rules restored to the shipped defaults and re-signed.")
        if approvals or limits:
            for line in amend(Path(a.owner_dir), pw, approvals, limits, audit=AuditLog(Path(a.audit))):
                print("changed", line)
    except OwnerError as exc:
        raise SystemExit(f"Refused: {exc}")


def cmd_hub(a: argparse.Namespace) -> None:
    import sys

    from .guardian import DEFAULT_CONSTITUTION, AuditLog, Guardian
    from .hub import HubBusy, load_tasks, run_hub, schedule_help
    from .owner import is_initialised

    if a.print_schedule:
        print(schedule_help(sys.executable))
        return
    owner_dir = Path(a.owner_dir)
    if not is_initialised(owner_dir):
        raise SystemExit("No owner passphrase is set up yet. Run:  python -m ouroboros init-owner")
    _require_anchor(owner_dir)
    tasks_file = Path(a.tasks) if a.tasks else owner_dir / "tasks.json"
    try:
        tasks = load_tasks(tasks_file)
    except ValueError as exc:
        raise SystemExit(f"{exc}\nExpected a file like: " + '[{"id": "daily", "goal": "list the files in the workspace"}]')
    audit = AuditLog(Path(a.audit))
    ws = Path(a.workspace)

    def make_guardian():
        return Guardian(DEFAULT_CONSTITUTION, ws, audit, owner_dir=owner_dir)

    try:
        report = run_hub(tasks, make_guardian, _brain(a), Path(a.audit), owner_dir / "reports",
                         owner_dir / "hub.lock", deadline_s=a.deadline_minutes * 60)
    except HubBusy as exc:
        raise SystemExit(str(exc))
    for r in report["tasks"]:
        print(f"[{r['status']}] {r['id']}: {r['detail'][:200]}")
    print(f"\nreport: {report['path']}  (audit intact: {report['audit_intact']})")


def cmd_snn(a: argparse.Namespace) -> None:
    from .snn import raster, train_snn

    model, res, (x, y) = train_snn(seed=a.seed, epochs=a.epochs, hidden=a.hidden, log=print)
    print(f"\ntrain accuracy {res.train_accuracy:.1%}   test accuracy {res.test_accuracy:.1%}   (chance 25%)")
    print(f"firing rate: hidden {res.hidden_rate:.1%} of neuron-steps, output {res.out_rate:.1%}")
    import torch

    with torch.no_grad():
        logits, spikes = model(x[:, :1])
    print(f"\nExample: true class {int(y[0])}, predicted {int(logits.argmax())}")
    print("Hidden-layer spike raster (rows = neurons, columns = time):\n" + raster(spikes["hidden"][:, 0]))


def cmd_audit(a: argparse.Namespace) -> None:
    from .guardian import AuditLog

    ok, n = AuditLog(Path(a.log)).verify()
    print(f"audit log intact: {n} entries" if ok else f"TAMPERING DETECTED at entry {n}")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser(prog="ouroboros")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--corpus", type=Path, default=None, help="text file (default: Tiny Shakespeare)")
        sp.add_argument("--device", default=None)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--prompt", default="ROMEO:")

    t = sub.add_parser("train", help="train one model from scratch")
    common(t)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--head-dim", type=int, default=32)
    t.add_argument("--block-size", type=int, default=128)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--out", default="runs/base")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evolve", help="recursive self-improvement run")
    common(e)
    d = EvolveConfig()
    e.add_argument("--generations", type=int, default=d.generations)
    e.add_argument("--children", type=int, default=d.children)
    e.add_argument("--steps", type=int, default=d.steps)
    e.add_argument("--seed-steps", type=int, default=None)
    e.add_argument("--max-params", type=int, default=d.max_params)
    e.add_argument("--max-flops", type=float, default=d.max_flops)
    e.add_argument("--margin", type=float, default=d.margin)
    e.add_argument("--patience", type=int, default=d.patience)
    e.add_argument("--escalations", type=int, default=d.escalations,
                   help="when stuck, widen the search this many times before stopping")
    e.add_argument("--time-budget", type=float, default=None, help="seconds")
    e.add_argument("--no-control", action="store_true")
    e.add_argument("--out", default="runs/evo")
    e.set_defaults(fn=cmd_evolve)

    s = sub.add_parser("sample", help="generate text from a checkpoint")
    common(s)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--tokens", type=int, default=400)
    s.add_argument("--temperature", type=float, default=0.8)
    s.set_defaults(fn=cmd_sample)

    se = sub.add_parser("selfedit", help="permission-gated self-improving code")
    se.add_argument("--target", type=Path,
                    default=Path(__file__).resolve().parent / "organism.py")
    se.add_argument("--generations", type=int, default=20)
    se.add_argument("--population", type=int, default=12)
    se.add_argument("--n-edits", type=int, default=1)
    se.add_argument("--patience", type=int, default=8)
    se.add_argument("--seed", type=int, default=0)
    se.add_argument("--out", default="runs/selfedit")
    se.add_argument("--apply", action="store_true",
                    help="after proposing, ask to rewrite the target file in place")
    se.set_defaults(fn=cmd_selfedit)

    home = Path.home() / ".becky"
    b = sub.add_parser("becky", help="run the guardian-supervised local agent")
    b.add_argument("goal")
    b.add_argument("--workspace", default="becky_workspace")
    b.add_argument("--audit", default=str(home / "audit.jsonl"))
    b.add_argument("--brain", choices=["own", "ollama"], default="own",
                   help="own = the model trained here (default); ollama = an outside model, only if you ask")
    b.add_argument("--brain-path", default=None)
    b.add_argument("--model", default="llama3.2", help="only used with --brain ollama")
    b.add_argument("--owner-dir", default=str(home))
    b.add_argument("--unsigned", action="store_true", help="run without owner-signed rules")
    b.add_argument("--claude-review", action="store_true",
                   help="ask Claude for a second opinion on risky actions (needs ANTHROPIC_API_KEY)")
    b.set_defaults(fn=cmd_becky)

    io = sub.add_parser("init-owner", help="set the owner passphrase (typed at a hidden prompt)")
    io.add_argument("--owner-dir", default=str(home))
    io.set_defaults(fn=cmd_init_owner)

    am = sub.add_parser("amend", help="change Becky's tunable rules (needs the owner passphrase)")
    am.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="e.g. write_file=allow, read_file=ask, max_steps=40")
    am.add_argument("--reset", action="store_true", help="restore the shipped default rules")
    am.add_argument("--owner-dir", default=str(home))
    am.add_argument("--audit", default=str(home / "audit.jsonl"))
    am.set_defaults(fn=cmd_amend)

    hb = sub.add_parser("hub", help="run the task list once, bounded (for a scheduler)")
    hb.add_argument("--tasks", default=None, help="default: <owner dir>/tasks.json")
    hb.add_argument("--workspace", default="becky_workspace")
    hb.add_argument("--audit", default=str(home / "audit.jsonl"))
    hb.add_argument("--owner-dir", default=str(home))
    hb.add_argument("--brain", choices=["own", "ollama"], default="own")
    hb.add_argument("--brain-path", default=None)
    hb.add_argument("--model", default="llama3.2", help="only used with --brain ollama")
    hb.add_argument("--deadline-minutes", type=float, default=30.0)
    hb.add_argument("--print-schedule", action="store_true", help="print how to schedule it, then exit")
    hb.set_defaults(fn=cmd_hub)

    sn = sub.add_parser("snn-demo", help="train and inspect a spiking neural network")
    sn.add_argument("--seed", type=int, default=0)
    sn.add_argument("--epochs", type=int, default=25)
    sn.add_argument("--hidden", type=int, default=64)
    sn.set_defaults(fn=cmd_snn)

    bt = sub.add_parser("brain-train", help="train Becky's own brain from scratch (replaces ouroboros/brain.pt)")
    bt.add_argument("--steps", type=int, default=3000)
    bt.add_argument("--examples", type=int, default=24000)
    bt.add_argument("--seed", type=int, default=0)
    bt.add_argument("--lr", type=float, default=3e-3)
    bt.add_argument("--resume", action="store_true", help="keep training the saved brain instead of starting over")
    bt.add_argument("--out", default=None)
    bt.set_defaults(fn=cmd_brain_train)

    bi = sub.add_parser("brain-improve", help="keep improving Becky's brain; change tactics when stuck")
    bi.add_argument("--rounds", type=int, default=8)
    bi.add_argument("--steps", type=int, default=600, help="training steps per round")
    bi.add_argument("--time-budget", type=float, default=None, help="seconds")
    bi.add_argument("--eval-n", type=int, default=120)
    bi.add_argument("--max-params", type=int, default=3_000_000)
    bi.add_argument("--seed", type=int, default=0)
    bi.add_argument("--out", default="runs/brain-improve")
    bi.add_argument("--apply", action="store_true", help="afterwards, ask whether to adopt the improved brain")
    bi.set_defaults(fn=cmd_brain_improve)

    ms = sub.add_parser("memory-search", help="find the notes most relevant to a query")
    ms.add_argument("query")
    ms.add_argument("-k", type=int, default=5)
    ms.add_argument("--workspace", default="becky_workspace")
    ms.set_defaults(fn=cmd_memory_search)

    md = sub.add_parser("memory-dedupe", help="remove repeated notes from memory.md")
    md.add_argument("--workspace", default="becky_workspace")
    md.set_defaults(fn=cmd_memory_dedupe)

    ch = sub.add_parser("chat", help="talk to Becky: type requests, or `improve`")
    ch.add_argument("--workspace", default="becky_workspace")
    ch.add_argument("--audit", default=str(home / "audit.jsonl"))
    ch.add_argument("--owner-dir", default=str(home))
    ch.add_argument("--runs", default="runs")
    ch.set_defaults(fn=cmd_chat)

    an = sub.add_parser("anchor", help="fingerprint and sign her brain, rules and code (needs your passphrase)")
    an.add_argument("--owner-dir", default=str(home))
    an.set_defaults(fn=cmd_anchor)

    va = sub.add_parser("verify-anchor", help="check that nothing changed since you anchored it")
    va.add_argument("--owner-dir", default=str(home))
    va.set_defaults(fn=cmd_verify_anchor)

    co = sub.add_parser("compress", help="shrink her brain to int8 and measure the cost")
    co.add_argument("--path", default=None)
    co.add_argument("--out", default="brain-int8.pt")
    co.add_argument("--eval-n", type=int, default=100)
    co.set_defaults(fn=cmd_compress)

    rv = sub.add_parser("review", help="Becky reviews her own layers, speed and size (changes nothing)")
    rv.add_argument("--path", default=None)
    rv.add_argument("--eval-n", type=int, default=40)
    rv.set_defaults(fn=cmd_review)

    be = sub.add_parser("brain-eval", help="score Becky's brain on fresh examples")
    be.add_argument("--path", default=None)
    be.set_defaults(fn=cmd_brain_eval)

    ta = sub.add_parser("task-add", help="add a task to the hub's task list")
    ta.add_argument("id")
    ta.add_argument("goal")
    ta.add_argument("--owner-dir", default=str(home))
    ta.set_defaults(fn=cmd_task_add)

    au = sub.add_parser("audit", help="verify the audit log's hash chain")
    au.add_argument("--log", default=str(home / "audit.jsonl"))
    au.set_defaults(fn=cmd_audit)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
