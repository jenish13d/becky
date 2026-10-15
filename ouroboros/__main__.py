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
        time_budget_s=a.time_budget, control=not a.no_control, seed=a.seed,
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


def cmd_becky(a: argparse.Namespace) -> None:
    from .agent import Agent, OllamaLLM
    from .guardian import DEFAULT_CONSTITUTION, AuditLog, ClaudeReviewer, Guardian
    from .owner import DEFAULT_OWNER_DIR, OwnerError, is_initialised

    owner_dir = Path(a.owner_dir)
    if not is_initialised(owner_dir) and not a.unsigned:
        raise SystemExit("No owner passphrase is set up yet. Run:  python -m ouroboros init-owner\n"
                         "(or pass --unsigned to run with unsigned rules)")
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
    print("\n" + Agent(OllamaLLM(a.model), g, confirm).run(a.goal))


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

    from .agent import OllamaLLM
    from .guardian import DEFAULT_CONSTITUTION, AuditLog, Guardian
    from .hub import HubBusy, load_tasks, run_hub, schedule_help
    from .owner import is_initialised

    if a.print_schedule:
        print(schedule_help(sys.executable))
        return
    owner_dir = Path(a.owner_dir)
    if not is_initialised(owner_dir):
        raise SystemExit("No owner passphrase is set up yet. Run:  python -m ouroboros init-owner")
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
        report = run_hub(tasks, make_guardian, OllamaLLM(a.model), Path(a.audit), owner_dir / "reports",
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
    b.add_argument("--model", default="llama3.2", help="a model served by local Ollama")
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
    hb.add_argument("--model", default="llama3.2")
    hb.add_argument("--deadline-minutes", type=float, default=30.0)
    hb.add_argument("--print-schedule", action="store_true", help="print how to schedule it, then exit")
    hb.set_defaults(fn=cmd_hub)

    sn = sub.add_parser("snn-demo", help="train and inspect a spiking neural network")
    sn.add_argument("--seed", type=int, default=0)
    sn.add_argument("--epochs", type=int, default=25)
    sn.add_argument("--hidden", type=int, default=64)
    sn.set_defaults(fn=cmd_snn)

    au = sub.add_parser("audit", help="verify the audit log's hash chain")
    au.add_argument("--log", default=str(home / "audit.jsonl"))
    au.set_defaults(fn=cmd_audit)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
