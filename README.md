# Ouroboros

**A small GPT built from scratch, and a lineage of them that designs and trains its own successors.**

Each generation's best model (the *champion*) proposes designs for the next generation, builds them, starts them from its own weights, and teaches them by distillation. A successor takes over only if it beats its parent on held-out text. The thing doing the proposing, the *designer*, is inherited and mutated along with the model, so the improvement process is itself improving. That is the recursive part.

It runs on a laptop CPU in minutes, and every decision it makes is written to disk.

## What's in it

| File | What it does |
| --- | --- |
| `ouroboros/model.py` | Decoder-only transformer from scratch: byte-level vocab, causal multi-head attention, pre-LN blocks, tied output head, sampling with temperature / top-k. `inherit_weights` copies a parent's layers into a differently-shaped child. |
| `ouroboros/data.py` | UTF-8 byte tokenizer (no training, nothing unknown), Tiny Shakespeare download, train/val/test split, and `bits_per_byte`, which scores every byte exactly once regardless of a model's context length, so different designs are compared fairly. |
| `ouroboros/train.py` | AdamW, warmup + cosine LR, gradient clipping, and **distillation**: `(1-α)·CE(data) + α·T²·KL(teacher ‖ student)`. |
| `ouroboros/evolve.py` | The self-improvement loop: `Genome` (design + training recipe), `Designer` (the policy that proposes successors), budgets, promotion, lineage log. |
| `ouroboros/selfedit.py` + `organism.py` | **Permission-gated self-improving *code*:** a program that reads its own source, proposes an upgrade to a bounded region of itself, proves it on a held-out task, and rewrites itself only after a human approves the diff. |
| `tests/` | 40 offline tests: causality of attention, the parameter-count formula, a proof that distillation transfers a teacher's behaviour, end-to-end evolution runs, and the self-editor's safety gates (it rejects smuggled imports/calls, sandboxes crashes, and never touches the target without confirmation). |

## How one generation works

```
champion (model + genome + designer)
   │
   ├─ designer.spawn() ×N ─► mutated designers ─► propose child genomes (within budget)
   ├─ control child: the champion's own genome, unchanged
   │
   for each child:
   │   build model ─► inherit_weights(child, champion) ─► train on data,
   │   distilling from the champion ─► score on validation (bits per byte)
   │
   └─ best child beats champion by `margin`?  yes → it becomes champion; its designer
                                                    learns a drift from the step that worked
                                              no  → champion stays; after `patience` misses, stop
```

**What the designer controls** (`SPACE` in `evolve.py`): depth, heads, head width, context length, dropout, learning rate, batch size, weight decay, and how much and how softly to learn from the parent.

**How the designer improves itself.** Each designer carries per-knob mutation probabilities and step sizes, which are randomly perturbed every time it spawns a copy. Designers whose proposals win are the ones that survive. On top of that, a winning designer records the step that worked as a *drift* direction and keeps pushing that way. Good search strategies are therefore selected and reinforced, not hand-tuned.

**Why there is a control child.** Successors inherit weights and get extra training, so they would usually beat the parent even with an identical design. The control child is exactly that: the same design trained the same extra steps. A designed child only wins if its *design* beats "more of the same".

## Growing code, with a hand on the switch

`evolve` improves a model's *design*. `selfedit` improves a program's *source code* — the closest honest version of "a system that analyses itself and generates its own upgrades." You raise it like a kid: it proposes, you approve.

```bash
# propose only — writes the winning source to runs/selfedit/champion.py, never touches the original
python -m ouroboros selfedit --generations 60 --population 20

# let it rewrite itself — shows you the diff and waits for you to type 'y'
python -m ouroboros selfedit --apply
```

The organism starts as a deliberately wrong formula (`0·x² + 0·x + 0`) and has to learn its way to a hidden target by tuning constants and swapping arithmetic operators. What makes it safe rather than a runaway:

- **Bounded edit surface** — only the code between `# EVOLVE-REGION` markers in one function is ever changed.
- **Safe by construction** — edits happen at the syntax-tree level and can only touch numbers and `+ - * /`. The mutation space cannot express an import, a call, a file write, or an attribute access; every candidate is re-validated to be sure, and anything else is rejected.
- **Sandboxed** — each candidate runs in a separate process with CPU, memory and wall-clock limits and restricted builtins, so a broken or slow candidate is killed and discarded.
- **Test-gated, then human-gated** — a candidate is kept only if it runs cleanly and beats the incumbent on held-out data; and the improved code is written back into the file *only* through `apply()`, which refuses until a `confirm(diff)` callback returns true. No network, no self-replication, one machine, one file.

It is not AGI and it is not sentient. It is a program that improves a small, well-defined piece of itself while you hold the approval switch.

## Built-in limits

The loop is deliberately bounded. That is what makes its results meaningful and keeps it safe to leave running.

- **No self-written code.** The search space is declarative data. Models change their *configuration*, never the program.
- **Budgets set by the human:** `--max-params`, `--max-flops` per candidate (6·params·tokens), `--generations`, `--patience`, `--time-budget`. Proposals outside the budget are rejected before training.
- **Held-out selection:** successors are chosen on the validation split. The test split is touched exactly once, at the end, to report the final champion.
- **Everything recorded:** `runs/<name>/lineage.json` holds every candidate (accepted or not), what it changed, and its score. Every champion is checkpointed as `genNN.pt`, so any generation can be inspected or rolled back.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -e .

python -m pytest                                   # ~1 minute, offline

python -m ouroboros train  --steps 2000            # one model from scratch
python -m ouroboros evolve --generations 8 --children 3 --steps 300 --seed-steps 600 \
       --max-params 800000 --max-flops 5e12 --time-budget 2400 --out runs/evo
python -m ouroboros sample --ckpt runs/evo/champion.pt --prompt "ROMEO:" --tokens 400
```

Any UTF-8 text file works as a corpus: `--corpus my_text.txt`.

## Results

A run on Tiny Shakespeare (≈1 MB), 8 generations, 3 designed children + 1 control each, ~4 minutes on 4 CPU cores:

```
python -m ouroboros evolve --generations 8 --children 3 --steps 300 --seed-steps 600 \
       --max-params 800000 --max-flops 5e12 --out runs/evo
```

| Gen | Design | Val (bits/byte) |
| --: | --- | --: |
| 0 | seed, no teacher | 2.991 |
| 1 | — (control won: more training) | 2.847 |
| 2 | +distillation α=0.10 | 2.747 |
| 3 | | 2.680 |
| 4 | | 2.631 |
| 5 | | 2.570 |
| 6 | +distillation α=0.15 | 2.514 |
| 7 | | 2.489 |
| **8** | lr / batch / weight-decay tuning | **2.436** |

Held-out loss fell every generation, **2.991 → 2.436 bpb** (test: 2.571), each step a model the previous champion built and taught. The designer discovered on its own that turning *on* distillation helped (α rose from 0 to 0.15) and kept the architecture small — it never needed to spend its parameter budget to improve. Generation 8's sample:

```
ROMEO:
Is have mean what be a of his proclaise,
And thy clangers for than my honour these of shall bed,
I say man you the please away
That off an which give a gray.

SICINIUS:
The worth now, and their by de
```

Not Shakespeare — it's a 120k-parameter model — but it has learned characters, line breaks, capitalised speaker names and roughly word-shaped tokens, entirely from bytes. The point is the *trajectory*: a lineage that improved itself, under a fixed budget, with every decision on disk in `runs/evo/lineage.json`.

## What this is and isn't

This is a real language model and a real recursive improvement loop, at a scale where you can watch every step. It is not a frontier model. A few honest limits:

- **Scale is the ceiling.** A model of a million parameters trained on 1 MB of Shakespeare learns spelling, word shapes and play formatting, not facts or reasoning. The loop finds better designs *within* a budget; it cannot escape the budget.
- **"No mistakes" isn't on the table** for any language model. What the loop guarantees is narrower and checkable: a champion is never replaced by something that scores worse on held-out text.
- **Merging every LLM ever made isn't possible.** Most strong models have closed weights, and models with different architectures can't be merged. The practical route is the one already built in here: distillation. Swap the champion-as-teacher for an open-weight model (Llama, Qwen, Mistral) and the student learns from that model's predictions.

## Where to take it next

1. **Bigger corpus + GPU:** `--corpus` and `--device cuda` already work; raise the budgets.
2. **External teacher:** distil generation 0 from an open-weight model via Hugging Face `transformers` (needs a shared tokenizer or logit mapping).
3. **Richer designer:** replace the evolution-strategy designer with a learned surrogate (e.g. a small regressor on `lineage.json`) that predicts which designs will win before paying to train them.

## License

MIT
