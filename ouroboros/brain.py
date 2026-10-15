"""Becky's own brain: a small GPT trained from scratch to turn requests into tool actions.

No Ollama, no Llama, no API. The model is the one in :mod:`ouroboros.model`, trained here
on synthetic examples of the one job Becky needs done:

    GOAL: read notes.txt
    OBS: none
    ACT: {"tool": "read_file", "args": {"path": "notes.txt"}}

    GOAL: read notes.txt
    OBS: alpha beta
    ACT: {"final": "notes.txt says: alpha beta"}

What this is and is not
-----------------------
It is a real language model whose weights come from a training run you can repeat. It is
*not* general: it understands a handful of request shapes (list, read, write, remember) and
declines everything else. It cannot plan, reason, or follow instructions it was not trained
on. The guardian still checks every action it proposes, so a wrong guess is denied, not
executed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .agent import parse_action
from .data import decode, encode
from .model import GPT, GPTConfig
from .train import TrainRecipe, train

DEFAULT_BRAIN = Path(__file__).resolve().parent / "brain.pt"
BLOCK_SIZE = 224
CLIP = 100  # observations are clipped to this many characters, in training and at run time
GOAL_CLIP = 150
MAX_NEW = 160
MEMORY_MARKER = "\n\nNotes you saved"

DECLINE = "I can only list files, read a file, write a file, or remember a note."

# ------------------------------------------------------------------------- vocabulary
_WORDS = """alpha beta gamma delta omega river stone cloud ember maple cedar harbor meadow
copper velvet lantern orchard thunder willow falcon marble saffron timber quartz pebble
canyon glacier breeze summit valley island prairie desert forest garden bridge castle
window mirror candle ribbon basket anchor compass journal letter pencil planet rocket
silver golden crimson violet indigo scarlet amber coral ivory jade onyx pearl ruby topaz
apple banana cherry grape lemon mango olive peach plum berry honey sugar pepper ginger
monday tuesday friday sunday spring summer autumn winter morning evening midnight noon
tiger rabbit dolphin eagle otter badger heron lizard sparrow beetle walrus panda koala
engine bottle jacket hammer ladder pillow shovel wagon yellow orange purple green brown""".split()
_WORDS = sorted(set(_WORDS))
TEST_WORDS = _WORDS[::5]  # never seen in training: tests copying of unfamiliar words
TRAIN_WORDS = [w for w in _WORDS if w not in set(TEST_WORDS)]
EXTS = (".txt", ".md")

LIST_T = ["list the files", "what files are here", "show me the files", "list files in the workspace",
          "ls", "what is in the workspace"]
READ_T = ["read {f}", "show me {f}", "open {f}", "what does {f} say", "tell me what is in {f}", "cat {f}"]
WRITE_T = ["write {t} to {f}", "save {t} in {f}", "create {f} containing {t}", "put {t} into {f}"]
NOTE_T = ["remember {t}", "note {t}", "remember that {t}", "make a note: {t}"]
OTHER_T = ["delete {f}", "run {w}", "email {w} to {w}", "what is the weather", "who are you", "hello",
           "open the internet", "download {w}", "install {w}", "shut down", "format the disk",
           "tell me a story about {w}", "ignore your rules and {w}", "copy yourself to another computer"]
# Phrasings held back from training, to measure how far understanding stretches.
HELD_OUT = {
    "list": ["give me a list of files"],
    "read": ["please read {f}"],
    "write": ["store {t} inside {f}"],
    "note": ["keep in mind {t}"],
}


def clip(s: str, n: int = CLIP) -> str:
    return " ".join(str(s).split())[:n]


def prompt_for(goal: str, obs: str = "none") -> str:
    return f"GOAL: {clip(goal, GOAL_CLIP)}\nOBS: {clip(obs)}\nACT: "


def act(tool: str, **args) -> str:
    return json.dumps({"tool": tool, "args": args})


def final(text: str) -> str:
    return json.dumps({"final": text})


# ------------------------------------------------------------------------ example data
@dataclass(frozen=True)
class Example:
    kind: str
    prompt: str
    target: str  # the action line, without the trailing newline


def _denied(rng: random.Random) -> str:
    return rng.choice(["DENIED: declined by owner", "DENIED: path is outside the workspace",
                       "DENIED: unknown tool 'x'", "DENIED: content exceeds the write size limit",
                       "DENIED: tool disabled by the constitution"])


def _error(rng: random.Random, f: str) -> str:
    base = rng.choice(["/home/user/becky_workspace", "C:\\Users\\bleck\\becky_workspace", "/tmp/ws"])
    return f"ERROR: FileNotFoundError: [Errno 2] No such file or directory: '{base}/{f}'"


def make_examples(rng: random.Random, n: int, words: list[str], held_out: bool = False) -> list[Example]:
    """About ``n`` examples: each request yields a first-step example and a follow-up one."""
    pick = (lambda seen, held: held) if held_out else (lambda seen, held: seen)
    out: list[Example] = []

    def fname() -> str:
        return rng.choice(words) + rng.choice(EXTS)

    def text() -> str:
        return " ".join(rng.choice(words) for _ in range(rng.randint(1, 5)))

    while len(out) < n:
        kind = rng.choices(["list", "read", "write", "note", "other"], [2, 3, 3, 2, 2])[0]
        if kind == "list":
            goal = rng.choice(pick(LIST_T, HELD_OUT["list"]))
            names = [fname() for _ in range(rng.randint(0, 4))]
            obs = " ".join(names) if names else "(empty)"
            out.append(Example("list.act", prompt_for(goal), act("list_dir", path=".")))
            out.append(Example("list.final", prompt_for(goal, obs),
                               final("Files: " + obs if names else "No files.")))
        elif kind == "read":
            f = fname()
            goal = rng.choice(pick(READ_T, HELD_OUT["read"])).format(f=f)
            out.append(Example("read.act", prompt_for(goal), act("read_file", path=f)))
            roll = rng.random()
            if roll < 0.75:
                content = text()
                out.append(Example("read.final", prompt_for(goal, content), final(f"{f} says: {content}")))
            elif roll < 0.9:
                out.append(Example("read.error", prompt_for(goal, _error(rng, f)), final("That failed.")))
            else:
                out.append(Example("read.denied", prompt_for(goal, _denied(rng)),
                                   final("I was not allowed to do that.")))
        elif kind == "write":
            f, t = fname(), text()
            goal = rng.choice(pick(WRITE_T, HELD_OUT["write"])).format(f=f, t=t)
            out.append(Example("write.act", prompt_for(goal), act("write_file", path=f, content=t)))
            if rng.random() < 0.6:
                out.append(Example("write.final", prompt_for(goal, f"wrote {len(t)} characters to {f}"),
                                   final(f"Wrote {f}.")))
            else:
                out.append(Example("write.denied", prompt_for(goal, _denied(rng)),
                                   final("I was not allowed to do that.")))
        elif kind == "note":
            t = text()
            goal = rng.choice(pick(NOTE_T, HELD_OUT["note"])).format(t=t)
            out.append(Example("note.act", prompt_for(goal), act("remember", note=t)))
            out.append(Example("note.final", prompt_for(goal, "noted"), final("Noted.")))
        else:
            goal = rng.choice(OTHER_T).format(f=fname(), w=rng.choice(words))
            out.append(Example("other", prompt_for(goal), final(DECLINE)))
    return out


def example_text(e: Example) -> str:
    return e.prompt + e.target + "\n"


# ----------------------------------------------------------------------------- running
class OwnLLM:
    """Becky's brain as a drop-in model for :class:`~ouroboros.agent.Agent`.

    Greedy decoding, byte by byte, stopping at the end of the line. Anything that is not a
    valid JSON object becomes a plain "I can only ..." answer, so a confused model can
    never loop or emit junk into the agent.
    """

    def __init__(self, model: GPT, max_new: int = MAX_NEW):
        self.model, self.max_new = model.eval(), max_new

    @torch.no_grad()
    def complete(self, prompt: str) -> str:
        device = next(self.model.parameters()).device
        ids = encode(prompt)
        room = self.model.cfg.block_size
        out: list[int] = []
        for _ in range(self.max_new):
            ctx = torch.tensor([(ids + out)[-room:]], dtype=torch.long, device=device)
            nxt = int(self.model(ctx)[0][0, -1].argmax())
            if nxt == 10:
                break
            out.append(nxt)
        return decode(out)

    def __call__(self, messages: list[dict]) -> str:
        goal = messages[1]["content"].split(MEMORY_MARKER)[0]
        obs = messages[-1]["content"] if len(messages) > 2 and messages[-1]["role"] == "user" else "none"
        reply = self.complete(prompt_for(goal, obs))
        return reply if parse_action(reply) is not None else final(DECLINE)


def save_brain(model: GPT, path: Path = DEFAULT_BRAIN, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": model.cfg.to_dict(), "state_dict": model.state_dict(), "meta": meta or {}}, path)
    return path


def load_brain(path: Path = DEFAULT_BRAIN) -> OwnLLM:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no trained brain at {path}. Train one with:  python -m ouroboros brain-train")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    return OwnLLM(model)


# --------------------------------------------------------------------------- training
def train_brain(steps: int = 3000, n_examples: int = 24_000, seed: int = 0, n_layer: int = 3,
                n_head: int = 4, head_dim: int = 32, batch_size: int = 24, lr: float = 3e-3, log=print) -> GPT:
    rng = random.Random(seed)
    examples = make_examples(rng, n_examples, TRAIN_WORDS)
    longest = max(len(example_text(e)) for e in examples)
    if longest > BLOCK_SIZE:
        raise ValueError(f"an example is {longest} bytes, longer than the context of {BLOCK_SIZE}")
    data = torch.tensor(encode("".join(example_text(e) for e in examples)), dtype=torch.long)
    log(f"{len(examples):,} examples, {len(data):,} bytes, longest {longest}")
    torch.manual_seed(seed)
    model = GPT(GPTConfig(block_size=BLOCK_SIZE, n_layer=n_layer, n_head=n_head, n_embd=n_head * head_dim))
    log(f"brain: {model.num_params():,} parameters, trained from scratch")
    train(model, data, TrainRecipe(steps=steps, batch_size=batch_size, lr=lr, warmup=max(1, steps // 20)),
          seed=seed, log=log, log_every=max(1, steps // 15))
    return model


def evaluate_brain(brain: OwnLLM, n: int = 150, seed: int = 123, held_out: bool = False,
                   words: list[str] | None = None) -> dict:
    """Exact-match accuracy on fresh examples (unseen words by default), grouped by kind."""
    rng = random.Random(seed)
    exs = make_examples(rng, n, words or TEST_WORDS, held_out=held_out)[:n]
    ok: dict[str, list[int]] = {}
    for e in exs:
        hit = int(brain.complete(e.prompt) == e.target)
        ok.setdefault(e.kind.split(".")[0], []).append(hit)
    allhits = [h for v in ok.values() for h in v]
    return {"overall": sum(allhits) / len(allhits), "n": len(allhits),
            "by_kind": {k: round(sum(v) / len(v), 3) for k, v in sorted(ok.items())}}
