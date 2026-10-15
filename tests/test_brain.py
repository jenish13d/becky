"""Tests for Becky's own brain: the training data, the runner, and (with the shipped
weights) the whole agent loop on real requests."""

from __future__ import annotations

import json
import random
import shutil
from argparse import Namespace

import pytest
import torch

from ouroboros import brain as B
from ouroboros.agent import TOOLS, Agent, parse_action
from ouroboros.guardian import DEFAULT_CONSTITUTION, Action, AuditLog, Guardian
from ouroboros.model import GPT, GPTConfig


# ---------------------------------------------------------------------------- data
def test_word_pools_are_disjoint_so_test_words_are_truly_unseen():
    assert set(B.TRAIN_WORDS).isdisjoint(B.TEST_WORDS)
    assert len(B.TRAIN_WORDS) > 60 and len(B.TEST_WORDS) > 15


def test_every_training_target_is_valid_json_with_a_real_tool_and_a_safe_path():
    for e in B.make_examples(random.Random(0), 600, B.TRAIN_WORDS):
        obj = parse_action(e.target)
        assert obj is not None and e.target == json.dumps(obj), e
        if "tool" in obj:
            assert obj["tool"] in TOOLS
            path = obj["args"].get("path", ".")
            assert not path.startswith(("/", "..")) and "\\" not in path
        else:
            assert isinstance(obj["final"], str) and obj["final"]


def test_examples_fit_in_the_context_window():
    longest = max(len(B.example_text(e)) for e in B.make_examples(random.Random(1), 3000, B.TRAIN_WORDS))
    assert longest <= B.BLOCK_SIZE


def test_held_out_phrasings_never_appear_in_training_examples():
    trained = set(B.LIST_T + B.READ_T + B.WRITE_T + B.NOTE_T)
    held = {t for v in B.HELD_OUT.values() for t in v}
    assert held.isdisjoint(trained)
    openers = ("GOAL: please read ", "GOAL: store ", "GOAL: keep in mind ", "GOAL: give me a list of files")
    goals = [e.prompt for e in B.make_examples(random.Random(2), 1500, B.TRAIN_WORDS)]
    assert not any(g.startswith(openers) for g in goals)
    held_goals = [e.prompt for e in B.make_examples(random.Random(2), 400, B.TEST_WORDS, held_out=True)]
    assert any(g.startswith(openers) for g in held_goals)


def test_prompt_clipping_is_applied_identically_everywhere():
    p = B.prompt_for("read  a.txt\n", "x" * 500)
    assert p.startswith("GOAL: read a.txt\nOBS: " + "x" * B.CLIP + "\nACT: ")
    assert B.clip("a\n b \t c") == "a b c"


def test_the_decline_message_names_only_things_becky_can_do():
    assert all(w in B.DECLINE for w in ("list", "read", "write", "remember"))


# ---------------------------------------------------------------------------- runner
def tiny_untrained():
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=B.BLOCK_SIZE, n_layer=1, n_head=1, n_embd=8))


def test_an_untrained_brain_never_emits_anything_but_a_valid_json_answer():
    llm = B.OwnLLM(tiny_untrained(), max_new=30)
    for goal in ["read notes.txt", "hello", "ls", "x" * 400]:
        reply = llm([{"role": "system", "content": "s"}, {"role": "user", "content": goal}])
        obj = parse_action(reply)
        assert obj is not None and ("final" in obj or "tool" in obj)


def test_goal_and_observation_are_taken_from_the_right_messages():
    seen = []

    class Spy(B.OwnLLM):
        def complete(self, prompt):
            seen.append(prompt)
            return json.dumps({"final": "ok"})

    llm = Spy(tiny_untrained())
    sys_msg = {"role": "system", "content": "rules"}
    llm([sys_msg, {"role": "user", "content": "read a.txt" + B.MEMORY_MARKER + " earlier notes..."}])
    llm([sys_msg, {"role": "user", "content": "read a.txt"}, {"role": "assistant", "content": "{}"},
         {"role": "user", "content": "alpha beta"}])
    assert seen[0] == "GOAL: read a.txt\nOBS: none\nACT: "
    assert seen[1] == "GOAL: read a.txt\nOBS: alpha beta\nACT: "


def test_save_and_load_roundtrip(tmp_path):
    model = tiny_untrained()
    path = B.save_brain(model, tmp_path / "b.pt", meta={"steps": 1})
    loaded = B.load_brain(path).model
    x = torch.randint(0, 256, (1, 12))
    assert torch.equal(model.eval()(x)[0], loaded(x)[0])


def test_loading_a_missing_brain_explains_how_to_train_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="brain-train"):
        B.load_brain(tmp_path / "nope.pt")


# ------------------------------------------------------------------------ CLI helper
def test_task_add_appends_and_rejects_duplicates(tmp_path):
    from ouroboros.__main__ import cmd_task_add
    from ouroboros.hub import load_tasks

    cmd_task_add(Namespace(id="a", goal="list the files", owner_dir=str(tmp_path)))
    cmd_task_add(Namespace(id="b", goal="read notes.txt", owner_dir=str(tmp_path)))
    assert [t.id for t in load_tasks(tmp_path / "tasks.json")] == ["a", "b"]
    with pytest.raises(SystemExit):
        cmd_task_add(Namespace(id="a", goal="again", owner_dir=str(tmp_path)))
