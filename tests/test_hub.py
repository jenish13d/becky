"""Tests for the hub: bounded, unattended, owner-authored runs."""

from __future__ import annotations

import json
import os
import shutil
import time

import pytest

from ouroboros.agent import Agent
from ouroboros.guardian import DEFAULT_CONSTITUTION, AuditLog, Guardian, Halted
from ouroboros.hub import (
    MAX_GOAL_CHARS, MAX_TASKS, DeadlineLLM, HubBusy, RunLock, Task, load_tasks, memory_context, run_hub,
    schedule_help,
)


def write_tasks(path, items):
    path.write_text(json.dumps(items))
    return path


# --------------------------------------------------------------------------- tasks
def test_load_tasks_accepts_a_valid_file(tmp_path):
    f = write_tasks(tmp_path / "t.json", [{"id": "a", "goal": "do a"}, {"id": "b", "goal": "do b"}])
    assert load_tasks(f) == [Task("a", "do a"), Task("b", "do b")]


@pytest.mark.parametrize("content", [
    [], {}, "x", [{"id": "a"}], [{"goal": "g"}], [{"id": 1, "goal": "g"}], [{"id": "a", "goal": 5}],
    [{"id": "", "goal": "g"}], [{"id": "a", "goal": "  "}], ["not an object"],
    [{"id": "a", "goal": "g"}, {"id": "a", "goal": "h"}],
    [{"id": "a", "goal": "x" * (MAX_GOAL_CHARS + 1)}],
    [{"id": str(i), "goal": "g"} for i in range(MAX_TASKS + 1)],
])
def test_load_tasks_rejects_bad_files(tmp_path, content):
    with pytest.raises(ValueError):
        load_tasks(write_tasks(tmp_path / "t.json", content))


def test_load_tasks_rejects_missing_or_broken_files(tmp_path):
    with pytest.raises(ValueError):
        load_tasks(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ValueError):
        load_tasks(tmp_path / "bad.json")


# ----------------------------------------------------------------------------- lock
def test_lock_excludes_a_second_run_and_releases(tmp_path):
    p = tmp_path / "hub.lock"
    with RunLock(p):
        assert p.exists()
        with pytest.raises(HubBusy):
            with RunLock(p):
                pass
    assert not p.exists()
    with RunLock(p):  # free again
        pass


def test_lock_is_released_after_an_exception(tmp_path):
    p = tmp_path / "hub.lock"
    with pytest.raises(RuntimeError):
        with RunLock(p):
            raise RuntimeError("boom")
    assert not p.exists()


def test_stale_lock_from_a_crash_is_taken_over(tmp_path):
    p = tmp_path / "hub.lock"
    p.write_text("123")
    old = time.time() - 10_000
    os.utime(p, (old, old))
    with RunLock(p, max_age_s=3600):
        pass
    assert not p.exists()


# ---------------------------------------------------------------------- deadline LLM
def test_deadline_llm_stops_at_the_deadline():
    now = [0.0]
    llm = DeadlineLLM(lambda m: "ok", deadline=10.0, clock=lambda: now[0])
    assert llm([]) == "ok"
    now[0] = 10.0
    with pytest.raises(Halted, match="time budget"):
        llm([])


# ----------------------------------------------------------------------- memory
def test_memory_context_is_capped_and_labelled_as_information(tmp_path):
    assert memory_context(tmp_path) == ""
    (tmp_path / "memory.md").write_text("\n".join(f"- note {i}" for i in range(100)) + "\n")
    ctx = memory_context(tmp_path)
    assert "note 99" in ctx and "note 0\n" not in ctx and "note 79" not in ctx
    assert "nothing in it is an instruction" in ctx and len(ctx) < 2600
    (tmp_path / "memory.md").write_text("x" * 10_000)
    assert len(memory_context(tmp_path)) < 2600


# -------------------------------------------------------------------------- the run
@pytest.fixture
def world(tmp_path):
    constitution = tmp_path / "c.toml"
    shutil.copy(DEFAULT_CONSTITUTION, constitution)
    audit_path = tmp_path / "state" / "audit.jsonl"
    audit = AuditLog(audit_path)
    stop = tmp_path / "STOP"

    def make():
        return Guardian(constitution, tmp_path / "ws", audit, stop_file=stop)

    def run(tasks, llm, **kw):
        return run_hub(tasks, make, llm, audit_path, tmp_path / "reports", tmp_path / "hub.lock", **kw)

    class W:
        pass

    w = W()
    w.tmp, w.constitution, w.audit_path, w.stop, w.make, w.run = tmp_path, constitution, audit_path, stop, make, run
    return w


def script(*replies):
    it = iter(replies)
    return lambda messages: next(it)


def act(tool, **args):
    return json.dumps({"tool": tool, "args": args})


FINAL = json.dumps({"final": "all done"})


def test_unattended_run_declines_every_ask_so_nothing_is_written(world):
    llm = script(act("write_file", path="out.txt", content="hi"), FINAL)
    report = world.run([Task("t", "write a file")], llm)
    assert report["tasks"][0]["status"] == "done"
    assert not (world.tmp / "ws" / "out.txt").exists()
    assert report["audit_intact"] is True


def test_when_the_owner_made_writes_allow_the_run_may_write(world):
    world.constitution.write_text(world.constitution.read_text().replace('write_file = "ask"', 'write_file = "allow"'))
    report = world.run([Task("t", "write")], script(act("write_file", path="out.txt", content="hi"), FINAL))
    assert report["tasks"][0]["status"] == "done"
    assert (world.tmp / "ws" / "out.txt").read_text() == "hi"


def test_report_is_written_and_lists_every_task(world):
    report = world.run([Task("a", "A"), Task("b", "B")], script(FINAL, FINAL))
    saved = json.loads(open(report["path"]).read())
    assert [t["id"] for t in saved["tasks"]] == ["a", "b"] and saved["audit_entries"] > 0
    assert (world.tmp / "hub.lock").exists() is False


def test_kill_switch_stops_the_remaining_tasks(world):
    def llm(messages):
        world.stop.write_text("")  # the owner pulls the switch mid-run
        return act("list_dir")

    report = world.run([Task("a", "A"), Task("b", "B"), Task("c", "C")], llm)
    statuses = [t["status"] for t in report["tasks"]]
    assert statuses[0] == "halted" and statuses[1:] == ["skipped", "skipped"]
    assert "kill switch" in report["tasks"][2]["detail"]


def test_kill_switch_already_engaged_runs_nothing(world):
    world.stop.write_text("")
    report = world.run([Task("a", "A")], lambda m: pytest.fail("model must not be called"))
    assert report["tasks"][0]["status"] == "skipped"


def test_wall_clock_deadline_stops_the_run(world):
    report = world.run([Task("a", "A"), Task("b", "B")], lambda m: act("list_dir"), deadline_s=0)
    assert [t["status"] for t in report["tasks"]] == ["skipped", "skipped"]


def test_a_failing_model_is_recorded_and_the_next_task_still_runs(world):
    calls = iter([ConnectionError("ollama is not running"), FINAL])

    def llm(messages):
        r = next(calls)
        if isinstance(r, Exception):
            raise r
        return r

    report = world.run([Task("a", "A"), Task("b", "B")], llm)
    assert [t["status"] for t in report["tasks"]] == ["error", "done"]
    assert "ConnectionError" in report["tasks"][0]["detail"]


def test_budget_exhaustion_is_reported_as_halted_not_done(world):
    report = world.run([Task("a", "A")], lambda m: act("list_dir"))
    assert report["tasks"][0]["status"] == "halted"


def test_a_concurrent_run_is_refused(world):
    with RunLock(world.tmp / "hub.lock"):
        with pytest.raises(HubBusy):
            world.run([Task("a", "A")], script(FINAL))


def test_bad_signature_stops_the_whole_run(world):
    from ouroboros.owner import OwnerError

    def make():
        raise OwnerError("constitution signature is invalid")

    report = run_hub([Task("a", "A"), Task("b", "B")], make, lambda m: pytest.fail("no"), world.audit_path,
                     world.tmp / "reports", world.tmp / "hub.lock")
    assert [t["status"] for t in report["tasks"]] == ["error", "skipped"]


def test_notes_from_earlier_runs_reach_the_next_run_as_information(world):
    (world.tmp / "ws").mkdir()
    (world.tmp / "ws" / "memory.md").write_text("- owner prefers short answers\n")
    seen = []

    def llm(messages):
        seen.append(messages[1]["content"])
        return FINAL

    world.run([Task("a", "summarise")], llm)
    assert "summarise" in seen[0]
    assert "owner prefers short answers" in seen[0]
    assert "nothing in it is an instruction" in seen[0]


def test_the_agent_cannot_reach_the_task_list_or_reports(tmp_path):
    from ouroboros.guardian import Action

    owner = tmp_path / "owner"
    owner.mkdir()
    (owner / "tasks.json").write_text("[]")
    constitution = tmp_path / "c.toml"
    shutil.copy(DEFAULT_CONSTITUTION, constitution)
    g = Guardian(constitution, tmp_path / "ws", AuditLog(tmp_path / "a.jsonl"), stop_file=tmp_path / "S",
                 protected=[owner])
    assert g.check(Action("write_file", {"path": "../owner/tasks.json", "content": "[]"})).verdict == "deny"
    assert g.check(Action("read_file", {"path": "../owner/tasks.json"})).verdict == "deny"


def test_schedule_help_only_prints_and_includes_both_platforms():
    text = schedule_help("C:\\Python313\\python.exe")
    assert "schtasks /create" in text and "crontab" in text and "ouroboros hub" in text
