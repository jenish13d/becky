"""Tests for Becky online: authentication, lockout, input limits, headers, and a real request."""

from __future__ import annotations

import http.client
import json
import shutil
import threading

import pytest

from ouroboros import web
from ouroboros.agent import Agent
from ouroboros.guardian import DEFAULT_CONSTITUTION, AuditLog, Guardian

TOKEN = "t" * 24 + "-secret"


class Running:
    def __init__(self, run_request, **kw):
        self.calls = []
        self.run = run_request or (lambda goal: self.calls.append(goal) or f"echo: {goal}")
        self.server = web.make_server("127.0.0.1", 0, TOKEN, self.run, **kw)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def request(self, method, path, body=None, headers=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        hdrs = dict(headers or {})
        if host:
            hdrs["Host"] = host
        conn.request(method, path, body=body, headers=hdrs)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, dict(r.getheaders()), data

    def ask(self, goal, token=TOKEN, **kw):
        body = json.dumps({"goal": goal}).encode()
        h = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        if token is not None:
            h["Authorization"] = f"Bearer {token}"
        status, _, data = self.request("POST", "/api/ask", body, h, **kw)
        return status, json.loads(data)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def srv():
    s = Running(None)
    yield s
    s.close()


def test_page_and_health_are_public_and_the_page_never_uses_innerhtml(srv):
    status, headers, body = srv.request("GET", "/")
    assert status == 200 and b"<title>Becky</title>" in body
    assert b"innerHTML" not in body and b"textContent" in body  # answers can never inject markup
    assert srv.request("GET", "/api/health")[0] == 200
    assert srv.request("GET", "/nope")[0] == 404


def test_security_headers_are_set_on_every_response(srv):
    for path in ("/", "/api/health", "/nope"):
        _, h, _ = srv.request("GET", path)
        assert "default-src 'none'" in h["Content-Security-Policy"]
        assert h["X-Frame-Options"] == "DENY" and h["Cache-Control"] == "no-store"
        assert h["X-Content-Type-Options"] == "nosniff"


def test_a_correct_token_gets_an_answer(srv):
    assert srv.ask("list the files") == (200, {"answer": "echo: list the files"})
    assert srv.calls == ["list the files"]


@pytest.mark.parametrize("token", [None, "", "wrong", TOKEN + "x", TOKEN[:-1], TOKEN.upper()])
def test_wrong_or_missing_tokens_are_refused_and_nothing_runs(srv, token):
    status, body = srv.ask("list the files", token=token)
    assert status == 401 and "answer" not in body and srv.calls == []


def test_the_scheme_must_be_bearer(srv):
    body = json.dumps({"goal": "x"}).encode()
    status, _, _ = srv.request("POST", "/api/ask", body, {"Authorization": TOKEN, "Content-Length": str(len(body))})
    assert status == 401 and srv.calls == []


def test_five_wrong_tokens_lock_the_address_out_even_for_the_right_token():
    now = [1000.0]
    s = Running(None, clock=lambda: now[0])
    try:
        for _ in range(web.MAX_FAILS):
            assert s.ask("x", token="nope")[0] == 401
        status, body = s.ask("x")  # the correct token, but locked out
        assert status == 429 and "try again" in body["error"] and s.calls == []
        now[0] += web.LOCKOUT_S + 1
        assert s.ask("x")[0] == 200
    finally:
        s.close()


def test_a_correct_token_resets_the_failure_count():
    s = Running(None)
    try:
        for _ in range(web.MAX_FAILS - 1):
            s.ask("x", token="nope")
        assert s.ask("x")[0] == 200
        for _ in range(web.MAX_FAILS - 1):
            assert s.ask("x", token="nope")[0] == 401  # count restarted, so still not locked
    finally:
        s.close()


def test_oversized_malformed_and_odd_requests_are_refused(srv):
    h = {"Authorization": f"Bearer {TOKEN}"}
    big = b"x" * (web.MAX_BODY + 1)
    assert srv.request("POST", "/api/ask", big, {**h, "Content-Length": str(len(big))})[0] == 413
    for body in (b"not json", b"[]", b'{"nope": 1}', b'{"goal": 5}', b'{"goal": ""}', b'{"goal": "   "}',
                 json.dumps({"goal": "x" * (web.MAX_GOAL + 1)}).encode()):
        assert srv.request("POST", "/api/ask", body, {**h, "Content-Length": str(len(body))})[0] == 400, body
    assert srv.request("POST", "/api/ask", None, h)[0] == 400  # empty body (the client sends Content-Length: 0)
    import socket

    with socket.create_connection(("127.0.0.1", srv.port), timeout=10) as sock:
        # no Content-Length header at all
        sock.sendall(f"POST /api/ask HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {TOKEN}\r\n\r\n".encode())
        assert b" 411 " in sock.recv(4096).split(b"\r\n")[0]
    assert srv.request("POST", "/api/other", b"{}", {**h, "Content-Length": "2"})[0] == 404
    for method in ("PUT", "DELETE", "PATCH", "OPTIONS"):
        assert srv.request(method, "/api/ask", None, h)[0] == 405
    assert srv.calls == []


def test_an_internal_error_is_reported_without_leaking_details(capsys):
    def boom(goal):
        raise RuntimeError("secret internal detail")

    s = Running(boom)
    try:
        status, body = s.ask("x")
        assert status == 500 and "secret internal detail" not in json.dumps(body)
        assert "secret internal detail" in capsys.readouterr().err  # but the owner sees it
        assert s.ask("x")[0] == 500  # and the server keeps running
    finally:
        s.close()


def test_other_websites_cannot_reach_a_local_copy_through_a_foreign_host_name(srv):
    assert srv.request("GET", "/", host="evil.example.com")[0] == 403
    assert srv.request("POST", "/api/ask", b"{}", {"Content-Length": "2"}, host="evil.example.com:8080")[0] == 403
    assert srv.request("GET", "/", host="localhost:8080")[0] == 200
    assert srv.request("GET", "/", host="127.0.0.1")[0] == 200


def test_a_tunnel_host_must_be_allowed_explicitly():
    s = Running(None, allowed_hosts=("becky.example.org",))
    try:
        assert s.request("GET", "/", host="becky.example.org")[0] == 200
        assert s.request("GET", "/", host="other.example.org")[0] == 403
    finally:
        s.close()


@pytest.mark.parametrize("value,expected", [("localhost:8080", "localhost"), ("[::1]:8080", "[::1]"),
                                            ("Becky.Example.org", "becky.example.org"), (None, ""), ("", "")])
def test_host_header_parsing(value, expected):
    assert web.host_name(value) == expected


def test_a_short_token_is_refused_at_startup():
    with pytest.raises(ValueError, match="at least"):
        web.make_server("127.0.0.1", 0, "short", lambda g: "x")


def test_the_token_never_appears_in_responses_or_the_page(srv):
    _, _, page = srv.request("GET", "/")
    status, body = srv.ask("x", token="nope")
    assert TOKEN.encode() not in page and TOKEN not in json.dumps(body)


def test_requests_are_handled_one_at_a_time():
    active, overlap = [0], []
    started = threading.Event()

    def slow(goal):
        active[0] += 1
        overlap.append(active[0])
        started.set()
        threading.Event().wait(0.1)
        active[0] -= 1
        return "done"

    s = Running(slow)
    try:
        threads = [threading.Thread(target=lambda: s.ask("x")) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert max(overlap) == 1  # the audit log's hash chain is not safe to write from two places at once
    finally:
        s.close()


def test_a_real_request_reads_but_a_write_is_declined(tmp_path):
    shutil.copy(DEFAULT_CONSTITUTION, tmp_path / "c.toml")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.txt").write_text("meeting at noon")
    audit = AuditLog(tmp_path / "a.jsonl")
    from ouroboros.brain import load_brain

    brain = load_brain()

    def run_request(goal):
        g = Guardian(tmp_path / "c.toml", ws, audit, stop_file=tmp_path / "STOP")
        return Agent(brain, g, lambda action, decision: False).run(goal)  # unattended: nothing that asks is allowed

    s = Running(run_request)
    try:
        assert "notes.txt says: meeting at noon" in s.ask("read notes.txt")[1]["answer"]
        assert "not allowed" in s.ask("write hello world to greeting.txt")[1]["answer"]
        assert not (ws / "greeting.txt").exists()
        assert "I can only" in s.ask("delete notes.txt")[1]["answer"] and (ws / "notes.txt").exists()
        assert audit.verify()[0]
    finally:
        s.close()
