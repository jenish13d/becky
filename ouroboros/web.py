"""Becky online: a small authenticated web page and API.

    python -m ouroboros serve

Safe by default:

* **Token required.** Every request to ``/api/ask`` must carry ``Authorization: Bearer <token>``.
  The token is random per run (or set by you) and is never written to disk or to logs.
* **Lockout.** Five wrong tokens from one address lock it out for a minute.
* **Local only unless you say otherwise.** It listens on 127.0.0.1. Reaching it from a phone or
  the internet is a deliberate step (see GETTING_STARTED.md), and a Host-header check blocks
  other websites from talking to the local copy.
* **Nobody is there to approve, so nothing that asks is allowed.** Reading and listing work;
  writes are declined unless you changed her rules with ``amend`` first.
* **Same guardian as always:** signed rules, confinement to her workspace, kill switch, step
  budget, audit log. No shell, no code execution, no outbound internet access for her.
* **Small inputs.** Bodies over 4 KB and requests over 300 characters are refused.

What it does not do: it has no HTTPS of its own. A token sent over plain HTTP across a network
can be read by anyone on that network, so use it through a private network (Tailscale) or an
HTTPS tunnel, never over open Wi-Fi.
"""

from __future__ import annotations

import hmac
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock
from typing import Callable

MAX_BODY = 4096
MAX_GOAL = 300
MAX_FAILS = 5
LOCKOUT_S = 60.0
LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "::1"}
MIN_TOKEN = 16

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Becky</title>
<style>
 body{font:16px system-ui,sans-serif;max-width:640px;margin:0 auto;padding:16px;background:#111;color:#eee}
 #log{min-height:240px;border:1px solid #333;border-radius:8px;padding:12px;margin:12px 0;white-space:pre-wrap}
 .me{color:#8ab4ff}.bot{color:#b5f5b5}.err{color:#ff9b9b}
 input,button{font:inherit;padding:10px;border-radius:8px;border:1px solid #444;background:#1c1c1c;color:#eee}
 input{width:100%;box-sizing:border-box;margin-bottom:8px} button{cursor:pointer}
</style></head><body>
<h1>Becky</h1>
<input id="token" type="password" placeholder="access token" autocomplete="off">
<div id="log"></div>
<input id="goal" placeholder="list the files" maxlength="300" autocomplete="off">
<button id="send">Send</button>
<script>
const log=document.getElementById('log'),tok=document.getElementById('token'),goal=document.getElementById('goal');
tok.value=sessionStorage.getItem('becky-token')||'';
function add(cls,text){const d=document.createElement('div');d.className=cls;d.textContent=text;log.appendChild(d);}
async function send(){
  const g=goal.value.trim(); if(!g) return;
  sessionStorage.setItem('becky-token',tok.value);
  add('me','you> '+g); goal.value='';
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Authorization':'Bearer '+tok.value,
      'Content-Type':'application/json'},body:JSON.stringify({goal:g})});
    const j=await r.json();
    add(r.ok?'bot':'err',(r.ok?'becky> '+j.answer:'error: '+(j.error||r.status)));
  }catch(e){add('err','error: could not reach Becky');}
}
document.getElementById('send').onclick=send;
goal.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
</script></body></html>
"""

HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                               "connect-src 'self'; base-uri 'none'; form-action 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


class Limiter:
    """Locks an address out after repeated wrong tokens."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock, self.fails, self.until = clock, {}, {}

    def remaining(self, who: str) -> float:
        return max(0.0, self.until.get(who, 0.0) - self.clock())

    def failed(self, who: str) -> None:
        self.fails[who] = self.fails.get(who, 0) + 1
        if self.fails[who] >= MAX_FAILS:
            self.until[who], self.fails[who] = self.clock() + LOCKOUT_S, 0

    def succeeded(self, who: str) -> None:
        self.fails.pop(who, None)


def host_name(header: str | None) -> str:
    if not header:
        return ""
    h = header.strip().lower()
    return h[: h.index("]") + 1] if h.startswith("[") and "]" in h else h.split(":")[0]


def make_server(host: str, port: int, token: str, run_request: Callable[[str], str],
                allowed_hosts: tuple[str, ...] = (), clock: Callable[[], float] = time.monotonic) -> HTTPServer:
    if len(token) < MIN_TOKEN:
        raise ValueError(f"the access token must be at least {MIN_TOKEN} characters")
    limiter, lock = Limiter(clock), Lock()
    token_bytes = token.encode()
    loopback_only = host in LOOPBACK
    allowed = LOOPBACK | {h.lower() for h in allowed_hosts}

    class Handler(BaseHTTPRequestHandler):
        server_version = "Becky"
        sys_version = ""

        def log_message(self, fmt, *args):  # request lines only, never headers; keep the console quiet
            pass

        # ---------------------------------------------------------------- helpers
        def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            for k, v in HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: dict) -> None:
            self._send(status, json.dumps(obj).encode())

        def _host_ok(self) -> bool:
            return (not loopback_only) or host_name(self.headers.get("Host")) in allowed

        # ----------------------------------------------------------------- routes
        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "unexpected Host header"})
            if self.path == "/":
                return self._send(200, PAGE.encode(), "text/html")
            if self.path == "/api/health":
                return self._json(200, {"ok": True})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "unexpected Host header"})
            if self.path != "/api/ask":
                return self._json(404, {"error": "not found"})
            who = self.client_address[0]
            wait = limiter.remaining(who)
            if wait > 0:
                return self._json(429, {"error": f"too many wrong tokens; try again in {int(wait) + 1}s"})
            auth = self.headers.get("Authorization", "")
            given = auth[7:].encode() if auth.startswith("Bearer ") else b""
            if not hmac.compare_digest(given, token_bytes):
                limiter.failed(who)
                return self._json(401, {"error": "wrong or missing token"})
            limiter.succeeded(who)
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._json(411, {"error": "Content-Length required"})
            if length < 0 or length > MAX_BODY:
                return self._json(413, {"error": "request too large"})
            try:
                goal = json.loads(self.rfile.read(length))["goal"]
            except (ValueError, KeyError, TypeError):
                return self._json(400, {"error": 'send JSON like {"goal": "list the files"}'})
            if not isinstance(goal, str) or not goal.strip() or len(goal) > MAX_GOAL:
                return self._json(400, {"error": f"goal must be 1-{MAX_GOAL} characters of text"})
            try:
                with lock:
                    answer = run_request(goal.strip())
            except Exception:  # never leak internals to the client, but do show them to the owner
                traceback.print_exc()
                return self._json(500, {"error": "Becky hit an internal error; see the console"})
            self._json(200, {"answer": answer})

        def _not_allowed(self):
            self._json(405, {"error": "method not allowed"})

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _not_allowed

    return HTTPServer((host, port), Handler)
