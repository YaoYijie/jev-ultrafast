"""Loopback-only web UI for jev-auto.

Same posture as the upstream inspector: bound to 127.0.0.1, Host and Origin checked, a per-process
token injected into the page and required on every write, standard library only.

It exists mostly for one thing a terminal cannot do well: the gate stops the run to ask before an
action commits something, and that question has to reach whoever is actually watching.
"""

import json
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import autopilot

ROOT = Path(__file__).parent
PORT = int(os.environ.get("JEV_AUTO_PORT", "8767"))
ORIGIN = f"http://127.0.0.1:{PORT}"
TOKEN = secrets.token_urlsafe(32)

LOCK = threading.Lock()
ANSWERED = threading.Event()
STATE = {"status": "idle", "lines": [], "question": None, "answer": None, "result": None, "error": None}


def _reset():
    # run() announces the need itself; adding it here printed it twice.
    STATE.update(status="running", lines=[], question=None, answer=None, result=None, error=None)
    ANSWERED.clear()


def say(*parts):
    line = " ".join(str(p) for p in parts)
    with LOCK:
        STATE["lines"].append(line)


def ask(question):
    """Block the run until the page answers. A closed page must not become a silent yes."""
    with LOCK:
        STATE["question"] = question.strip()
        STATE["answer"] = None
    ANSWERED.clear()
    ANSWERED.wait()
    with LOCK:
        answer = bool(STATE["answer"])
        STATE["question"] = None
    return answer


def _worker(params):
    try:
        result = autopilot.run(
            params["need"],
            url=params.get("url") or None,
            chunk=int(params.get("chunk") or 6),
            max_steps=int(params.get("max_steps") or 30),
            allow_commit=bool(params.get("allow_commit")),
            say=say,
            ask=ask,
        )
        with LOCK:
            STATE["result"] = result
            STATE["status"] = "done"
    except Exception as exc:
        with LOCK:
            STATE["error"] = f"{type(exc).__name__}: {exc}"
            STATE["status"] = "error"
    finally:
        ANSWERED.set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def send(self, status, content, mime="application/json"):
        content = content if isinstance(content, bytes) else content.encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _local(self):
        return self.headers.get("Host") == f"127.0.0.1:{PORT}"

    def do_GET(self):
        if not self._local():
            return self.send(403, "Forbidden", "text/plain")
        path = urlparse(self.path).path
        if path == "/api/state":
            with LOCK:
                return self.send(200, json.dumps({
                    "status": STATE["status"],
                    "lines": STATE["lines"],
                    "question": STATE["question"],
                    "result": STATE["result"],
                    "error": STATE["error"],
                }, ensure_ascii=False))
        if path != "/":
            return self.send(404, "Not found", "text/plain")
        page = (ROOT / "static" / "auto.html").read_text().replace("__TOKEN__", TOKEN)
        self.send(200, page, "text/html; charset=utf-8")

    def do_POST(self):
        if (
            not self._local()
            or self.headers.get("X-Jev-Token") != TOKEN
            or self.headers.get("Origin") not in (None, ORIGIN)
        ):
            return self.send(403, json.dumps({"error": "Local requests only"}))
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or "{}")
        except ValueError:
            return self.send(400, json.dumps({"error": "Bad JSON"}))
        path = urlparse(self.path).path

        if path == "/api/start":
            need = (body.get("need") or "").strip()
            if not need:
                return self.send(400, json.dumps({"error": "需求不能为空"}))
            with LOCK:
                if STATE["status"] == "running":
                    return self.send(409, json.dumps({"error": "已有任务在运行"}))
                _reset()
            threading.Thread(target=_worker, args=(body,), daemon=True).start()
            return self.send(200, json.dumps({"ok": True}))

        if path == "/api/answer":
            with LOCK:
                if STATE["question"] is None:
                    return self.send(409, json.dumps({"error": "当前没有待确认的动作"}))
                STATE["answer"] = bool(body.get("approve"))
            ANSWERED.set()
            return self.send(200, json.dumps({"ok": True}))

        return self.send(404, json.dumps({"error": "Not found"}))


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"jev-auto UI: {ORIGIN}")
    try:
        webbrowser.open(ORIGIN)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
