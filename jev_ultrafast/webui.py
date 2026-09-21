"""Loopback-only web UI for jev-auto.

Same posture as the upstream inspector: bound to 127.0.0.1, Host and Origin checked, a per-process
token injected into the page and required on every write, standard library only.

It exists mostly for one thing neither a terminal nor an MCP tool can do: the gate stops the run
to ask before an action commits something, and that question has to reach a person. This is the
only surface that can answer it — a run started from MCP is unwatched and declines instead.
"""

import json
import os
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import runs

ROOT = Path(__file__).parent
PORT = int(os.environ.get("JEV_AUTO_PORT", "8767"))
ORIGIN = f"http://127.0.0.1:{PORT}"
TOKEN = secrets.token_urlsafe(32)


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

    def json(self, status, payload):
        return self.send(status, json.dumps(payload, ensure_ascii=False))

    def _local(self):
        return self.headers.get("Host") == f"127.0.0.1:{PORT}"

    def do_GET(self):
        if not self._local():
            return self.send(403, "Forbidden", "text/plain")
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path

        if path == "/api/state":
            run_id = (query.get("run_id") or [""])[0]
            if not run_id:
                return self.json(200, {"status": "idle"})
            try:
                since = int((query.get("since") or ["0"])[0])
            except ValueError:
                since = 0
            try:
                return self.json(200, runs.poll(run_id, since=since))
            except KeyError:
                # Evicted from memory mid-poll; the finished record is still readable.
                try:
                    return self.json(200, {**runs.load(run_id), "next_line": 0, "question": None})
                except KeyError:
                    return self.json(404, {"error": "没有这个任务"})

        if path == "/api/history":
            return self.json(200, {"runs": runs.history(limit=30)})

        if path == "/api/result":
            run_id = (query.get("run_id") or [""])[0]
            try:
                return self.json(200, runs.load(run_id))
            except KeyError:
                return self.json(404, {"error": "没有这个任务"})

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
            return self.json(403, {"error": "Local requests only"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or "{}")
        except ValueError:
            return self.json(400, {"error": "Bad JSON"})
        path = urlparse(self.path).path

        if path == "/api/start":
            try:
                run = runs.start(
                    body.get("need") or "",
                    url=(body.get("url") or "").strip() or None,
                    chunk=int(body.get("chunk") or 6),
                    max_steps=int(body.get("max_steps") or 30),
                    # Someone is looking at this page, so this run may ask before it commits.
                    watched=True,
                    allow_commit=bool(body.get("allow_commit")),
                )
            except ValueError as exc:
                return self.json(400, {"error": str(exc)})
            return self.json(200, {"ok": True, "run_id": run.id})

        if path == "/api/answer":
            try:
                runs.get(body.get("run_id") or "").reply(bool(body.get("approve")))
            except KeyError:
                return self.json(404, {"error": "没有这个任务"})
            except ValueError as exc:
                return self.json(409, {"error": str(exc)})
            return self.json(200, {"ok": True})

        if path == "/api/cancel":
            try:
                runs.get(body.get("run_id") or "").cancel()
            except KeyError:
                return self.json(404, {"error": "没有这个任务"})
            return self.json(200, {"ok": True})

        return self.json(404, {"error": "Not found"})


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
