"""Runs: jev-auto as a background job, so a browsing task outlives the call that started it.

A run is started, polled and read back by id. Nothing blocks for minutes waiting on a browser,
the gate's question reaches whoever is actually watching instead of a closed stdin, and every
finished run is written to disk, so what a task found survives the process that ran it.

One rule here is structural rather than configured. A run nobody is watching cannot approve a
committing action: start(watched=False) hands the loop an ask() that declines immediately,
records what it declined, and never blocks. Approval needs a caller that can reach a person,
which is why answer() is reachable from the local page and not from the MCP tools.
"""

import json
import threading
import time
import uuid
from pathlib import Path

from . import autopilot

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "runs"
INDEX = ARTIFACTS / "index.jsonl"

# Finished runs stay in memory so a poll right after completion is free. Disk is the real
# history, so evicting one only costs a file read.
MAX_RESIDENT = 8
# Each run drives a browser tab, and session.MAX_LIVE_SESSIONS closes the oldest when there are
# too many — so an unbounded auto_start loop would quietly pull the tab out from under a run that
# is still going. Refusing here is the honest failure.
MAX_CONCURRENT = 2

_REGISTRY: dict[str, "Run"] = {}
_REGISTRY_LOCK = threading.Lock()
_DISK_LOCK = threading.Lock()


class Run:
    """One jev-auto task: its log as it happens, and its report once there is one."""

    def __init__(self, need: str, url: str | None, chunk: int, max_steps: int,
                 watched: bool, allow_commit: bool):
        self.id = uuid.uuid4().hex[:12]
        self.need = need
        self.url = url or None
        self.chunk = chunk
        self.max_steps = max_steps
        self.watched = watched
        # Unwatched runs have nobody to ask, and a question nobody answers must never decay into
        # a yes. Refusing to commit is the only honest reading of an empty room.
        self.allow_commit = bool(allow_commit and watched)
        self.status = "running"
        self.lines: list[str] = []
        self.question: str | None = None
        self.answer: bool | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self.started_at = time.time()
        self.ended_at: float | None = None
        self.cancelled = False
        self.answered = threading.Event()
        # A terminal status flips before the file exists, so anything that reads the record back
        # from disk has to wait for this rather than for the status.
        self.persisted = threading.Event()
        self.lock = threading.RLock()

    # -- callbacks handed to the loop ----------------------------------------

    def say(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        with self.lock:
            self.lines.append(line)

    def ask(self, question: str) -> bool:
        if not self.watched:
            return False
        with self.lock:
            self.status = "waiting"
            self.question = question.strip()
            self.answer = None
        self.answered.clear()
        self.answered.wait()
        with self.lock:
            approved = bool(self.answer)
            self.question = None
            self.status = "running" if self.status == "waiting" else self.status
        return approved

    def should_stop(self) -> bool:
        return self.cancelled

    # -- control -------------------------------------------------------------

    def reply(self, approve: bool) -> None:
        with self.lock:
            if self.question is None:
                raise ValueError("当前没有待确认的动作")
            self.answer = bool(approve)
        self.answered.set()

    def cancel(self) -> None:
        with self.lock:
            if self.status in {"done", "error", "cancelled"}:
                return
            self.cancelled = True
            # A run parked on a question would otherwise sit there forever.
            if self.question is not None:
                self.answer = False
        self.answered.set()

    # -- views ---------------------------------------------------------------

    def summary(self) -> dict:
        with self.lock:
            result = self.result or {}
            return {
                "run_id": self.id,
                "need": self.need,
                "status": self.status,
                "watched": self.watched,
                "allow_commit": self.allow_commit,
                # Sub-second precision, because history() orders by this and two runs started
                # in the same second would otherwise come back in arbitrary order.
                "started_at": round(self.started_at, 3),
                "elapsed_s": round((self.ended_at or time.time()) - self.started_at, 1),
                "total_steps": result.get("total_steps"),
                "pages": len(result.get("visited", [])),
                "declined": len(result.get("declined", [])),
                "error": self.error,
            }

    def record(self, *, text: bool = False) -> dict:
        """The whole run. Page text is opt-in: four thousand characters a page fills a context."""
        with self.lock:
            result = dict(self.result or {})
            visited = result.get("visited", [])
            result["visited"] = [
                {k: v for k, v in page.items() if text or k != "text"} for page in visited
            ]
            return {**self.summary(), "lines": list(self.lines), **result}


# -- store -------------------------------------------------------------------


def _evict() -> None:
    finished = [r for r in _REGISTRY.values() if r.status in {"done", "error", "cancelled"}]
    for run in sorted(finished, key=lambda r: r.ended_at or 0)[: max(0, len(finished) - MAX_RESIDENT)]:
        _REGISTRY.pop(run.id, None)


def _persist(run: Run) -> None:
    """Write the run, then its index line: an indexed run that cannot be read back is worse."""
    try:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with _DISK_LOCK:
            (ARTIFACTS / f"{run.id}.json").write_text(
                json.dumps(run.record(text=True), ensure_ascii=False, indent=2), encoding="utf-8")
            with open(INDEX, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(run.summary(), ensure_ascii=False) + "\n")
    except OSError as exc:
        run.say(f"   ⚠ 运行记录写入失败：{exc}")


def _worker(run: Run) -> None:
    try:
        result = autopilot.run(
            run.need,
            url=run.url,
            chunk=run.chunk,
            max_steps=run.max_steps,
            allow_commit=run.allow_commit,
            say=run.say,
            ask=run.ask,
            should_stop=run.should_stop,
        )
        with run.lock:
            run.result = result
            run.status = "cancelled" if run.cancelled else "done"
    except Exception as exc:
        with run.lock:
            run.error = f"{type(exc).__name__}: {exc}"
            run.status = "error"
    finally:
        with run.lock:
            run.ended_at = time.time()
            run.question = None
        # Nothing may be left waiting on a run that has stopped.
        run.answered.set()
        _persist(run)
        run.persisted.set()


def start(need: str, url: str | None = None, chunk: int = 6, max_steps: int = 40,
          watched: bool = False, allow_commit: bool = False) -> Run:
    need = (need or "").strip()
    if not need:
        raise ValueError("需求不能为空")
    run = Run(need, url, chunk, max_steps, watched, allow_commit)
    with _REGISTRY_LOCK:
        busy = [r for r in _REGISTRY.values() if r.status in {"running", "waiting"}]
        if len(busy) >= MAX_CONCURRENT:
            raise ValueError(
                f"已有 {len(busy)} 个任务在跑（上限 {MAX_CONCURRENT}）。等它们结束，或先取消一个："
                + "、".join(r.id for r in busy))
        _evict()
        _REGISTRY[run.id] = run
    threading.Thread(target=_worker, args=(run,), daemon=True).start()
    return run


def get(run_id: str) -> Run:
    with _REGISTRY_LOCK:
        run = _REGISTRY.get(run_id)
    if run is None:
        raise KeyError(f"Unknown run {run_id!r}")
    return run


def load(run_id: str) -> dict:
    """A run by id, from memory if it is still there and from disk if it is not."""
    try:
        return get(run_id).record(text=True)
    except KeyError:
        pass
    path = ARTIFACTS / f"{run_id}.json"
    if not path.exists():
        raise KeyError(f"Unknown run {run_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def poll(run_id: str, since: int = 0) -> dict:
    run = get(run_id)
    with run.lock:
        lines = run.lines[max(0, since):]
        return {
            **run.summary(),
            "lines": lines,
            "next_line": max(0, since) + len(lines),
            "question": run.question,
            "answer": run.result.get("answer") if run.result else None,
        }


def history(limit: int = 20) -> list[dict]:
    """Newest first. Live runs come from memory; the rest from the index, last entry per id."""
    seen: dict[str, dict] = {}
    if INDEX.exists():
        for line in INDEX.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            seen[entry["run_id"]] = entry
    with _REGISTRY_LOCK:
        for run in _REGISTRY.values():
            seen[run.id] = run.summary()
    return sorted(seen.values(), key=lambda e: e["started_at"], reverse=True)[:limit]


def live() -> list[dict]:
    with _REGISTRY_LOCK:
        return [r.summary() for r in _REGISTRY.values() if r.status in {"running", "waiting"}]
