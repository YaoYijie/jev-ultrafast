"""Supervised Jev sessions: Jev chooses every action, the host only supervises at checkpoints.

A session keeps one live browser tab across tool calls, so the host can advance the run a few
steps at a time, inspect Jev's own decisions and confidences, and hand a new sub-goal back in
without restarting the tab. Nothing here generates an action or reads the page with a model:
the only model call inside a step is Jev's choice, plus the upstream text helper when Jev picks
TYPE_TEXT and a field value has to be written.
"""

import base64
import os
import threading
import time
import uuid
from pathlib import Path

from .agent import Agent
from .browser import StalePage
from .launcher import ensure_chrome_running
from .model import action_space

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "sessions"

# Hand control back before Jev burns the run on guesses. 0.99 -> 0.37 with an unchanged URL is
# the shape of a stuck run; both thresholds exist so the host sees it at step 2, not at step 6.
MIN_CONFIDENCE = 0.55
STALL_LIMIT = 2
STEP_BUDGET = 40
STALE_RETRIES = 3

MAX_LIVE_SESSIONS = 4
IDLE_TIMEOUT_S = 1800

_REGISTRY: dict[str, "Session"] = {}
_REGISTRY_LOCK = threading.Lock()


def load_env() -> None:
    """Load .env without overriding anything the process was already given."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _decision_view(decision: dict, page: dict) -> dict:
    """What Jev decided, in the form the host needs to judge whether to let it through."""
    choice = decision["choice"]
    action = next((a for a in page["actions"] if a["id"] == choice), None)
    ranked = sorted(decision.get("probabilities", {}).items(), key=lambda kv: -kv[1])[:5]
    return {
        "operation": decision["operation"],
        "choice": choice,
        "target_label": action["label"][:120] if action else None,
        "target_kind": action["kind"] if action else None,
        "confidence": decision["confidence"],
        "target_confidence": decision.get("target_confidence"),
        "top_candidates": [{"id": i, "p": round(p, 4)} for i, p in ranked],
        "model": decision.get("model"),
        "latency_ms": decision.get("latency_ms"),
    }


class Session:
    def __init__(self, url: str, goal: str, screenshots: bool = True):
        self.id = uuid.uuid4().hex[:12]
        self.dir = ARTIFACTS / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.agent = Agent(url, goal, screenshots=screenshots)
        self.created_at = time.time()
        self.touched_at = time.time()
        self.total_steps = 0
        self.shot_index = 0
        self.legs = [{"goal": goal, "started_at_step": 0}]
        self.archive: list[dict] = []
        self.pending_decision: dict | None = None
        self.last_error: str | None = None
        self.closed = False

    # -- internals -----------------------------------------------------------

    def _reobserve(self) -> None:
        """Mirror Agent.tick's stale-page recovery: wait out a navigation, then look again."""
        state = self.agent.state
        state["decision"] = None
        state["status"] = "ready"
        deadline = time.monotonic() + 6.0
        while True:
            try:
                state["page"] = self.agent.browser.observe(screenshot=self.agent.screenshots)
                return
            except StalePage as err:
                if time.monotonic() >= deadline or "navigating" not in str(err).lower():
                    raise
                time.sleep(0.05)

    def _save_shot(self) -> str | None:
        page = self.agent.state["page"]
        data = page.get("screenshot")
        if not data:
            return None
        self.shot_index += 1
        out = self.dir / f"{self.shot_index:03d}.jpg"
        out.write_bytes(base64.b64decode(data))
        return str(out)

    def _trailing_stall(self) -> int:
        count = 0
        for entry in reversed(self.agent.state["history"]):
            if entry.get("kind") == "wait" or entry.get("page_changed") is not False:
                break
            count += 1
        return count

    # -- public surface ------------------------------------------------------

    def observation(self, text_chars: int = 4000, elements: int = 30, screenshot: bool = True) -> dict:
        page = self.agent.state["page"]
        space = action_space(page["actions"])[0]
        text = page.get("text", "")
        return {
            "url": page.get("url"),
            "title": page.get("title"),
            "status": "awaiting_review"
            if self.agent.state["status"] == "predicted"
            else self.agent.state["status"],
            "text": text[:text_chars],
            "text_length": len(text),
            "text_truncated": len(text) > text_chars,
            "elements": [
                {k: e[k] for k in ("index", "label", "role", "value", "operations") if k in e}
                for e in space[:elements]
            ],
            "elements_total": len(space),
            "screenshot_path": self._save_shot() if screenshot else None,
        }

    def step(
        self,
        steps: int = 3,
        min_confidence: float = MIN_CONFIDENCE,
        stall_limit: int = STALL_LIMIT,
        force: bool = False,
    ) -> dict:
        """Let Jev decide and act up to `steps` times, stopping early at anything worth a look."""
        with self.lock:
            if self.closed:
                raise ValueError(f"Session {self.id} is closed. Start a new one.")
            self.touched_at = time.time()
            self.pending_decision = None
            self.last_error = None
            agent = self.agent
            executed: list[dict] = []
            retries = 0
            stop = None

            while len(executed) < max(1, steps):
                if agent.state["status"] in {"done", "blocked"}:
                    stop = agent.state["status"]
                    break
                if self.total_steps >= STEP_BUDGET:
                    stop = "step_budget"
                    break

                try:
                    agent.command("predict", {})
                except StalePage:
                    if retries >= STALE_RETRIES:
                        stop = "stale_page"
                        break
                    retries += 1
                    self._reobserve()
                    continue
                except (ValueError, RuntimeError) as exc:
                    stop, self.last_error = "error", str(exc)
                    break

                decision = dict(agent.state["decision"])
                page = agent.state["page"]
                terminal = decision["choice"] in {"DONE", "BLOCKED"}
                if not terminal and not force and decision["confidence"] < min_confidence:
                    # Do not execute a guess. Show the host what Jev wanted and let it decide.
                    stop = "low_confidence"
                    self.pending_decision = _decision_view(decision, page)
                    break

                before = len(agent.state["history"])
                try:
                    agent.command("act", {"fingerprint": page["fingerprint"]})
                except StalePage:
                    if retries >= STALE_RETRIES:
                        stop = "stale_page"
                        break
                    retries += 1
                    self._reobserve()
                    continue
                except (ValueError, RuntimeError) as exc:
                    stop, self.last_error = "error", str(exc)
                    break

                retries = 0
                if len(agent.state["history"]) == before:
                    # DONE / BLOCKED settle the status without touching the page.
                    executed.append(
                        {
                            "terminal": agent.state["status"],
                            "operation": decision["operation"],
                            "confidence": decision["confidence"],
                        }
                    )
                    stop = agent.state["status"]
                    break

                entry = agent.state["history"][-1]
                self.total_steps += 1
                executed.append(
                    {
                        "step": self.total_steps,
                        "operation": entry.get("operation"),
                        "action": (entry.get("action") or "")[:120],
                        "kind": entry.get("kind"),
                        "text": entry.get("text"),
                        "confidence": entry.get("confidence"),
                        "page_changed": entry.get("page_changed"),
                        "url": entry.get("url"),
                        "screenshot_path": self._save_shot(),
                        "elapsed_ms": entry.get("elapsed_ms"),
                    }
                )
                if self._trailing_stall() >= stall_limit:
                    stop = "stalled"
                    break

            return self._report(stop or "steps_exhausted", executed)

    def retarget(self, goal: str) -> dict:
        """Give Jev a fresh atomic sub-goal on the same tab.

        Planning is the host's job; choosing among observed elements is Jev's. The finished leg is
        archived and the action history cleared so a previous stall cannot immediately re-block the
        new leg — the session-wide step budget still bounds the whole run.
        """
        with self.lock:
            if self.closed:
                raise ValueError(f"Session {self.id} is closed. Start a new one.")
            goal = goal.strip()
            if not goal:
                raise ValueError("Supply a goal")
            self.touched_at = time.time()
            state = self.agent.state
            self.archive.append({"goal": state["goal"], "history": list(state["history"])})
            state["history"] = []
            state["goal"] = goal
            state["plan"] = [goal]
            state["plan_index"] = 0
            state["decision"] = None
            state["status"] = "ready"
            self.pending_decision = None
            self.legs.append({"goal": goal, "started_at_step": self.total_steps})
            if not self.agent.browser.fresh(state["page"]):
                self._reobserve()
            return self._report("retargeted", [])

    def close(self) -> dict:
        with self.lock:
            report = self._report("finished", [])
            if not self.closed:
                self.closed = True
                try:
                    self.agent.close()
                except Exception as exc:  # a dead tab must not block cleanup
                    report["close_error"] = str(exc)
            return report

    def brief(self, stop_reason: str = "retargeted") -> dict:
        """The same report a step returns, without executing anything."""
        with self.lock:
            return self._report(stop_reason, [])

    def _report(self, stop_reason: str, executed: list[dict]) -> dict:
        agent = self.agent
        # "predicted" is an internal half-state: a decision exists but was held back unexecuted.
        status = "awaiting_review" if agent.state["status"] == "predicted" else agent.state["status"]
        report = {
            "session_id": self.id,
            "goal": agent.state["goal"],
            "status": status,
            "stop_reason": stop_reason,
            "steps_executed": executed,
            "total_steps": self.total_steps,
            "step_budget": STEP_BUDGET,
            "legs": len(self.legs),
            "observation": self.observation(),
            "next": NEXT_HINTS.get(stop_reason, NEXT_HINTS["steps_exhausted"]),
        }
        if self.pending_decision:
            report["pending_decision"] = self.pending_decision
        if self.last_error:
            report["error"] = self.last_error
        return report


NEXT_HINTS = {
    "steps_exhausted": "Jev is still making progress. Call ultrafast_step again to continue.",
    "low_confidence": "Jev's next choice is below the confidence floor and was NOT executed. Read "
    "pending_decision and elements: call ultrafast_step(force=true) to let it through, "
    "ultrafast_retarget with a narrower sub-goal, or ultrafast_finish.",
    "stalled": "The page stopped changing. Retarget with a narrower sub-goal, or finish and start "
    "a new session from a more specific URL.",
    "done": "Jev reports the goal is visibly satisfied. Verify with ultrafast_read, then "
    "ultrafast_finish.",
    "blocked": "Jev found no supported operation. Retarget with a narrower sub-goal, or finish.",
    "step_budget": "Session step budget reached. Read what you have, then finish.",
    "stale_page": "The page kept navigating. Call ultrafast_step again, or retarget.",
    "error": "The run hit an error; see `error`. The tab is still open — retarget or finish.",
    "retargeted": "New sub-goal set on the same tab. Call ultrafast_step to run it.",
    "finished": "Session closed.",
}


# -- registry ----------------------------------------------------------------


def _reap(now: float | None = None) -> None:
    now = now or time.time()
    stale = [s for s in _REGISTRY.values() if now - s.touched_at > IDLE_TIMEOUT_S]
    for session in stale:
        session.close()
        _REGISTRY.pop(session.id, None)
    if len(_REGISTRY) >= MAX_LIVE_SESSIONS:
        for session in sorted(_REGISTRY.values(), key=lambda s: s.touched_at)[
            : len(_REGISTRY) - MAX_LIVE_SESSIONS + 1
        ]:
            session.close()
            _REGISTRY.pop(session.id, None)


def start(url: str, goal: str, screenshots: bool = True) -> Session:
    load_env()
    ensure_chrome_running()
    with _REGISTRY_LOCK:
        _reap()
        session = Session(url, goal, screenshots=screenshots)
        _REGISTRY[session.id] = session
        return session


def get(session_id: str) -> Session:
    with _REGISTRY_LOCK:
        session = _REGISTRY.get(session_id)
    if session is None:
        raise KeyError(f"Unknown session {session_id!r}. Live sessions: {sorted(_REGISTRY)}")
    return session


def finish(session_id: str) -> dict:
    session = get(session_id)
    report = session.close()
    with _REGISTRY_LOCK:
        _REGISTRY.pop(session_id, None)
    return report


def live() -> list[dict]:
    with _REGISTRY_LOCK:
        return [
            {
                "session_id": s.id,
                "goal": s.agent.state["goal"],
                "status": s.agent.state["status"],
                "url": s.agent.state["page"].get("url"),
                "total_steps": s.total_steps,
                "idle_s": round(time.time() - s.touched_at),
            }
            for s in _REGISTRY.values()
        ]
