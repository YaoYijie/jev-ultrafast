"""Supervised Jev sessions: Jev chooses every action, the host only supervises at checkpoints.

A session keeps one live browser tab across tool calls, so the host can advance the run a few
steps at a time, inspect Jev's own decisions and confidences, and hand a new sub-goal back in
without restarting the tab. Nothing here generates an action or reads the page with a model:
the only model call inside a step is Jev's choice, plus the upstream text helper when Jev picks
TYPE_TEXT and a field value has to be written.
"""

import base64
import json
import os
import threading
import time
import uuid
from pathlib import Path

from . import guard, job_patrol, launcher, ownership
from .agent import Agent
from .browser import StalePage
from .launcher import ensure_chrome_running
from .model import action_space

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "sessions"

# Hand control back before Jev burns the run on guesses. A stuck run looks like 0.52, 0.48, 0.45,
# 0.37 with the page never changing, so the stall limit is the primary signal and the confidence
# floor only catches wild guesses: on a real goal, correct choices are observed as low as 0.42,
# while the ones worth stopping sit near 0.2. A floor above ~0.4 interrupts decisions that are right.
MIN_CONFIDENCE = 0.35
STALL_LIMIT = 2
# A click that reopens the same document in a new tab changes the fingerprint every time, so
# page_changed cannot see that nothing is advancing. Repeating one choice is the other stall.
REPEAT_LIMIT = 3
STEP_BUDGET = 40
# Scrolling and waiting neither navigate nor mutate, and reading a long list is mostly scrolling.
# Holding those back on confidence costs a round trip for nothing; running out of page stops them.
SOFT_KINDS = {"scroll", "wait"}
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
        value = value.strip()
        # A value may be wrapped in quotes so that `uv run --env-file` keeps the quotes inside it;
        # read directly, those wrappers are not part of the value.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


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
    def __init__(
        self,
        url: str,
        goal: str,
        screenshots: bool = True,
        mode: str = "standard",
        allowed_platform: str | None = None,
        action_delay_s: float = 0,
        step_budget: int = STEP_BUDGET,
    ):
        if mode not in {"standard", job_patrol.MODE}:
            raise ValueError(f"Unknown session mode: {mode}")
        if mode == job_patrol.MODE:
            allowed_platform = job_patrol.require_platform_url(url, allowed_platform)
        self.id = uuid.uuid4().hex[:12]
        self.dir = ARTIFACTS / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        browser_options = None
        if mode == job_patrol.MODE:
            browser_options = {
                "daemon_name": job_patrol.DAEMON_NAME,
                "daemon_env": job_patrol.DAEMON_ENV,
                "strict_fresh": True,
            }
        self.agent = Agent(url, goal, screenshots=screenshots, browser_options=browser_options)
        self.created_at = time.time()
        self.touched_at = time.time()
        self.total_steps = 0
        self.mode = mode
        self.allowed_platform = allowed_platform
        self.action_delay_s = max(2.0 if mode == job_patrol.MODE else 0.0, float(action_delay_s))
        self.step_budget = max(1, min(int(step_budget), 20) if mode == job_patrol.MODE else int(step_budget))
        self.last_action_at: float | None = None
        self.blocked_reason: str | None = None
        self.shot_index = 0
        self.legs = [{"goal": goal, "started_at_step": 0}]
        self.archive: list[dict] = []
        self.pending_decision: dict | None = None
        self.rechecked_blocked = False
        self.last_error: str | None = None
        self.closed = False
        self.patrol_stop_reason: str | None = None
        self.observed_pages: list[dict] = []
        self._observed_keys: set[str] = set()
        self._remember_page(self.agent.state["page"])
        if mode == job_patrol.MODE:
            self.agent.page_guard = self._guard_page

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
                self._remember_page(state["page"])
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

    def _trailing_repeat(self) -> int:
        history = [h for h in self.agent.state["history"] if h.get("kind") not in SOFT_KINDS]
        if not history:
            return 0
        label = history[-1].get("action")
        count = 0
        for entry in reversed(history):
            if entry.get("action") != label:
                break
            count += 1
        return count

    def _remember_page(self, page):
        # Keep every observed viewport, including intermediate pages inside one checkpoint.
        # Replacing by URL loses SPA city changes and scrolling through the same result list.
        if not page.get("url") or page["url"].startswith("about:"):
            return
        evidence = {k: page.get(k) for k in ("url", "title", "text", "links")}
        evidence["text"] = (evidence["text"] or "")[:6000]
        evidence["links"] = (evidence["links"] or [])[:250]
        key = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        if key not in self._observed_keys and len(self.observed_pages) < 256:
            self._observed_keys.add(key)
            self.observed_pages.append(evidence)

    def _guard_page(self, page):
        self._remember_page(page)
        stopped = job_patrol.page_stop(page, self.allowed_platform)
        if stopped:
            self.patrol_stop_reason, self.blocked_reason = stopped
            raise job_patrol.PatrolStopped(*stopped)

    def _account_actions(self, before, executed):
        # Agent logs input before observation. Even if that observation fails, the action counts
        # toward the budget and the next input must still respect the pacing interval.
        for entry in self.agent.state["history"][before:]:
            self.total_steps += 1
            self.last_action_at = time.monotonic()
            executed.append({
                "step": self.total_steps,
                "operation": entry.get("operation"),
                "action": (entry.get("action") or "")[:120],
                "kind": entry.get("kind"),
                "text": entry.get("text"),
                "confidence": entry.get("confidence"),
                "page_changed": entry.get("page_changed"),
                "url": entry.get("url"),
                "elapsed_ms": entry.get("elapsed_ms"),
            })
        self._remember_page(self.agent.state["page"])

    def _patrol_stop(self) -> str | None:
        if self.mode != job_patrol.MODE:
            return None
        if self.blocked_reason:
            return getattr(self, "patrol_stop_reason", None) or "platform_blocked"
        stopped = job_patrol.page_stop(self.agent.state["page"], self.allowed_platform)
        if stopped:
            self.patrol_stop_reason, self.blocked_reason = stopped
            return self.patrol_stop_reason
        return None

    def _pace(self) -> None:
        if self.mode != job_patrol.MODE or self.last_action_at is None:
            return
        remaining = self.action_delay_s - (time.monotonic() - self.last_action_at)
        if remaining > 0:
            time.sleep(remaining)

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
            "links": page.get("links", [])[:elements],
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
        gated: bool = False,
        approved: str | None = None,
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

            steps = min(steps, 2) if self.mode == job_patrol.MODE else steps
            while len(executed) < max(1, steps):
                patrol_stop = self._patrol_stop()
                if patrol_stop:
                    stop = patrol_stop
                    break
                if agent.state["status"] in {"done", "blocked"}:
                    stop = agent.state["status"]
                    break
                if self.total_steps >= self.step_budget:
                    stop = "step_budget"
                    break

                self._pace()

                try:
                    agent.command("predict", {})
                except job_patrol.PatrolStopped as exc:
                    stop = exc.reason
                    break
                except StalePage:
                    if retries >= STALE_RETRIES:
                        stop = "stale_page"
                        break
                    retries += 1
                    self._reobserve()
                    continue
                except (ValueError, RuntimeError) as exc:
                    # Choosing has no side effect, so a dropped connection is safe to retry here.
                    if "connection failed" in str(exc).lower() and retries < STALE_RETRIES:
                        retries += 1
                        time.sleep(0.4 * retries)
                        continue
                    stop, self.last_error = "error", str(exc)
                    break

                self._remember_page(agent.state["page"])
                patrol_stop = self._patrol_stop()
                if patrol_stop:
                    stop = patrol_stop
                    break
                decision = dict(agent.state["decision"])
                page = agent.state["page"]
                if (
                    decision["choice"] == "BLOCKED"
                    and not agent.state["history"]
                    and not self.rechecked_blocked
                ):
                    # A leg that gives up before acting has usually just looked too early: results
                    # a click away are still rendering. Look once more before believing it.
                    self.rechecked_blocked = True
                    time.sleep(1.5)
                    self._reobserve()
                    continue
                chosen = next((a for a in page["actions"] if a["id"] == decision["choice"]), None)
                soft = chosen is not None and chosen.get("kind") in SOFT_KINDS
                if gated or self.mode == job_patrol.MODE:
                    verdict, reason = guard.gate(decision, chosen, page, mode=self.mode)
                    approved_this = approved is not None and chosen is not None and approved == chosen.get("label")
                    if verdict == guard.BLOCK or (verdict == guard.CONFIRM and not approved_this):
                        stop = "blocked_action" if verdict == guard.BLOCK else "needs_confirmation"
                        self.pending_decision = {**_decision_view(decision, page), "gate_reason": reason}
                        if self.mode == job_patrol.MODE:
                            self.patrol_stop_reason, self.blocked_reason = "blocked_action", reason
                        break
                    approved = None
                if not force and not soft and decision["confidence"] < min_confidence:
                    # Do not act on a guess, and that includes giving up: Jev is measurably less
                    # sure when it says DONE (median 0.49) than when it clicks (median 0.93), so
                    # exempting terminals ended legs after one action on a 0.16 "done".
                    stop = "low_confidence"
                    self.pending_decision = _decision_view(decision, page)
                    break

                before = len(agent.state["history"])
                stale = False
                try:
                    agent.command("act", {"fingerprint": page["fingerprint"]})
                except StalePage:
                    stale = True
                except (ValueError, RuntimeError) as exc:
                    stop, self.last_error = "error", str(exc)
                finally:
                    self._account_actions(before, executed)
                if stop:
                    break
                if stale:
                    if retries >= STALE_RETRIES:
                        stop = "stale_page"
                        break
                    retries += 1
                    self._reobserve()
                    stop = self._patrol_stop()
                    if stop:
                        break
                    continue

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

                executed[-1]["screenshot_path"] = self._save_shot()
                patrol_stop = self._patrol_stop()
                if patrol_stop:
                    stop = patrol_stop
                    break
                if self._trailing_stall() >= stall_limit:
                    stop = "stalled"
                    break
                if self._trailing_repeat() >= REPEAT_LIMIT:
                    stop = "looping"
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
            if self._patrol_stop():
                raise ValueError(f"Patrol has stopped: {self.blocked_reason}")
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
            self.rechecked_blocked = False
            if not self.agent.browser.fresh(state["page"]):
                self._reobserve()
            return self._report("retargeted", [])

    def navigate(self, url: str, goal: str | None = None) -> dict:
        """Point the same tab at a different site.

        Retargeting can only restate the goal for the page you are on. When the site itself cannot
        do the job, a supervisor that can only retarget will say so forever and never move.
        """
        with self.lock:
            if self.closed:
                raise ValueError(f"Session {self.id} is closed. Start a new one.")
            url = url.strip()
            if not url.startswith(("http://", "https://")):
                raise ValueError("navigate needs an http(s) URL")
            if self.mode == job_patrol.MODE:
                if self._patrol_stop():
                    raise ValueError(f"Patrol has stopped: {self.blocked_reason}")
                job_patrol.require_platform_url(url, self.allowed_platform)
            self.touched_at = time.time()
            browser = self.agent.browser
            state = self.agent.state
            self.archive.append({"goal": state["goal"], "history": list(state["history"])})
            state["history"] = []
            if goal and goal.strip():
                state["goal"] = goal.strip()
                state["plan"] = [state["goal"]]
                state["plan_index"] = 0
            self._pace()
            browser.navigate(url)
            self.last_action_at = time.monotonic()
            state["decision"] = None
            state["status"] = "ready"
            state["page"] = browser.observe(screenshot=self.agent.screenshots)
            self._remember_page(state["page"])
            self.pending_decision = None
            self.rechecked_blocked = False
            self.legs.append({"goal": state["goal"], "started_at_step": self.total_steps})
            return self._report("navigated", [])

    def close(self) -> dict:
        with self.lock:
            report = {}
            try:
                report = self._report("finished", [])
                return report
            finally:
                if not self.closed:
                    self.closed = True
                    try:
                        self.agent.close()
                    except Exception as exc:  # a dead tab must not block cleanup
                        report["close_error"] = str(exc)
                    finally:
                        lease = getattr(self, "_browser_lease", None)
                        if lease is not None:
                            lease.close()

    def _granularity_advice(self, stop_reason: str, executed: list[dict]) -> str | None:
        """Tell the host when its sub-goal left Jev nothing to choose.

        A leg that ends after one near-certain action is the shape of a goal that named the element
        instead of the outcome. Jev confirms rather than decides, and the host is back to doing the
        choosing it delegated.
        """
        if stop_reason != "done":
            return None
        actions = [e for e in executed if "step" in e]
        if len(actions) > 1 or not actions:
            return None
        if min(a["confidence"] for a in actions) < 0.95:
            return None
        return (
            "This leg finished in a single near-certain action, which usually means the sub-goal "
            "named the element to click instead of the page state to reach — Jev confirmed a choice "
            "you had already made. Give the next leg an outcome to reach and more steps, and let "
            "Jev find the way there."
        )

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
            "step_budget": self.step_budget,
            "mode": self.mode,
            "platform": self.allowed_platform,
            "legs": len(self.legs),
            "observation": self.observation(),
            "next": NEXT_HINTS.get(stop_reason, NEXT_HINTS["steps_exhausted"]),
        }
        advice = self._granularity_advice(stop_reason, executed)
        if advice:
            report["advice"] = advice
        if self.pending_decision:
            report["pending_decision"] = self.pending_decision
        if self.last_error:
            report["error"] = self.last_error
        if self.blocked_reason:
            report["blocked_reason"] = self.blocked_reason
        return report


NEXT_HINTS = {
    "steps_exhausted": "Jev is still making progress. Call ultrafast_step again to continue.",
    "low_confidence": "Jev's next choice is below the confidence floor and was NOT executed. Read "
    "pending_decision and elements: call ultrafast_step(force=true) to let it through, "
    "ultrafast_retarget with a narrower sub-goal, or ultrafast_finish. When the held-back "
    "operation is DONE, Jev is unsure the goal is actually satisfied — check the page before "
    "accepting it, and prefer retargeting the rest of the goal over forcing.",
    "stalled": "The page stopped changing. Retarget with a narrower sub-goal, or finish and start "
    "a new session from a more specific URL.",
    "looping": "Jev chose the same element several times in a row without advancing — often a link "
    "that reopens the same document in a new tab. Retarget with a sub-goal that names a "
    "different next outcome, or read what is on the page and finish.",
    "done": "Jev reports the goal is visibly satisfied. Verify with ultrafast_read, then "
    "ultrafast_finish.",
    "blocked": "Jev found no supported operation. Retarget with a narrower sub-goal, or finish.",
    "step_budget": "Session step budget reached. Read what you have, then finish.",
    "stale_page": "The page kept navigating. Call ultrafast_step again, or retarget.",
    "error": "The run hit an error; see `error`. The tab is still open — retarget or finish.",
    "needs_confirmation": "Jev wants to run an action that commits something (submit, order, pay, "
    "send, sign in, delete). It was NOT executed. Show the user pending_decision and gate_reason, "
    "and only re-run with approved set to that exact target_label if they say yes.",
    "blocked_action": "Jev wants to type into a field that asks for a secret (password, card, ID). "
    "This is never executed. Tell the user to fill it themselves, then continue.",
    "platform_blocked": "The recruiting platform showed a login, verification, rate-limit, or "
    "risk-control signal. Stop this platform for the current patrol; do not retry with another route.",
    "left_platform": "The job patrol left its fixed platform. Stop the run instead of following "
    "the external page or switching sites.",
    "retargeted": "New sub-goal set on the same tab. Call ultrafast_step to run it.",
    "navigated": "The tab now points at a different site. Call ultrafast_step to run the goal there.",
    "finished": "Session closed.",
}


# -- registry ----------------------------------------------------------------


def _reap(now: float | None = None, *, capacity: bool = True) -> None:
    now = now or time.time()
    stale = [s for s in _REGISTRY.values() if now - s.touched_at > IDLE_TIMEOUT_S]
    for session in stale:
        session.close()
        _REGISTRY.pop(session.id, None)
    if capacity and len(_REGISTRY) >= MAX_LIVE_SESSIONS:
        for session in sorted(_REGISTRY.values(), key=lambda s: s.touched_at)[
            : len(_REGISTRY) - MAX_LIVE_SESSIONS + 1
        ]:
            session.close()
            _REGISTRY.pop(session.id, None)


def prepare(mode: str = "standard") -> dict:
    """Prepare Chrome synchronously so a background run can fail before it is dispatched."""
    load_env()
    if mode not in {"standard", job_patrol.MODE}:
        raise ValueError(f"Unknown session mode: {mode}")
    if mode == job_patrol.MODE:
        return launcher.ensure_user_browser(job_patrol.DAEMON_NAME, job_patrol.DAEMON_ENV)
    ensure_chrome_running()
    return launcher.describe()


def start(
    url: str,
    goal: str,
    screenshots: bool = True,
    mode: str = "standard",
    allowed_platform: str | None = None,
    action_delay_s: float = 0,
    step_budget: int = STEP_BUDGET,
) -> Session:
    if mode == job_patrol.MODE:
        allowed_platform = job_patrol.require_platform_url(url, allowed_platform)
    with _REGISTRY_LOCK:
        _reap(capacity=False)
    lease = ownership.acquire(ARTIFACTS.parent / "browser.lock", exclusive=mode == job_patrol.MODE)
    try:
        prepare(mode)
        with _REGISTRY_LOCK:
            _reap()
            session = Session(
                url,
                goal,
                screenshots=screenshots,
                mode=mode,
                allowed_platform=allowed_platform,
                action_delay_s=action_delay_s,
                step_budget=step_budget,
            )
            session._browser_lease = lease
            _REGISTRY[session.id] = session
            return session
    except BaseException:
        lease.close()
        raise


def get(session_id: str) -> Session:
    with _REGISTRY_LOCK:
        session = _REGISTRY.get(session_id)
    if session is None:
        raise KeyError(f"Unknown session {session_id!r}. Live sessions: {sorted(_REGISTRY)}")
    return session


def finish(session_id: str) -> dict:
    session = get(session_id)
    try:
        return session.close()
    finally:
        with _REGISTRY_LOCK:
            _REGISTRY.pop(session_id, None)


def live() -> list[dict]:
    with _REGISTRY_LOCK:
        _reap(capacity=False)
        return [
            {
                "session_id": s.id,
                "goal": s.agent.state["goal"],
                "status": s.agent.state["status"],
                "url": s.agent.state["page"].get("url"),
                "mode": s.mode,
                "platform": s.allowed_platform,
                "total_steps": s.total_steps,
                "idle_s": round(time.time() - s.touched_at),
            }
            for s in _REGISTRY.values()
        ]
