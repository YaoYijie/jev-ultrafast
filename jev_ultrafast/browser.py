"""Observed actions through Browser Harness; one CDP session, no per-step subprocess."""

import hashlib
import json
import sys
import time
from pathlib import Path

from browser_harness import _ipc as harness_ipc
from browser_harness.admin import ensure_daemon, require_existing_daemon
from browser_harness.helpers import cdp

# Atomically read visible content and controls, preserving actual DOM node identity.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"

class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


def _cdp(method, session_id=None, daemon_name=None, **params):
    """Use Browser Harness's default daemon or an explicitly isolated named daemon."""
    if daemon_name is None:
        return cdp(method, session_id=session_id, **params)
    client, token = harness_ipc.connect(daemon_name, timeout=5.0)
    try:
        client.settimeout(60.0 if method == "Page.captureScreenshot" else 5.0)
        request = {"method": method, "params": params, "session_id": session_id}
        response = harness_ipc.request(client, token, request)
    finally:
        client.close()
    if "error" in response:
        raise RuntimeError(response["error"])
    return response.get("result", {})


class Browser:
    def __init__(self, url, daemon_name=None, daemon_env=None, strict_fresh=False):
        if daemon_name is None:
            ensure_daemon(env=daemon_env)
        else:
            # A named connection has already been prepared and identity-checked. Do not silently
            # replace it with a different Chrome if it dies between preflight and tab creation.
            require_existing_daemon(daemon_name)
        self.daemon_name = daemon_name
        self.strict_fresh = strict_fresh
        self.opened = []
        self.target = None
        try:
            self.target = self._cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
            self.session = self._cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
            self.equip()
            self.navigate(url)
        except Exception:
            self.close()
            raise

    def navigate(self, url):
        self.after_input = None
        result = self.call("Page.navigate", url=url)
        if result.get("errorText"):
            raise RuntimeError(f"Navigation failed: {result['errorText']}")
        self.settle()

    def equip(self):
        self.call("Emulation.setDeviceMetricsOverride", width=1120, height=780, deviceScaleFactor=1, mobile=False)
        # Keep rAF/menus rendering in an owned background tab, without activating the user's Chrome tab.
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)

    def settle(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                if self.evaluate("document.readyState") == "complete":
                    return
            except StalePage:
                pass
            time.sleep(0.02)

    def adopt_popup(self, timeout=0.0):
        """Follow a tab this page opened, so target="_blank" is not mistaken for a dead click.

        One agent owns one tab. A link that opens a new tab leaves the observed page untouched,
        which is indistinguishable from a click that did nothing, and a correct choice then looks
        like a stall. Only the caller that saw no change pays for this lookup.
        """
        deadline = time.monotonic() + timeout
        while True:
            for info in self._cdp("Target.getTargets")["targetInfos"]:
                if info.get("type") == "page" and info.get("openerId") == self.target:
                    self.opened.append(self.target)
                    self.target = info["targetId"]
                    self.session = self._cdp(
                        "Target.attachToTarget", targetId=self.target, flatten=True
                    )["sessionId"]
                    self.after_input = None
                    self.equip()
                    # A fresh popup is "complete" while still on about:blank; settling there would
                    # hand the model an empty page and call it progress.
                    blank = time.monotonic() + 3.0
                    while time.monotonic() < blank:
                        try:
                            if self.evaluate("location.href") not in (None, "about:blank"):
                                break
                        except StalePage:
                            pass
                        time.sleep(0.05)
                    self.settle()
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def call(self, method, **params):
        return self._cdp(method, session_id=self.session, **params)

    def _cdp(self, method, session_id=None, **params):
        return _cdp(method, session_id=session_id, daemon_name=self.daemon_name, **params)

    def evaluate(self, expression):
        response = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def observe(self, screenshot=True):
        if getattr(self, "after_input", None):
            action, self.after_input = self.after_input, None
            # This is read-only and happens after execution was logged, even if navigation interrupts it.
            try:
                self.call(
                    "Runtime.evaluate",
                    expression="""(action => new Promise(resolve => {
                      const field=window.__jevFast?.nodes.get(action.node);
                      const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
                      let frames=0, stopped=false;
                      const finish=()=>{stopped=true;resolve()};
                      setTimeout(finish,autocomplete ? 200 : 50);
                      const ready=()=>{
                        if (stopped) return;
                        const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                          .split(/\\s+/).filter(Boolean);
                        const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                        const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                        if (++frames>=2 && (!autocomplete || options.some(e=>{
                          const r=e.getBoundingClientRect();
                          return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                            e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                        }))) finish();
                        else requestAnimationFrame(ready);
                      };
                      requestAnimationFrame(ready);
                    }))(""" + json.dumps(action) + ")",
                    awaitPromise=True,
                    returnByValue=True,
                )
            except RuntimeError:
                pass
        for attempt in range(10):
            try:
                return browser_operation(
                    {
                        "operation": "observe",
                        "session": self.session,
                        "screenshot": screenshot,
                        "daemon_name": self.daemon_name,
                    }
                )
            except StalePage:
                if attempt == 9:
                    raise
                time.sleep(0.02)
        raise StalePage("Page did not settle")

    def fresh(self, page, action=None):
        if (not getattr(self, "strict_fresh", False)
                and action is not None and action["kind"] in {"click", "select"}):
            node = action["node"]
            if type(node) is not int:
                return False
            current = self.evaluate(
                "(() => { const c=window.__jevFast; "
                f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
            )
            return current == [page["page_key"], page["guards"].get(str(node))]
        return self.evaluate(MARKER) == page["marker"]

    def act(self, action, page, text=None):
        if not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        if action["kind"] == "wait":
            time.sleep(0.1)
        result = browser_operation({
            "operation": "act",
            "session": self.session,
            "action": action,
            "text": text,
            "daemon_name": self.daemon_name,
        })
        self.after_input = action if action["kind"] != "wait" else None
        return result

    def close(self):
        for target in [*self.opened, self.target]:
            if target:
                try:
                    self._cdp("Target.closeTarget", targetId=target)
                except Exception:
                    pass
        self.opened, self.target = [], None


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
    operation = request["operation"]
    session = request["session"]

    def call(method, **params):
        return _cdp(method, session_id=session, daemon_name=request.get("daemon_name"), **params)

    def evaluate(expression):
        result = call("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            if operation == "act" and request["action"]["kind"] == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.")
            raise StalePage("Document changed during evaluation")
        return result.get("result", {}).get("value")

    if operation == "act":
        action = request["action"]
        kind = action["kind"]
        if kind == "scroll":
            call("Input.dispatchMouseEvent", type="mouseWheel", x=550, y=650, deltaX=0, deltaY=action["delta"])
        elif kind != "wait":
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            # Code-owned node IDs refer to actual observed elements, never model-generated selectors.
            target = evaluate("""(action => {
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                  !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
              if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
              if (!e.contains(document.elementFromPoint(x,y))) return null;
              if (action.kind==='select') {
                if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                    !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                e.value=action.value;
                e.dispatchEvent(new Event('input',{bubbles:true}));
                e.dispatchEvent(new Event('change',{bubbles:true}));
              }
              return {x,y};
            })(""" + json.dumps(action) + ")")
            if target is None:
                if kind == "select":
                    raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
                raise StalePage("Target changed or is covered. Observe again.")
            if kind != "select":
                x, y = target["x"], target["y"]
                for event in ("mousePressed", "mouseReleased"):
                    call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
                if kind == "fill":
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyDown",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                        commands=["selectAll"],
                    )
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyUp",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                    )
                    call("Input.insertText", text=request["text"])
        return {"executed": action["id"]}

    info = evaluate(READ_STATE)
    if info is None:
        raise StalePage("Document is navigating")
    info["fingerprint"] = fingerprint(info)
    if request.get("screenshot", True):
        info["screenshot"] = call("Page.captureScreenshot", format="jpeg", quality=72)["data"]
    return info
