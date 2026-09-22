"""Chrome process lifecycle manager for Jev Ultrafast automation.

Two things here are easy to get wrong and expensive to debug. A port can be held by a process
that answers the connection and nothing else, so reachability on a hardcoded host proves nothing.
And a Chrome answering on the expected port is not necessarily the Chrome that was meant: attach
to the wrong one and the run gets a clean profile that is logged into nothing, which looks like
every site suddenly demanding a login rather than like a wiring mistake.
"""

import json
import os
import platform
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from browser_harness.admin import daemon_browser_kind
from browser_harness.admin import ensure_daemon as ensure_harness_daemon
from browser_harness.admin import require_existing_daemon as require_harness_daemon

# One Chrome binds 127.0.0.1, the next finds IPv4 taken and binds [::1]. Both are "port 9222".
CDP_HOSTS = ("127.0.0.1", "[::1]", "localhost")

HARNESS_PROFILE = Path.home() / ".config" / "browser-harness" / "chrome-profile"
_HARNESS_ENV_LOCK = threading.Lock()
_REMOTE_BROWSER_ENV = ("BU_CDP_URL", "BU_CDP_WS", "BU_BROWSER_ID")


def _version(base: str, timeout: float = 1.5) -> dict | None:
    """The /json/version of a Chrome really speaking CDP at this base URL, else None."""
    try:
        with urllib.request.urlopen(f"{base}/json/version", timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None
    return data if ("Browser" in data or "webSocketDebuggerUrl" in data) else None


def cdp_endpoint(port: int = 9222) -> str | None:
    """Base URL of whichever Chrome actually answers CDP on this port, or None."""
    for host in CDP_HOSTS:
        base = f"http://{host}:{port}"
        if _version(base):
            return base
    return None


def is_chrome_cdp_ready(port: int = 9222) -> bool:
    return cdp_endpoint(port) is not None


def get_default_chrome_path() -> str:
    system = platform.system()
    if system == "Darwin":
        candidate = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.exists(candidate):
            return candidate
        # Fallback to Chromium or Canary
        for alt in [
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]:
            if os.path.exists(alt):
                return alt
    elif system == "Linux":
        for bin_name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
            path = subprocess.getoutput(f"which {bin_name} 2>/dev/null").strip()
            if path and os.path.exists(path):
                return path
    elif system == "Windows":
        for cand in [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]:
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError("Could not find Google Chrome binary on this system.")


def default_user_data_dir() -> Path | None:
    """Where the user's real Chrome keeps its profile: the one carrying their logins."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    if system == "Linux":
        return Path.home() / ".config" / "google-chrome"
    if system == "Windows":
        return Path(os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data"))
    return None


def get_user_chrome_port_file() -> Path | None:
    root = default_user_data_dir()
    if root is None:
        return None
    port_file = root / "DevToolsActivePort"
    return port_file if port_file.exists() else None


def user_chrome_endpoint() -> str | None:
    """The user's own Chrome, verified by identity rather than by port.

    DevToolsActivePort holds a port and that browser's GUID. The port alone proves nothing: it
    outlives the Chrome that wrote it, and a different Chrome is usually sitting on the same port
    by then. Matching the GUID against webSocketDebuggerUrl is what makes this the user's browser
    instead of whoever happened to answer.
    """
    port_file = get_user_chrome_port_file()
    if port_file is None:
        return None
    try:
        lines = port_file.read_text().splitlines()
        port, guid = int(lines[0].strip()), lines[1].strip()
    except (OSError, ValueError, IndexError):
        return None
    if not guid:
        return None
    for host in CDP_HOSTS:
        base = f"http://{host}:{port}"
        data = _version(base)
        if data and str(data.get("webSocketDebuggerUrl", "")).endswith(guid):
            return base
    return None


def is_user_chrome_ready() -> bool:
    return user_chrome_endpoint() is not None


def chosen_user_data_dir(profile_dir: str | None = None) -> Path:
    """Which profile a launch should use.

    JEV_CHROME_USER_DATA_DIR=user means the real, logged-in profile; a path means that path;
    unset keeps the isolated harness profile, which is logged into nothing.
    """
    if profile_dir:
        return Path(profile_dir).expanduser()
    configured = (os.environ.get("JEV_CHROME_USER_DATA_DIR") or "").strip()
    if not configured:
        return HARNESS_PROFILE
    if configured.lower() in {"user", "default", "real"}:
        real = default_user_data_dir()
        if real is None:
            raise RuntimeError(
                "JEV_CHROME_USER_DATA_DIR=user，但这个平台上找不到 Chrome 的默认 profile 目录。"
                "请直接写绝对路径。")
        return real
    return Path(configured).expanduser()


def profile_holder(user_data_dir: Path) -> str | None:
    """Chrome locks a profile to one process; SingletonLock names who holds it."""
    try:
        return os.readlink(user_data_dir / "SingletonLock")
    except OSError:
        return None


def _relaunch_instructions(chrome_bin: str, port: int, profile: Path, holder: str | None) -> str:
    held = f"（SingletonLock -> {holder}）" if holder else ""
    return (
        f"需要用这个 profile 的登录态，但它已经被一个没开远程调试的 Chrome 占着{held}：\n"
        f"  {profile}\n"
        f"Chrome 不允许两个进程共用一个 user-data-dir，所以只能先退出再带调试端口重开：\n"
        f"  1. 完全退出 Chrome（Cmd-Q，确认没有残留进程）\n"
        f'  2. "{chrome_bin}" --remote-debugging-port={port} &\n'
        f"重开之后这里会自动认出它。"
    )


def ensure_chrome_running(port: int = 9222, profile_dir: str | None = None,
                          headless: bool = False) -> str:
    """Ensure a CDP-speaking Chrome, preferring the profile that carries the user's logins."""
    # 1. The user's own Chrome, identity-checked. Nothing beats this: it has the logins.
    endpoint = user_chrome_endpoint()
    if endpoint:
        os.environ["BU_CDP_URL"] = endpoint
        return endpoint

    chrome_bin = get_default_chrome_path()
    profile_path = chosen_user_data_dir(profile_dir)
    real = default_user_data_dir()
    wants_logins = real is not None and profile_path == real

    # 2. Something else is on the port. Reusing it is right for a throwaway profile and wrong
    #    when logins were asked for: a silent attach to the wrong Chrome is the whole bug.
    endpoint = cdp_endpoint(port)
    if endpoint and not wants_logins:
        os.environ["BU_CDP_URL"] = endpoint
        return endpoint
    if endpoint:
        raise RuntimeError(
            f"{port} 端口上有 Chrome 在应答，但它不是你那个登录过的 profile（"
            f"DevToolsActivePort 里的 GUID 对不上）。直接用它会得到一个什么都没登录的浏览器。\n"
            + _relaunch_instructions(chrome_bin, port, profile_path, profile_holder(profile_path))
        )

    holder = profile_holder(profile_path)
    if holder:
        raise RuntimeError(_relaunch_instructions(chrome_bin, port, profile_path, holder))

    profile_path.mkdir(parents=True, exist_ok=True)
    args = [
        chrome_bin,
        f"--remote-debugging-port={port}",
        "--remote-debugging-address=127.0.0.1",
        f"--user-data-dir={profile_path}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    if headless:
        args.append("--headless=new")

    # Launch daemon in a separate session so it survives child exits
    subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.time() + 10
    while time.time() < deadline:
        endpoint = user_chrome_endpoint() if wants_logins else cdp_endpoint(port)
        if endpoint:
            os.environ["BU_CDP_URL"] = endpoint
            return endpoint
        time.sleep(0.3)

    raise RuntimeError(f"Chrome failed to start and bind CDP on port {port} within 10s.")


def _note(connected: bool, wants_logins: bool) -> str:
    """Three states, not two: configured for the real profile but unreachable is its own case."""
    if connected:
        return "在用你自己的 Chrome，登录态可用。"
    if wants_logins:
        return ("已配置成用你自己的 profile，但它现在没开远程调试，连不上。"
                "退出 Chrome 后用 --remote-debugging-port=9222 重开即可。")
    return "在用独立 profile，没有任何登录态；需要登录的站点会被门禁挡下。"


def ensure_user_browser(
    daemon_name: str,
    daemon_env: dict[str, str],
) -> dict:
    """Connect a named Browser Harness daemon to the user's visible local Chrome."""
    # Browser Harness 0.1.13 checks both its explicit env and this process's env when deciding
    # whether to run the local Chrome permission flow. Mask inherited remote endpoints only while
    # starting the isolated named daemon, then restore them for ordinary sessions.
    with _HARNESS_ENV_LOCK:
        saved = {key: os.environ.pop(key) for key in _REMOTE_BROWSER_ENV if key in os.environ}
        try:
            ensure_harness_daemon(wait=20, name=daemon_name, env=daemon_env)
        finally:
            os.environ.update(saved)
    kind = daemon_browser_kind(daemon_name)
    if kind != "local":
        raise RuntimeError(
            "job_patrol 必须通过 Browser Harness 连接用户自己的可见 Chrome；"
            f"当前 daemon 类型为 {kind or 'unknown'}"
        )
    # Browser Harness's connection_status also requires its previously attached tab to exist. Jev
    # does not use that tab: every patrol creates and owns a fresh target. The old tab can therefore
    # be gone while the browser transport is healthy (Target.getTargets succeeds), even though
    # daemon_browser_ready reports false for the now-closed old tab. Verify the browser-level CDP
    # transport instead, or every later patrol is rejected before it can create its own tab.
    require_harness_daemon(daemon_name)
    profile = default_user_data_dir()
    return {
        "endpoint": None,
        "is_user_profile": True,
        "profile_dir": str(profile) if profile else None,
        "profile_has_logins": True,
        "port_answered_by_someone_else": False,
        "browser_kind": kind,
        "daemon_name": daemon_name,
        "note": "Browser Harness 已连接你当前可见的本地 Chrome，登录态可用。",
    }


def describe(port: int = 9222) -> dict:
    """What a run is actually about to drive — above all, whether it has the user's logins."""
    user = user_chrome_endpoint()
    any_cdp = cdp_endpoint(port)
    real = default_user_data_dir()
    # When identity verification proves the user's Chrome is connected, report that actual
    # profile rather than the configured fallback that would be used for a future launch.
    profile = real if user and real is not None else chosen_user_data_dir()
    return {
        "endpoint": user or any_cdp,
        "is_user_profile": bool(user),
        "profile_dir": str(profile),
        "profile_has_logins": real is not None and profile == real,
        "port_answered_by_someone_else": bool(any_cdp and not user),
        "note": _note(bool(user), real is not None and profile == real),
    }


if __name__ == "__main__":
    print(json.dumps(describe(), ensure_ascii=False, indent=2))
    print(f"Chrome CDP active at {ensure_chrome_running()}")
