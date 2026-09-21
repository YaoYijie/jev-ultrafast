"""Chrome process lifecycle manager for Jev Ultrafast automation."""

import json
import os
import platform
import subprocess
import time
import urllib.request
from pathlib import Path


def is_chrome_cdp_ready(port: int = 9222) -> bool:
    for host in ["localhost", "127.0.0.1", "[::1]"]:
        try:
            req = urllib.request.Request(f"http://{host}:{port}/json/version")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode())
                if "Browser" in data or "webSocketDebuggerUrl" in data:
                    return True
        except Exception:
            continue
    return False


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


def get_user_chrome_port_file() -> Path | None:
    system = platform.system()
    if system == "Darwin":
        p = Path.home() / "Library/Application Support/Google/Chrome/DevToolsActivePort"
        if p.exists():
            return p
    elif system == "Linux":
        p = Path.home() / ".config/google-chrome/DevToolsActivePort"
        if p.exists():
            return p
    return None


def is_user_chrome_ready() -> bool:
    port_file = get_user_chrome_port_file()
    if not port_file:
        return False
    try:
        lines = port_file.read_text().splitlines()
        if not lines:
            return False
        port = int(lines[0].strip())
        req = urllib.request.Request(f"http://127.0.0.1:{port}/json/version")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode())
            return "Browser" in data or "webSocketDebuggerUrl" in data
    except Exception:
        return False


def ensure_chrome_running(port: int = 9222, profile_dir: str | None = None, headless: bool = False) -> str:
    """Ensures Chrome CDP is ready. Prioritizes user's already open Chrome profile."""
    # 1. If user's existing Chrome has remote debugging turned on, use it directly!
    if is_user_chrome_ready():
        os.environ.pop("BU_CDP_URL", None)
        return "user_default_chrome"

    endpoint = f"http://localhost:{port}"
    os.environ["BU_CDP_URL"] = endpoint

    if is_chrome_cdp_ready(port):
        return endpoint

    chrome_bin = get_default_chrome_path()
    if not profile_dir:
        profile_path = Path.home() / ".config" / "browser-harness" / "chrome-profile"
    else:
        profile_path = Path(profile_dir)
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
        if is_chrome_cdp_ready(port):
            return endpoint
        time.sleep(0.3)

    raise RuntimeError(f"Chrome failed to start and bind CDP on port {port} within 10s.")


if __name__ == "__main__":
    url = ensure_chrome_running()
    print(f"Chrome CDP active at {url}")
