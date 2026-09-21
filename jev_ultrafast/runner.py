"""One-shot CLI over a supervised Jev session, for smoke tests outside MCP.

Agents drive the browser through the MCP tools in jev_ultrafast.mcp_server, not through this
script. It exists so a run can be reproduced and watched from a terminal. It never asks a model
to read the page: it prints Jev's decisions and the raw page text, and you do the extraction.
"""

import argparse
import json
import sys

from . import session as sessions

HANDBACK = {"low_confidence", "stalled", "step_budget", "stale_page", "error"}
TERMINAL = {"done", "blocked", "finished"}


def run_browse_task(
    url: str,
    goal: str,
    max_steps: int = 20,
    chunk: int = 3,
    min_confidence: float = sessions.MIN_CONFIDENCE,
    stall_limit: int = sessions.STALL_LIMIT,
    keep_open: bool = False,
    on_report=None,
) -> dict:
    """Advance one goal to a handback or a terminal state, reporting every checkpoint."""
    session = sessions.start(url, goal)
    reports = []
    try:
        while session.total_steps < max_steps:
            report = session.step(steps=chunk, min_confidence=min_confidence, stall_limit=stall_limit)
            reports.append(report)
            if on_report:
                on_report(report)
            if report["stop_reason"] in HANDBACK | TERMINAL:
                break
        final = session.brief(reports[-1]["stop_reason"] if reports else "steps_exhausted")
        page = session.agent.state["page"]
        return {
            "session_id": session.id,
            "success": final["status"] == "done",
            "status": final["status"],
            "stop_reason": final["stop_reason"],
            "url": page.get("url"),
            "title": page.get("title"),
            "total_steps": session.total_steps,
            "checkpoints": reports,
            "page_text": page.get("text", ""),
            "screenshot_path": final["observation"]["screenshot_path"],
            "pending_decision": final.get("pending_decision"),
        }
    finally:
        if not keep_open:
            sessions.finish(session.id)


def _print_report(report: dict) -> None:
    print(f"\n--- checkpoint: {report['stop_reason']} (status={report['status']}) ---", file=sys.stderr)
    for step in report["steps_executed"]:
        if "terminal" in step:
            print(f"  [terminal] {step['terminal']} conf={step['confidence']}", file=sys.stderr)
            continue
        changed = "page changed" if step["page_changed"] else "no change"
        text = f" text={step['text']!r}" if step.get("text") else ""
        print(
            f"  [{step['step']:2d}] {step['operation']:9s} conf={step['confidence']:.2f} "
            f"{changed:11s} {step['action'][:60]}{text}",
            file=sys.stderr,
        )
    if report.get("pending_decision"):
        pending = report["pending_decision"]
        print(
            f"  [held back] {pending['operation']} -> {pending['target_label']!r} "
            f"conf={pending['confidence']}",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(description="Jev Ultrafast supervised runner (smoke test CLI)")
    parser.add_argument("--url", required=True, help="Starting URL")
    parser.add_argument("--goal", required=True, help="One atomic sub-goal for this leg")
    parser.add_argument("--max-steps", type=int, default=20, help="Total actions before stopping (default 20)")
    parser.add_argument("--chunk", type=int, default=3, help="Actions per checkpoint (default 3)")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=sessions.MIN_CONFIDENCE,
        help=f"Hold back choices below this confidence (default {sessions.MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--stall-limit",
        type=int,
        default=sessions.STALL_LIMIT,
        help=f"Stop after this many actions that do not change the page (default {sessions.STALL_LIMIT})",
    )
    parser.add_argument("--text-chars", type=int, default=4000, help="Page text to print in the JSON result")
    parser.add_argument("--keep-open", action="store_true", help="Leave the tab open after finishing")
    args = parser.parse_args()

    result = run_browse_task(
        url=args.url,
        goal=args.goal,
        max_steps=args.max_steps,
        chunk=args.chunk,
        min_confidence=args.min_confidence,
        stall_limit=args.stall_limit,
        keep_open=args.keep_open,
        on_report=_print_report,
    )
    result["page_text"] = result["page_text"][: args.text_chars]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
