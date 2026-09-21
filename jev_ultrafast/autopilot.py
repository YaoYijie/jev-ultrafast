"""Standalone driver: state a need, Jev browses, you are asked before anything commits.

The division of labour this program exists to keep: the supervisor decides what to cover and
what to do at a checkpoint, Jev decides every click, and a deterministic gate decides what may
run without asking. Nothing here chooses an element.
"""

import argparse
import json
import os
import sys
import time

from . import session as sessions
from . import supervisor

CHECKPOINTS_PER_LEG = 6
# Checkpoints that execute nothing are pure cost: one failed run spent 16 model calls and
# zero browser actions restating the same goal on a site that could not do the job.
IDLE_CHECKPOINTS = 3
# One trial run bounced between four flight sites because an impossible leg reads like a wrong
# site. One change of site per leg is enough to correct a genuinely wrong one.
NAVIGATIONS_PER_LEG = 1


def _say(*parts):
    print(*parts, file=sys.stderr, flush=True)


def _ask(question: str) -> bool:
    if not sys.stdin.isatty():
        _say("   stdin 不是终端，按拒绝处理。")
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return False


def _collect(store: dict, session) -> None:
    page = session.agent.state["page"]
    url = page.get("url")
    if not url or url.startswith("about:"):
        return
    text = page.get("text", "")
    kept = store.get(url)
    if kept is None or len(text) > len(kept["text"]):
        store[url] = {"url": url, "title": page.get("title"), "text": text[:4000]}


def _show(report: dict) -> None:
    for entry in report.get("steps_executed", []):
        if "step" in entry:
            changed = "→" if entry["page_changed"] else "·"
            text = f"  «{entry['text']}»" if entry.get("text") else ""
            _say(f"   {changed} [{entry['step']:2d}] {entry['operation']:11s} "
                 f"{entry['confidence']:.2f}  {entry['action'][:52]}{text}")
        else:
            _say(f"   ■ {entry['terminal']} ({entry['confidence']:.2f})")


def run(need: str, url: str | None = None, chunk: int = 6, max_steps: int = 40,
        allow_commit: bool = True) -> dict:
    started = time.time()
    sessions.load_env()
    _say(f"需求: {need}")
    _say(f"监督模型: {supervisor.model_name()}    填值模型: {os.environ.get('TEXT_MODEL', '?')}")
    _say("规划中…")
    plan = supervisor.plan(need, url)
    legs = plan["legs"]
    for i, leg in enumerate(legs, 1):
        _say(f"   {i}. {leg['goal']}")

    session = sessions.start(plan["start_url"], legs[0]["goal"])
    _say(f"\n会话 {session.id} @ {plan['start_url']}")
    collected: dict = {}
    trace: list = []
    tried: set = {plan["start_url"]}
    approved = None
    index = 0
    finished = False

    try:
        while index < len(legs) and session.total_steps < max_steps:
            _say(f"\n── leg {index + 1}/{len(legs)}: {legs[index]['goal']}")
            idle = 0
            hops = NAVIGATIONS_PER_LEG
            for _ in range(CHECKPOINTS_PER_LEG):
                before = session.total_steps
                report = session.step(steps=chunk, gated=True, approved=approved)
                approved = None
                _show(report)
                _collect(collected, session)
                trace.append({"leg": index + 1, "stop_reason": report["stop_reason"],
                              "steps": report["steps_executed"]})
                stop = report["stop_reason"]
                pending = report.get("pending_decision") or {}
                _say(f"   ⟂ {stop}")

                if stop == "blocked_action":
                    _say(f"   ⛔ {pending.get('gate_reason', '')}")
                    _say("      这类字段不代填。请在 Chrome 里自己填好。")
                    if not _ask("      填好了，继续这一段？"):
                        break
                    continue

                if stop == "needs_confirmation":
                    _say(f"   ⏸  {pending.get('gate_reason', '')}")
                    _say(f"      动作: {pending.get('operation')} → {pending.get('target_label')}")
                    _say(f"      页面: {report['observation']['url']}")
                    if not allow_commit:
                        _say("      --no-commit 模式，已拒绝。")
                        break
                    if _ask("      执行这个动作？"):
                        approved = pending.get("target_label")
                        continue
                    _say("      已拒绝，转交监督者处理。")

                idle = idle + 1 if session.total_steps == before else 0
                verdict = supervisor.judge(need, legs[index]["goal"], report, len(legs) - index - 1,
                                           tried=sorted(tried), navigations_left=hops)
                _say(f"   ⟳ {verdict['action']}  {verdict['note']}")
                if idle >= IDLE_CHECKPOINTS and verdict["action"] not in {"navigate", "finish"}:
                    _say(f"   ✗ 连续 {idle} 个检查点没有执行任何动作，放弃这一段。")
                    break
                if verdict["action"] == "continue":
                    continue
                if verdict["action"] == "continue" and stop in {"looping", "stalled"}:
                    # The loop and stall detectors already proved more of the same goal does not
                    # advance. "continue" here just re-enters the loop the detector caught.
                    _say(f"   ⚠ {stop} 之后不能 continue，改为换目标。")
                    verdict["action"] = "retarget" if verdict["goal"] else "next"
                if verdict["action"] == "force" and not pending:
                    # force only releases a held-back low-confidence choice. A weak supervisor
                    # reaches for it to mean "type something else", which it cannot do.
                    _say("   ⚠ 没有被扣住的决策，force 无效，改为换目标。")
                    verdict["action"] = "retarget" if verdict["goal"] else "next"
                if verdict["action"] == "force":
                    approved = pending.get("target_label")
                    report = session.step(steps=1, force=True, gated=True, approved=approved)
                    approved = None
                    _show(report)
                    _collect(collected, session)
                    continue
                if verdict["action"] == "navigate" and verdict["url"] and hops > 0:
                    _say(f"   ↪ 换站点: {verdict['url']}")
                    tried.add(verdict["url"])
                    hops -= 1
                    session.navigate(verdict["url"], verdict["goal"] or None)
                    legs[index] = {"goal": session.agent.state["goal"]}
                    idle = 0
                    continue
                if verdict["action"] == "retarget" and verdict["goal"]:
                    session.retarget(verdict["goal"])
                    legs[index] = {"goal": verdict["goal"]}
                    continue
                finished = verdict["action"] == "finish"
                break
            else:
                _say("   本段检查点用尽，进入下一段。")

            if finished:
                break
            index += 1
            if index < len(legs):
                session.retarget(legs[index]["goal"])

        _say(f"\n共 {session.total_steps} 步，访问 {len(collected)} 个页面，汇总中…")
        answer = supervisor.write_report(need, list(collected.values()))
    finally:
        sessions.finish(session.id)

    return {
        "need": need,
        "start_url": plan["start_url"],
        "legs": legs,
        "total_steps": session.total_steps,
        "elapsed_s": round(time.time() - started, 1),
        "visited": list(collected.values()),
        "trace": trace,
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="用自然语言描述需求，Jev 自主操作浏览器；任何提交类动作会先问你。")
    parser.add_argument("need", help="你的需求，自然语言")
    parser.add_argument("--url", default=None, help="起始网址（不给则由规划器决定）")
    parser.add_argument("--chunk", type=int, default=6, help="每个检查点之间让 Jev 跑几步（默认 6）")
    parser.add_argument("--max-steps", type=int, default=40, help="整个任务的动作上限（默认 40）")
    parser.add_argument("--no-commit", action="store_true",
                        help="只读运行：遇到提交类动作直接拒绝，不询问")
    parser.add_argument("--json", dest="json_path", default=None, help="把完整记录写到这个文件")
    args = parser.parse_args()

    result = run(args.need, url=args.url, chunk=args.chunk, max_steps=args.max_steps,
                 allow_commit=not args.no_commit)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        _say(f"记录已写入 {args.json_path}")
    print("\n" + result["answer"])


if __name__ == "__main__":
    main()
