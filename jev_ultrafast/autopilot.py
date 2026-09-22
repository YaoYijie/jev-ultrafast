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

from . import job_patrol, launcher, supervisor
from . import session as sessions

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


def _refusal(index: int, report: dict, pending: dict, why: str) -> dict:
    """What the gate stopped, in the report. An unattended run's whole value is saying this."""
    return {
        "leg": index + 1,
        "why": why,
        "operation": pending.get("operation"),
        "target_label": pending.get("target_label"),
        "gate_reason": pending.get("gate_reason"),
        "url": report.get("observation", {}).get("url"),
    }


def _collect(store: dict, session) -> None:
    page = session.agent.state["page"]
    url = page.get("url")
    if not url or url.startswith("about:"):
        return
    text = page.get("text", "")
    kept = store.get(url)
    if kept is None or len(text) > len(kept["text"]):
        store[url] = {"url": url, "title": page.get("title"), "text": text[:4000]}


def _show(report: dict, say=_say) -> None:
    for entry in report.get("steps_executed", []):
        if "step" in entry:
            changed = "→" if entry["page_changed"] else "·"
            text = f"  «{entry['text']}»" if entry.get("text") else ""
            _say(f"   {changed} [{entry['step']:2d}] {entry['operation']:11s} "
                 f"{entry['confidence']:.2f}  {entry['action'][:52]}{text}")
        else:
            _say(f"   ■ {entry['terminal']} ({entry['confidence']:.2f})")


def run(
    need: str,
    url: str | None = None,
    chunk: int = 6,
    max_steps: int = 40,
    allow_commit: bool = True,
    say=None,
    ask=None,
    should_stop=None,
    mode: str = "standard",
) -> dict:
    # The CLI talks to a terminal and the web UI talks to a browser; the loop itself should not
    # care which, and must never reach for stdin on its own.
    say = say or _say
    ask = ask or _ask
    # A run started in the background needs a way out that is not killing the process; a cancelled
    # run still reports, because the pages it already read are worth as much as before.
    should_stop = should_stop or (lambda: False)
    started = time.time()
    sessions.load_env()
    platform = None
    if mode == job_patrol.MODE:
        if not url:
            raise ValueError("job_patrol 必须提供一个明确的招聘平台起始 URL")
        platform = job_patrol.require_platform_url(url)
        chunk = min(max(1, int(chunk)), 2)
        max_steps = min(max(1, int(max_steps)), 20)
        allow_commit = False
        browser = sessions.prepare(mode)
    elif mode == "standard":
        browser = launcher.describe()
    else:
        raise ValueError(f"Unknown autopilot mode: {mode}")
    say(f"需求: {need}")
    say(f"监督模型: {supervisor.model_name()}    填值模型: {os.environ.get('TEXT_MODEL', '?')}")
    # Which browser this drives decides whether half the web is even reachable, so it is the
    # first thing the log says rather than something to infer from a wall of login pages.
    say(f"浏览器: {browser['note']}")
    say("规划中…")
    try:
        plan = supervisor.plan(need, url)
    except Exception as exc:
        if not url:
            raise
        say(f"   ⚠ 规划失败（{type(exc).__name__}），退回单段模式。")
        plan = {"start_url": url, "legs": [{"goal": need}]}
    if mode == job_patrol.MODE:
        # The supervisor may propose a different site. A patrol stays on the exact platform the
        # caller named and never treats another recruiting site as a fallback.
        plan["start_url"] = url
    legs = plan["legs"]
    for i, leg in enumerate(legs, 1):
        say(f"   {i}. {leg['goal']}")

    session = sessions.start(
        plan["start_url"],
        legs[0]["goal"],
        mode=mode,
        allowed_platform=platform,
        action_delay_s=2.0 if mode == job_patrol.MODE else 0,
        step_budget=max_steps,
    )
    say(f"\n会话 {session.id} @ {plan['start_url']}")
    collected: dict = {}
    trace: list = []
    tried: set = {plan["start_url"]}
    approved = None
    index = 0
    finished = False
    declined: list = []
    platform_stop: dict | None = None

    try:
        while index < len(legs) and session.total_steps < max_steps:
            if should_stop():
                say("\n✗ 已取消，用已经读到的页面汇总。")
                break
            say(f"\n── leg {index + 1}/{len(legs)}: {legs[index]['goal']}")
            idle = 0
            hops = 0 if mode == job_patrol.MODE else NAVIGATIONS_PER_LEG
            for _ in range(CHECKPOINTS_PER_LEG):
                if should_stop():
                    break
                before = session.total_steps
                report = session.step(steps=chunk, gated=True, approved=approved)
                approved = None
                _show(report, say)
                _collect(collected, session)
                trace.append({
                    "leg": index + 1,
                    "stop_reason": report["stop_reason"],
                    "steps": report["steps_executed"],
                    "blocked_reason": report.get("blocked_reason"),
                })
                stop = report["stop_reason"]
                pending = report.get("pending_decision") or {}
                say(f"   ⟂ {stop}")

                if mode == job_patrol.MODE and stop in {"platform_blocked", "left_platform"}:
                    platform_stop = {
                        "reason": stop,
                        "detail": report.get("blocked_reason"),
                        "url": report.get("observation", {}).get("url"),
                    }
                    say(f"   ⛔ {platform_stop['detail']}")
                    say("      本轮停止该平台，不换站点、账号、IP 或工具继续访问。")
                    finished = True
                    break

                if stop == "blocked_action":
                    say(f"   ⛔ {pending.get('gate_reason', '')}")
                    declined.append(_refusal(index, report, pending, "只读门禁拒绝"))
                    if mode == job_patrol.MODE:
                        platform_stop = {
                            "reason": "read_only_action",
                            "detail": pending.get("gate_reason"),
                            "url": report.get("observation", {}).get("url"),
                        }
                        say("      岗位巡检不会人工补做或批准该动作，本次运行到此为止。")
                        finished = True
                        break
                    say("      这类字段不代填。请在 Chrome 里自己填好。")
                    if not ask("      填好了，继续这一段？"):
                        break
                    continue

                if stop == "needs_confirmation":
                    say(f"   ⏸  {pending.get('gate_reason', '')}")
                    say(f"      动作: {pending.get('operation')} → {pending.get('target_label')}")
                    say(f"      页面: {report['observation']['url']}")
                    if not allow_commit:
                        say("      未获授权执行提交类动作，停在这里。")
                        declined.append(_refusal(index, report, pending, "未获授权"))
                        if mode == job_patrol.MODE:
                            platform_stop = {
                                "reason": "read_only_action",
                                "detail": pending.get("gate_reason"),
                                "url": report.get("observation", {}).get("url"),
                            }
                            finished = True
                        break
                    if ask("      执行这个动作？"):
                        approved = pending.get("target_label")
                        continue
                    say("      已拒绝，转交监督者处理。")
                    declined.append(_refusal(index, report, pending, "人工拒绝"))

                idle = idle + 1 if session.total_steps == before else 0
                try:
                    verdict = supervisor.judge(need, legs[index]["goal"], report, len(legs) - index - 1,
                                               tried=sorted(tried), navigations_left=hops)
                except Exception as exc:
                    # Losing the supervisor must not throw away a browsing run that is going fine.
                    say(f"   ⚠ 监督模型调用失败（{type(exc).__name__}），本段到此为止。")
                    break
                say(f"   ⟳ {verdict['action']}  {verdict['note']}")
                if idle >= IDLE_CHECKPOINTS and verdict["action"] not in {"navigate", "finish"}:
                    say(f"   ✗ 连续 {idle} 个检查点没有执行任何动作，放弃这一段。")
                    break
                if verdict["action"] == "continue" and stop not in {"steps_exhausted", "stale_page"}:
                    # More of the same goal cannot help here. The detectors already proved it for
                    # looping and stalled, and after done or blocked the session refuses to step
                    # at all until the goal is replaced, so "continue" executes literally nothing.
                    say(f"   ⚠ {stop} 之后 continue 不会执行任何动作，改为换目标。")
                    verdict["action"] = "retarget" if verdict["goal"] else "next"
                if verdict["action"] == "continue":
                    continue
                if verdict["action"] == "force" and not pending:
                    # force only releases a held-back low-confidence choice. A weak supervisor
                    # reaches for it to mean "type something else", which it cannot do.
                    say("   ⚠ 没有被扣住的决策，force 无效，改为换目标。")
                    verdict["action"] = "retarget" if verdict["goal"] else "next"
                if verdict["action"] == "force":
                    approved = pending.get("target_label")
                    report = session.step(steps=1, force=True, gated=True, approved=approved)
                    approved = None
                    _show(report, say)
                    _collect(collected, session)
                    continue
                if verdict["action"] == "navigate" and verdict["url"] and hops > 0:
                    say(f"   ↪ 换站点: {verdict['url']}")
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
                say("   本段检查点用尽，进入下一段。")

            if finished:
                break
            index += 1
            if index < len(legs):
                session.retarget(legs[index]["goal"])

        say(f"\n共 {session.total_steps} 步，访问 {len(collected)} 个页面，汇总中…")
        try:
            answer = supervisor.write_report(need, list(collected.values()))
        except Exception as exc:
            answer = (f"（汇总调用失败：{type(exc).__name__}。以下是实际访问到的页面，原文在 --json 记录里。）\n"
                      + "\n".join(f"- {c['title']} — {c['url']}" for c in collected.values()))
    finally:
        sessions.finish(session.id)

    return {
        "need": need,
        "mode": mode,
        "platform": platform,
        "browser": browser,
        "platform_stop": platform_stop,
        "start_url": plan["start_url"],
        "legs": legs,
        "total_steps": session.total_steps,
        "elapsed_s": round(time.time() - started, 1),
        "visited": list(collected.values()),
        "trace": trace,
        "declined": declined,
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
    parser.add_argument(
        "--job-patrol",
        action="store_true",
        help="招聘平台只读巡检：固定单平台、复用用户 Chrome、限速并在风险信号处停止",
    )
    parser.add_argument("--json", dest="json_path", default=None, help="把完整记录写到这个文件")
    args = parser.parse_args()

    result = run(args.need, url=args.url, chunk=args.chunk, max_steps=args.max_steps,
                 allow_commit=not args.no_commit,
                 mode=job_patrol.MODE if args.job_patrol else "standard")
    for item in result.get("declined", []):
        _say(f"⛔ 未执行: {item['operation']} → {item['target_label']}  ({item['why']})  {item['url']}")
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        _say(f"记录已写入 {args.json_path}")
    print("\n" + result["answer"])


if __name__ == "__main__":
    main()
