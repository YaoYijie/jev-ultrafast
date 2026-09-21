"""A run nobody is watching must not be able to approve anything.

Everything else here is bookkeeping, but that one rule is the reason the MCP tools are allowed
to exist at all, so it is the first thing that should fail if someone loosens it.
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from jev_ultrafast import runs


def fake_loop(result=None, hold=None):
    """Stand in for autopilot.run: no browser, no models, no network."""
    def inner(need, url=None, chunk=6, max_steps=40, allow_commit=True,
              say=None, ask=None, should_stop=None):
        if hold is not None:
            hold(say, ask, should_stop)
        return {"need": need, "start_url": url or "about:blank", "legs": [], "total_steps": 3,
                "elapsed_s": 0.1, "visited": [{"url": "https://e.test", "title": "T", "text": "body"}],
                "trace": [], "declined": [], "answer": "answer", **(result or {})}
    return inner


class Isolated(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for name, value in (("ARTIFACTS", root), ("INDEX", root / "index.jsonl")):
            patch = mock.patch.object(runs, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        runs._REGISTRY.clear()
        self.addCleanup(runs._REGISTRY.clear)

    def finished(self, **kwargs):
        """Wait for the record to be on disk, not merely for the status to flip."""
        run = runs.start("need", **kwargs)
        if not run.persisted.wait(timeout=5):
            raise AssertionError(f"run never finished: {run.status}")
        return run


class UnwatchedRunsCannotApprove(Isolated):
    def test_asking_for_commit_permission_is_refused_even_if_requested(self):
        run = runs.Run("need", None, 6, 40, watched=False, allow_commit=True)
        self.assertFalse(run.allow_commit)

    def test_a_question_to_an_empty_room_is_a_no_and_does_not_block(self):
        run = runs.Run("need", None, 6, 40, watched=False, allow_commit=False)
        done = []
        thread = threading.Thread(target=lambda: done.append(run.ask("提交订单？")))
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive(), "ask() blocked with nobody to answer")
        self.assertEqual(done, [False])

    def test_the_loop_is_told_commits_are_off(self):
        seen = {}

        def record(need, url=None, allow_commit=True, **kwargs):
            seen["allow_commit"] = allow_commit
            return fake_loop()(need, url=url, allow_commit=allow_commit, **kwargs)

        with mock.patch.object(runs.autopilot, "run", record):
            self.finished(watched=False, allow_commit=True)
        self.assertIs(seen["allow_commit"], False)


class WatchedRunsWaitForAPerson(Isolated):
    def test_ask_blocks_until_the_page_answers(self):
        run = runs.Run("need", None, 6, 40, watched=True, allow_commit=True)
        out = []
        thread = threading.Thread(target=lambda: out.append(run.ask("提交订单？")))
        thread.start()
        for _ in range(200):
            if run.question:
                break
            time.sleep(0.01)
        self.assertEqual(run.question, "提交订单？")
        self.assertEqual(run.status, "waiting")
        run.reply(True)
        thread.join(timeout=2)
        self.assertEqual(out, [True])
        self.assertIsNone(run.question)

    def test_answering_nothing_is_an_error_not_a_yes(self):
        run = runs.Run("need", None, 6, 40, watched=True, allow_commit=True)
        with self.assertRaises(ValueError):
            run.reply(True)


class Cancelling(Isolated):
    def test_cancel_reaches_the_loop_and_still_reports(self):
        gate = threading.Event()

        def hold(say, ask, should_stop):
            gate.wait(2)
            assert should_stop(), "cancel never reached the loop"

        with mock.patch.object(runs.autopilot, "run", fake_loop(hold=hold)):
            run = runs.start("need")
            run.cancel()
            gate.set()
            for _ in range(200):
                if run.status != "running":
                    break
                time.sleep(0.01)
        self.assertEqual(run.status, "cancelled")
        self.assertEqual(run.result["answer"], "answer")

    def test_cancel_releases_a_run_parked_on_a_question(self):
        run = runs.Run("need", None, 6, 40, watched=True, allow_commit=True)
        out = []
        thread = threading.Thread(target=lambda: out.append(run.ask("提交？")))
        thread.start()
        for _ in range(200):
            if run.question:
                break
            time.sleep(0.01)
        run.cancel()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(out, [False])


class PollingAndHistory(Isolated):
    def test_since_returns_only_what_is_new(self):
        run = runs.Run("need", None, 6, 40, watched=False, allow_commit=False)
        runs._REGISTRY[run.id] = run
        run.say("one")
        run.say("two")
        first = runs.poll(run.id)
        self.assertEqual(first["lines"], ["one", "two"])
        self.assertEqual(first["next_line"], 2)
        run.say("three")
        second = runs.poll(run.id, since=first["next_line"])
        self.assertEqual(second["lines"], ["three"])
        self.assertEqual(second["next_line"], 3)

    def test_a_finished_run_is_readable_after_it_leaves_memory(self):
        with mock.patch.object(runs.autopilot, "run", fake_loop()):
            run = self.finished()
        runs._REGISTRY.clear()
        record = runs.load(run.id)
        self.assertEqual(record["answer"], "answer")
        self.assertEqual(record["need"], "need")
        self.assertEqual(record["visited"][0]["text"], "body")

    def test_history_survives_the_process_and_is_newest_first(self):
        with mock.patch.object(runs.autopilot, "run", fake_loop()):
            first = self.finished()
            time.sleep(0.01)
            second = self.finished()
        runs._REGISTRY.clear()
        ids = [entry["run_id"] for entry in runs.history()]
        self.assertEqual(ids[:2], [second.id, first.id])

    def test_a_corrupt_index_line_does_not_lose_the_rest(self):
        with mock.patch.object(runs.autopilot, "run", fake_loop()):
            run = self.finished()
        with open(runs.INDEX, "a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        runs._REGISTRY.clear()
        self.assertEqual([e["run_id"] for e in runs.history()], [run.id])

    def test_page_text_is_opt_in_because_it_fills_a_context(self):
        run = runs.Run("need", None, 6, 40, watched=False, allow_commit=False)
        run.result = {"visited": [{"url": "u", "title": "t", "text": "x" * 4000}]}
        self.assertNotIn("text", run.record()["visited"][0])
        self.assertIn("text", run.record(text=True)["visited"][0])

    def test_too_many_at_once_is_refused_rather_than_stealing_a_tab(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        with mock.patch.object(runs.autopilot, "run", fake_loop(hold=lambda *_: gate.wait(3))):
            held = [runs.start("need") for _ in range(runs.MAX_CONCURRENT)]
            with self.assertRaises(ValueError) as caught:
                runs.start("one too many")
            self.assertIn(held[0].id, str(caught.exception))
            gate.set()
            for run in held:
                run.persisted.wait(timeout=5)
            runs.start("now there is room").persisted.wait(timeout=5)

    def test_an_empty_need_is_refused_before_a_browser_opens(self):
        with self.assertRaises(ValueError):
            runs.start("   ")


class Persistence(Isolated):
    def test_the_record_on_disk_is_the_whole_run(self):
        with mock.patch.object(runs.autopilot, "run", fake_loop()):
            run = self.finished()
        record = json.loads((runs.ARTIFACTS / f"{run.id}.json").read_text(encoding="utf-8"))
        self.assertEqual(record["run_id"], run.id)
        self.assertEqual(record["status"], "done")
        self.assertIn("lines", record)

    def test_a_crash_is_recorded_rather_than_swallowed(self):
        def boom(*_args, **_kwargs):
            raise RuntimeError("supervisor died")

        with mock.patch.object(runs.autopilot, "run", boom):
            run = self.finished()
        self.assertEqual(run.status, "error")
        self.assertIn("supervisor died", run.error)
        self.assertIn("supervisor died", runs.load(run.id)["error"])


if __name__ == "__main__":
    unittest.main()
