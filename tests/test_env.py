"""Config that arrives mangled is worse than config that is missing: it fails somewhere else.

`uv run --env-file` strips unquoted double quotes, so a JSON header map written bare in .env
reaches the process as {k:v} and never parses. Quoting the value fixes that path and breaks the
other one — reading .env directly — unless the wrapper quotes are stripped there too.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jev_ultrafast import session
from jev_ultrafast.model import extra_headers


class LoadEnv(unittest.TestCase):
    def load(self, text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / ".env").write_text(text)
        with mock.patch.object(session, "ROOT", root), \
                mock.patch.dict("os.environ", {}, clear=True):
            session.load_env()
            import os
            return dict(os.environ)

    def test_wrapper_quotes_are_not_part_of_the_value(self):
        for quote in ("'", '"'):
            got = self.load(f'A_HEADERS={quote}{{"k": "v"}}{quote}\n')
            self.assertEqual(got["A_HEADERS"], '{"k": "v"}', quote)

    def test_an_unquoted_value_is_left_alone(self):
        self.assertEqual(self.load("MODEL=deepseek-v4.1-flash\n")["MODEL"], "deepseek-v4.1-flash")

    def test_a_lone_quote_is_not_treated_as_a_wrapper(self):
        self.assertEqual(self.load("X=\"unbalanced\n")["X"], '"unbalanced')

    def test_comments_and_blanks_are_skipped(self):
        got = self.load("# note\n\nK=v\n")
        self.assertEqual(got["K"], "v")
        self.assertNotIn("# note", got)


class HeaderParsing(unittest.TestCase):
    def test_a_quoted_json_object_round_trips(self):
        with mock.patch.dict("os.environ", {"T_HEADERS": '{"k": "v"}'}):
            self.assertEqual(extra_headers("T"), {"k": "v"})

    def test_the_stripped_quote_case_says_how_to_fix_it(self):
        # What uv actually delivers for T_HEADERS={"k":"v"}.
        with mock.patch.dict("os.environ", {"T_HEADERS": "{k:v}"}):
            with self.assertRaises(ValueError) as caught:
                extra_headers("T")
        self.assertIn("single quotes", str(caught.exception))

    def test_a_json_scalar_is_still_refused(self):
        with mock.patch.dict("os.environ", {"T_HEADERS": '"nope"'}):
            with self.assertRaises(ValueError):
                extra_headers("T")

    def test_no_variable_means_no_headers(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(extra_headers("T", "U"), {})


if __name__ == "__main__":
    unittest.main()
