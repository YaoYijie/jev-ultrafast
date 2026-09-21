"""Attaching to the wrong Chrome is silent: the run just finds itself logged into nothing.

These tests are mostly about refusing to do that. Reachability on a port is not identity, and a
stale DevToolsActivePort plus a second Chrome on the same port is the normal case, not an edge one.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jev_ultrafast import launcher

GUID = "81b195e1-f0b0-4bbf-b31f-3a50ada142c8"
OTHER = "fbaa4cbf-0692-4a86-b979-87444ec52f1b"


def answers(mapping):
    """Fake _version: only the listed base URLs speak CDP, each with its own browser GUID."""
    def inner(base, timeout=1.5):
        guid = mapping.get(base)
        return None if guid is None else {
            "Browser": "Chrome/153", "webSocketDebuggerUrl": f"ws://x/devtools/browser/{guid}"}
    return inner


class PortFile(unittest.TestCase):
    """DevToolsActivePort with the given contents, as the user's real profile would have it."""

    def port_file(self, text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "DevToolsActivePort"
        path.write_text(text)
        patch = mock.patch.object(launcher, "get_user_chrome_port_file", lambda: path)
        patch.start()
        self.addCleanup(patch.stop)
        return path


class FindingWhoAnswers(unittest.TestCase):
    def test_a_dead_listener_on_one_stack_does_not_hide_a_live_one(self):
        with mock.patch.object(launcher, "_version", answers({"http://[::1]:9222": OTHER})):
            self.assertEqual(launcher.cdp_endpoint(9222), "http://[::1]:9222")
            self.assertTrue(launcher.is_chrome_cdp_ready(9222))

    def test_nothing_answering_is_not_ready(self):
        with mock.patch.object(launcher, "_version", answers({})):
            self.assertIsNone(launcher.cdp_endpoint(9222))
            self.assertFalse(launcher.is_chrome_cdp_ready(9222))


class IdentifyingTheUsersChrome(PortFile):
    def test_a_stale_port_file_plus_a_squatter_is_not_the_users_chrome(self):
        # Exactly the observed failure: the file records 9222 and one GUID, a different Chrome
        # answers there. Attaching would hand the run a profile with no logins.
        self.port_file(f"9222\n/devtools/browser/{GUID}\n")
        with mock.patch.object(launcher, "_version", answers({"http://[::1]:9222": OTHER})):
            self.assertIsNone(launcher.user_chrome_endpoint())
            self.assertFalse(launcher.is_user_chrome_ready())

    def test_a_matching_guid_on_any_stack_is_the_users_chrome(self):
        self.port_file(f"9222\n/devtools/browser/{GUID}\n")
        with mock.patch.object(launcher, "_version", answers({"http://[::1]:9222": GUID})):
            self.assertEqual(launcher.user_chrome_endpoint(), "http://[::1]:9222")

    def test_a_port_file_without_a_guid_proves_nothing(self):
        self.port_file("9222\n")
        with mock.patch.object(launcher, "_version", answers({"http://127.0.0.1:9222": GUID})):
            self.assertIsNone(launcher.user_chrome_endpoint())

    def test_an_unreadable_port_file_is_not_an_exception(self):
        self.port_file("not a port\n")
        with mock.patch.object(launcher, "_version", answers({})):
            self.assertIsNone(launcher.user_chrome_endpoint())


class ChoosingAProfile(unittest.TestCase):
    def test_unset_keeps_the_throwaway_profile(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("JEV_CHROME_USER_DATA_DIR", None)
            self.assertEqual(launcher.chosen_user_data_dir(), launcher.HARNESS_PROFILE)

    def test_user_means_the_profile_with_the_logins(self):
        with mock.patch.dict("os.environ", {"JEV_CHROME_USER_DATA_DIR": "user"}):
            self.assertEqual(launcher.chosen_user_data_dir(), launcher.default_user_data_dir())

    def test_a_path_is_taken_literally(self):
        with mock.patch.dict("os.environ", {"JEV_CHROME_USER_DATA_DIR": "~/somewhere"}):
            self.assertEqual(launcher.chosen_user_data_dir(), Path.home() / "somewhere")

    def test_an_explicit_argument_wins_over_the_environment(self):
        with mock.patch.dict("os.environ", {"JEV_CHROME_USER_DATA_DIR": "user"}):
            self.assertEqual(launcher.chosen_user_data_dir("/tmp/p"), Path("/tmp/p"))


class Describing(PortFile):
    def test_configured_but_unreachable_is_its_own_state_not_throwaway(self):
        self.port_file(f"9222\n/devtools/browser/{GUID}\n")
        with mock.patch.dict("os.environ", {"JEV_CHROME_USER_DATA_DIR": "user"}), \
                mock.patch.object(launcher, "_version", answers({"http://[::1]:9222": OTHER})):
            got = launcher.describe()
        self.assertFalse(got["is_user_profile"])
        self.assertTrue(got["profile_has_logins"])
        self.assertTrue(got["port_answered_by_someone_else"])
        self.assertIn("remote-debugging-port", got["note"])

    def test_connected_to_the_users_chrome_says_so(self):
        self.port_file(f"9222\n/devtools/browser/{GUID}\n")
        with mock.patch.object(launcher, "_version", answers({"http://127.0.0.1:9222": GUID})):
            got = launcher.describe()
        self.assertTrue(got["is_user_profile"])
        self.assertIn("登录态可用", got["note"])


class EnsuringChrome(PortFile):
    def setUp(self):
        for name, value in (("get_default_chrome_path", lambda: "/chrome"),
                            ("profile_holder", lambda _p: None)):
            patch = mock.patch.object(launcher, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.popen = mock.patch.object(launcher.subprocess, "Popen").start()
        self.addCleanup(mock.patch.stopall)

    def test_the_users_own_chrome_wins_outright(self):
        self.port_file(f"9222\n/devtools/browser/{GUID}\n")
        with mock.patch.object(launcher, "_version", answers({"http://[::1]:9222": GUID})):
            self.assertEqual(launcher.ensure_chrome_running(), "http://[::1]:9222")
        self.popen.assert_not_called()

    def test_a_foreign_chrome_is_fine_for_a_throwaway_profile(self):
        self.port_file("9222\n")
        with mock.patch.dict("os.environ", {"JEV_CHROME_USER_DATA_DIR": ""}), \
                mock.patch.object(launcher, "_version", answers({"http://[::1]:9222": OTHER})):
            self.assertEqual(launcher.ensure_chrome_running(), "http://[::1]:9222")
        self.popen.assert_not_called()

    def test_a_foreign_chrome_is_refused_when_logins_were_asked_for(self):
        self.port_file(f"9222\n/devtools/browser/{GUID}\n")
        with mock.patch.dict("os.environ", {"JEV_CHROME_USER_DATA_DIR": "user"}), \
                mock.patch.object(launcher, "_version", answers({"http://[::1]:9222": OTHER})):
            with self.assertRaises(RuntimeError) as caught:
                launcher.ensure_chrome_running()
        self.assertIn("--remote-debugging-port=9222", str(caught.exception))
        self.popen.assert_not_called()

    def test_a_profile_another_chrome_holds_is_not_launched_over(self):
        self.port_file("9222\n")
        with mock.patch.dict("os.environ", {"JEV_CHROME_USER_DATA_DIR": "user"}), \
                mock.patch.object(launcher, "_version", answers({})), \
                mock.patch.object(launcher, "profile_holder", lambda _p: "mac-83141"):
            with self.assertRaises(RuntimeError) as caught:
                launcher.ensure_chrome_running()
        self.assertIn("mac-83141", str(caught.exception))
        self.popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
