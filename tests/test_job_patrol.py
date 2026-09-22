"""Offline policy checks for the dedicated recruiting-platform mode."""

import threading
import unittest
from unittest import mock

from jev_ultrafast import job_patrol, session


class PlatformScope(unittest.TestCase):
    def test_supported_subdomains_are_recognized(self):
        cases = {
            "https://www.zhipin.com/web/geek/job": "boss",
            "https://we.51job.com/pc/search": "51job",
            "https://www.liepin.com/zhaopin/": "liepin",
            "https://sou.zhaopin.com/": "zhaopin",
            "https://cn.linkedin.com/jobs/": "linkedin",
        }
        for url, expected in cases.items():
            self.assertEqual(job_patrol.platform_for_url(url), expected, url)

    def test_lookalike_and_non_http_hosts_are_rejected(self):
        for url in ["https://evilzhipin.com/jobs", "javascript:alert(1)", "https://example.com/zhipin.com"]:
            self.assertIsNone(job_patrol.platform_for_url(url), url)

    def test_a_patrol_cannot_switch_platforms(self):
        with self.assertRaises(ValueError):
            job_patrol.require_platform_url("https://www.liepin.com/zhaopin/", expected="boss")


class StopSignals(unittest.TestCase):
    def page(self, url="https://www.zhipin.com/web/geek/job", title="职位搜索", text="登录 注册 搜索"):
        return {"url": url, "title": title, "text": text}

    def test_a_header_login_link_is_not_mistaken_for_a_login_wall(self):
        self.assertIsNone(job_patrol.page_stop(self.page(), "boss"))

    def test_risk_control_text_stops_the_platform(self):
        stop = job_patrol.page_stop(self.page(text="检测到账号异常，请完成安全验证"), "boss")
        self.assertEqual(stop[0], "platform_blocked")
        self.assertIn("安全验证", stop[1])

    def test_login_path_stops_the_platform(self):
        stop = job_patrol.page_stop(self.page(url="https://www.zhipin.com/web/user/login"), "boss")
        self.assertEqual(stop[0], "platform_blocked")

    def test_bare_http_error_title_stops_the_platform(self):
        stop = job_patrol.page_stop(self.page(title="429"), "boss")
        self.assertEqual(stop[0], "platform_blocked")

    def test_boss_security_check_query_stops_before_supervisor_judgment(self):
        stop = job_patrol.page_stop(self.page(
            url=("https://www.zhipin.com/web/geek/jobs?query=AI%20Application%20Engineer"
                 "&city=101210100&_security_check=1_1790040471628"),
            text="没有更多职位，尝试登录查看全部职位",
        ), "boss")
        self.assertEqual(stop[0], "platform_blocked")
        self.assertIn("_security_check", stop[1])

    def test_external_navigation_stops_the_run(self):
        stop = job_patrol.page_stop(self.page(url="https://careers.example.com/job/1"), "boss")
        self.assertEqual(stop[0], "left_platform")


class BrowserPreparation(unittest.TestCase):
    def test_job_patrol_selects_and_verifies_the_real_user_profile(self):
        browser = {"is_user_profile": True, "note": "ok"}
        with mock.patch.object(session, "load_env"), \
                mock.patch.object(session.launcher, "ensure_user_browser", return_value=browser) as ensure:
            self.assertEqual(session.prepare("job_patrol"), browser)
            ensure.assert_called_once_with(job_patrol.DAEMON_NAME, job_patrol.DAEMON_ENV)

    def test_job_patrol_propagates_a_local_browser_preflight_failure(self):
        with mock.patch.object(session, "load_env"), \
                mock.patch.object(
                    session.launcher, "ensure_user_browser", side_effect=RuntimeError("permission-blocked")
                ), \
                self.assertRaises(RuntimeError):
            session.prepare("job_patrol")


class SessionEnforcement(unittest.TestCase):
    def shell(self, page):
        current = session.Session.__new__(session.Session)
        current.lock = threading.RLock()
        current.closed = False
        current.touched_at = 0
        current.pending_decision = None
        current.last_error = None
        current.blocked_reason = None
        current.mode = "job_patrol"
        current.allowed_platform = "boss"
        current.action_delay_s = 2
        current.last_action_at = None
        current.total_steps = 0
        current.step_budget = 20
        current.agent = mock.Mock()
        current.agent.state = {"status": "ready", "page": page}
        return current

    def test_risk_page_stops_before_any_model_decision(self):
        current = self.shell({
            "url": "https://www.zhipin.com/web/geek/job",
            "title": "安全验证",
            "text": "检测到账号异常",
        })
        with mock.patch.object(current, "_report", side_effect=lambda reason, _steps: {"stop_reason": reason}):
            report = current.step(steps=2, gated=True)
        self.assertEqual(report["stop_reason"], "platform_blocked")
        current.agent.command.assert_not_called()

    def test_navigation_to_another_platform_is_rejected_before_browser_input(self):
        current = self.shell({})
        with self.assertRaises(ValueError):
            current.navigate("https://www.liepin.com/zhaopin/")
        current.agent.browser.call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
