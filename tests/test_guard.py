"""The gate is the only thing standing between an autonomous run and a submitted form."""

import unittest

from jev_ultrafast import guard


def verdict(kind, label, mode="standard"):
    return guard.gate({}, {"kind": kind, "label": label}, {}, mode=mode)[0]


class GateAllowsReadingAndMoving(unittest.TestCase):
    def test_navigation_and_reading_run_unattended(self):
        for label in ["人力资源类", "制度流程", "下一页", "Next page", "More results"]:
            self.assertEqual(verdict("click", label), guard.ALLOW, label)

    def test_search_is_not_a_commitment(self):
        for label in ["搜索", "查询航班", "查找", "筛选", "Search flights", "Find"]:
            self.assertEqual(verdict("click", label), guard.ALLOW, label)

    def test_scroll_and_wait_are_always_allowed(self):
        self.assertEqual(verdict("scroll", "Scroll down"), guard.ALLOW)
        self.assertEqual(verdict("wait", "Wait for the page to update"), guard.ALLOW)

    def test_ordinary_fields_are_filled(self):
        for label in ["出发地", "目的地", "出发日期", "Departure city", "搜索关键词"]:
            self.assertEqual(verdict("fill", label), guard.ALLOW, label)

    def test_terminal_decisions_touch_nothing(self):
        self.assertEqual(guard.gate({"choice": "DONE"}, None, {})[0], guard.ALLOW)


class GateStopsCommitments(unittest.TestCase):
    def test_spending_money_asks_first(self):
        for label in ["立即预订", "去支付", "确认下单", "立即购买", "Book now", "Checkout", "Pay now"]:
            self.assertEqual(verdict("click", label), guard.CONFIRM, label)

    def test_sending_and_applying_ask_first(self):
        for label in ["提交申请", "发送", "投递简历", "报名", "Submit", "Send", "Apply now"]:
            self.assertEqual(verdict("click", label), guard.CONFIRM, label)

    def test_identity_and_destruction_ask_first(self):
        for label in ["登录", "注册", "删除", "注销", "Sign in", "Register", "Delete"]:
            self.assertEqual(verdict("click", label), guard.CONFIRM, label)

    def test_a_commitment_inside_a_longer_label_still_asks(self):
        self.assertEqual(verdict("click", "同意条款并提交订单"), guard.CONFIRM)

    def test_confirming_gives_a_reason_naming_the_match(self):
        result, reason = guard.gate({}, {"kind": "click", "label": "立即预订"}, {})
        self.assertEqual(result, guard.CONFIRM)
        self.assertIn("预订", reason)


class GateNeverTypesSecrets(unittest.TestCase):
    def test_credentials_and_payment_fields_are_blocked_outright(self):
        for label in ["密码", "支付密码", "银行卡号", "信用卡有效期", "身份证号",
                      "Password", "Card number", "CVV", "SSN"]:
            self.assertEqual(verdict("fill", label), guard.BLOCK, label)

    def test_blocking_is_not_merely_a_confirmation(self):
        self.assertNotEqual(verdict("fill", "密码"), guard.CONFIRM)


class JobPatrolIsStrictlyReadOnly(unittest.TestCase):
    def test_recruiting_state_changes_are_blocked(self):
        for label in ["立即沟通", "打招呼", "聊一聊", "收藏职位", "关注招聘者", "保存", "完成验证"]:
            self.assertEqual(verdict("click", label, mode="job_patrol"), guard.BLOCK, label)

    def test_account_fields_are_blocked(self):
        for label in ["手机号", "用户名", "Email"]:
            self.assertEqual(verdict("fill", label, mode="job_patrol"), guard.BLOCK, label)

    def test_search_controls_stay_available(self):
        for kind, label in [("fill", "搜索关键词"), ("select", "城市"), ("click", "搜索职位")]:
            self.assertEqual(verdict(kind, label, mode="job_patrol"), guard.ALLOW, label)


if __name__ == "__main__":
    unittest.main()
