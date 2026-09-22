"""A deterministic gate on what the browser may do without asking.

Deliberately not a model. A guard you can argue with is not a guard, and "do not click submit"
written into a goal is a suggestion to a model that chooses among the elements it sees. This is
keyword and kind matching over the already-observed action, and it runs before every execution.

Upstream already keeps password, file and hidden inputs out of the action space entirely
(snapshot.js), so those can never be filled no matter what this returns.
"""

from . import job_patrol

ALLOW = "allow"
CONFIRM = "confirm"
BLOCK = "block"

# Reading, moving, scrolling and filling are reversible. These are not: they spend money, send
# something, bind an identity, or destroy data. Matched case-insensitively as substrings.
COMMIT_WORDS = (
    "提交", "下单", "支付", "付款", "购买", "预订", "预定", "订购", "结算", "抢购",
    "申请", "办理", "发送", "发布", "投递", "报名", "签署", "同意并", "确认支付",
    "确认下单", "确认提交", "立即购买", "立即预订", "去支付", "去结算",
    "登录", "登陆", "注册", "删除", "注销", "解绑", "退订", "取消订单",
    "submit", "place order", "pay now", "checkout", "buy", "book now", "reserve now",
    "sign in", "log in", "sign up", "register", "delete", "send", "apply now",
    "subscribe", "agree and", "confirm and", "purchase", "add to cart",
)

# Never typed, with or without a confirmation. Entering these is the user's own job.
SECRET_WORDS = (
    "密码", "口令", "卡号", "银行卡", "信用卡", "有效期", "安全码", "身份证", "证件号",
    "社保号", "支付密码", "验证码",
    "password", "passcode", "card number", "cardnumber", "cvv", "cvc", "security code",
    "expiry", "ssn", "social security", "iban", "routing number", "one-time code",
)


def _hit(text: str, words) -> str | None:
    lowered = (text or "").lower()
    return next((w for w in words if w.lower() in lowered), None)


def gate(decision: dict, action: dict | None, page: dict, mode: str = "standard") -> tuple[str, str]:
    """Return (verdict, reason) for one chosen action, before it is executed."""
    if action is None:
        # DONE / BLOCKED and the scroll and wait controls touch nothing.
        return ALLOW, ""
    kind = action.get("kind")
    label = action.get("label", "")
    if kind in {"scroll", "wait"}:
        return ALLOW, ""
    if kind in {"fill", "select"}:
        secret = _hit(label, SECRET_WORDS)
        if secret:
            return BLOCK, f"字段“{label[:40]}”看起来要求机密信息（匹配“{secret}”），不会代填"
        if mode == job_patrol.MODE:
            blocked = job_patrol.blocked_action(kind, label)
            if blocked:
                return BLOCK, f"岗位巡检只读模式禁止填写“{label[:60]}”（匹配“{blocked}”）"
        return ALLOW, ""
    if mode == job_patrol.MODE:
        blocked = _hit(label, COMMIT_WORDS) or job_patrol.blocked_action(kind, label)
        if blocked:
            return BLOCK, f"岗位巡检只读模式禁止执行“{label[:60]}”（匹配“{blocked}”）"
    commit = _hit(label, COMMIT_WORDS)
    if commit:
        return CONFIRM, f"“{label[:60]}”匹配提交类关键词“{commit}”"
    return ALLOW, ""


def describe(action: dict | None) -> str:
    if action is None:
        return "(terminal decision)"
    return f"{action.get('kind')} → {action.get('label', '')[:80]}"
