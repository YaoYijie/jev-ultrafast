"""Deterministic boundaries for low-frequency, read-only job-platform patrols."""

from urllib.parse import parse_qsl, urlparse


class PatrolStopped(ValueError):
    """A terminal platform condition observed before any further browser input."""

    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


MODE = "job_patrol"
DAEMON_NAME = "jev-job-patrol"
# Empty values override any inherited remote or dedicated-CDP configuration in the named daemon.
# Browser Harness then uses its local Chrome flow and the user's explicit remote-debugging toggle.
DAEMON_ENV = {"BU_CDP_URL": "", "BU_CDP_WS": "", "BU_BROWSER_ID": ""}

PLATFORM_DOMAINS = {
    "boss": ("zhipin.com",),
    "51job": ("51job.com",),
    "liepin": ("liepin.com",),
    "zhaopin": ("zhaopin.com",),
    "linkedin": ("linkedin.com",),
}

RISK_MARKERS = (
    "验证码", "安全验证", "人机验证", "访问验证", "滑块验证", "异常 ip", "ip 异常",
    "账号异常", "行为异常", "访问过于频繁", "请求过于频繁", "操作频繁", "账号存在风险",
    "检测到异常", "访问被拒绝", "403 forbidden", "429 too many requests", "access verification",
    "security verification", "captcha", "too many requests", "unusual activity", "suspicious activity",
    "verify you are human", "account restricted", "rate limit", "401 unauthorized", "403 access denied",
    "error 401", "error 403", "error 429", "http 401", "http 403", "http 429",
)

LOGIN_WALL_MARKERS = (
    "请登录后", "登录后查看", "登录后继续", "请先登录", "重新登录", "登录状态已失效", "登录已过期",
    "session expired", "sign in to continue", "log in to continue", "please sign in",
)

LOGIN_PATH_PARTS = ("login", "signin", "passport", "checkpoint", "challenge")

READ_ONLY_ACTION_WORDS = (
    "沟通", "联系", "招呼", "聊一聊", "发送", "投递", "申请", "收藏", "保存", "关注", "订阅",
    "感兴趣", "不感兴趣", "屏蔽", "举报", "上传", "更新简历", "编辑简历", "求职状态", "登录",
    "注册", "验证", "同意", "授权", "点赞", "评论", "分享", "转发", "邀约", "apply", "message",
    "chat", "contact", "connect", "inmail", "save", "follow", "subscribe", "upload", "edit profile",
    "sign in", "log in", "register", "verify", "agree", "like", "share", "withdraw",
)

READ_ONLY_INPUT_WORDS = (
    "手机号", "手机号码", "用户名", "账号", "邮箱", "验证码", "密码", "姓名", "phone", "mobile",
    "username", "email", "verification code", "password",
)


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def platform_for_url(url: str) -> str | None:
    """Return the supported recruiting platform for an exact host boundary."""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.rstrip(".").lower()
    for platform, domains in PLATFORM_DOMAINS.items():
        if any(_host_matches(host, domain) for domain in domains):
            return platform
    return None


def require_platform_url(url: str, expected: str | None = None) -> str:
    platform = platform_for_url(url)
    if platform is None:
        supported = "、".join(PLATFORM_DOMAINS)
        raise ValueError(f"job_patrol 只接受已登记招聘平台 URL（{supported}）")
    if expected is not None and platform != expected:
        raise ValueError(f"job_patrol 已固定在 {expected}，不能跳转到 {platform}")
    return platform


def page_stop(page: dict, expected_platform: str) -> tuple[str, str] | None:
    """Return a terminal patrol stop when the page leaves scope or shows a risk wall."""
    url = str(page.get("url") or "")
    current = platform_for_url(url)
    if current != expected_platform:
        return "left_platform", f"页面已离开 {expected_platform}：{url or '(empty URL)'}"

    title = str(page.get("title") or "")
    text = str(page.get("text") or "")
    query_keys = {key.lower() for key, _value in parse_qsl(urlparse(url).query, keep_blank_values=True)}
    if "_security_check" in query_keys:
        return "platform_blocked", "页面 URL 出现风险控制参数“_security_check”"
    title_lower = title.strip().lower()
    for status in ("401", "403", "429"):
        if title_lower == status or title_lower.startswith(f"{status} ") or title_lower.startswith(f"error {status}"):
            return "platform_blocked", f"页面标题显示 HTTP {status} 拒绝"
    combined = f"{title}\n{text}".lower()
    for marker in RISK_MARKERS:
        if marker.lower() in combined:
            return "platform_blocked", f"页面出现风险控制信号“{marker}”"
    for marker in LOGIN_WALL_MARKERS:
        if marker.lower() in combined:
            return "platform_blocked", f"页面出现登录阻断信号“{marker}”"

    path = urlparse(url).path.lower()
    if any(part in path for part in LOGIN_PATH_PARTS):
        return "platform_blocked", f"页面进入登录或验证路径：{path}"
    return None


def blocked_action(kind: str, label: str) -> str | None:
    """Name the marker that makes an action incompatible with a read-only patrol."""
    lowered = (label or "").lower()
    words = READ_ONLY_INPUT_WORDS if kind in {"fill", "select"} else READ_ONLY_ACTION_WORDS
    return next((word for word in words if word.lower() in lowered), None)
