"""Regression tests through real Session/Agent and autopilot boundaries; no network/models."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent, autopilot, browser, session

URL = 'https://www.zhipin.com/web/geek/job'


def page(text='results', url=URL, label='Search'):
    p = dict(url=url, title='Jobs', text=text, scroll={'y': 0}, actions=[
        {'id': 'e1', 'node': 1, 'kind': 'click', 'role': 'button', 'label': label}])
    p['fingerprint'] = browser.fingerprint(p)
    return p


def decision(confidence=1):
    return dict(choice='e1', operation='CLICK', target='1', confidence=confidence,
                probabilities={'e1': 1.0}, latency_ms=0, usage={})


@pytest.fixture
def make_session(monkeypatch, tmp_path):
    monkeypatch.setattr(session, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(agent, 'choose', Mock(return_value=decision()))
    def make(initial=None, **kwargs):
        b = Mock()
        b.observe.return_value = deepcopy(initial or page())
        b.fresh.return_value = True
        b.adopt_popup.return_value = False
        monkeypatch.setattr(agent, 'Browser', Mock(return_value=b))
        s = session.Session(URL, 'read jobs', screenshots=False, mode='job_patrol', **kwargs)
        return s, b
    return make


def test_refreshed_risk_page_stops_before_click(make_session):
    s, b = make_session()
    b.fresh.return_value = False
    b.observe.return_value = page('请完成安全验证')
    result = s.step(steps=1, gated=True)
    assert result['stop_reason'] == 'platform_blocked'
    b.act.assert_not_called()


def test_executed_action_is_counted_when_observation_goes_stale(make_session):
    s, b = make_session(step_budget=1)
    b.observe.side_effect = [browser.StalePage('Document is navigating'), page('results loaded'), page('more')]
    result = s.step(steps=2, gated=True)
    assert b.act.call_count == 1, 'an executed click was omitted from the budget'
    assert s.total_steps == 1
    assert len(result['steps_executed']) == 1


def test_patrol_gate_is_mandatory_even_without_gated_argument(make_session):
    s, b = make_session(page(label='立即沟通'))
    result = s.step(steps=1)
    assert result['stop_reason'] == 'blocked_action'
    b.act.assert_not_called()


def test_terminal_stop_cannot_be_navigated_away(make_session):
    s, b = make_session(page('安全验证'))
    s.step(steps=1, gated=True)
    with pytest.raises(ValueError):
        s.navigate(URL + '?city=next')
    b.call.assert_not_called()


def test_stop_reason_survives_repeat_calls(make_session):
    s, _ = make_session(page(url='https://outside.test/'))
    assert s.step(steps=1)['stop_reason'] == 'left_platform'
    assert s.step(steps=1)['stop_reason'] == 'left_platform'


def test_close_releases_browser_even_when_report_fails(make_session, monkeypatch):
    s, b = make_session()
    monkeypatch.setattr(s, '_report', Mock(side_effect=OSError('disk full')))
    try:
        s.close()
    except OSError:
        pass
    b.close.assert_called_once()


def test_browser_failed_initialization_closes_owned_tab(monkeypatch):
    calls = []
    def cdp(self, method, **kw):
        calls.append((method, kw))
        if method == 'Target.createTarget':
            return {'targetId': 'owned'}
        if method == 'Target.attachToTarget':
            raise RuntimeError('attach failed')
        return {}
    monkeypatch.setattr(browser, 'require_existing_daemon', Mock())
    monkeypatch.setattr(browser.Browser, '_cdp', cdp)
    with pytest.raises(RuntimeError, match='attach failed'):
        browser.Browser('about:blank', daemon_name='patrol')
    assert any(m == 'Target.closeTarget' and p['targetId'] == 'owned' for m, p in calls)


def setup_autopilot(monkeypatch, s):
    monkeypatch.setattr(autopilot.sessions, 'load_env', Mock())
    monkeypatch.setattr(autopilot.sessions, 'prepare', Mock(return_value={'note': 'fixture'}))
    monkeypatch.setattr(autopilot.sessions, 'start', Mock(return_value=s))
    monkeypatch.setattr(autopilot.sessions, 'finish', Mock())
    monkeypatch.setattr(autopilot.supervisor, 'plan', Mock(return_value={
        'start_url': URL, 'legs': [{'goal': 'read jobs'}]}))
    monkeypatch.setattr(autopilot.supervisor, 'write_report', Mock(return_value='report'))


def test_force_result_is_checked_even_on_final_allowed_step(make_session, monkeypatch):
    s, b = make_session(step_budget=1)
    monkeypatch.setattr(agent, 'choose', Mock(return_value=decision(0.1)))
    b.observe.return_value = page('安全验证')
    setup_autopilot(monkeypatch, s)
    monkeypatch.setattr(autopilot, 'CHECKPOINTS_PER_LEG', 2)
    monkeypatch.setattr(autopilot.supervisor, 'judge', Mock(side_effect=[
        {'action': 'continue', 'note': 'waiting', 'goal': 'retry', 'url': ''}, {
        'action': 'force', 'note': 'release confidence only', 'goal': '', 'url': ''}]))
    result = autopilot.run('read jobs', url=URL, mode='job_patrol', max_steps=1, say=lambda *a: None)
    assert result['platform_stop'] and result['platform_stop']['reason'] == 'platform_blocked'
    assert any(item['stop_reason'] == 'platform_blocked' for item in result['trace'])


def test_intermediate_results_are_kept_with_their_links(make_session, monkeypatch):
    s, b = make_session()
    middle = page('Candidate A', url=URL+'?city=first')
    middle['links'] = [{'label': 'Candidate A', 'url': 'https://www.zhipin.com/job_detail/a.html'}]
    b.observe.side_effect = [middle, page('Candidate B', url=URL+'?city=second')]
    setup_autopilot(monkeypatch, s)
    monkeypatch.setattr(autopilot.supervisor, 'judge', Mock(return_value={
        'action': 'finish', 'note': 'done', 'goal': '', 'url': ''}))
    result = autopilot.run('read jobs', url=URL, mode='job_patrol', max_steps=2, say=lambda *a: None)
    found = [p for p in result['visited'] if p['text'] == 'Candidate A']
    assert found, 'first result page in a two-action checkpoint was lost'
    assert found[0]['links'] == middle['links']


def test_step_log_goes_to_run_callback(capsys):
    logged = []
    autopilot._show({'steps_executed': [{'terminal': 'done', 'confidence': 1.0}]},
                    say=lambda *a: logged.append(a))
    assert logged, 'auto_poll loses all per-action log lines'
    assert not capsys.readouterr().err


def test_cancel_during_planning_does_not_open_a_platform(make_session, monkeypatch):
    s, _ = make_session()
    setup_autopilot(monkeypatch, s)
    cancelled = False
    def plan(*_args):
        nonlocal cancelled
        cancelled = True
        return {'start_url': URL, 'legs': [{'goal': 'read jobs'}]}
    monkeypatch.setattr(autopilot.supervisor, 'plan', plan)
    monkeypatch.setattr(autopilot.supervisor, 'judge', Mock(return_value={
        'action': 'finish', 'note': '', 'goal': '', 'url': ''}))
    autopilot.run('read', url=URL, mode='job_patrol', should_stop=lambda: cancelled, say=lambda *a: None)
    autopilot.sessions.start.assert_not_called()


def test_finished_run_can_be_polled_from_another_mcp_process(tmp_path, monkeypatch):
    import json

    from jev_ultrafast import runs
    monkeypatch.setattr(runs, 'ARTIFACTS', tmp_path)
    record = {'run_id': 'abcdef123456', 'status': 'done', 'lines': ['one', 'two'], 'answer': 'found'}
    (tmp_path/'abcdef123456.json').write_text(json.dumps(record))
    result = runs.poll('abcdef123456', since=1)
    assert result['status'] == 'done'
    assert result['lines'] == ['two']
    assert result['next_line'] == 2


def test_idle_session_is_reaped_before_acquiring_patrol_lease(make_session, monkeypatch):
    make_session()
    monkeypatch.setattr(session, '_REGISTRY', {})
    monkeypatch.setattr(session, 'prepare', Mock())
    first = session.start(URL, 'read', screenshots=False, mode='job_patrol')
    first.touched_at = 1
    try:
        second = session.start(URL, 'read', screenshots=False, mode='job_patrol')
        assert first.closed
        session.finish(second.id)
    finally:
        if not first.closed:
            session.finish(first.id)


def test_live_does_not_leave_an_expired_session_blocking_mcp(make_session, monkeypatch):
    s, _ = make_session()
    s.touched_at = 1
    monkeypatch.setattr(session, '_REGISTRY', {s.id: s})
    assert session.live() == []
    assert s.closed


def test_mcp_rejects_source_drift_before_opening_browser(monkeypatch):
    import json

    from jev_ultrafast import mcp_server

    monkeypatch.setattr(mcp_server, '_source_id', lambda: 'changed')
    prepare = Mock()
    monkeypatch.setattr(mcp_server.sessions, 'prepare', prepare)
    result = json.loads(mcp_server.auto_start('read', URL, mode='job_patrol'))
    assert result['ok'] is False
    assert 'source_changed' in result['error']
    assert result['runtime']['current_source_id'] == 'changed'
    prepare.assert_not_called()
