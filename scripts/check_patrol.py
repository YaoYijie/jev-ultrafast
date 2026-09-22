"""Real MCP -> run -> Session -> Chrome checks against loopback pages only.

Models are deterministic offline fixtures. No recruiting sites or paid APIs are contacted.
Run: uv run python scripts/check_patrol.py
"""
import asyncio
import json
import sys
import tempfile
import threading
import time
from contextlib import AsyncExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'jev_ultrafast').exists():
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def worker(directory):
    from jev_ultrafast import agent, job_patrol, runs, session, supervisor
    # Only this test process admits loopback as a fixture platform. Production allowlists stay intact.
    job_patrol.PLATFORM_DOMAINS['fixture'] = ('127.0.0.1',)
    session.ARTIFACTS = Path(directory) / 'sessions'
    runs.ARTIFACTS = Path(directory) / 'runs'
    runs.INDEX = runs.ARTIFACTS / 'index.jsonl'

    def choose(page, goal, _history):
        if goal == 'hold':
            time.sleep(1)
        terminal = 'Fixture evidence complete' in page['text']
        action = next((a for a in page['actions'] if a['kind'] == 'click'), None)
        choice = 'DONE' if terminal else action['id'] if action else 'BLOCKED'
        return dict(choice=choice, operation='DONE' if terminal else 'CLICK', target='1',
                    confidence=1.0, probabilities={choice: 1.0}, latency_ms=0, usage={})

    def plan(need, url):
        if need == 'cancel':
            time.sleep(1)
        return {'start_url': url, 'legs': [{'goal': need}]}

    agent.choose = choose
    agent.field_text = lambda *_: (_ for _ in ()).throw(AssertionError('Unexpected text model call'))
    supervisor.plan = plan
    supervisor.judge = lambda _need, _goal, report, *_a, **_kw: {
        'action': 'finish' if report['stop_reason'] == 'done' else 'continue',
        'note': 'fixture judgment', 'goal': '', 'url': '',
    }
    supervisor.write_report = lambda _need, pages: json.dumps(pages, ensure_ascii=False)
    from jev_ultrafast.mcp_server import mcp
    mcp.run(transport='stdio')


class Fixture(BaseHTTPRequestHandler):
    hits = []

    def do_GET(self):
        type(self).hits.append(self.path)
        if self.path.startswith('/risk'):
            body = """<button onclick="document.body.innerHTML='<h1>安全验证</h1><button>Continue</button>'">
            Show status</button>"""
        elif self.path.startswith('/results'):
            body = '<p>Candidate Alpha, Shanghai, 30K</p><a target="_blank" href="/detail">Open detail</a>'
        elif self.path.startswith('/detail'):
            body = '<h1>Fixture evidence complete</h1><p>Candidate Alpha details: build agent systems.</p>'
        else:
            body = '<h1>Fixture search</h1><a href="/results">Read results</a>'
        data = ('<!doctype html><meta charset="UTF-8"><title>Patrol fixture</title>'
                '<style>body{padding:40px;font:20px sans-serif}button,a{display:block;margin:20px}</style>'
                + body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=UTF-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


async def call(client, name, **kwargs):
    response = await client.call_tool(name, kwargs)
    result = json.loads(next(c.text for c in response.content if c.type == 'text'))
    assert result.get('ok'), (name, result)
    runtime = result['runtime']
    assert runtime['loaded_source_id'] == runtime['current_source_id'], runtime
    return result


async def finished(client, run_id):
    for _ in range(300):
        result = await call(client, 'auto_poll', run_id=run_id)
        if result['status'] in {'done', 'error', 'cancelled'}:
            return result
        await asyncio.sleep(0.1)
    raise AssertionError('Fixture run did not finish within 30 seconds')


async def check(origin, directory):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    passed = []
    params = StdioServerParameters(command=sys.executable,
        args=['-B', str(Path(__file__).resolve()), '--worker', directory], cwd=str(ROOT))
    with open(Path(directory) / 'mcp.stderr.log', 'w') as errors:
        async with AsyncExitStack() as stack:
            clients = []
            for _ in range(2):
                read, write = await stack.enter_async_context(stdio_client(params, errlog=errors))
                client = await stack.enter_async_context(ClientSession(read, write))
                await client.initialize()
                clients.append(client)
            first, second = clients
            schema = await first.list_tools()
            start_schema = next(t.input_schema for t in schema.tools if t.name == 'auto_start')
            assert 'job_patrol' in start_schema['properties']['mode']['enum']
            passed.append('MCP schema exposes job_patrol')

            start = await call(first, 'auto_start', need='hold', url=origin+'/start', mode='job_patrol')
            assert start['browser']['is_user_profile'] is True
            # Wait for the first process to hold a real session, then challenge from another MCP.
            for _ in range(100):
                if '/start' in Fixture.hits:
                    break
                await asyncio.sleep(0.05)
            assert '/start' in Fixture.hits
            conflict = await call(second, 'auto_start', need='read', url=origin+'/start', mode='job_patrol')
            rejected = await finished(second, conflict['run_id'])
            assert rejected['status'] == 'error' and 'browser_busy' in rejected['error'], rejected
            passed.append('second MCP cannot overlap patrol')

            done = await finished(first, start['run_id'])
            assert done['status'] == 'done', done
            assert done['total_steps'] == 2, done
            assert any('CLICK' in line for line in done['lines']), done['lines']
            result = await call(first, 'auto_result', run_id=start['run_id'], page_text=True)
            intermediate = [p for p in result['visited'] if '/results' in p['url']]
            assert intermediate and any(link['url'] == origin+'/detail' for link in intermediate[0]['links'])
            assert any('Fixture evidence complete' in p['text'] for p in result['visited'])
            assert result['platform_stop'] is None
            passed.append('real clicks, popup adoption, intermediate evidence and links retained')

            # Same URLs, new session: no lingering lease or dead-tab preflight failure.
            repeat = await call(first, 'auto_start', need='read', url=origin+'/start', mode='job_patrol')
            assert (await finished(first, repeat['run_id']))['status'] == 'done'
            passed.append('second sequential run succeeds on same Chrome daemon')

            risk = await call(first, 'auto_start', need='read', url=origin+'/risk', mode='job_patrol')
            assert (await finished(first, risk['run_id']))['status'] == 'done'
            result = await call(first, 'auto_result', run_id=risk['run_id'], page_text=True)
            assert result['platform_stop']['reason'] == 'platform_blocked', result
            assert result['total_steps'] == 1, result
            passed.append('simulated risk wall stops before another input')

            before = len(Fixture.hits)
            cancel = await call(first, 'auto_start', need='cancel', url=origin+'/cancel', mode='job_patrol')
            await call(first, 'auto_cancel', run_id=cancel['run_id'])
            assert (await finished(first, cancel['run_id']))['status'] == 'cancelled'
            assert not any(p.startswith('/cancel') for p in Fixture.hits[before:])
            passed.append('cancel during planning never opens target')

            # Read the persisted run through another process that never owned it.
            result = await call(second, 'auto_poll', run_id=start['run_id'])
            assert result['status'] == 'done'
            result = await call(second, 'auto_result', run_id=start['run_id'], page_text=True)
            assert result['total_steps'] == 2
            passed.append('poll and report survive transfer to another MCP process')

    from jev_ultrafast.browser import Browser, StalePage, _cdp
    from jev_ultrafast.job_patrol import DAEMON_NAME

    guard = Browser(origin+'/start', daemon_name=DAEMON_NAME, strict_fresh=True)
    try:
        page = guard.observe(screenshot=False)
        action = next(a for a in page['actions'] if a['kind'] == 'click')
        guard.evaluate("const risk=document.createElement('aside'); "
                       "risk.textContent='安全验证'; document.body.append(risk)")
        try:
            guard.act(action, page)
        except StalePage:
            pass
        else:
            raise AssertionError('Changed risk page still accepted the old click')
        assert guard.evaluate('location.href') == origin+'/start'
        passed.append('risk text appearing after prediction invalidates input')
    finally:
        guard.close()
    remaining = _cdp('Target.getTargets', daemon_name=DAEMON_NAME)['targetInfos']
    assert not any(t.get('url', '').startswith(origin) for t in remaining), remaining
    passed.append('all fixture tabs closed after completion and cancellation')
    return passed


def main():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Fixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='jev-patrol-smoke-') as directory:
            origin = f'http://127.0.0.1:{server.server_port}'
            try:
                passed = asyncio.run(asyncio.wait_for(check(origin, directory), timeout=90))
            except BaseException:
                log = Path(directory) / 'mcp.stderr.log'
                if log.exists():
                    print(log.read_text()[-4000:], file=sys.stderr)
                raise
            for item in passed:
                print('PASS:', item)
            print(f'PASS: {len(passed)} integration checks; only loopback pages; no model API calls')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        main()
