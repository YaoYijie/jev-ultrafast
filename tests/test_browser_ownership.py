"""Exercise session.start from independent processes, with only I/O dependencies stubbed."""
import subprocess
import sys

import pytest

WORKER = '''
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from jev_ultrafast import session
session.ARTIFACTS = Path(sys.argv[1]) / 'sessions'
fake=Mock()
fake.state={'page': {'url': 'https://www.zhipin.com/', 'title': '', 'text': ''}}
with patch.object(session, 'prepare'), patch.object(session, 'Agent', return_value=fake):
    try:
        current=session.start('https://www.zhipin.com/', 'read', mode=sys.argv[2], screenshots=False)
    except ValueError:
        print('busy', flush=True)
    else:
        print('acquired', flush=True)
        sys.stdin.readline()
        current._report=Mock(return_value={})
        session.finish(current.id)
'''


def spawn(path, mode):
    return subprocess.Popen([sys.executable, '-B', '-c', WORKER, str(path), mode],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@pytest.mark.parametrize('held,requested', [('job_patrol','job_patrol'),
    ('job_patrol','standard'), ('standard','job_patrol')])
def test_other_process_cannot_overlap_patrol(tmp_path, held, requested):
    first=spawn(tmp_path, held)
    try:
        assert first.stdout.readline().strip() == 'acquired'
        second=spawn(tmp_path, requested)
        out, err=second.communicate('\n', timeout=5)
        assert second.returncode == 0, err
        assert out.strip() == 'busy', 'two processes acquired the browser during a patrol'
    finally:
        first.communicate('\n', timeout=5)
    third=spawn(tmp_path, requested)
    out,err=third.communicate('\n', timeout=5)
    assert third.returncode == 0, err
    assert out.strip() == 'acquired'


def test_process_exit_releases_browser_without_manual_cleanup(tmp_path):
    first=spawn(tmp_path, 'job_patrol')
    try:
        assert first.stdout.readline().strip() == 'acquired'
    finally:
        first.terminate()
        first.communicate(timeout=5)
    second=spawn(tmp_path, 'job_patrol')
    out,err=second.communicate('\n', timeout=5)
    assert second.returncode == 0, err
    assert out.strip() == 'acquired'
