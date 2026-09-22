"""Cross-process browser ownership: patrols are exclusive; normal sessions may share.

The OS releases these locks on process death. Never unlink the lock file: concurrent callers
must keep locking the same inode. On Windows the stdlib lock is exclusive for all sessions.
"""

import os
from pathlib import Path


def acquire(path: Path, *, exclusive: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            kind = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle.fileno(), kind | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise ValueError("browser_busy: 岗位巡检必须串行运行；其他 Jev 进程仍在使用浏览器") from exc
    return handle
