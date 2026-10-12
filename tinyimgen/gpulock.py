"""Cross-process GPU mutex (stdlib only, so the 3D venv can import it too).

The image server and 3D jobs run in different processes; a 6 GB card fits one of them at a time.
"""
import contextlib
import fcntl
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", ".gpu.lock")


@contextlib.contextmanager
def gpu_lock():
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    fd = os.open(PATH, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
