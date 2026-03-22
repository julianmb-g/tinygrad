import atexit
import os
import signal
import subprocess

os.environ["DISABLE_COMPILER_CACHE"] = "1"

active_pids = set()

_original_popen = subprocess.Popen

class _TrackedPopen(_original_popen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        active_pids.add(self.pid)

    def wait(self, *args, **kwargs):
        if "timeout" not in kwargs and len(args) == 0:
            kwargs["timeout"] = 15.0
        try:
            ret = super().wait(*args, **kwargs)
        except subprocess.TimeoutExpired:
            try: os.kill(self.pid, signal.SIGKILL)
            except OSError: pass
            raise
        active_pids.discard(self.pid)
        return ret

    def poll(self):
        ret = super().poll()
        if ret is not None:
            active_pids.discard(self.pid)
        return ret

    def communicate(self, *args, **kwargs):
        if "timeout" not in kwargs and len(args) == 0:
            kwargs["timeout"] = 15.0
        try:
            ret = super().communicate(*args, **kwargs)
        except subprocess.TimeoutExpired:
            try: os.kill(self.pid, signal.SIGKILL)
            except OSError: pass
            raise
        active_pids.discard(self.pid)
        return ret

subprocess.Popen = _TrackedPopen

def teardown_worker_group():
    for pid in list(active_pids):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

def pytest_configure(config):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None:
        os.setpgrp()
        atexit.register(teardown_worker_group)
