import atexit
import os
import signal

os.environ["DISABLE_COMPILER_CACHE"] = "1"

active_pids = set()

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
        
        import subprocess
        original_popen = subprocess.Popen
        class TrackedPopen(original_popen):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                active_pids.add(self.pid)
        subprocess.Popen = TrackedPopen
