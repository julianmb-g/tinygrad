import atexit
import os
import signal
import multiprocessing
import pytest
import subprocess
import glob

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

os.environ["DISABLE_COMPILER_CACHE"] = "1"

active_pids = set()

def teardown_worker_group():
    try:
        for pid in list(active_pids):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    except (AttributeError, KeyError):
        pass

def pytest_configure(config):
    if getattr(config, "workerinput", None) is not None:
        os.setpgrp()
        atexit.register(teardown_worker_group)

        m = pytest.MonkeyPatch()
        original_popen = subprocess.Popen
        class TrackedPopen(original_popen):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                active_pids.add(self.pid)
        m.setattr(subprocess, "Popen", TrackedPopen)
        config.worker_monkeypatch = m

def pytest_unconfigure(config):
    if hasattr(config, "worker_monkeypatch"):
        config.worker_monkeypatch.undo()
def pytest_sessionfinish(session, exitstatus):
    if not hasattr(session.config, "workerinput"):
        try:
            for shm_path in glob.glob("/dev/shm/psm_*"):
                try: os.unlink(shm_path)
                except OSError: pass
        except (AttributeError, KeyError, OSError, FileNotFoundError): pass
