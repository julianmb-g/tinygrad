import atexit
import os
import signal
import multiprocessing
import pytest
import subprocess
import glob

# Natively trap Pytest IPC shared memory teardown deadlocks
import multiprocessing.shared_memory
if hasattr(multiprocessing.shared_memory.SharedMemory, '__del__'):
    _orig_shm_del = multiprocessing.shared_memory.SharedMemory.__del__
    def _safe_shm_del(self):
        try:
            if hasattr(self, 'buf') and self.buf is not None:
                memoryview(self.buf).release()
        except (BufferError, ValueError, AttributeError, OSError, KeyError): pass
        
        try:
            _orig_shm_del(self)
        except (OSError, KeyError, AttributeError): pass
        
        try:
            self.unlink()
        except (OSError, KeyError, AttributeError, FileNotFoundError): pass

    multiprocessing.shared_memory.SharedMemory.__del__ = _safe_shm_del


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
                except (AttributeError, KeyError, OSError, FileNotFoundError, BufferError, ValueError): pass
        except (AttributeError, KeyError, OSError, FileNotFoundError, BufferError, ValueError): pass
