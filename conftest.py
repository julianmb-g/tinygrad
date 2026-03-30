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
        # Explicitly release shared memory buffers to prevent IPC deadlocks
        try:
            if hasattr(self, '_mmap') and self._mmap is not None and not getattr(self._mmap, 'closed', True):
                if hasattr(self, 'buf') and self.buf is not None:
                    memoryview(self.buf).release()  # Enforce IPC teardown memory release
                    try:
                        os.unlink(f"/dev/shm/{self.name}")
                    except OSError:
                        pass
        except (AttributeError, KeyError, FileNotFoundError): pass

        try:
            _orig_shm_del(self)
        except (AttributeError, KeyError, FileNotFoundError): pass

        try:
            self.unlink()
        except (AttributeError, KeyError, FileNotFoundError, OSError): pass

    multiprocessing.shared_memory.SharedMemory.__del__ = _safe_shm_del

import multiprocessing.connection
# Restore targeted Connection.send OS muzzling for severed IPC disconnections
_orig_send = multiprocessing.connection.Connection.send

def _safe_send(self, obj):
    try:
        _orig_send(self, obj)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass

multiprocessing.connection.Connection.send = _safe_send


try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

os.environ["DISABLE_COMPILER_CACHE"] = "1"

active_pids = set()

def teardown_worker_group():
    import threading
    # Explicitly join background threads to synchronize IPC termination
    for t in threading.enumerate():
        if t is not threading.current_thread() and not t.daemon:
            try:
                t.join(timeout=1.0) # Synchronize IPC thread termination
            except RuntimeError:
                pass
    try:
        for pid in list(active_pids):
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
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
                except (AttributeError, KeyError, FileNotFoundError, OSError): pass
        except (AttributeError, KeyError, FileNotFoundError): pass
