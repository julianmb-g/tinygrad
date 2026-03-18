import os


def pytest_configure(config):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None:
        from tinygrad.helpers import init_worker_process_group
        init_worker_process_group()
