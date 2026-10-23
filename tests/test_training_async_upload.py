"""Background uploads keep order, bound the queue and surface errors in the training loop."""
import threading
import time

import pytest

from signtranslator.training.async_upload import Uploader


class SlowStore:
    def __init__(self, fail_on=None):
        self.order, self.fail_on, self.lock = [], fail_on, threading.Lock()

    def put(self, key, payload):
        time.sleep(.02)
        if key == self.fail_on:
            raise OSError("R2 unavailable")
        with self.lock:
            self.order.append(key)


def test_uploads_run_in_order_and_drain():
    store = SlowStore()
    u = Uploader(store, max_pending=2)
    for i in range(5):
        u.submit([(f"ck-{i}", b"x"), (f"best-{i}", b"y")])
    u.drain()
    assert store.order == [k for i in range(5) for k in (f"ck-{i}", f"best-{i}")]
    assert not u.pending


def test_an_upload_error_reaches_the_training_loop():
    u = Uploader(SlowStore(fail_on="ck-0"))
    u.submit([("ck-0", b"x")])
    with pytest.raises(OSError):
        u.drain()
