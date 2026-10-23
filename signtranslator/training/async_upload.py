"""Checkpoint uploads off the training loop's critical path.

A full checkpoint (weights, EMA and optimiser state) of the token prior is about 600 MB. Uploading it to R2 took
2-4 minutes, during which the GPU sat idle. That was about 30% of wall time when every validation set a new best.
``Uploader`` puts objects from one background thread, in submission order, so a checkpoint is always written before
the ``best.json`` that names it. At most ``max_pending`` jobs are queued: submitting more waits for the oldest, which
bounds memory. Every upload error re-raises in the training loop at the next ``submit``, or at ``drain``, so a
failed upload never passes silently.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor


class Uploader:
    def __init__(self, store, *, max_pending: int = 2) -> None:
        self.store, self.max_pending = store, max_pending
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="upload")
        self.pending: list[Future] = []

    def _check(self, *, wait_all: bool = False) -> None:
        while self.pending and (wait_all or self.pending[0].done() or len(self.pending) >= self.max_pending):
            self.pending.pop(0).result()                     # re-raises an upload error here

    def submit(self, items: list[tuple[str, bytes]]) -> None:
        """Put ``(key, payload)`` items in order, after every earlier submission."""
        self._check()

        def put_all():
            for key, payload in items:
                self.store.put(key, payload)
        self.pending.append(self.pool.submit(put_all))

    def drain(self) -> None:
        self._check(wait_all=True)


__all__ = ["Uploader"]
