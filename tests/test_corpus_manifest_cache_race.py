"""ContentCache tolerates entries removed concurrently (the recogniser v3 gates crash, 2026-10-29)."""
import hashlib
import threading

from signtranslator.data_engineering.corpus_manifest import ContentCache


def test_size_and_evict_tolerate_concurrent_removal(tmp_path, monkeypatch):
    cache = ContentCache(tmp_path, max_bytes=50)
    blobs = [bytes([i]) * 20 for i in range(8)]
    for b in blobs:
        cache.get("k", hashlib.sha256(b).hexdigest(), lambda _k, b=b: b)
    real_stat = type(tmp_path).stat
    hit = {"n": 0}

    def flaky(self, *a, **k):                      # the first entry vanishes between listing and stat
        if self.parent == tmp_path and len(self.name) == 64 and hit["n"] == 0:
            hit["n"] += 1
            self.unlink(missing_ok=True)
            raise FileNotFoundError(str(self))
        return real_stat(self, *a, **k)
    monkeypatch.setattr(type(tmp_path), "stat", flaky)
    assert cache.size() >= 0
    extra = b"z" * 20
    cache.get("k", hashlib.sha256(extra).hexdigest(), lambda _k: extra)
    assert cache.size() <= 50 + 20


def test_parallel_gets_do_not_raise(tmp_path):
    cache = ContentCache(tmp_path, max_bytes=100)
    blobs = [bytes([i % 251]) * 30 + i.to_bytes(4, "big") for i in range(200)]
    errors = []

    def work(chunk):
        try:
            for b in chunk:
                cache.get("k", hashlib.sha256(b).hexdigest(), lambda _k, b=b: b)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=work, args=(blobs[i::8],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_heavy_parallel_gets_with_tiny_cache_do_not_raise(tmp_path):
    """16 threads on a cache far smaller than the working set: reads race evictions (the reviewer's stress case)."""
    cache = ContentCache(tmp_path, max_bytes=200)
    blobs = [bytes([i % 251]) * 40 + i.to_bytes(4, "big") for i in range(64)]
    errors = []

    def work(seed):
        try:
            for j in range(400):
                b = blobs[(seed * 7 + j) % len(blobs)]
                assert cache.get("k", hashlib.sha256(b).hexdigest(), lambda _k, b=b: b) == b
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
