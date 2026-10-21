"""Clip claims of the Sapiens fleet: a lost race skips the clip, R2's same-object 503 counts as a lost race."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("sapiens_runpod", Path(__file__).resolve().parents[1] / "scripts/sapiens_keypoints_runpod.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class Store:
    def __init__(self, error=None):
        self.objects, self.error = {}, error
    def exists(self, key):
        return key in self.objects
    def get(self, key):
        return self.objects[key]
    def put(self, key, payload):
        if self.error:
            raise self.error
        self.objects[key] = payload


def test_a_fresh_claim_blocks_others_and_an_old_one_is_taken_over():
    import time
    store = Store()
    store.objects["c/a"] = json.dumps(dict(time=time.time(), host="another-pod")).encode()
    assert not module.claim(store, "c/a")                              # another pod is on it
    store.objects["c/b"] = json.dumps(dict(time=0., host="another-pod")).encode()   # a crashed pod's claim
    assert module.claim(store, "c/b")


def test_a_restarted_job_takes_back_its_own_fresh_claims():
    store = Store()
    assert module.claim(store, "c/a")
    assert module.claim(store, "c/a")             # same host: the clip it was processing when it restarted


def test_concurrent_write_refusal_is_a_lost_race_and_other_errors_still_raise():
    refused = Exception("An error occurred (ServiceUnavailable) when calling the PutObject operation: "
                        "Reduce your concurrent request rate for the same object.")
    assert module.claim(Store(refused), "c/x") is False
    with pytest.raises(RuntimeError):                                  # not a race: surfaces
        module.claim(Store(RuntimeError("AccessDenied")), "c/y")
