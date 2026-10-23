"""Serving: pinned releases and the licence gate, the service's timings and checks, the HTTP front end."""
import base64
import hashlib
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from signtranslator.serving.release import Corpus, GeneratorSpec, Pinned, Release, fetch_pinned
from signtranslator.serving.service import TranslatorService
from tests.test_pose_resample import state

SHA = "a" * 64


def release(**kw):
    base = dict(name="r", version="1", generator=GeneratorSpec(Pinned("g.pt", SHA), SHA),
                provenance=(Corpus("How2Sign", "CC BY-NC 4.0", False, True),))
    base.update(kw)
    return Release(**base)


def test_licence_gate_and_round_trip():
    r = release()
    assert r.public_release_allowed and not r.commercial_use_allowed
    research = release(provenance=(Corpus("ASLLRP", "terms", True, permission="requested 2026-10-22"),))
    assert not research.public_release_allowed and not research.commercial_use_allowed
    cleared = release(provenance=(Corpus("ASLLRP", "terms", True, permission="granted 2026-11-01 (email)"),))
    assert cleared.public_release_allowed
    assert Release.from_dict(json.loads(json.dumps(r.to_dict()))) == r
    with pytest.raises(ValueError):
        release(generator=GeneratorSpec(Pinned("g.pt", SHA), SHA, condition="plan"))      # plan needs a planner
    with pytest.raises(ValueError):
        Pinned("g.pt", "short")
    with pytest.raises(ValueError):
        release(provenance=())


def test_pinned_models_are_hash_checked():
    class Store:
        def get(self, key):
            return b"model"
    assert fetch_pinned(Store(), Pinned("k", hashlib.sha256(b"model").hexdigest())) == b"model"
    with pytest.raises(ValueError, match="pinned"):
        fetch_pinned(Store(), Pinned("k", SHA))


def fake_service(delay_event=None):
    calls = []

    def sampler(text, fps, seconds, guidance, steps, seed):
        calls.append((text, fps, seconds, seed))
        if delay_event is not None:
            delay_event.wait(5)
        return state(np.arange(int(round(seconds * fps))) / fps)
    return TranslatorService(release(), None, sampler), calls


def test_translate_times_stages_and_checks_text():
    service, calls = fake_service()
    out = service.translate("  hello   there  ", seed=3)
    assert calls == [("hello there", 24., 1., 3)] and out.frames == 24 and out.plan_line is None
    assert set(out.timings_ms) == {"generate", "total"} and out.state_npz[:2] == b"PK"
    for bad in ("", "   ", None, "x" * 301):
        with pytest.raises(ValueError):
            service.check_text(bad)


def test_http_front_end_answers_and_applies_backpressure():
    from http.server import ThreadingHTTPServer
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from serve_translator import make_handler
    gate = threading.Event()
    service, _ = fake_service(delay_event=gate)
    r = service.release
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, r.to_dict(), r.sha256(), max_inflight=1))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def post(body):
        req = urllib.request.Request(url + "/translate", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        health = json.loads(urllib.request.urlopen(url + "/health", timeout=5).read())
        assert health["status"] == "ok" and health["release_sha256"] == r.sha256()
        assert post({"text": ""})[0] == 400
        first = {}
        t = threading.Thread(target=lambda: first.setdefault("r", post({"text": "hello", "include_state": True})))
        t.start()
        import time
        for _ in range(100):                              # wait until the first request holds the only slot
            if service._generate.locked():
                break
            time.sleep(.02)
        assert post({"text": "second"})[0] == 503          # bounded: refused, not queued
        gate.set(); t.join(10)
        code, body = first["r"]
        assert code == 200 and body["frames"] > 0 and base64.b64decode(body["state_npz_base64"])[:2] == b"PK"
    finally:
        gate.set(); server.shutdown()


def test_release_post_processing_round_trips_and_is_applied():
    from signtranslator.generation.postprocess import calibrate_face, smooth_rotation_channel
    from signtranslator.pose.multichannel import MultichannelMotion
    s0 = state(np.arange(48) / 24.)
    labels = list(s0.channels["face"].labels)
    calibration = dict(labels=labels, center_generated=[.1] * len(labels), center_real=[.2] * len(labels),
                       gain=[2.] * len(labels))
    plain = release()
    post = release(generator=GeneratorSpec(Pinned("g.pt", SHA), SHA, head_smoothing=.2, face_calibration=calibration))
    assert "head_smoothing" not in plain.to_dict()["generator"]          # unused: omitted, old SHA kept
    assert Release.from_dict(json.loads(json.dumps(post.to_dict()))) == post and post.sha256() != plain.sha256()
    with pytest.raises(ValueError):
        GeneratorSpec(Pinned("g.pt", SHA), SHA, face_calibration=dict(gain=[1.]))

    def sampler(text, fps, seconds, guidance, steps, seed):
        return s0
    out = TranslatorService(post, None, sampler).translate("hello")
    expected = calibrate_face(smooth_rotation_channel(s0, "head_rotation", .2), calibration)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "s.npz"; p.write_bytes(out.state_npz)
        got = MultichannelMotion.load(p, expected_sha256=hashlib.sha256(out.state_npz).hexdigest())
    np.testing.assert_allclose(got.channels["face"].values, expected.channels["face"].values, atol=1e-6)
    np.testing.assert_allclose(got.channels["head_rotation"].values, expected.channels["head_rotation"].values, atol=1e-6)
