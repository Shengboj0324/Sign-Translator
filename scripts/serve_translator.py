"""HTTP front end for the translation service (plan item 6): ``python scripts/serve_translator.py --release release.json``.

Endpoints (JSON):

- ``GET /health`` -> ``{"status": "ok", "release": name, "version": ..., "release_sha256": ...}``;
- ``GET /release`` -> the release manifest, including ``public_release_allowed`` (the licence gate);
- ``POST /translate`` with ``{"text": "...", "seed": 0, "render": false, "include_state": false}`` -> the plan line,
  glosses, duration, frames and per-stage timings; ``include_state`` adds the generated state (``.npz``, base64) and
  ``render`` the preview video (``.mp4``, base64).

**Backpressure.** At most ``--max-inflight`` requests are admitted (a bounded semaphore, non-blocking): the next one is
answered ``503`` with ``Retry-After`` instead of queueing without bound (docs/DEPLOYMENT.md §3: the queue stays bounded,
so latency stays bounded). Bodies over 16 kB are refused (413), malformed requests get 400 with the reason. The server
binds to 127.0.0.1 unless ``--host`` says otherwise. A release whose ``public_release_allowed`` is false
(research-only training data) refuses to bind to a non-loopback address unless ``--research-use`` is given.
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_BODY = 16 * 1024


def make_handler(service, release_dict: dict, release_sha: str, max_inflight: int):
    slots = threading.BoundedSemaphore(max_inflight)

    class Handler(BaseHTTPRequestHandler):
        server_version = "SignTranslator/1"

        def log_message(self, fmt, *args):          # one line per request on stderr
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, payload: dict, headers: dict | None = None):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, dict(status="ok", release=release_dict["name"], version=release_dict["version"],
                                            release_sha256=release_sha))
            if self.path == "/release":
                return self._send(200, release_dict)
            return self._send(404, dict(error="not found"))

        def do_POST(self):
            if self.path != "/translate":
                return self._send(404, dict(error="not found"))
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, dict(error="request body too large"))
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(request, dict):
                    raise ValueError("body must be a JSON object")
                seed = int(request.get("seed", 0))
                render, include_state = bool(request.get("render", False)), bool(request.get("include_state", False))
                text = service.check_text(request.get("text"))
            except (ValueError, TypeError) as e:
                return self._send(400, dict(error=str(e)))
            if not slots.acquire(blocking=False):
                return self._send(503, dict(error="busy"), {"Retry-After": "2"})
            try:
                result = service.translate(text, seed=seed, render=render)
            except Exception as e:                    # a model failure is the server's, not the client's
                return self._send(500, dict(error=f"{type(e).__name__}: {e}"))
            finally:
                slots.release()
            out = result.summary()
            if include_state:
                out["state_npz_base64"] = base64.b64encode(result.state_npz).decode()
            if result.video_mp4:
                out["video_mp4_base64"] = base64.b64encode(result.video_mp4).decode()
            return self._send(200, out)

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-inflight", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--research-use", action="store_true",
                        help="allow a research-only release on a non-loopback address (never a public service)")
    args = parser.parse_args(argv)
    from signtranslator.serving.release import Release
    from signtranslator.serving.service import TranslatorService
    from train_text_to_motion import make_store
    release = Release.from_dict(json.loads(args.release.read_text()))
    if not release.public_release_allowed and args.host not in ("127.0.0.1", "localhost", "::1") and not args.research_use:
        raise SystemExit("this release has research-only training data: bind to loopback or pass --research-use")
    service = TranslatorService.from_release(make_store(args.store), release, device=args.device)
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(service, release.to_dict(), release.sha256(), args.max_inflight))
    print(json.dumps(dict(listening=f"http://{args.host}:{args.port}", release=release.name, version=release.version,
                          public_release_allowed=release.public_release_allowed)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
