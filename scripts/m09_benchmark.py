"""M09 measured capacity benchmark on this host; measurements only, no claims.

Each workload runs in its own subprocess so peak resident memory is attributable
to that workload. Timings use a monotonic clock after explicit warm-up and report
order statistics over repeated runs. Extrapolations are labelled as such and
are linear in measured throughput, nothing more. The canonical-state kernel is
profiled at declared, realistic channel sizes on synthetic tensors: it measures
arithmetic cost, not a governed run, data loading or model quality.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

# Declared realistic sizes: 10 s at 30 fps, 40 reviewed events, a 22-joint body,
# 15 joints per hand and a 52-coefficient ARKit-style face basis.
SIZES = dict(batch=8, frames=300, events=40, body=22, hand=15, face=52)


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def timings(values):
    return dict(n=len(values), p50=percentile(values, .5), p95=percentile(values, .95),
                p99=percentile(values, .99), mean=statistics.fmean(values),
                min=min(values), max=max(values))


def peak_rss_bytes():
    # ru_maxrss is bytes on macOS and kibibytes on Linux.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


# ---------------------------------------------------------------------------
# workloads (each executed in a child process)
# ---------------------------------------------------------------------------
def workload_legacy_epoch(_device):
    import tempfile
    import torch
    sys.path.insert(0, str(ROOT))
    from signtranslator.run import run_pipeline
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as directory:
        start = time.monotonic()
        result = run_pipeline(os.path.join(directory, "corpus"), epochs=1, batch_size=32,
                              regenerate=True, do_analyze=False, verbose=False)
        elapsed = time.monotonic() - start
    steps = result["trainer"].global_step
    return dict(epoch_seconds=elapsed, optimizer_steps=steps, steps_per_second=steps / elapsed,
                parameters=result["model"].num_parameters(),
                note="includes synthetic corpus generation and validation")


def _canonical_inputs(device, *, batch, frames, events):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT))
    from signtranslator.data.multichannel import ChannelBatch, ChannelLayout, ChannelProvenance
    from signtranslator.generation.canonical_state import (
        CanonicalChannelBinding, CanonicalStateConfig, CanonicalStateSIRDecoder,
    )
    from signtranslator.pose.multichannel import CHANNELS
    counts = dict(body=SIZES["body"], left_hand=SIZES["hand"], right_hand=SIZES["hand"],
                  face=SIZES["face"])
    bindings = tuple(CanonicalChannelBinding(
        name, tuple(f"{name}-{i}" for i in range(counts.get(name, 1))), "bench-frame",
        units, "bench-convention") for name, (_, _, units) in CHANNELS.items())
    config = CanonicalStateConfig(
        hidden_dim=128, event_dim=128, attention_heads=4, frame_frequencies=6,
        translation_scale_metres=.05, coefficient_scale=.5, velocity_weight=1.,
        max_frames=4096, max_events=128, lexicon_sha256="0" * 64,
        convention_sha256="1" * 64, locus_count=8, bindings=bindings)
    # The profiling kernel builds exactly the governed model's layers.
    model = CanonicalStateSIRDecoder.profiling_kernel(config, label_count=64)
    cfg = model.model_cfg
    model = model.to(device)
    generator = torch.Generator().manual_seed(0)
    frames_t = torch.rand((batch, frames, 1 + 2 * cfg.frame_frequencies), generator=generator)
    events_t = torch.randn((batch, events, cfg.event_dim), generator=generator)
    starts = torch.randint(0, frames - 20, (batch, events), generator=generator)
    membership = torch.zeros((batch, events, frames), dtype=torch.bool)
    for b in range(batch):
        for e in range(events):
            membership[b, e, starts[b, e]:starts[b, e] + 20] = True
    lengths = torch.full((batch,), frames, dtype=torch.int64)
    frame_valid = torch.ones((batch, frames), dtype=torch.bool)
    timestamps = (torch.arange(frames, dtype=torch.float64) / 30.).expand(batch, frames).clone()
    channels = {}
    for binding in bindings:
        kind = binding.kind
        shape = (batch, frames, binding.elements, binding.width)
        if kind == "rotation_6d_columns":
            values = torch.tensor([1., 0, 0, 0, 1, 0], dtype=torch.float64).expand(shape).clone()
        elif kind == "unit_vector":
            values = torch.tensor([0., 0, 1], dtype=torch.float64).expand(shape).clone()
        elif kind == "bounded_scalar":
            values = torch.full(shape, .5, dtype=torch.float64)
        else:
            values = torch.zeros(shape, dtype=torch.float64)
        mask = torch.ones(shape[:3], dtype=torch.bool)
        channels[binding.name] = ChannelBatch(
            values.to(device), mask.to(device), mask.to(device),
            torch.ones(shape[:3], dtype=torch.float64, device=device),
            ChannelLayout(binding.labels, binding.coordinate_frame, binding.units,
                          binding.convention),
            tuple(ChannelProvenance("bench", "2" * 64, None) for _ in range(batch)))
    del np
    return (model, frames_t.to(device), events_t.to(device), membership.to(device), lengths,
            frame_valid.to(device), timestamps.to(device), channels)


def workload_canonical_train(device):
    import torch
    sys.path.insert(0, str(ROOT))
    from signtranslator.generation.canonical_state import (
        CANONICAL_BRANCHES, channel_loss_per_example,
    )
    torch.manual_seed(0)
    model, frames, events, membership, lengths, frame_valid, timestamps, channels = \
        _canonical_inputs(device, batch=SIZES["batch"], frames=SIZES["frames"],
                          events=SIZES["events"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        optimizer.zero_grad(set_to_none=True)
        predictions = model.heads_for(model.condition(frames, events, membership, lengths,
                                                      frame_valid))
        total = 0.
        for name in CANONICAL_BRANCHES:
            values, _, _ = channel_loss_per_example(
                name, predictions[name], channels[name], frame_valid,
                config=model.model_cfg, timestamps=timestamps)
            total = total + values.mean()
        total.backward()
        optimizer.step()
        if device != "cpu":
            torch.mps.synchronize()

    for _ in range(3):
        step()
    measured = []
    for _ in range(20):
        start = time.monotonic()
        step()
        measured.append(time.monotonic() - start)
    stats = timings(measured)
    clips = SIZES["batch"] / stats["p50"]
    return dict(step_seconds=stats, clips_per_second_p50=clips,
                frames_per_second_p50=clips * SIZES["frames"],
                parameters=sum(p.numel() for p in model.parameters()), sizes=SIZES)


def workload_canonical_infer(device):
    import torch
    torch.manual_seed(0)
    model, frames, events, membership, lengths, frame_valid, _, _ = \
        _canonical_inputs(device, batch=1, frames=SIZES["frames"], events=SIZES["events"])
    model.eval()

    def run():
        with torch.no_grad():
            model.heads_for(model.condition(frames, events, membership, lengths, frame_valid))
        if device != "cpu":
            torch.mps.synchronize()

    for _ in range(5):
        run()
    measured = []
    for _ in range(50):
        start = time.monotonic()
        run()
        measured.append(time.monotonic() - start)
    stats = timings(measured)
    return dict(clip_seconds_of_motion=SIZES["frames"] / 30., latency_seconds=stats,
                real_time_factor_p95=stats["p95"] / (SIZES["frames"] / 30.))


def workload_rig_pose(_device):
    import torch
    sys.path.insert(0, str(ROOT))
    from signtranslator.avatar_render.makehuman import load_makehuman_rig
    assets = (ROOT / "Sign Translator Stage Documentation" / "evidence"
              / "w1-w2-2026-09-26/source-intake/makehuman")
    start = time.monotonic()
    rig = load_makehuman_rig(assets)
    load_seconds = time.monotonic() - start
    identity = torch.eye(3, dtype=rig.rest_vertices.dtype).repeat(len(rig.bone_names), 1, 1)
    for _ in range(2):
        rig.pose(identity)
    measured = []
    for _ in range(15):
        start = time.monotonic()
        rig.pose(identity)
        measured.append(time.monotonic() - start)
    stats = timings(measured)
    return dict(rig_load_seconds=load_seconds, pose_seconds=stats,
                frames_per_second_p50=1. / stats["p50"], bones=len(rig.bone_names),
                vertices=int(rig.rest_vertices.shape[0]), dtype=str(rig.rest_vertices.dtype),
                real_time_30fps_feasible_p95=stats["p95"] <= 1. / 30.)


WORKLOADS = dict(legacy_epoch=workload_legacy_epoch,
                 canonical_train=workload_canonical_train,
                 canonical_infer=workload_canonical_infer,
                 rig_pose=workload_rig_pose)


def child(name, device):
    try:
        result = dict(status="measured", **WORKLOADS[name](device))
    except Exception as error:  # report, never hide, a failing workload
        result = dict(status="failed", error=f"{type(error).__name__}: {error}")
    result["peak_rss_bytes"] = peak_rss_bytes()
    print(json.dumps(result))


def environment():
    import torch
    usage = shutil.disk_usage(ROOT)
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return None
    return dict(platform=platform.platform(), machine=platform.machine(),
                cpu=sysctl("machdep.cpu.brand_string"), logical_cpus=os.cpu_count(),
                memory_bytes=int(sysctl("hw.memsize") or 0), python=platform.python_version(),
                torch=torch.__version__, torch_threads=torch.get_num_threads(),
                mps_available=bool(torch.backends.mps.is_available()),
                disk_total_bytes=usage.total, disk_free_bytes=usage.free)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=2, metavar=("WORKLOAD", "DEVICE"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.child:
        child(*args.child)
        return 0
    if args.output is None:
        parser.error("--output is required")
    env = environment()
    plan = [("legacy_epoch", "cpu"), ("canonical_train", "cpu"), ("canonical_infer", "cpu"),
            ("rig_pose", "cpu")]
    if env["mps_available"]:
        plan += [("canonical_train", "mps"), ("canonical_infer", "mps")]
    results = {}
    for name, device in plan:
        start = time.monotonic()
        completed = subprocess.run([sys.executable, "-I", __file__, "--child", name, device],
                                   capture_output=True, text=True, cwd=ROOT)
        wall = time.monotonic() - start
        lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        record = json.loads(lines[-1]) if lines else dict(
            status="failed", error=completed.stderr.strip()[-2000:])
        record["process_wall_seconds"] = wall
        results[f"{name}:{device}"] = record
        print(f"{name}:{device} -> {record['status']} ({wall:.1f}s)", flush=True)
    train = results.get("canonical_train:cpu", {})
    extrapolation = None
    if train.get("status") == "measured":
        # Linear extrapolation in measured clip throughput only.
        clips = 31044
        extrapolation = dict(
            assumption="31,044 trainable How2Sign-scale clips of 10 s each, CPU, this kernel only",
            seconds_per_epoch=clips / train["clips_per_second_p50"],
            hours_per_100_epochs=100 * clips / train["clips_per_second_p50"] / 3600,
            excludes=["data loading and admission", "validation", "checkpoint I/O",
                      "seeds and ablations", "restarts"])
    record = dict(schema_version=1, environment=env, sizes=SIZES, results=results,
                  linear_extrapolation=extrapolation,
                  limitations=[
                      "Synthetic tensors measure arithmetic cost only, not model quality.",
                      "No real source conversion was available to measure.",
                      "Single host; thermal state and background load are uncontrolled.",
                      "Extrapolations are linear and exclude the listed costs."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
