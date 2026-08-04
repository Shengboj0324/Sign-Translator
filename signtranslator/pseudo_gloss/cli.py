"""Offline CLI for verified pseudo-gloss model bundles and single-record inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .artifacts import (
    load_pipeline_bundle,
    verify_bundle,
    verify_candidate_batch,
)
from .inference import (
    InferenceRequest,
    load_inference_context,
    run_inference,
)
from .corpus import run_corpus_inference
from .readiness import assess_activation, load_activation_charter


def _infer(args: argparse.Namespace) -> int:
    context = load_inference_context(args.model_bundle, args.dataset_authorization)
    run_inference(context, InferenceRequest(
        transcript_file=Path(args.transcript_file),
        source_video=Path(args.source_video), landmark_track=Path(args.landmark_track),
        sample_id=args.sample_id, created_at=args.created_at, output=Path(args.output)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline, fail-closed pseudo-gloss candidate research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_model = subparsers.add_parser("verify-model")
    verify_model.add_argument("model_bundle")
    verify_batch = subparsers.add_parser("verify-batch")
    verify_batch.add_argument("candidate_batch")
    verify_batch.add_argument("model_bundle")
    verify_batch.add_argument("dataset_authorization")
    infer = subparsers.add_parser("infer-one")
    infer.add_argument("--model-bundle", required=True)
    infer.add_argument("--transcript-file", required=True)
    infer.add_argument("--source-video", required=True)
    infer.add_argument("--landmark-track", required=True)
    infer.add_argument("--dataset-authorization", required=True)
    infer.add_argument("--sample-id", required=True)
    infer.add_argument("--created-at", required=True,
                       help="fixed ISO-8601 provenance timestamp")
    infer.add_argument("--output", required=True)
    infer_corpus = subparsers.add_parser("infer-corpus")
    infer_corpus.add_argument("--input-manifest", required=True)
    infer_corpus.add_argument("--output", required=True)
    infer_corpus.add_argument("--model-bundle", required=True)
    infer_corpus.add_argument("--dataset-authorization", required=True)
    infer_corpus.add_argument("--activation-charter", required=True)
    infer_corpus.add_argument("--resume", action="store_true")
    readiness = subparsers.add_parser("assess-readiness")
    readiness.add_argument("charter")
    args = parser.parse_args(argv)
    if args.command == "verify-model":
        verify_bundle(args.model_bundle)
        load_pipeline_bundle(args.model_bundle)
        return 0
    if args.command == "verify-batch":
        verify_candidate_batch(
            args.candidate_batch, model_bundle=args.model_bundle,
            dataset_authorization=args.dataset_authorization)
        return 0
    if args.command == "assess-readiness":
        report = assess_activation(load_activation_charter(args.charter))
        print(json.dumps({
            "activation_approved": report.activation_approved,
            "linguistic_validation_approved": report.linguistic_validation_approved,
            "production_gloss_export_approved": report.production_gloss_export_approved,
            "checks": [asdict(check) for check in report.checks],
        }, sort_keys=True))
        return 0 if report.activation_approved else 2
    if args.command == "infer-corpus":
        run_corpus_inference(
            input_manifest=args.input_manifest, output=args.output,
            model_bundle=args.model_bundle,
            dataset_authorization=args.dataset_authorization,
            activation_charter=args.activation_charter, resume=args.resume)
        return 0
    return _infer(args)
