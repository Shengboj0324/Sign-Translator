"""Frozen sentence embeddings for the retrieval evaluator's text side (``TextMotionRetrievalConfig.text_embedding_key``).

Every distinct sentence of the given manifests - a record's ``text`` and, for plan records, its ``english`` (meaning is
judged against English) - is embedded once with a pretrained sentence encoder (default
``sentence-transformers/all-MiniLM-L6-v2``, Apache-2.0: mean-pooled last hidden states, L2-normalised, as the model
card specifies) and stored as one ``.npz`` (``keys``: ``text_motion_retrieval.text_key`` of each sentence, i.e. the
SHA-256 of the bytes the generators' encoder sees; ``vectors``: float32). The encoder runs here, once; training and
scoring pods read only the table, so they need no extra dependency. Embedding a sentence uses no motion, so
validation and test sentences may be in the table: the retrieval model is still trained on training pairs only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.generation.text_motion_retrieval import text_key  # noqa: E402


def sentences_of(manifest) -> list[str]:
    out = []
    for r in manifest["records"]:
        for field in ("text", "english"):
            t = (r.get(field) or "").strip()
            if t:
                out.append(t)
    return out


def embed(texts: list[str], model_name: str, *, batch: int = 128) -> tuple[np.ndarray, str]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    revision = getattr(model.config, "_commit_hash", None) or "unknown"
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tokenizer(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"][..., None].to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            out.append(torch.nn.functional.normalize(pooled, dim=-1).numpy())
    return np.concatenate(out).astype(np.float32), revision


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest", action="append", required=True, help="KEY@SHA (repeatable)")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-key", required=True)
    args = parser.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    texts = []
    for spec in args.manifest:
        key, _, sha = spec.rpartition("@")
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.json"; p.write_bytes(store.get(key) + b"\n")
            texts += sentences_of(load_manifest(p, expected_sha256=sha))
    by_key = {}
    for t in texts:
        by_key.setdefault(text_key(t), t)
    keys = sorted(by_key)
    vectors, revision = embed([by_key[k] for k in keys], args.model)
    buffer = io.BytesIO()
    np.savez(buffer, keys=np.array(keys), vectors=vectors)
    if store.exists(args.output_key):
        raise SystemExit("refusing to overwrite an existing embedding table")
    digest = store.put(args.output_key, buffer.getvalue())
    meta = dict(model=args.model, revision=revision, texts=len(keys), dim=int(vectors.shape[1]),
                manifests=args.manifest, table_sha256=digest)
    store.put(args.output_key + ".json", json.dumps(meta, indent=1).encode())
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
