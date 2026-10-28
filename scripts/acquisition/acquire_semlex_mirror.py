"""Acquire the Sem-Lex Benchmark pose files into R2 from a public Hugging Face mirror (owner-provided, 2026-10-28).

The publisher's own download links (Sem-Lex, Kezar et al. 2023, arXiv 2310.00196; terms CC BY-NC-SA 4.0 accepted by the
owner on the publisher's form) open a Google Drive restricted to Boston University. The mirror
`tylerxia8/sem-lex-pilot-mirror` holds four files: `semlex_metadata.csv` and `train/val/test.tar.gz`. It labels itself
Apache-2.0, which is not the publisher's licence: the data is used under the publisher's CC BY-NC-SA 4.0 terms only
(research, non-commercial, attribution, no redistribution from our private store).

Why it is accepted as the publisher's data (checked 2026-10-28, recorded in each object's provenance):
1. the metadata has exactly the publisher's 91,148 clips (README: "91,148 isolated sign videos"), with the paper's
   ASL-LEX phonological columns, 44 signer ids and ASL-LEX / SignBank / free-text label types;
2. the archives' members are owned by user ``lee`` and dated 2023-10-23 (the paper's month), one ``<video_id>.npy``
   per clip: MediaPipe Holistic landmarks, (frames, 553, 3) float16 (478 face mesh, 33 pose, 2 x 21 hands; normalised
   image x, y and relative z; NaN when undetected), about 30 fps; the "pose estimation files" the publisher's README
   announces; every sampled member's id is in the metadata;
3. every file's SHA-256 must equal the mirror's Git LFS id at the pinned revision; the streamed bytes are hashed, the
   stored R2 object is re-read and re-hashed (acquire_resumable.py).
It is a third-party copy, so its completeness against the publisher's is checked after transfer (member count per split
against the metadata), not assumed.

Usage: python scripts/acquisition/acquire_semlex_mirror.py [LEDGER.jsonl]
"""
import csv, hashlib, io, json, subprocess, sys, urllib.request
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
PY = sys.executable
HERE = Path(__file__).resolve().parent
LEDGER = sys.argv[1] if len(sys.argv) > 1 else 'acquisition-ledger.jsonl'
MIRROR = 'tylerxia8/sem-lex-pilot-mirror'
REVISION = 'bf340cbfc4522ecc4dcd0fc247e33c4cb3498275'          # pinned commit (2026-05-25)
BASE = f'https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/'
UA = {'User-Agent': 'Sign-Translator-research-acquisition/1.0'}
META_SHA256 = 'f4250b2877e738f028e4b9517922952a350ec8952e59d53ceabdd778938fc3c3'
FILES = ('data/semlex_metadata.csv', 'data/test.tar.gz', 'data/val.tar.gz', 'data/train.tar.gz')


def paths_info(path):
    """(size, sha256) the mirror declares for a file at the pinned revision."""
    url = f'https://huggingface.co/api/datasets/{MIRROR}/paths-info/{REVISION}'
    req = urllib.request.Request(url, data=json.dumps({'paths': [path]}).encode(), headers={**UA, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        info = json.load(r)[0]
    return info['size'], info['lfs']['oid']


with urllib.request.urlopen(urllib.request.Request(BASE + FILES[0], headers=UA), timeout=120) as r:
    meta = r.read()
if hashlib.sha256(meta).hexdigest() != META_SHA256:
    raise SystemExit('mirror metadata changed since it was checked')
rows = list(csv.DictReader(io.StringIO(meta.decode())))
if len(rows) != 91148:
    raise SystemExit(f'metadata has {len(rows)} clips, the publisher states 91,148')
print('mirror metadata: 91,148 clips, SHA-256 as checked', flush=True)

for path in FILES:
    size, sha256 = paths_info(path)
    evidence = dict(mirror=MIRROR, revision=REVISION, mirror_lfs_sha256=sha256, metadata_rows=len(rows),
                    checked='sha256 equal to mirror LFS id; metadata 91,148 clips (publisher README); members owned by '
                            'lee, dated 2023-10-23, MediaPipe Holistic (553 points)',
                    mirror_label='apache-2.0 (incorrect; publisher terms apply)')
    print(f'== {path} {size:,} bytes sha256 {sha256}', flush=True)
    r = subprocess.run([PY, '-I', str(HERE / 'acquire_resumable.py'), REPO, BASE + path,
                        'sem-lex-mirror-v1/' + path, '--expected-size', str(size), '--expected-sha256', sha256,
                        '--verify-existing', '--license', 'CC BY-NC-SA 4.0 (Sem-Lex Benchmark terms of use)',
                        '--source-page', 'https://github.com/leekezar/SemLex',
                        '--mirror-of', 'https://github.com/leekezar/SemLex (publisher download links)',
                        '--evidence', json.dumps(evidence), '--ledger', LEDGER], capture_output=True, text=True)
    out = [l for l in (r.stdout + r.stderr).splitlines() if not l.endswith(' MiB')]
    print('\n'.join(out[-3:]), flush=True)
    if r.returncode:
        sys.exit(r.returncode)
print('SEMLEX-MIRROR-DONE', flush=True)
