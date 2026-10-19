"""Acquire the How2Sign RGB clip zips into R2 from a byte-identical Hugging Face mirror.

The publisher's Google Drive copies are behind an exhausted download quota, and the
institutional copy (CORA, doi:10.34810/DATA33) sits behind an anti-bot challenge.
Neither is circumvented. The public mirror `aipieces/How2Sign` holds the publisher's
files under `original/`; it is accepted only because, and only while, it is provably
the same bytes:

1. its six translation CSVs are SHA-256-identical to the copies acquired from the
   publisher's Drive (checked again here before anything is transferred);
2. every zip has exactly the byte count the publisher's Drive reported for it
   (Content-Range, 2026-10-18);
3. the mirror publishes each zip's SHA-256 (Git LFS object id) at a pinned commit, and
   the streamed bytes must hash to it, then the stored R2 object is re-read and
   re-hashed (acquire_resumable.py);
4. after transfer every zip member's CRC-32 is checked from R2 (verify_how2sign_zips.py)
   and clips are compared with an independently uploaded mirror.

Usage: python scripts/acquisition/acquire_how2sign_mirror.py [LEDGER.jsonl]

Objects land at the same keys a Drive acquisition would use, so nothing downstream
changes. Bytes go HTTP -> memory -> R2 multipart; nothing is staged on local disk.
"""
import hashlib, json, subprocess, sys, urllib.request
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])           # holds .env.r2.local (never committed)
PY = sys.executable
HERE = Path(__file__).resolve().parent
LEDGER = sys.argv[1] if len(sys.argv) > 1 else 'acquisition-ledger.jsonl'
MIRROR = 'aipieces/How2Sign'
REVISION = '1231830fc1e8d77555a245ca22353b144e168111'   # pinned commit (2024-02-07)
BASE = f'https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/original/'
DRIVE = {  # name -> (publisher Drive file id, byte count Drive reported 2026-10-18)
    'val_rgb_side_clips.zip': ('1mxL7kJPNUzJ6zoaqJyxF1Krnjo4F-eQG', 1_237_300_839),
    'test_rgb_side_clips.zip': ('1j9v9P7UdMJ0_FVWg8H95cqx4DMSsrdbH', 1_694_343_193),
    'val_rgb_front_clips.zip': ('1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO', 1_778_074_855),
    'test_rgb_front_clips.zip': ('1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG', 2_413_152_782),
    'train_rgb_side_clips.zip': ('1oiw861NGp4CKKFO3iuHGSCgTyQ-DXHW7', 23_273_015_704),
    'train_rgb_front_clips.zip': ('1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW', 33_033_658_988),
}
FOLDER = {'front': 'green-screen-rgb-clips-frontal-view/', 'side': 'green-screen-rgb-clips-side-view/'}
CSV = {  # mirror path -> SHA-256 of the copy acquired from the publisher's Drive
    'english-translation/how2sign_val.csv': 'f6f0cef01c3f891ca624a45c71eb7f7748b77362d7418c47567a984384f1537c',
    'english-translation/how2sign_test.csv': 'ecc73526d01bfc56839c1ad09f804d1aa21c2616b42cffc066d03d01d67afc3e',
    'english-translation/how2sign_train.csv': 'dcec3edab931833bdaeb8469c264d42aee5744a9a999a35a9f8ac15afa3376f1',
    'realigned-english-translation/how2sign_realigned_val.csv': '1bf5d57ce90d61571776d06a179335d99648d8c94fc22b327d2abb328d26137c',
    'realigned-english-translation/how2sign_realigned_test.csv': 'd1799dcbf100eda822eeb1a0e6e754d82b12b2a57756bd47073c4e4b8a02ae4d',
    'realigned-english-translation/how2sign_realigned_train.csv': '67a8f4fb6dc066f38006b89ac84c27a4b05d7fa07d5e6952f997042724d7a847',
}
UA = {'User-Agent': 'Sign-Translator-research-acquisition/1.0'}


def lfs(path):
    """(size, sha256) the mirror declares for a file at the pinned revision."""
    url = f'https://huggingface.co/api/datasets/{MIRROR}/paths-info/{REVISION}'
    body = json.dumps({'paths': [path]}).encode()
    req = urllib.request.Request(url, data=body, headers={**UA, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        info = json.load(r)[0]
    return info['size'], info['lfs']['oid']


for path, expected in CSV.items():
    with urllib.request.urlopen(urllib.request.Request(BASE + path, headers=UA), timeout=120) as r:
        digest = hashlib.sha256(r.read()).hexdigest()
    if digest != expected:
        raise SystemExit(f'mirror {path} differs from the publisher copy: {digest}')
print('mirror translation CSVs: 6/6 SHA-256-identical to the publisher copies', flush=True)

for name, (drive_id, drive_size) in DRIVE.items():
    path = FOLDER['side' if '_side_' in name else 'front'] + name
    size, sha256 = lfs('original/' + path)
    if size != drive_size:
        raise SystemExit(f'{name}: mirror size {size} != publisher size {drive_size}')
    evidence = dict(mirror=MIRROR, revision=REVISION, mirror_lfs_sha256=sha256, publisher_drive_size=drive_size,
                    translation_csvs_identical=len(CSV), checked='size equal to publisher; sha256 equal to mirror LFS id')
    print(f'== {name} {size:,} bytes sha256 {sha256}', flush=True)
    r = subprocess.run([PY, '-I', str(HERE / 'acquire_resumable.py'), REPO, BASE + path, f'how2sign-public/{name}',
                        '--expected-size', str(size), '--expected-sha256', sha256, '--always-range', '--verify-existing',
                        '--license', 'CC BY-NC 4.0', '--source-page', 'https://how2sign.github.io/',
                        '--mirror-of', f'https://drive.google.com/file/d/{drive_id}', '--evidence', json.dumps(evidence),
                        '--ledger', LEDGER], capture_output=True, text=True)
    out = [l for l in (r.stdout + r.stderr).splitlines() if not l.endswith(' MiB')]
    print('\n'.join(out[-3:]), flush=True)
    if r.returncode:
        sys.exit(r.returncode)
