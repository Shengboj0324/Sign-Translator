"""Acquire How2Sign raw 30 fps recordings (both cameras) into R2 from the mirror's shards.

Needed to recover take-5 sessions, whose released lateral clips are mis-cut (audit B80;
``scripts/recut_how2sign_take5.py``). The public mirror `aipieces/How2Sign` holds the
publisher's raw recordings re-packed as per-split shards (``*_raw_videos/`` frontal,
``*_side_raw_videos/`` lateral; each a zip of whole recordings). Only val (both cameras)
and test (frontal) also have the publisher's own zip under ``original/``. A shard is
accepted only when

1. its bytes hash to the mirror's Git LFS SHA-256 at the pinned commit (streamed, then
   the stored R2 object re-read and re-hashed: ``acquire_resumable.py``);
2. where the publisher's zip is mirrored under ``original/``, every shard member has the
   CRC-32 and byte count of the same recording there (central directories only).

Downstream the recut checks every raw frontal recording frame by frame against the
publisher-verified frontal clips, and the lateral one by epipolar geometry.

Usage: acquire_how2sign_raw.py --split val [--view side] [--recordings REGEX] [--ledger L] [--shard i --shards n]

Shards land at ``acquisition/2026-10-17-v1/how2sign-raw/<mirror folder>/<shard>.zip``.
With ``--recordings`` only shards holding a matching recording are transferred.
Credentials come from ``.env.r2.local`` or, on a pod, the environment.
"""
import argparse
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MIRROR = 'aipieces/How2Sign'
REVISION = '1231830fc1e8d77555a245ca22353b144e168111'   # pinned commit (2024-02-07), as for the clip zips
BASE = f'https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/'
RAW = 'how2sign-raw'                                    # under acquisition/2026-10-17-v1/
FOLDERS = {('train', 'front'): 'train_raw_videos', ('val', 'front'): 'val_raw_videos',
           ('test', 'front'): 'test_raw_videos', ('train', 'side'): 'train_side_raw_videos',
           ('val', 'side'): 'val_rgb_side_raw_videos', ('test', 'side'): 'test_rgb_side_raw_videos'}
PUBLISHER = {('val', 'front'): 'original/green-screen-rgb-videos-frontal-view/val_raw_videos.zip',
             ('val', 'side'): 'original/green-screen-rgb-videos-side-view/val_rgb_side_raw_videos.zip',
             ('test', 'front'): 'original/green-screen-rgb-videos-frontal-view/test_raw_videos.zip'}
UA = {'User-Agent': 'Sign-Translator-research-acquisition/1.0'}


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file by HTTP range requests (zip central directories)."""

    def __init__(self, url):
        self.url, self.position = url, 0
        request = urllib.request.Request(url, headers={**UA, 'Range': 'bytes=0-0'})
        with urllib.request.urlopen(request, timeout=120) as response:
            self.size = int(response.headers['Content-Range'].rsplit('/', 1)[1])
            self.url = response.geturl()                # follow the CDN redirect once

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size is None or size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b''
        request = urllib.request.Request(self.url, headers={**UA, 'Range': f'bytes={self.position}-{self.position + size - 1}'})
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
        if len(data) != size:
            raise OSError('short ranged read')
        self.position += size
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def members(url) -> dict:
    """Recording file name -> (byte count, CRC-32) from a remote zip's central directory."""
    with zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), 1 << 20)) as archive:
        return {PurePosixPath(i.filename).name: (i.file_size, i.CRC) for i in archive.infolist()
                if i.filename.endswith('.mp4')}


def shards(folder) -> list[tuple[str, int, str]]:
    """(mirror path, size, LFS SHA-256) of every shard in a mirror folder at the pinned commit."""
    url = f'https://huggingface.co/api/datasets/{MIRROR}/tree/{REVISION}/{folder}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as response:
        entries = json.load(response)
    return sorted((e['path'], e['size'], e['lfs']['oid']) for e in entries if e['type'] == 'file' and e['path'].endswith('.zip'))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', required=True, choices=('train', 'val', 'test'))
    parser.add_argument('--view', choices=('front', 'side', 'both'), default='both')
    parser.add_argument('--recordings', help='only shards holding a recording whose file name matches this regex')
    parser.add_argument('--ledger', type=Path, default=Path('acquisition-ledger.jsonl'))
    parser.add_argument('--shard', type=int, default=0, help='with --shards: only mirror shards whose index is this modulo')
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        raise SystemExit('shard must be in [0, shards)')
    for view in ('front', 'side') if args.view == 'both' else (args.view,):
        folder = FOLDERS[args.split, view]
        publisher = PUBLISHER.get((args.split, view))
        reference = members(BASE + publisher) if publisher else None
        for index, (path, size, sha256) in enumerate(shards(folder)):
            if index % args.shards != args.shard:
                continue
            listing = members(BASE + path)
            if args.recordings and not any(re.search(args.recordings, name) for name in listing):
                continue
            checked = None
            if reference is not None:
                differing = sorted(n for n, v in listing.items() if reference.get(n) != v)
                if differing:
                    raise SystemExit(f'{path}: {len(differing)} recordings differ from the publisher zip, e.g. {differing[:3]}')
                checked = len(listing)
            evidence = dict(mirror=MIRROR, revision=REVISION, mirror_lfs_sha256=sha256, recordings=len(listing),
                            publisher_zip=publisher, publisher_crc32_and_size_equal=checked,
                            checked='sha256 equal to mirror LFS id' + ('; every member CRC-32 and size equal to the '
                                                                        'publisher zip' if checked else ''))
            print(f'== {path} {size:,} bytes sha256 {sha256} ({len(listing)} recordings'
                  + (f', all equal to the publisher zip)' if checked else ')'), flush=True)
            result = subprocess.run([sys.executable, '-I', str(HERE / 'acquire_resumable.py'), str(REPO), BASE + path,
                                     f'{RAW}/{path}', '--expected-size', str(size), '--expected-sha256', sha256,
                                     '--verify-existing', '--license', 'CC BY-NC 4.0',
                                     '--source-page', 'https://how2sign.github.io/', '--evidence', json.dumps(evidence),
                                     '--ledger', str(args.ledger)], capture_output=True, text=True)
            lines = [line for line in (result.stdout + result.stderr).splitlines() if not line.endswith(' MiB')]
            print('\n'.join(lines[-3:]), flush=True)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
