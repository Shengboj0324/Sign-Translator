"""Stream one approved public file into R2 with end-to-end SHA-256 verification.

Usage:
  acquire.py REPO URL KEY --expected-size N [--expected-sha256 H] --license L
             --source-page P --ledger LEDGER.jsonl

Bytes go HTTP -> memory parts -> R2 multipart upload; nothing is staged on disk.
The streamed SHA-256 and byte count must equal the publisher's values when given.
On any mismatch or error the multipart upload is aborted, so no unverified object
exists. After completion the object is re-read from R2 and re-hashed. A
provenance record is stored beside the object and appended to the ledger.
"""
import argparse, hashlib, json, os, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config

PREFIX = 'acquisition/2026-10-17-v1'
PART = 64 * 1024 * 1024
USER_AGENT = 'Sign-Translator-research-acquisition/1.0'

parser = argparse.ArgumentParser()
parser.add_argument('repo', type=Path)
parser.add_argument('url')
parser.add_argument('key')
parser.add_argument('--expected-size', type=int)
parser.add_argument('--expected-sha256')
parser.add_argument('--expected-md5')
parser.add_argument('--always-range', action='store_true', help='request bytes=N- even at offset 0 (Drive serves an HTML warning otherwise)')
parser.add_argument('--license', required=True)
parser.add_argument('--source-page', required=True)
parser.add_argument('--ledger', type=Path, required=True)
parser.add_argument('--header', action='append', default=[])
parser.add_argument('--verify-existing', action='store_true',
                    help='an object left without provenance (re-read interrupted): re-hash it from R2 and accept it '
                         'only if size and SHA-256 equal the expected values, else delete it')
parser.add_argument('--mirror-of', help='publisher URL this byte-identical mirror copy stands in for')
parser.add_argument('--evidence', help='JSON object: why the mirror is byte-identical to the publisher file')
args = parser.parse_args()

if any(part in ('', '.', '..') for part in args.key.split('/')):
    raise SystemExit('unsafe object key')
key = f'{PREFIX}/{args.key}'
local = args.repo / '.env.r2.local'          # absent on pods: secrets come from the environment
env = dict(l.split('=', 1) for l in local.read_text().splitlines() if l) if local.exists() else os.environ
s3 = boto3.client('s3', endpoint_url=env['R2_ENDPOINT_URL'], region_name='auto',
                  aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                  aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                  config=Config(connect_timeout=20, read_timeout=600,
                                retries={'mode': 'standard', 'max_attempts': 8}))
bucket = env['R2_BUCKET']

from botocore.exceptions import ClientError


def reread(key, size):
    """SHA-256 and MD5 of the stored object via ranged GETs, each range retried on its own."""
    sha, md, position, block = hashlib.sha256(), hashlib.md5(), 0, 256 * 1024 * 1024
    while position < size:
        end = min(position + block, size) - 1
        for attempt in range(8):
            try:
                piece = s3.get_object(Bucket=bucket, Key=key, Range=f'bytes={position}-{end}')['Body'].read()
                if len(piece) != end - position + 1:
                    raise OSError('short ranged read')
                break
            except Exception as error:  # noqa: BLE001 - network errors of every kind are retried
                if attempt == 7:
                    raise
                print(f'  re-read retry at byte {position:,}: {error}', flush=True)
                time.sleep(2 ** attempt)
        sha.update(piece)
        md.update(piece)
        position = end + 1
    return sha.hexdigest(), md.hexdigest()

try:
    s3.head_object(Bucket=bucket, Key=key)
    present = True
except ClientError as error:
    if error.response['Error']['Code'] not in ('404', 'NoSuchKey', 'NotFound'):
        raise
    present = False
if present:
    try:
        prior = json.loads(s3.get_object(Bucket=bucket, Key=key + '.provenance.json')['Body'].read())
    except ClientError:
        if not (args.verify_existing and args.expected_sha256 and args.expected_size):
            raise SystemExit(f'{key} exists without provenance; refusing to overwrite')
        stored = s3.head_object(Bucket=bucket, Key=key)['ContentLength']
        started = time.monotonic()
        sha256, md5_hex = reread(key, stored) if stored == args.expected_size else (None, None)
        if stored != args.expected_size or sha256 != args.expected_sha256:
            s3.delete_object(Bucket=bucket, Key=key)
            raise SystemExit(f'{key}: stored object ({stored} bytes, sha256 {sha256}) is not the expected file; removed')
        record = dict(key=key, url=args.url, final_url=None, source_page=args.source_page, license=args.license,
                      bytes=stored, sha256=sha256, publisher_sha256=args.expected_sha256,
                      publisher_size=args.expected_size, md5=md5_hex, publisher_md5=args.expected_md5,
                      content_type=None, verified_by_r2_reread=True, seconds=round(time.monotonic() - started, 1),
                      resumes=None, acquired_at=datetime.now(timezone.utc).isoformat(),
                      note='transfer completed (stream hash checked before completion); first re-read was interrupted; '
                           'object re-read and re-hashed by ranged GETs',
                      mirror_of=args.mirror_of, mirror_evidence=json.loads(args.evidence) if args.evidence else None)
        s3.put_object(Bucket=bucket, Key=key + '.provenance.json',
                      Body=json.dumps(record, indent=2, sort_keys=True).encode())
        with args.ledger.open('a') as ledger:
            ledger.write(json.dumps(record, sort_keys=True) + '\n')
        print(json.dumps({k: record[k] for k in ('key', 'bytes', 'sha256', 'seconds')}))
        sys.exit(0)
    if prior.get('verified_by_r2_reread') and (
            (args.expected_sha256 and prior.get('sha256') == args.expected_sha256)
            or (not args.expected_sha256 and args.expected_md5 and prior.get('md5') == args.expected_md5
                and prior.get('bytes') == args.expected_size)):
        print(json.dumps({'key': key, 'status': 'already_present_and_verified'}))
        sys.exit(0)
    raise SystemExit(f'refusing to overwrite existing object {key}')

started = time.monotonic()
upload = s3.create_multipart_upload(Bucket=bucket, Key=key, Metadata={'source-url': args.url[:1000]})
parts, digest, total, number = [], hashlib.sha256(), 0, 1
md5 = hashlib.md5()
resumes = 0
try:
    buffer = bytearray()
    final_url = content_type = None
    while True:
        headers = {'User-Agent': USER_AGENT, **dict(h.split(': ', 1) for h in args.header)}
        if total or args.always_range:
            headers['Range'] = f'bytes={total}-'
        request = urllib.request.Request(args.url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                if 'text/html' in (response.headers.get('Content-Type') or ''):
                    raise ValueError('server returned an HTML page, not the file')
                if args.always_range and response.status != 206:
                    raise ValueError(f'range request answered with status {response.status}')
                if total and response.status == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {total}-'):
                        raise ValueError('resume returned a different byte range')
                elif total and response.status == 200:
                    # No range support: re-read the prefix and prove it is the
                    # same bytes already uploaded before continuing after it.
                    check, remaining = hashlib.sha256(), total
                    while remaining:
                        piece = response.read(min(1 << 20, remaining))
                        if not piece:
                            raise OSError('stream ended while re-reading the verified prefix')
                        check.update(piece)
                        remaining -= len(piece)
                    if check.hexdigest() != prefix_digest.hexdigest():
                        raise ValueError('re-read prefix differs from the uploaded bytes')
                elif total:
                    raise ValueError(f'resume refused: status {response.status}')
                final_url = final_url or response.geturl()
                content_type = content_type or response.headers.get('Content-Type')
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    buffer += chunk
                    digest.update(chunk)
                    md5.update(chunk)
                    total += len(chunk)
                    # R2 requires every non-trailing part to have the same length,
                    # and network reads may be short, so cut exact PART slices.
                    while len(buffer) >= PART:
                        etag = s3.upload_part(Bucket=bucket, Key=key, UploadId=upload['UploadId'],
                                              PartNumber=number, Body=bytes(buffer[:PART]))['ETag']
                        parts.append({'ETag': etag, 'PartNumber': number})
                        number += 1
                        del buffer[:PART]
                        print(f'  {key}: {total / 2**20:,.0f} MiB', flush=True)
        except (OSError, ConnectionError) as error:  # includes timeouts and resets
            print(f'  transfer interrupted at {total:,} bytes: {error}', flush=True)
        prefix_digest = digest.copy()
        prefix_md5 = md5.copy()
        if args.expected_size is None or total >= args.expected_size:
            break
        resumes += 1
        if resumes > 20:
            raise ValueError(f'gave up after {resumes} resumes at {total} bytes')
        print(f'  resuming from byte {total:,} (attempt {resumes})', flush=True)
        time.sleep(min(60, 2 ** resumes))
    if buffer:
        etag = s3.upload_part(Bucket=bucket, Key=key, UploadId=upload['UploadId'],
                              PartNumber=number, Body=bytes(buffer))['ETag']
        parts.append({'ETag': etag, 'PartNumber': number})
    sha256 = digest.hexdigest()
    if total == 0:
        raise ValueError('empty response')
    if args.expected_size is not None and total != args.expected_size:
        raise ValueError(f'size {total} != expected {args.expected_size}')
    if args.expected_sha256 and sha256 != args.expected_sha256:
        raise ValueError(f'sha256 {sha256} != expected {args.expected_sha256}')
    if args.expected_md5 and md5.hexdigest() != args.expected_md5:
        raise ValueError(f'md5 {md5.hexdigest()} != publisher {args.expected_md5}')
    s3.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload['UploadId'],
                                 MultipartUpload={'Parts': parts})
except BaseException:
    s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload['UploadId'])
    raise

# Re-read from R2: the stored object, not the network stream, must match.
size = s3.head_object(Bucket=bucket, Key=key)['ContentLength']
reread_sha256 = reread(key, size)[0] if size == total else None
if reread_sha256 != sha256 or size != total:
    s3.delete_object(Bucket=bucket, Key=key)
    raise SystemExit(f'R2 re-read mismatch for {key}; object removed')

record = dict(key=key, url=args.url, final_url=final_url, source_page=args.source_page,
              license=args.license, bytes=total, sha256=sha256,
              publisher_sha256=args.expected_sha256, publisher_size=args.expected_size,
              md5=md5.hexdigest(), publisher_md5=args.expected_md5,
              content_type=content_type, verified_by_r2_reread=True,
              seconds=round(time.monotonic() - started, 1), resumes=resumes,
              acquired_at=datetime.now(timezone.utc).isoformat(),
              mirror_of=args.mirror_of, mirror_evidence=json.loads(args.evidence) if args.evidence else None)
s3.put_object(Bucket=bucket, Key=key + '.provenance.json',
              Body=json.dumps(record, indent=2, sort_keys=True).encode())
with args.ledger.open('a') as ledger:
    ledger.write(json.dumps(record, sort_keys=True) + '\n')
print(json.dumps({k: record[k] for k in ('key', 'bytes', 'sha256', 'seconds')}))
