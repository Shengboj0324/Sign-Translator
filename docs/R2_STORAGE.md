# Private R2 storage and local data retrieval

The configured account is `e3d9647571bd8bb6027db63db3197fd0`, bucket
`signingdatatranslator`, endpoint
`https://e3d9647571bd8bb6027db63db3197fd0.r2.cloudflarestorage.com`, region `auto`.
The bucket name is a separate S3 parameter, not part of the endpoint URL.

Credentials are in `.env.r2.local` (owner-only, Git-ignored). Do not upload this
file, print it in logs, or commit it. The transfer environment is isolated under
`.r2-migration/venv`; model dependencies have not been changed.

## Migration in progress

The October 7 inventory contains 5,115,482 regular files / 67,396,481,715 bytes
(62.77 GiB) across `/Users/jiangshengbo/Volumes` and the repository's
`Sign Translator Stage Documentation/evidence`. Finder `.DS_Store` files are
excluded. This does not migrate unrelated user directories, source code, Git
history, Python environments or credentials.

Object prefix: `migration/2026-10-07-v1/`. Up to 10,000 files or approximately
256 MiB of input are grouped per gzip tar archive; a larger single file remains
whole. Every archived file retains its path, bytes, timestamp and SHA-256 in a
member index. Archives and indexes are uploaded then downloaded for SHA-256
verification. Source identities are checked before and after packing. A synced
append-only local catalog records verified parts; unfinished parts can be retried.

`upload` never deletes source files. An upload is complete only when
`.r2-migration/complete.json` exists and its catalog identity has been verified.
A running process, uploaded object count or successful connectivity probe does
not establish a complete migration. Current logs: `.r2-migration/upload.log`.

```sh
.r2-migration/venv/bin/python scripts/r2_data.py upload
```

## Cloud-backed source reconciliation

The remaining evidence files included macOS dataless placeholders. Reading their
contents changed some nanosecond timestamps, causing exact inventory checks to
stop packing. The repair reads every remaining file before reconciling metadata,
then rehashes it with stable identity checks. Reconciled rows include SHA-256;
packing must match that hash as well as the exact source identity. Previously
verified inventory rows and archives are preserved. The original inventory had
no content hashes, so reconciliation does not assert historical byte equality.
The audit is retained locally and in R2 as `reconciliation-hydrated.json`.

## Retrieval and loader integration

Current dataset readers use regular local files, including hash-bound paths.
There is no transparent R2 filesystem or direct network training loader. Restore
only the needed source subtree before using the existing readers:

```sh
.r2-migration/venv/bin/python scripts/r2_data.py restore \
  --prefix data/how2sign_audit \
  --destination /Users/jiangshengbo/Volumes
```

This restores `how2sign_audit/...` under the destination. Use an exact file prefix
for one file or a directory prefix for its descendants. Existing identical files
are reused; differing files are never overwritten. The archive and each restored
file are verified. Publication uses atomic no-clobber links. Re-admit any governed
corpus against restored files before training; cloud storage does not confer
permissions, accepted calibration, or ASL review.

On a replacement computer, recover indexes using the independently retained
catalog SHA-256, then run restore:

```sh
.r2-migration/venv/bin/python scripts/r2_data.py bootstrap \
  --catalog-sha256 ACTUAL_VERIFIED_CATALOG_SHA256
```

The final catalog hash will be recorded after upload completion. Keep the local
catalog and member indexes: these are small recovery metadata compared with the
data and are intentionally retained after cleanup. Credentials are also required
on the replacement host. No automatic recurring sync has been configured; new or
changed source files require a separately inventoried snapshot. This migration
refuses a changed inventory rather than mixing versions.

## Local cleanup gate

Cleanup is restricted to the inventoried `/Users/jiangshengbo/Volumes` files.
Project evidence stays in the checkout. It requires the completed catalog and a
successful real restore test bound to that catalog. Before deletion, each remote
archive identity is rechecked and each local file is rehashed against its member
index. Changed files stop cleanup. Checks are not a concurrent filesystem
snapshot; source writers must remain stopped during migration and cleanup.

```sh
.r2-migration/venv/bin/python scripts/r2_data.py cleanup \
  --root /Users/jiangshengbo/Volumes
```

Do not create the restore-test marker to bypass the gate. Do not run cleanup
until the actual restore test and complete upload have been independently checked.

## Guarded completion worker

`scripts/r2_finish.py --wait-pid PID` waits for the named upload process to exit.
It then requires a complete, verified catalog, checks every remote archive and
local member index, and restores a representative file from every inventoried
source/evidence group to a temporary directory. Successful file-hash comparisons
produce the catalog-bound restore-test marker. Only then does it call cleanup.
Missing completion evidence or any failed check stops the worker without bypassing
the gate. If cleanup has already started, earlier verified deletions remain valid
and later failures stop further deletion; there is no multi-file transaction.

Completion status is recorded in `.r2-migration/finished.json` and uploaded to
`migration/2026-10-07-v1/finished.json`. The completion worker reports any remaining
non-Finder local files. It retains project evidence, source code, credentials,
catalog and member indexes. Logs are `.r2-migration/upload.log` and
`.r2-migration/finish.log`; the existence of these logs is not proof of completion.
