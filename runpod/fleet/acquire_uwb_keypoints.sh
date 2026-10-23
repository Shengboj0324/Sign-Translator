#!/usr/bin/env bash
# acquire_uwb_keypoints.sh — plan item 7 fallback (YouTube refuses datacenter downloads even at one request per 24 s):
# the YouTube-ASL Clip Keypoint Dataset (Železný, Hrůz, Straka, Gueuwou; UWB / JSALT 2024; LINDAT handle 11234/1-5898,
# CC BY 4.0): MediaPipe Holistic 2D keypoints of 390,547 caption-aligned YouTube-ASL sentence clips + their English
# translations, hosted by LINDAT (not YouTube). Every file is streamed HTTP -> R2 multipart by
# scripts/acquisition/acquire_resumable.py with the repository's size and MD5 verified (no local staging), at most
# three streams at once (polite to an academic repository). File list, sizes and MD5s: the repository's bitstream
# listing of 2026-10-22. UWB-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH
cd /root/work/Sign-Translator
API=https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams
PAGE=https://hdl.handle.net/11234/1-5898
LEDGER=/root/uwb-ledger.jsonl; : > $LEDGER
FILES="
  YT.translations.dev.json d5d23d31-c93a-4752-8e5c-e20548f13da0 5792366 26e96e563f0c50f933b021c8089eeece
  YT.translations.train.json f8460818-3605-4f05-9832-90ddc68f22e6 53098164 7b7b36e50f384f09aea7cf818c00b83c
  YT.translations.all.json ec3632aa-ba9a-400e-b14b-1d15de9b6061 58890528 34b884205edf609baa97f92bc64cbd4c
  raw_keypoints_10.zip d96b874e-72fa-4830-b2f0-a072bb6be31d 37055905174 0019b603f9ebd7594fce8fae8dc65167
  raw_keypoints_2.zip bfa38e27-bd46-48ae-bf9a-eb1eafaabb95 37282065086 c891a51901ca17fa6f42529e5657df67
  raw_keypoints_4.zip 1a4ace36-ed9a-4bfb-a80f-483c0463e02d 37303234091 890dd2fa779b56cd90c6ae39fa67faef
  raw_keypoints_8.zip 43a9146e-0abf-47ec-a3b3-90c01a4d9380 37343201379 b6bcc2e8517c2dcf8347347bbd74800c
  raw_keypoints_6.zip 3b4ef094-bc95-4efb-b8e0-3bc4f63e57b0 37362757430 403958646966402245f69f9d473c4346
  raw_keypoints_3.zip 0c59bda5-908d-4194-93bf-13e13de2ef10 37431373349 0713086c142d52c31cff1b4be9f4f82a
  raw_keypoints_5.zip 6b405702-9f74-4729-8202-55eca36adaea 37434055308 e935642e8bb5f9b594af74a8ba75d97f
  raw_keypoints_9.zip 05385788-d459-4f35-92b4-4908b9d86de6 37449567437 9d52caab1aa2db4218188819485f92ab
  raw_keypoints_7.zip 74e99da7-c580-4fdf-8363-8024d7a7adf1 37472907471 786fe0067d1e4d8665b1ddbb8628a17c
  raw_keypoints_1.zip 3dc57bf4-c5fb-491c-8ab2-a9215e0e2fe5 37504799288 75223dd5e7e9b6ccb9f34c5792fc1e6d
"
export API PAGE LEDGER
fetch() { local name=$1 uuid=$2 size=$3 md5=$4
  for attempt in 1 2 3; do
    python scripts/acquisition/acquire_resumable.py . "$API/$uuid/content" "youtube-asl-uwb-keypoints-v1/$name" \
      --expected-size "$size" --expected-md5 "$md5" --license "CC BY 4.0 (YouTube-ASL Clip Keypoint Dataset, UWB); derived from YouTube videos owned by their creators: research use, private store, no redistribution" \
      --source-page "$PAGE" --ledger "$LEDGER" > "/root/uwb-$name.log" 2>&1 && { echo "OK $name"; return 0; }
    echo "RETRY $name ($attempt): $(tail -1 /root/uwb-$name.log)"; sleep 60
  done
  echo "FAILED $name"; return 1; }
export -f fetch
echo "$FILES" | grep . | xargs -P 3 -L 1 bash -c 'fetch "$@"' _ 2>&1
python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put('acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/ledger.jsonl', open(sys.argv[1],'rb').read())" $LEDGER
grep -c . $LEDGER
echo UWB-DONE
