# Sapiens fleet on Runpod (subset run of 2026-10-20)

How the budget-sized Sapiens subset (`runpod/subset-v1.json`: 11.99 h, 6,723 clips,
467 sessions) was run on ten 1× L40S pods. No credential is stored here:

- **Runpod API key:** from `RUNPOD_API_KEY` or the Runpod MCP config (`rp.py`).
- **R2 credentials:** pods get them as Runpod secrets, `{{ RUNPOD_SECRET_Access_Key_ID }}`
  and `{{ RUNPOD_SECRET_Secret_Access_Key }}`. Endpoint and bucket come from the local
  `.env.r2.local` (`deploy.py`). Inside a pod, `r2env.sh` exports them from PID 1's
  environment for ssh sessions.
- **SSH key:** `pod_key` is a dedicated ed25519 key generated per run with
  `ssh-keygen -t ed25519 -f runpod/fleet/pod_key` and not committed.

| Step | Command |
|---|---|
| Deploy | `python3 -c "from deploy import deploy; print(deploy(name='st-1', gpu='NVIDIA L40S', dc='US-IL-1'))"` (secure cloud, 120 GB disk, no volume) |
| Boot | `./boot.sh POD_ID I N` uploads `setup.sh`, the code archive and `run_sapiens.sh`, then runs the setup (Python 3.12 venv via uv, torch 2.13 for CUDA 13, models from R2 with verified hashes) and Sapiens rotation I/N |
| Run | `run_sapiens.sh I N` runs `runpod/pipeline.sh sapiens` with `SUBSET`, `ROTATE=I/N`, `--no-flip --batch 24 --workers 16`. Pods sharing a zip start at different places and take each other's remainders; a finished clip is never redone |
| Monitor | `POST_POD=<id> python fleet_status.py` prints R2 progress and one line per pod, and stops each pod whose job printed `SAPIENS-DONE` (except the post pod) |
| Post | `post.sh` (on the post pod, after its Sapiens job) waits until both views are complete, then runs cameras → 16 parallel triangulation shards → manifest → tokenizer + prior (grammar features) → evaluation → locus-conditioned prior → evaluation → grammar tracks → two-view grammar measurement |

Measured throughput is 10–11 frames/s per L40S (the device-side batched path), about 40
clips per minute across ten pods. The clips are 30 fps.
