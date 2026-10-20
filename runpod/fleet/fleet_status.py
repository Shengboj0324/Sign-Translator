"""One status line per pod plus R2 progress; stops pods whose job printed SAPIENS-DONE."""
import glob, json, os, subprocess, sys
from pathlib import Path
from rp import gql
R = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
env = dict(l.split("=", 1) for l in Path(__file__).resolve().parents[2] / ".env.r2.local".read_text().splitlines() if "=" in l)
os.environ.update({k: v.strip() for k, v in env.items()})
import boto3
s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"], region_name="auto")
def count(prefix):
    n, token = 0, None
    while True:
        kw = dict(Bucket=os.environ["R2_BUCKET"], Prefix=prefix, MaxKeys=1000)
        if token: kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw); n += r.get("KeyCount", 0)
        if not r.get("IsTruncated"): return n
        token = r["NextContinuationToken"]
base = "acquisition/2026-10-17-v1/derived/subset-v1"
front, side = count(f"{base}/how2sign-sapiens-front-v1/keypoints/"), count(f"{base}/how2sign-sapiens-side-v1/keypoints/")
lines = [f"R2 keypoints: front {front}/6639, side {side}/6639"]
for f in sorted(glob.glob(str(R / "ssh-*.txt"))):
    pid, ip, port = open(f).read().split()
    status = gql('query($id: String!) { pod(input: {podId: $id}) { desiredStatus } }', {"id": pid})["pod"]
    if not status or status["desiredStatus"] != "RUNNING":
        lines.append(f"{pid}: {status and status['desiredStatus']}"); continue
    cmd = ["ssh", "-i", str(R / "pod_key"), "-o", f"UserKnownHostsFile={R/'known_hosts'}", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ConnectTimeout=15", "-p", port, f"root@{ip}",
           "tail -c 300 /root/sapiens.log 2>/dev/null | tail -1; grep -c Traceback /root/sapiens.log 2>/dev/null; tail -1 /root/setup.log 2>/dev/null; "
           "tail -1 /root/post.log 2>/dev/null; "
           "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=40).stdout.strip().splitlines()
    except subprocess.TimeoutExpired:
        out = ["ssh timeout"]
    text = " | ".join(o[-90:] for o in out)
    if any("SAPIENS-DONE" in o for o in out) and pid != os.environ.get("POST_POD", ""):     # the post pod continues with post.sh
        gql('mutation($id: String!) { podStop(input: {podId: $id}) { id } }', {"id": pid}); text += " -> STOPPED (done)"
    lines.append(f"{pid}: {text}")
print("\n".join(lines), flush=True)
