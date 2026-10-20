"""Deploy a pod that reads R2 through the user's Runpod secrets; endpoint and bucket come from the local config."""
import json, sys
from pathlib import Path
from rp import gql
env_file = Path(__file__).resolve().parents[2] / ".env.r2.local"
cfg = dict(l.split("=", 1) for l in env_file.read_text().splitlines() if "=" in l and not l.startswith("#"))
def deploy(name, gpu, count=1, dc=None, disk=120, cloud="SECURE", volume=None):
    env = [{"key": "PUBLIC_KEY", "value": open(Path(__file__).with_name("pod_key.pub")).read().strip()},
           {"key": "R2_ENDPOINT_URL", "value": cfg["R2_ENDPOINT_URL"].strip()},
           {"key": "R2_BUCKET", "value": cfg["R2_BUCKET"].strip()},
           {"key": "AWS_ACCESS_KEY_ID", "value": "{{ RUNPOD_SECRET_Access_Key_ID }}"},
           {"key": "AWS_SECRET_ACCESS_KEY", "value": "{{ RUNPOD_SECRET_Secret_Access_Key }}"}]
    spec = dict(cloudType=cloud, gpuCount=count, gpuTypeId=gpu, name=name,
                imageName="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
                containerDiskInGb=disk, ports="22/tcp", startSsh=True, env=env)
    if dc: spec["dataCenterId"] = dc
    if volume: spec.update(networkVolumeId=volume, volumeMountPath="/workspace")
    return gql("""mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) {
      id desiredStatus costPerHr machine { gpuDisplayName location } } }""", {"input": spec})["podFindAndDeployOnDemand"]
if __name__ == "__main__":
    a = json.loads(sys.argv[1]); print(json.dumps(deploy(**a)))
