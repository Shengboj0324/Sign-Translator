"""Print 'ip port' of a pod's public SSH endpoint once it is up."""
import sys, time
from rp import gql
pid = sys.argv[1]
for _ in range(60):
    p = gql('query($id: String!) { pod(input: {podId: $id}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }', {"id": pid})["pod"]
    ports = [x for x in ((p.get("runtime") or {}).get("ports") or []) if x["privatePort"] == 22 and x["isIpPublic"]]
    if ports:
        print(ports[0]["ip"], ports[0]["publicPort"]); break
    time.sleep(10)
else:
    raise SystemExit("no ssh endpoint")
