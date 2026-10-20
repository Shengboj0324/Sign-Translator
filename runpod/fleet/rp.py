"""Minimal Runpod GraphQL client. The API key comes from RUNPOD_API_KEY or the Runpod MCP config; it is never printed."""
import json, sys, urllib.request, urllib.error
from pathlib import Path

def key():
    import os
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    cfg = json.loads((Path.home() / ".claude.json").read_text())   # the Runpod MCP server's configured key
    def find(o):
        if isinstance(o, dict):
            if "runpod" in o and isinstance(o["runpod"], dict) and "env" in o["runpod"]:
                return o["runpod"]["env"].get("RUNPOD_API_KEY")
            for v in o.values():
                r = find(v)
                if r: return r
        return None
    k = find(cfg)
    if not k: raise SystemExit("no RUNPOD_API_KEY in MCP config")
    return k

def gql(query, variables=None):
    req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": query, "variables": variables or {}}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key()}",
                                          "User-Agent": "sign-translator/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:1500]}")
    if out.get("errors"): raise SystemExit(json.dumps(out["errors"])[:1500])
    return out["data"]

if __name__ == "__main__":
    print(json.dumps(gql(sys.stdin.read()), indent=1))
