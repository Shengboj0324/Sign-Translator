"""prefetch_cache.py MANIFEST_KEY [MANIFEST_KEY ...] — fill the trainer's content cache (SIGN_CACHE_DIR or /tmp/sign-translator-cache) with every
state the manifest names, 32 downloads at a time, each verified against its SHA-256 and written atomically under its
hash (the ContentCache layout). Safe to run beside a trainer that is using the same cache."""
import hashlib, json, os, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signtranslator.data.text_motion_dataset import R2Store
store = R2Store(os.environ)
cache = Path(os.environ.get("SIGN_CACHE_DIR", tempfile.gettempdir())) / "sign-translator-cache"
cache.mkdir(parents=True, exist_ok=True)
records = [r for key in sys.argv[1:] for r in json.loads(store.get(key))["records"]]
todo = {r["continuous"]["state_sha256"]: r["continuous"]["state_key"] for r in records
        if (r.get("continuous") or {}).get("state_key") and not (cache / r["continuous"]["state_sha256"]).is_file()}
print(f"{len(records)} records, {len(todo)} to fetch", flush=True)
done = [0]; t0 = time.time(); failed = []
def one(item):
    sha, key = item
    for attempt in range(6):
        try:
            data = store.get(key)
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError("hash mismatch")
            with tempfile.NamedTemporaryFile(dir=cache, delete=False, prefix=".partial-") as tmp:
                tmp.write(data)
            os.replace(tmp.name, cache / sha)
            break
        except Exception as e:
            time.sleep(2 + 3 * attempt)
    else:
        failed.append(key)                      # never silently: a training run would hit it later
    done[0] += 1
    if done[0] % 2000 == 0:
        print(f"{done[0]}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
with ThreadPoolExecutor(32) as pool:
    list(pool.map(one, todo.items()))
print(f"PREFETCH-DONE {done[0]} in {time.time() - t0:.0f}s", flush=True)
if failed:
    print(f"PREFETCH-FAILED {len(failed)} states missing or not matching the manifest, e.g. {sorted(failed)[:5]}", flush=True)
    sys.exit(3)
