"""Post-S2 A6's declaration, apart from A1 to A5's: ``comp_a6_declared.py COMMIT`` exits 0 only when ``runpod/fleet/comp_a6_declared.json`` names COMMIT, both of A6's rule texts and the running code with A6's job and chain
(``comp_a_declared.check``); ``comp_a6_declared.py record COMMIT`` prints that record at declaration time, and refuses while the A5 slot is empty. Nothing is declared by this file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import comp_a_declared as dc  # noqa: E402
from comp_a6_read import JOB_FILE, RULES  # noqa: E402

RECORD = "runpod/fleet/comp_a6_declared.json"


POWER_REPORT = "reports/comp_a6_hold_concentration.json"
SPAN_REPORT = "reports/wrist_by_signer.json"
JOB = "runpod/fleet/comp_a6.sh"


def check_hold_arms(root: Path, setting: str | None = None) -> None:
    """The declared HOLD_ARMS is the committed span report's, and the job's own line says the same."""
    from comp_a6_read import HOLD_ARMS
    setting = setting or HOLD_ARMS
    path = root / SPAN_REPORT
    span = (json.loads(path.read_text()).get("real_signer_span") or {}) if path.exists() else {}
    if span.get("hold_arms") != setting:
        raise SystemExit(f"HOLD_ARMS is {setting}, the span report's {span.get('hold_arms')}: the setting is read from the committed report")
    if f"\nHOLD_ARMS={setting}\n" not in (root / JOB).read_text():
        raise SystemExit(f"{JOB} does not declare HOLD_ARMS={setting}")


def check_power(root: Path) -> None:
    """A declaration publishes the power table, with a row for each rule."""
    path = root / POWER_REPORT
    table = json.loads(path.read_text()).get("power_table_at_declaration") if path.exists() else None
    if not table or set(table) != {"hold", "length"}:
        raise SystemExit(f"{POWER_REPORT}'s power_table_at_declaration is empty: fill it from A5's paired SD (comp_a6_hold_concentration.py power), a row for each rule")


def check_slot(root: Path, slot) -> None:
    """A record may not exist while the A5 slot or the power table is empty: the bars restate against A5's summary at declaration."""
    if not (root / RECORD).exists():
        return
    if not slot:
        raise SystemExit(f"{RECORD} exists while A5_SUMMARY_AT_DECLARATION is empty: the bars were not set against A5's summary")
    check_power(root)
    check_hold_arms(root)


def check(commit: str, root: Path = ROOT) -> str:
    from comp_a6_read import A5_SUMMARY_AT_DECLARATION
    check_slot(root, A5_SUMMARY_AT_DECLARATION)
    how = {dc.check(commit, test, root, record_path=RECORD, job=JOB_FILE, rules=RULES) for test in RULES}
    return how.pop()


def make_record(commit: str) -> dict:
    from comp_a6_read import A5_SUMMARY_AT_DECLARATION
    if not A5_SUMMARY_AT_DECLARATION:
        raise SystemExit("A5_SUMMARY_AT_DECLARATION is empty: the bars restate against A5's summary at declaration")
    check_power(ROOT)
    check_hold_arms(ROOT)
    return dc.make_record(commit, job=JOB_FILE, rules=RULES)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "slot":                 # what goes in comp_a6_read.A5_SUMMARY_AT_DECLARATION at declaration
        from comp_a6_read import slot_from_summary
        raw = Path(argv[1]).read_bytes()
        print(repr(slot_from_summary(json.loads(raw), hashlib.sha256(raw).hexdigest())), flush=True)
        return 0
    if len(argv) == 2 and argv[0] == "record":
        print(json.dumps(make_record(argv[1]), indent=1, sort_keys=True), flush=True)
        return 0
    if len(argv) != 1:
        raise SystemExit("usage: comp_a6_declared.py COMMIT, comp_a6_declared.py record COMMIT, or comp_a6_declared.py slot A5_SUMMARY_FILE")
    print(f"declared {argv[0]} a6 by {check(argv[0])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
