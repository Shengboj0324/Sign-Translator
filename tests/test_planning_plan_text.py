import json
from pathlib import Path
import random

from signtranslator.planning.asllrp_gloss import MARKERS, plan_target
from signtranslator.planning.plan_text import plan_from_text, plan_to_text


def m(*tiers):
    return [int(MARKERS[k] in tiers) for k in range(len(MARKERS))]


def test_plan_line_form():
    text = plan_to_text(["MOTHER", "IX-loc", "GIVE", "NOT", "LIKE"], [(0, 0), (0, 1), (1, 2), (0, 0), (0, 0)],
                        [m("topic/focus"), m(), m(), m("negative"), m("negative")])
    assert text == "[topic MOTHER ] IX-loc>1 1>GIVE>2 [neg NOT LIKE ]"


def test_round_trip_is_exact_including_crossing_and_nested_markers():
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(1, 12)
        glosses = [rng.choice(["A", "B-C", "#IF", "ns-fs-JO", "5\"wave"]) for _ in range(n)]
        loci = [(rng.choice([0, 0, 1, 2]), rng.choice([0, 0, 1, 2, 3])) for _ in range(n)]
        markers = [[0] * len(MARKERS) for _ in range(n)]
        for _ in range(rng.randint(0, 4)):                 # random runs, overlapping freely across tiers
            k, s = rng.randrange(len(MARKERS)), rng.randrange(n)
            for i in range(s, rng.randint(s + 1, n)):
                markers[i][k] = 1
        text = plan_to_text(glosses, loci, markers)
        assert plan_from_text(text) == (glosses, loci, markers), text


def test_round_trip_on_real_asllrp_pairs_when_available():
    p = Path("/private/tmp/claude-501/-Users-jiangshengbo-Desktop-Sign-Translator/c4be2423-91cc-415b-a2ac-5c909d21cd29/scratchpad/planner/pairs.json")
    if not p.exists():
        return
    for pair in json.loads(p.read_text())["pairs"]:
        t = plan_target(pair)
        g, l, mk = list(t.bases), [tuple(x) for x in t.loci], [list(x) for x in t.markers]
        assert plan_from_text(plan_to_text(g, l, mk)) == (g, l, mk)
