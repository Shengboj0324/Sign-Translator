"""Model card for a release (plan item 6; Mitchell et al. 2019 sections), generated from the release manifest and the
stored evaluation reports - never hand-written numbers.

Inputs: a release manifest (``serving.release.Release``), plus any of these reports in the store:
- the exit gate (``--gate-key``);
- the strict pass test (``--strict-key``);
- back-translation (``--bleu-key``);
- the sign-level score (``--sign-level-key``);
- the human study (``--human-key``, ``score_human_eval.py``).

The card states what each number is relative to (real motion is the ceiling, blank motion the floor) and lists every
training corpus with its licence and research-only flag, the citations, and the release gate's verdict
(``public_release_allowed`` / ``commercial_use_allowed``). Written to ``--output`` (Markdown).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CITATIONS = {
    "How2Sign": "Duarte et al., How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language, CVPR 2021.",
    "ASLLRP": ("C. Neidle, A. Opoku and D. Metaxas (2022), 'ASL Video Corpora & Sign Bank: Resources Available through the "
               "American Sign Language Linguistic Research Project (ASLLRP)', arXiv:2201.07899; Data Access Interface (DAI 2): "
               "https://dai.cs.rutgers.edu/dai/s/dai; ASLLRP ASL Sign Bank: https://dai.cs.rutgers.edu/dai/s/signbank; "
               "plus the citation given on each dataset's DAI page."),
    "ASLLVD": ("C. Neidle, A. Thangali and S. Sclaroff (2012), 'Challenges in Development of the American Sign Language Lexicon "
               "Video Dataset (ASLLVD) Corpus', 5th Workshop on the Representation and Processing of Sign Languages: "
               "Interactions between Corpus and Lexicon, LREC 2012, Istanbul, Turkey, May 27, 2012; ASLLRP ASL Sign Bank: "
               "https://dai.cs.rutgers.edu/dai/s/signbank."),
    "ASL Citizen": "Desai et al., ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition, NeurIPS 2023 Datasets.",
    "YouTube-ASL": "Uthus, Tanzer, Georg, YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus, NeurIPS 2023 (arXiv:2306.15162).",
    "YouTube-SL-25": "Tanzer & Zhang, YouTube-SL-25: A Large-Scale, Open-Domain Multilingual Sign Language Parallel Corpus, 2024 (arXiv:2407.11144; includes YouTube-ASL).",
    "MediaPipe": "Lugaresi et al., MediaPipe: A Framework for Building Perception Pipelines, 2019.",
    "T5": "Chung et al., Scaling Instruction-Finetuned Language Models (Flan-T5), 2022.",
}


# requested by Carol Neidle (ASLLRP), 2026-10-23/24; wording to be reviewed again by ASLLRP before any distribution
ASLLRP_ATTRIBUTION = (
    "**ASLLRP attribution.** This model was trained in part on data from the American Sign Language Linguistic Research "
    "Project (ASLLRP), Boston University, obtained through the Data Access Interface (DAI 2, "
    "https://dai.cs.rutgers.edu/dai/s/dai) and the ASLLRP ASL Sign Bank (https://dai.cs.rutgers.edu/dai/s/signbank), under "
    "the ASLLRP Sign Bank Data Terms of Use. We thank Carol Neidle, Dimitris Metaxas and the ASLLRP team. No data, video or "
    "annotation is redistributed. Please cite C. Neidle, A. Opoku and D. Metaxas (2022), arXiv:2201.07899; C. Neidle, "
    "A. Thangali and S. Sclaroff (2012), LREC 2012 workshop (ASLLVD); and each DAI dataset used (the ASLLRP continuous "
    "signing corpus and the RIT sentence collection) as its DAI page specifies. *This wording awaits ASLLRP's review "
    "before distribution.*")


# guidance from the YouTube-ASL / YouTube-SL-25 authors (Manfred Georg, Google), email of 2026-10-29 22:02 UTC; the
# authors stated it is practical guidance, not legal advice
YOUTUBE_ASL_GUIDANCE = (
    "**YouTube-ASL / YouTube-SL-25 terms (authors' guidance, 2026-10-29).** The video-ID lists and annotations of "
    "YouTube-ASL and YouTube-SL-25 are released under CC BY 4.0; the underlying videos and captions remain subject to "
    "the YouTube Terms of Service. A build trained on them may release its weights with attribution, but no raw "
    "videos, captions or full extracted feature sets are distributed with them. Cite Uthus, Tanzer and Georg (2023), "
    "arXiv:2306.15162, and/or Tanzer and Zhang (2024), arXiv:2407.11144 (YouTube-SL-25 includes YouTube-ASL). The "
    "authors note that the corpora's domain variance (many signers, recording set-ups and registers) can reduce "
    "generation consistency, that the static YouTube splits are unstable as videos are removed (How2Sign and "
    "FLEURS-ASL are the stable benchmarks), and that this guidance is practical, not legal advice.")
YOUTUBE_ABSENT_NOTE = ("*This release contains no YouTube-ASL data; the terms above apply to any later build "
                       "that does.*")

def cite(name: str) -> str | None:
    return next((v for k, v in CITATIONS.items() if k.lower() in name.lower()), None)


def table(rows: list[tuple], header: tuple) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def card(release: dict, reports: dict) -> str:
    g = release["generator"]
    lines = [f"# Model card: {release['name']} {release['version']}", "",
             "## Model details",
             f"- **System:** English text -> {'ASL sign plan (gloss planner) -> ' if release.get('planner') else ''}"
             "3D signing motion (body, hands, head, face blendshapes, gaze) -> avatar or skeleton rendering.",
             f"- **Generator:** `{g['checkpoint']['key']}` (SHA-256 `{g['checkpoint']['sha256'][:16]}...`), conditioned on "
             f"{'the plan line' if g['condition'] == 'plan' else 'the English sentence'}; guidance {g['guidance']}, "
             f"per-part temperatures {g.get('part_temperature') or 'none'}.",
             *([f"- **Post-processing:** head-rotation smoothing {g.get('head_smoothing', 0)} s"
                + (f", head excursions damped to {g['head_damping']}x around the clip mean" if g.get("head_damping", 1) != 1 else "")
                + ("; per-blendshape face calibration (gain = real / generated within-clip spread, clipped to [1, "
                   f"{g['face_calibration'].get('max_gain', 4)}], estimated on validation, confirmed once on test)"
                   if g.get("face_calibration") else "") + "."]
               if g.get("head_smoothing") or g.get("face_calibration") or g.get("head_damping", 1) != 1 else []),
             *([f"- **Known limits from the exit gate:** {', '.join(release['selection']['known_limits'])}."]
               if (release.get("selection") or {}).get("known_limits") else []),
             (f"- **Planner:** `{release['planner']['key']}` (SHA-256 `{release['planner']['sha256'][:16]}...`)."
              if release.get("planner") else "- **Planner:** none."),
             f"- **Timing:** {release['fps']} fps, {release['seconds_per_sign']} s per planned sign.",
             f"- **Release SHA-256:** `{reports.get('release_sha256', '')}`.", "",
             "## Intended use",
             "- Research on sign-language generation and its evaluation; demonstrations to and evaluation by Deaf signers.",
             "- **Not** a replacement for a qualified interpreter. **Not** for medical, legal, emergency or other "
             "safety-critical communication. **Not** for identifying signers.", "",
             "## Training data and licences", "",
             ASLLRP_ATTRIBUTION, "",
             YOUTUBE_ASL_GUIDANCE
             + ("" if any("youtube" in c["name"].lower() for c in release["provenance"]) else " " + YOUTUBE_ABSENT_NOTE),
             ""]
    rows = [(c["name"], c["licence"], "yes" if c["research_only"] else "no", "yes" if c.get("non_commercial") else "no",
             c.get("permission") or "-", cite(c["name"]) or "-") for c in release["provenance"]]
    lines += [table(rows, ("Corpus", "Licence / terms", "Research only", "Non-commercial", "Release permission",
                           "Citation")), "",
              f"- **Public release allowed by the licence gate:** {release.get('public_release_allowed')}; "
              f"**commercial use allowed:** {release.get('commercial_use_allowed')}.",
              "- Research-only corpora are released only with the data owners' written permission, recorded in the "
              "release provenance.", "", "## Evaluation", ""]
    gate = reports.get("gate")
    if gate:
        sel = gate["selected"]
        lines += [f"**Exit gate** (validation, declared before results): selected `{sel['run']}` / {sel['setting']}, "
                  f"Fréchet {sel['frechet']:.1f}; failing checks: {', '.join(sel['failing']) or 'none'}.", ""]
    strict = reports.get("strict")
    if strict:
        rows = []
        for name, r in strict["rules"].items():
            if name.startswith("R1"):
                detail = f"generated R@3 {r['generated']['r_at_3']} {r['generated']['ci95']} vs blank {r['blank']['r_at_3']} {r['blank']['ci95']}, real {r['real']['r_at_3']}"
            elif name.startswith("R2"):
                detail = f"generated BLEU {r['generated']['bleu']} {r['generated']['ci95']} vs blank {r['blank']['bleu']}, real {r['real']['bleu']}"
            elif name.startswith("R3"):
                detail = f"copy rate {r['copy_rate']} (median ratio {r['median_ratio_generated']} vs real {r['median_ratio_real']})"
            elif name.startswith("R5"):
                detail = f"top-5 {r['generated']['top5']} {r['generated']['ci95']} vs blank {r['blank']['top5']}, real {r['real']['top5']}"
            elif name.startswith("R6"):
                detail = f"test regressions: {', '.join(r['regressions']) or 'none'}"
            else:
                detail = ", ".join(f"{k}: {'pass' if v['pass'] else 'fail'}" for k, v in r.get("slices", {}).items())
            rows.append((name, "pass" if r["pass"] else "**fail**", detail))
        lines += ["**Strict pass test** (declared before results):", "",
                  table(rows, ("Rule", "Result", "Numbers")), "", f"Overall: **{'pass' if strict['pass'] else 'fail'}**.", ""]
    sign = reports.get("sign_level")
    if sign:
        lines += [f"**Sign level** (validated recogniser, {sign['glosses']} reliable glosses): generated top-1 "
                  f"{sign['generated']['top1']} vs real {sign['real']['top1']} and blank {sign['blank']['top1']}.", ""]
    human = reports.get("human")
    if human:
        rows = [(s, v["comprehension_choice"]["rate"], v["comprehension_choice"]["ci95"], (v.get("naturalness") or {}).get("mean"))
                for s, v in human["systems"].items()]
        lines += ["**Human evaluation** (Deaf ASL reviewers, blinded; docs/HUMAN_EVALUATION_PROTOCOL.md):", "",
                  table(rows, ("System", "Comprehension (4-way, chance 0.25)", "95% CI", "Naturalness 1-5")), ""]
    if not human:
        lines += ["**Human evaluation:** pending (reviewer packets built; no sheets returned yet).", ""]
    lines += ["All automatic scores are relative: real recorded motion is the ceiling of each instrument and blank "
              "(duration-only) motion its floor. They do not measure comprehension by Deaf signers.", "",
              "## Limitations",
              "- Motion is estimated from monocular or two-view video, not motion capture; fingers and depth are noisier than body.",
              "- Spatial grammar (placing referents in signing space) is weak: the planner's locus head is at the "
              "per-gloss baseline.",
              "- Fingerspelling, name signs and classifier constructions are under-represented.",
              "- Trained mostly on How2Sign instructional signing; other registers and dialects are less covered.",
              "- Signer demographics and regional variants are skewed towards the signers who appear in the source "
              "corpora (few signers in How2Sign; web video for YouTube-ASL builds).",
              "- A research artifact: it is **not** ready for accessibility deployment without evaluation with Deaf "
              "and Hard of Hearing users.", "",
              "## Ethical considerations",
              "- Built with and for Deaf signers' review; outputs must be presented as machine-generated.",
              "- Training videos belong to their signers and creators; no video, keypoints or captions are released.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--release", type=Path, required=True)
    for k in ("gate", "strict", "bleu", "sign-level", "human"):
        parser.add_argument(f"--{k}-key")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from signtranslator.serving.release import Release
    release = Release.from_dict(json.loads(args.release.read_text()))
    reports = dict(release_sha256=release.sha256())
    if any(getattr(args, f"{k.replace('-', '_')}_key") for k in ("gate", "strict", "bleu", "sign-level", "human")):
        from train_text_to_motion import make_store
        store = make_store(args.store)
        for k in ("gate", "strict", "bleu", "sign_level", "human"):
            key = getattr(args, f"{k}_key")
            if key:
                reports[k] = json.loads(store.get(key))
    raw = json.loads(args.release.read_text())
    args.output.write_text(card(release.to_dict() | {"selection": raw.get("selection")}, reports) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
