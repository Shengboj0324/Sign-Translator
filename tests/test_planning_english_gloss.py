import torch

from signtranslator.generation.motion_to_text import WordVocabulary
from signtranslator.planning.asllrp_gloss import (EOS, GLOSS_END, GlossVocabulary, MARKERS, canonical_loci,
                                                  encode_target, parse_surface_token, plan_target)
from signtranslator.planning.english_gloss import (EnglishGlossConfig, EnglishGlossPlanner, encode_sources,
                                                   evaluate_plans, plan_from_generation)


def test_surface_tokens_parse_to_base_and_loci():
    assert parse_surface_token("i:MOVE:j_____") == parse_surface_token("i:MOVE:j")
    s = parse_surface_token("i:MOVE:j"); assert (s.base, s.source, s.target) == ("MOVE", "i", "j")
    s = parse_surface_token("(2h)IX-3p:i"); assert (s.base, s.target) == ("IX-3p", "i")
    assert parse_surface_token("ICL:tight-C\"eating").base == "ICL:tight-C"
    assert parse_surface_token("5\\hesitation\\\"").base == "5"
    assert parse_surface_token("ns-fs-MARY__").base == "ns-fs-MARY"
    assert parse_surface_token("#NO+__").base == "#NO+"


def test_loci_are_renumbered_by_first_appearance():
    signs = [parse_surface_token(t) for t in ["IX-3p:j", "j:GIVE:i", "IX-3p:i", "BOOK"]]
    assert canonical_loci(signs) == [(0, 1), (1, 2), (0, 2), (0, 0)]


def test_vocabulary_spells_rare_glosses_exactly():
    v = GlossVocabulary(["BOOK"])
    ids = v.pieces("BOOK") + v.pieces("ns-fs-ZOË") + [EOS]
    assert v.decode(ids) == ["BOOK", "ns-fs-ZOË"]
    assert v.pieces("ns-fs-ZOË")[-1] == GLOSS_END


def _pair(english, surface, markers=()):
    return dict(english=english, surface=list(surface), markers=list(markers), partition="train")


def test_encoded_targets_carry_attributes_only_at_sign_ends():
    t = plan_target(_pair("x", ["IX-3p:i", "RARE-WORD"], [dict(tier="negative", label="neg", token_start=1, token_end=2)]))
    v = GlossVocabulary(["IX-3p"])
    ids, src, dst, mk, end = encode_target(v, t)
    assert sum(end) == 2 and ids[-1] == EOS
    k = MARKERS.index("negative")
    closing = [i for i, e in enumerate(end) if e]
    assert dst[closing[0]] == 1 and mk[closing[1]][k] == 1 and mk[closing[0]][k] == 0
    assert all(m == [0] * len(MARKERS) for m, e in zip(mk, end) if not e)


def test_planner_overfits_a_tiny_corpus_and_decodes_its_plans():
    torch.manual_seed(0)
    pairs = [_pair("he gives her the book", ["IX-3p:i", "i:GIVE:j", "BOOK"]),
             _pair("did you finish", ["IX-2p", "FINISH"], [dict(tier="y/n question", label="q", token_start=0, token_end=2)]),
             _pair("i do not like it", ["IX-1p", "NOT", "LIKE"], [dict(tier="negative", label="neg", token_start=1, token_end=3)])]
    targets = [plan_target(p) for p in pairs]
    gv = GlossVocabulary.build(targets, min_count=1)
    wv = WordVocabulary.build([p["english"] for p in pairs], min_count=1)
    cfg = EnglishGlossConfig(len(wv), len(gv), d_model=64, heads=4, encoder_layers=1, decoder_layers=1,
                             feedforward=128, dropout=0., label_smoothing=0.)
    model = EnglishGlossPlanner(cfg)
    enc = [encode_target(gv, t) for t in targets]
    L = max(len(e[0]) for e in enc)

    def pad(rows, fill):
        return torch.tensor([r + [fill] * (L - len(r)) for r in rows])
    tokens = pad([e[0] for e in enc], 0); src = pad([e[1] for e in enc], 0); dst = pad([e[2] for e in enc], 0)
    mk = pad([e[3] for e in enc], [0] * len(MARKERS)); end = pad([e[4] for e in enc], False)
    source = encode_sources(wv, [p["english"] for p in pairs], max_tokens=32)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss, _ = model.loss(source, tokens, src, dst, mk, end)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    plans = [plan_from_generation(gv, g) for g in model.generate(source, beam=2)]
    refs = [dict(glosses=list(t.bases), loci=list(t.loci), markers=[list(m) for m in t.markers]) for t in targets]
    scores = evaluate_plans(refs, plans)
    assert scores["gloss_wer"] == 0 and scores["locus_accuracy"] == 1 and scores["marker_f1_micro"] == 1


def test_evaluation_counts_substitutions_and_marker_misses():
    ref = [dict(glosses=["A", "B", "C"], loci=[(0, 0)] * 3, markers=[[0] * 7, [1] + [0] * 6, [0] * 7])]
    hyp = [dict(glosses=["A", "X", "C"], loci=[(0, 0)] * 3, markers=[[0] * 7] * 3)]
    s = evaluate_plans(ref, hyp)
    assert abs(s["gloss_wer"] - 1 / 3) < 1e-9 and s["marker_f1"]["topic/focus"] == 0


def test_split_quoted_descriptions_are_rejoined_and_markers_remapped():
    from signtranslator.planning.asllrp_gloss import merge_quoted
    surface = ["MOTHER", "SCL:bent-U-L\"person", "walking\"", "SELL+__", "\"WHAT\"_"]
    assert merge_quoted(surface)[0] == ["MOTHER", "SCL:bent-U-L\"person~walking\"", "SELL+__", "\"WHAT\"_"]
    t = plan_target(_pair("x", surface, [dict(tier="wh question", label="q", token_start=2, token_end=5)]))
    assert t.bases == ("MOTHER", "SCL:bent-U-L", "SELL+", "\"WHAT\"")
    k = MARKERS.index("wh question")
    assert [m[k] for m in t.markers] == [0, 1, 1, 1]       # the span started inside the merged sign
