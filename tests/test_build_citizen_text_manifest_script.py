"""ASL Citizen gloss labels -> English words."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_citizen_text_manifest import english  # noqa: E402


def test_gloss_labels_become_words():
    assert english("LIBRARY") == "library" and english("CANDY2") == "candy"
    assert english("THIS/IT") == "this" and english("STAND-UP") == "stand up" and english("W.H.A.T") == "what"
    assert english("SPLASH-CA") is None and english("WALK-TIGHTROPE-CL") is None and english("9OCLOCK") is None
    assert english("CHEW2MOUTH") is None
