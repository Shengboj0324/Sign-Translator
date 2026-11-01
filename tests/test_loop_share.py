"""The loop share's definition (reports/post_s2_plan.md, section 1): whitespace tokens, not BLEU's. comp_a_lib.is_loop has the same
rule, and these cases hold both in step."""
import pytest

from signtranslator.eval.loop_share import is_loop, loop_flags
from signtranslator.generation.motion_to_text import words


@pytest.mark.parametrize("text,expected", [("a b c a b c a b c", True), ("a a a a a", True), ("a b c a b c", False), ("", False), ("a b", False),
                                           ("a b c d a b c d e a b x", False), ("A B C. a b c, A b C", False), ("x y z. x y z. x y z.", True)])
def test_a_whitespace_three_gram_at_least_three_times(text, expected):
    assert is_loop(text) is expected


def test_a_contraction_is_one_token_as_the_plans_figures_read_it():
    text = "you've you've you've"
    assert len(words(text)) > 3                                        # BLEU's tokens split it, and would call this a loop
    assert not is_loop(text)
    assert is_loop("you've it's so you've it's so you've it's so")


def test_flags_are_floats_in_order():
    assert loop_flags(["a a a a a", "b"]) == [1., 0.]


@pytest.mark.parametrize("text", ["a b c a b c a b c", "a a a a a", "a b c a b c", "", "a b", "A B C. a b c, A b C", "x y z. x y z. x y z.", "you've you've you've", "you've it's so you've it's so you've it's so"])
def test_the_rule_is_comp_as_own(text):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import comp_a_lib
    assert comp_a_lib.is_loop(text) is is_loop(text)
