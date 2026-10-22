"""A sign plan as a line of text: the motion generator's conditioning for plan item 2.

The generators encode their condition as UTF-8 bytes (``text_to_motion.encode_texts``). Writing the plan - base
glosses, locus slots and the grammatical markers that span them - as one deterministic line lets the same encoder,
the same classifier-free guidance (the empty line) and every existing trainer learn plan -> motion, and lets the
planner's output drive generation without a second encoder. The form is

    [topic MOTHER ] IX-loc>1 1>GIVE>2 [neg NOT LIKE ]

- a marker opens with ``[<tag>`` before the first sign it spans and closes with `` ]`` after the last; markers that
  open together nest outer (longer) first; when an inner marker outlasts the one around it, the inner one is closed and
  reopened, so the line is always properly bracketed and every sign's marker set is exact;
- ``a>GLOSS>b`` marks an agreement sign moving from slot ``a`` to slot ``b``; ``GLOSS>b`` a sign directed at (an
  index or possessive pointing to) slot ``b``; ``a>GLOSS`` one leaving slot ``a``;
- glosses are the planner's base glosses, which contain no spaces, ``[`` or ``>``.

``plan_from_text`` parses the line back, so the encoding is checked to be lossless on every plan.
"""
from __future__ import annotations

from .asllrp_gloss import MARKERS

TAGS = {"topic/focus": "topic", "wh question": "wh", "y/n question": "yn", "rhq question": "rhq",
        "negative": "neg", "cond/when": "cond", "relative cl": "rel"}
_TIER = {v: k for k, v in TAGS.items()}


def plan_to_text(glosses: list[str], loci: list[tuple[int, int]], markers: list[list[int]]) -> str:
    if not (len(glosses) == len(loci) == len(markers)):
        raise ValueError("glosses, loci and markers must align")
    n = len(glosses)

    def run_end(i, k):                                  # first sign after i's run of marker k
        j = i
        while j < n and markers[j][k]:
            j += 1
        return j
    stack, parts = [], []                               # open markers, innermost last
    for i, (gloss, (a, b)) in enumerate(zip(glosses, loci)):
        active = {k for k in range(len(MARKERS)) if markers[i][k]}
        if any(k not in active for k in stack):         # close down to the deepest marker that ends here,
            keep = []                                   # then reopen the still-active ones popped on the way
            while any(k not in active for k in stack):
                k = stack.pop(); parts.append("]")
                if k in active:
                    keep.append(k)
            for k in reversed(keep):
                parts.append(f"[{TAGS[MARKERS[k]]}"); stack.append(k)
        new = sorted((k for k in active if k not in stack), key=lambda k: (-run_end(i, k), k))   # outer first
        for k in new:
            parts.append(f"[{TAGS[MARKERS[k]]}"); stack.append(k)
        token = gloss
        if a:
            token = f"{a}>{token}"
        if b:
            token = f"{token}>{b}"
        parts.append(token)
    parts += ["]"] * len(stack)
    return " ".join(parts)


def plan_from_text(text: str) -> tuple[list[str], list[tuple[int, int]], list[list[int]]]:
    glosses, loci, markers, open_tags = [], [], [], []
    for part in text.split():
        if part.startswith("[") and part[1:] in _TIER:
            open_tags.append(MARKERS.index(_TIER[part[1:]]))
        elif part == "]":
            if not open_tags:
                raise ValueError("unbalanced ]")
            open_tags.pop()
        else:
            a = b = 0
            pieces = part.split(">")
            if len(pieces) == 3:
                a, gloss, b = int(pieces[0]), pieces[1], int(pieces[2])
            elif len(pieces) == 2 and pieces[0].isdigit():
                a, gloss = int(pieces[0]), pieces[1]
            elif len(pieces) == 2 and pieces[1].isdigit():
                gloss, b = pieces[0], int(pieces[1])
            else:
                gloss = part
            m = [0] * len(MARKERS)
            for k in open_tags:
                m[k] = 1
            glosses.append(gloss); loci.append((a, b)); markers.append(m)
    if open_tags:
        raise ValueError("unbalanced [")
    return glosses, loci, markers


__all__ = ["TAGS", "plan_to_text", "plan_from_text"]
