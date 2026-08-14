# -*- coding: utf-8 -*-
"""
Check `letter_classes` against the actual classifier — run this after any
retrain, or whenever a copying grade looks wrong.

    cd backend
    python validate_letter_mapping.py

It feeds every reference letter shape in `letters drawing/letters_normalized/`
to the model and reports whether the class the backend mapped it to is the one
that comes back. The mapping is what makes a copying grade mean anything: if
ع/medial mapped to the wrong class, every correct ع would be marked wrong and
nothing else in the stack would notice.

How to read the output
----------------------
* A wrong TOKEN TABLE looks like one letter missing on ALL of its forms, and
  the predictions landing on some unrelated letter. That is a bug — fix
  TOKEN_FOR_LETTER.
* A hard MODEL/DATA case looks like the same letter one form away (ش medial
  predicted as B_sh) or a dotted pair (ت vs ث). That is not a mapping bug;
  the grader already gives partial credit for it (see copying_grader).

Baseline on 2026-08-13, checkpoint bgru_mha_final_model.h5:
    93 shapes, 92 mapped, 63 top-1 (68%), 29 differ, 1 documented gap (I_gh).
These reference files are idealised outlines, not human ink, so this number is
below the model's 83.6% test accuracy — it is a regression tripwire, not an
accuracy measurement.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
NORMALIZED = BACKEND.parent / "letters drawing" / "letters_normalized"

import letter_classes            # noqa: E402
import handwriting_model         # noqa: E402

# Filename tokens -> form / letter. Mirrors seed_letters.py, kept local so this
# script needs no database.
FORM_TOKENS = {
    "end": "final", "e": "final",
    "mid": "medial", "m": "medial", "middle": "medial", "midle": "medial",
    "iso": "isolated", "isolated": "isolated", "isolate": "isolated", "i": "isolated",
    "start": "initial", "s": "initial",
}
LETTER_TOKENS = {
    "alef": "ا", "ba": "ب", "ta": "ت", "tha": "ث", "thaa": "ث", "gem": "ج",
    "jeem": "ج", "haa": "ح", "khaa": "خ", "dal": "د", "zal": "ذ", "raa": "ر",
    "zin": "ز", "seen": "س", "sheen": "ش", "shen": "ش", "saad": "ص",
    "daad": "ض", "too": "ط", "zah": "ظ", "ain": "ع", "ghain": "غ", "faa": "ف",
    "fa": "ف", "qaf": "ق", "kaf": "ك", "kaaf": "ك", "lam": "ل", "meem": "م",
    "noon": "ن", "ha": "ه", "ya": "ي", "waw": "و",
}


def resolve(stem: str):
    """'B0006_2_1_001_Ain_S_points' -> ('ع', 'initial'). None if unreadable."""
    parts = [p for p in re.split(r"[_\s]+", stem) if p and p.lower() != "points"]
    for i in range(len(parts) - 1, 0, -1):
        form = FORM_TOKENS.get(parts[i].lower())
        letter = LETTER_TOKENS.get(parts[i - 1].lower())
        if form and letter:
            return letter, form
    return None


def main() -> int:
    if not NORMALIZED.is_dir():
        print(f"reference shapes not found: {NORMALIZED}")
        return 2

    rows, seen = [], set()
    for path in sorted(NORMALIZED.glob("*.json")):
        got = resolve(re.sub(r"\s*\(\d+\)", "", path.stem))
        if not got or got in seen:
            continue
        letter, form = got
        with io.open(path, encoding="utf-8") as fh:
            strokes = (json.load(fh) or {}).get("strokes") or []
        if not strokes:
            continue
        seen.add(got)

        expected = letter_classes.class_for(letter, form)
        result = handwriting_model.score_for_class(strokes, expected, top_k=3)
        rows.append((letter, form, expected, result))

    hit = miss = unmapped = 0
    wrong_letter: list[str] = []
    print(f"{'letter':6} {'form':9} {'mapped':7} {'predicted':10} "
          f"{'conf':>6} {'rank':>4}  verdict")
    for letter, form, expected, result in rows:
        predicted = result["predicted"] if result else "UNUSABLE"
        conf = f"{result['confidence']:.3f}" if result else "  -  "
        rank = str(result["target_rank"]) if result else "-"

        if expected is None:
            verdict, unmapped = "no class (documented gap)", unmapped + 1
        elif predicted == expected:
            verdict, hit = "MATCH", hit + 1
        else:
            miss += 1
            parsed = letter_classes.parse_class(predicted)
            if parsed and parsed[0] == letter:
                verdict = "same letter, other form"
            else:
                verdict = "DIFFERENT LETTER"
                wrong_letter.append(f"{letter}/{form}")

        print(f"{letter:6} {form:9} {str(expected):7} {predicted:10} "
              f"{conf:>6} {rank:>4}  {verdict}")

    total = hit + miss
    print(f"\nshapes: {len(rows)}   mapped: {total}   "
          f"top-1: {hit} ({100 * hit / max(total, 1):.0f}%)   "
          f"differs: {miss}   unmapped: {unmapped}")

    # A token-table error shows up as a letter whose shapes NEVER come back as
    # that letter — in any form. Landing on a different form of the same letter
    # (ط medial -> E_to) still confirms the token; only the form was confused.
    # One shape reading as something else is ordinary model error, so a letter
    # is only called suspect once it has failed on two or more of its shapes.
    evidence: dict[str, list[bool]] = {}
    for letter, _form, expected, result in rows:
        if expected is None or result is None:
            continue
        parsed = letter_classes.parse_class(result["predicted"])
        evidence.setdefault(letter, []).append(bool(parsed and parsed[0] == letter))

    suspect = sorted(l for l, ok in evidence.items() if not any(ok) and len(ok) >= 2)
    thin = sorted(l for l, ok in evidence.items() if not any(ok) and len(ok) == 1)

    if suspect:
        print(f"\nSUSPECT MAPPING — never read as itself on any of its "
              f"{len(evidence[suspect[0]])}+ shapes: {' '.join(suspect)}")
        print("Check TOKEN_FOR_LETTER in letter_classes.py for these letters.")
        return 1

    print(f"\nToken table looks sound: {len(evidence)} letters, each read as "
          f"itself on at least one shape.")
    if thin:
        print(f"  note: {' '.join(thin)} missed on their only reference shape — "
              f"too little evidence to judge, add more samples to be sure.")
    if wrong_letter:
        print(f"  note: {len(wrong_letter)} shapes read as a different letter "
              f"({', '.join(wrong_letter)}) — model/data difficulty, not mapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
