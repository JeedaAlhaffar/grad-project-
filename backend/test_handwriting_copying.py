# -*- coding: utf-8 -*-
"""
Checks for the copying (نسخ الحروف) grading pipeline.

Run from `backend/`:
    python test_handwriting_copying.py            # everything
    HANDWRITING_ENABLED=0 python test_handwriting_copying.py   # geometry only

Three groups:
  1. letter_classes  -- every letter/form the app teaches maps to a class that
     really exists in label_map.npy (this is the part that would silently mark
     correct answers wrong if the token table were guessed).
  2. trace scoring   -- a perfect trace beats a sloppy one beats a wrong letter.
  3. end-to-end      -- copying_grader.grade() over the three modes, using real
     reference strokes from `letters drawing/letters_normalized/`.

The recognition numbers are PRINTED, not asserted: the reference strokes are
idealised templates, not student handwriting, so treating them as ground truth
for a model trained on human ink would be a bogus test. Assertions are limited
to what must hold regardless (mapping validity, score ranges, ordering).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NORMALIZED = ROOT / "letters drawing" / "letters_normalized"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import letter_classes                                             # noqa: E402

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


class FakeLetter:
    """Stand-in for a models.Letter row (no DB needed)."""

    def __init__(self, arabic_letter: str, form: str, strokes: list):
        self.arabic_letter = arabic_letter
        self.form = form
        self.strokes = strokes
        self.stroke_count = len(strokes)


# --- test fixtures ----------------------------------------------------------
FORM_OF_SUFFIX = {"End": "final", "Iso": "isolated", "Mid": "medial", "Start": "initial"}
LETTER_OF_NAME = {"Ain": "ع", "Haa": "ح", "Khaa": "خ", "Ghain": "غ", "Gem": "ج"}


def load_reference(name: str, suffix: str):
    """Reference strokes for e.g. ("Ain", "Mid") -> (FakeLetter, path)."""
    path = NORMALIZED / f"{name}_{suffix}_points.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return FakeLetter(LETTER_OF_NAME[name], FORM_OF_SUFFIX[suffix],
                      data.get("strokes") or [])


def jitter(strokes: list, amount: float) -> list:
    """A sloppier version of the same shape (deterministic, no RNG seed games)."""
    import math

    out = []
    for si, stroke in enumerate(strokes):
        pts = []
        for i, (x, y) in enumerate(stroke):
            pts.append([x + amount * math.sin(0.7 * i + si),
                        y + amount * math.cos(0.5 * i + si)])
        out.append(pts)
    return out


# ===========================================================================
print("\n1) letter_classes — the mapping the grade depends on")
# ===========================================================================
label_map_path = Path(
    os.environ.get("HANDWRITING_MODEL_DIR")
    or ROOT / "beginer level copy letters" / "model"
) / "label_map.npy"

known_classes: set[str] = set()
if label_map_path.is_file():
    import numpy as np

    known_classes = set(np.load(label_map_path, allow_pickle=True).item().keys())
    check("label_map.npy loads", len(known_classes) == 107, f"{len(known_classes)} classes")
else:
    print(f"  [skip] {label_map_path} not found — mapping checked structurally only")

# The 29 letters of the app's 6 copying units (see seed_letters.py).
APP_ALPHABET = "ا د ذ ر ز و ب ت ث ط ظ ج ح خ ع غ س ش ص ض ك ل ن ف ق م ه ة ي".split()
check("every app letter has a token",
      all(l in letter_classes.TOKEN_FOR_LETTER for l in APP_ALPHABET),
      f"{len(APP_ALPHABET)} letters")

check("tokens are unique (no two letters share a class)",
      len(set(letter_classes.TOKEN_FOR_LETTER.values()))
      == len(letter_classes.TOKEN_FOR_LETTER))

if known_classes:
    missing, produced = [], 0
    for letter in APP_ALPHABET:
        for form in ("isolated", "initial", "medial", "final"):
            cls = letter_classes.class_for(letter, form)
            if cls is None:
                continue
            produced += 1
            if cls not in known_classes:
                missing.append(f"{letter}/{form}->{cls}")
    # I_gh is the one documented gap: the training set has no isolated غ.
    check("every produced class exists in the checkpoint",
          missing == ["غ/isolated->I_gh"] or not missing,
          f"{produced} shapes, unmatched: {missing or 'none'}")

    # Round-trip: a class the model can emit describes the letter we started from.
    round_trip_ok = all(
        letter_classes.parse_class(letter_classes.class_for(l, f) or "")
        in (None, (l, f))
        for l in APP_ALPHABET for f in ("isolated", "initial", "medial", "final")
    )
    check("class -> letter round-trips", round_trip_ok)

    # The non-connecting letters must have no initial/medial class at all —
    # this is the structural fingerprint that says the token table is right.
    non_connecting = "ا د ذ ر ز و ة".split()
    fingerprint = all(
        letter_classes.class_for(l, f) not in known_classes
        for l in non_connecting for f in ("initial", "medial")
    )
    check("non-connecting letters have no initial/medial class",
          fingerprint, " ".join(non_connecting))

print(f"  ع medial -> {letter_classes.class_for('ع', 'medial')} "
      f"({letter_classes.describe_letter('ع', 'medial')})")

# ===========================================================================
print("\n2) trace scoring — geometry only, no model")
# ===========================================================================
import copying_grader                                             # noqa: E402

ain_mid = load_reference("Ain", "Mid")
ain_end = load_reference("Ain", "End")
haa_mid = load_reference("Haa", "Mid")

if ain_mid is None or haa_mid is None:
    print("  [skip] reference strokes not found under letters_normalized/")
else:
    perfect = copying_grader.trace_score(ain_mid.strokes, ain_mid.strokes)
    sloppy = copying_grader.trace_score(jitter(ain_mid.strokes, 0.03), ain_mid.strokes)
    wrong = copying_grader.trace_score(haa_mid.strokes, ain_mid.strokes)

    print(f"  perfect={perfect}  sloppy={sloppy}  wrong-letter={wrong}")
    check("a perfect trace scores ~100", perfect is not None and perfect >= 99)
    check("sloppy < perfect", sloppy is not None and sloppy < perfect)
    check("wrong letter < sloppy", wrong is not None and wrong < sloppy)
    check("scores stay in 0..100",
          all(0 <= s <= 100 for s in (perfect, sloppy, wrong)))
    check("empty drawing has no trace score",
          copying_grader.trace_score([], ain_mid.strokes) is None)

# ===========================================================================
print("\n3) end-to-end — copying_grader.grade() over the 3 modes")
# ===========================================================================
use_model = os.environ.get("HANDWRITING_ENABLED", "1") == "1"
print(f"  handwriting model: {'on' if use_model else 'off (geometry only)'}")

if use_model:
    # The grader degrades to the trace score whenever the classifier fails, so
    # "the tests passed" must NOT be reachable with a checkpoint that never
    # loads — that is exactly how the Keras 3 `score_mode` breakage hid itself.
    import handwriting_model

    try:
        engine = handwriting_model.get_engine()
        check("checkpoint loads", True,
              f"{len(engine.idx_to_class)} classes, max_len={engine.max_len}")
        probe = handwriting_model.predict(
            [[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]], top_k=1)
        check("model returns a prediction", probe is not None
              and len(probe["probabilities"]) == 107)
    except Exception as exc:
        check("checkpoint loads", False, f"{type(exc).__name__}: {exc}")

if ain_mid is None:
    print("  [skip] no reference strokes")
else:
    empty = copying_grader.grade(ain_mid, [], "free_draw", use_model=use_model)
    check("empty drawing scores 0 and says so",
          empty is not None and empty[0] == 0.0 and empty[1]["errors"])

    for mode in ("bold_line", "dotted_line", "free_draw"):
        result = copying_grader.grade(ain_mid, ain_mid.strokes, mode,
                                      use_model=use_model)
        if result is None:
            check(f"{mode}: gradeable", False, "no signal available")
            continue
        score, fb = result
        rec = fb["recognition"]
        check(f"{mode}: score in 0..100", 0 <= score <= 100, f"score={score}")
        check(f"{mode}: feedback is populated",
              bool(fb["summary"]) and isinstance(fb["errors"], list))
        if use_model:
            check(f"{mode}: graded by the model, not the fallback",
                  not rec["degraded"] and rec["scores"]["recognition"] is not None)
        print(f"      recognition={rec['scores']['recognition']} "
              f"trace={rec['scores']['trace']} final={rec['scores']['final']}")
        if rec.get("predicted"):
            print(f"      model saw: {rec['predicted']['label']} "
                  f"({rec['predicted']['class']}, "
                  f"conf={rec['predicted']['confidence']:.3f}); "
                  f"target {rec['expected']['class']} "
                  f"rank={rec['target_rank']} p={rec['target_probability']}")

    # The mode weights must actually differentiate the three exercises.
    if ain_end is not None:
        wrong_shape = ain_end.strokes            # right letter, wrong form
        scores = {}
        for mode in ("bold_line", "dotted_line", "free_draw"):
            r = copying_grader.grade(ain_mid, wrong_shape, mode, use_model=use_model)
            scores[mode] = r[0] if r else None
        print(f"  wrong-form drawing: {scores}")
        check("guided modes penalise a drawing that ignores the guide",
              scores["bold_line"] is not None
              and scores["bold_line"] < 100)

    check("mode weights sum to 1",
          all(abs(sum(w) - 1.0) < 1e-9 for w in copying_grader.MODE_WEIGHTS.values()))

# --- the score bands, checked directly ------------------------------------
# The reference shapes don't happen to produce a rank>3 same-letter case, so
# exercise the partial-credit rule itself rather than hoping a fixture hits it.
_score = copying_grader._recognition_score
bands = {
    "exact (rank 1)": _score(0.90, 1, True, 0.95),
    "close (rank 3)": _score(0.30, 3, False, 0.40),
    "right letter, wrong form": _score(0.05, 6, True, 0.80),
    "far (rank 6)": _score(0.05, 6, False, 0.10),
    "absent (rank 40)": _score(0.001, 40, False, 0.01),
}
print("  bands: " + "  ".join(f"{k}={v}" for k, v in bands.items()))
check("exact > close > right-letter > far > absent",
      bands["exact (rank 1)"] > bands["close (rank 3)"]
      > bands["right letter, wrong form"] > bands["far (rank 6)"]
      > bands["absent (rank 40)"])
check("a recognisable letter in the wrong form still passes",
      bands["right letter, wrong form"] >= 50)
check("a letter the model does not see at all fails",
      bands["absent (rank 40)"] < 50)

check("letter_probability sums every form of the letter",
      abs(copying_grader.letter_probability(
          {"I_ay": 0.4, "M_ay": 0.3, "E_ay": 0.1, "I_ba": 0.2}, "ع") - 0.8) < 1e-9)

# --- the seam the API actually calls --------------------------------------
if ain_mid is not None:
    import ai_service

    score, fb = ai_service.evaluate_copying(ain_mid, ain_mid.strokes, "free_draw")
    check("ai_service.evaluate_copying returns a real grade",
          0 <= score <= 100 and "summary" in fb, f"score={score}")

    from schemas import EvaluationFeedback

    try:
        EvaluationFeedback(**fb)
        check("feedback validates against EvaluationFeedback", True)
    except Exception as exc:
        check("feedback validates against EvaluationFeedback", False, str(exc)[:150])

# ===========================================================================
print()
if _failures:
    print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
    sys.exit(1)
print("All checks passed.")
