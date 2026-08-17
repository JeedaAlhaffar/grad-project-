# -*- coding: utf-8 -*-
"""
Checks for the intermediate-level word recogniser (ADAB Conformer).

    python test_word_handwriting.py

Runs WITHOUT the real normalisation stats: the checks that need a loadable
model build a THROWAWAY feat_norm_adab.pt in a temp directory. That proves the
plumbing, not the accuracy — recognised text under fake statistics is
meaningless, so nothing here asserts on what the model reads.

The accuracy numbers that matter (CER 1.65% / WER 6.01%) come from the
notebook's held-out ADAB test set, not from this file.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adab_model"))

_fail = 0


def ok(cond, label, extra=""):
    global _fail
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + (f" — {extra}" if extra else ""))
    if not cond:
        _fail += 1


def fake_word(n_parts=3, seed=0):
    """A synthetic multi-stroke scribble shaped roughly like cursive ink."""
    rng = np.random.default_rng(seed)
    strokes = []
    for k in range(n_parts):
        t = np.linspace(0, 4 * np.pi, 90)
        x = t * 6 + k * 70
        y = 40 * np.sin(t) + 60 + rng.normal(0, 1.0, t.shape)
        strokes.append([[float(a), float(b)] for a, b in zip(x, y)])
    return strokes


print("1) feature extraction matches the trained input contract")
import word_handwriting_preprocessing as prep

feats = prep.drawing_to_features(fake_word())
ok(feats is not None, "a drawing produces features")
ok(feats.shape[1] == prep.N_FEATURES == 46, "feature width is 46", str(feats.shape))
ok(feats.dtype == np.float32, "features are float32", str(feats.dtype))
ok(bool(np.isfinite(feats).all()), "no NaN/inf reaches the model")
ok(prep.ADAB_SAMPLE_RATE == 125.0, "sample rate is ADAB's 125 Hz, not KHATT's 100")

print("\n2) malformed client payloads degrade instead of raising")
dict_pts = [[{"x": p[0], "y": p[1]} for p in fake_word()[0]]]
ok(prep.drawing_to_features(dict_pts) is not None, "{'x','y'} points are accepted")
mixed = [[[1, 2], [None, 3], "junk", [4, 5], [6, 7], [8, 9]]]
ok(prep.drawing_to_features(mixed) is not None, "junk points are dropped, not fatal")
ok(prep.drawing_to_features([]) is None, "an empty drawing yields None")
ok(prep.drawing_to_features([[]]) is None, "a drawing of empty strokes yields None")

print("\n3) the alphabet is the one the checkpoint was trained on")
os.environ["WORD_MODEL_DIR"] = MODEL_DIR
import importlib

import word_handwriting_model as wm
importlib.reload(wm)
ok(wm.NUM_CLASSES == 39, "39 output classes", str(wm.NUM_CLASSES))
ok(wm.ARABIC_CHARS[0] == "blank" and wm.BLANK == 0, "index 0 is the CTC blank")
ok(wm.unsupported_chars("الأمر") == set(), "curriculum word is spellable")
ok(wm.unsupported_chars("كتابٌ٣") == {"ٌ", "٣"},
   "diacritics and digits are flagged as unspellable")

print("\n4) a missing normalisation file is refused, never guessed around")
import word_grader
importlib.reload(word_grader)
has_norm = os.path.isfile(os.path.join(MODEL_DIR, "feat_norm_adab.pt"))
if has_norm:
    ok(wm.is_available(), "real stats present — model reports available")
else:
    ok(not wm.is_available(), "is_available() is False without the stats")
    ok(any("feat_norm" in p for p in wm.missing_files()),
       "missing_files() names the stats file")
    ok(word_grader.grade("الأمر", fake_word()) is None,
       "grade() falls back rather than scoring on unnormalised input")

print("\n5) end-to-end plumbing, under THROWAWAY statistics")
import torch

scratch = tempfile.mkdtemp(prefix="adab_test_")
try:
    shutil.copy(os.path.join(MODEL_DIR, "best_model_adab.pth"), scratch)
    torch.save({"mean": torch.zeros(46), "std": torch.ones(46)},
               os.path.join(scratch, "feat_norm_adab.pt"))
    os.environ["WORD_MODEL_DIR"] = scratch
    importlib.reload(wm)
    importlib.reload(word_grader)

    ok(wm.is_available(), "both files present -> available")
    rec = wm.recognise_drawing(fake_word())
    ok(rec is not None, "recognition returns a result")
    ok(isinstance(rec["text"], str), "result carries text")
    ok(0.0 <= rec["confidence"] <= 1.0, "confidence is a probability",
       f"{rec['confidence']:.3f}")
    ok(set("".join(rec["text"])) <= wm.SUPPORTED_CHARS,
       "decoded text only uses the 39-symbol alphabet")

    out = word_grader.grade("الأمر", fake_word())
    ok(out is not None, "grade() returns a grade")
    score, fb = out
    ok(0.0 <= score <= 100.0, "score is within 0..100", str(score))
    for key in ("summary", "strengths", "errors", "suggestions", "recognition"):
        ok(key in fb, f"feedback carries '{key}'")
    ok(fb["recognition"]["target"] == "الأمر", "feedback reports the target back")

    ok(word_grader.grade("كتابٌ٣", fake_word()) is None,
       "an unspellable target is refused, not mis-graded")

    s0, fb0 = word_grader.grade("الأمر", [])
    ok(s0 == 0.0 and not fb0["recognition"]["recognised"],
       "an empty drawing is an unreadable card, not a crash")

    # A perfect "student" — feed the target through the same grader path the
    # recogniser feeds, so a correct reading must score 100.
    import dictation_grader
    ok(dictation_grader.grade("الأمر", "الأمر")["score"] == 100.0,
       "a correctly-read word scores 100 through the shared grader")
finally:
    os.environ["WORD_MODEL_DIR"] = MODEL_DIR
    importlib.reload(wm)              # drop the scratch dir off the module
    shutil.rmtree(scratch, ignore_errors=True)
ok(not os.path.exists(scratch), "throwaway statistics were cleaned up")

print("\n6) the beginner stack is not involved")
ok("handwriting_model" not in sys.modules,
   "grading a word never imports the letter classifier")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "word_grader.py"), encoding="utf-8").read()
ok("import handwriting_model" not in src, "word_grader does not import the letter model")
ok(wm.MODEL_DIR.rstrip("\\/").endswith("adab_model"),
   "word model reads from adab_model/, not Jeeda/", wm.MODEL_DIR)

print()
if _fail:
    print(f"{_fail} check(s) FAILED.")
    sys.exit(1)
print("All checks passed.")
if not has_norm:
    print("\nNOTE: adab_model/feat_norm_adab.pt is still missing, so the model is")
    print("      not servable yet — checks above used throwaway statistics.")
