# -*- coding: utf-8 -*-
"""Checks for the dictation (إملاء) grader. No models, no network, ~1 s.

    python test_dictation.py

Covers the two production bugs this grader was written to fix, so neither can
come back silently:

  * a perfect answer scored 0/100, because the reference handed to the grader
    is the fully diacritized script and no student can type tashkeel;
  * a merged pair of words was charged as two errors and desynchronised the
    alignment for everything after it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dictation_grader as dg

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILURES.append(name)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


# ===========================================================================
print("\n1) the regression that broke the whole level")
# ===========================================================================
# What routers/attempts.py used to hand the grader, and what a student types.
DIAC = "مِمَّا يَزِيدُ الْأَمْرَ تَعْقِيداً"
PLAIN = "مما يزيد الأمر تعقيدا"

r = dg.grade(DIAC, PLAIN)
check("perfect answer vs diacritized reference scores 100",
      r["score"] == 100.0, f"score={r['score']}")
check("...and reports no errors", r["errors"] == [], f"{len(r['errors'])} errors")

check("perfect answer vs plain reference scores 100",
      dg.grade(PLAIN, PLAIN)["score"] == 100.0)
check("a student who does type tashkeel is not punished for it",
      dg.grade(DIAC, DIAC)["score"] == 100.0)

# ===========================================================================
print("\n2) error typing — one tag per known mistake")
# ===========================================================================
CASES = [
    ("الأمر", "الامر", "OH", "hamza dropped to bare alef"),
    ("إسعاف", "اسعاف", "OH", "hamzat qat' kasra dropped"),
    ("طائرة", "طايرة", "OH", "hamza seat written as its bare carrier"),
    ("المستعجلة", "المستعجله", "OT", "ta marbuta written as ha"),
    ("عناية", "عنايت", "OT", "ta marbuta written as ta maftuha"),
    ("يخشى", "يخشا", "OA", "alef maqsura written as alef"),
    ("على", "علي", "OA", "alef maqsura written as ya"),
    ("تعقيدا", "تعقيد", "ON", "tanwin alef dropped"),
    ("الثلوج", "اثلوج", "OM", "silent lam of lam shamsiyya dropped"),
    ("سرير", "سرر", "OM", "long vowel dropped"),
    ("سرير", "سريير", "OD", "spurious long vowel"),
    ("الرطوبة", "الرتوبة", "OR", "perceptual confusion ط/ت"),
    ("عناية", "عنياة", "OC", "two letters transposed"),
]
for ref, stu, want, why in CASES:
    got = dg.classify(ref, stu)
    check(f"{ref} → {stu} is {want} ({why})", got == want, f"got {got}")

check("an identical word yields no tag", dg.classify("كتاب", "كتاب") == "")

# ===========================================================================
print("\n3) soft vs hard — did the student mishear, or misspell?")
# ===========================================================================
check("hamza is soft (heard right, wrote the wrong seat)",
      dg.severity_class("OH") == "soft")
check("ta marbuta is soft", dg.severity_class("OT") == "soft")
check("wrong consonant is hard (the sound is not in the word)",
      dg.severity_class("OR") == "hard")
check("missed word is hard", dg.severity_class("XM") == "hard")

allsoft = dg.grade("طائرة إسعاف للحالات المستعجلة", "طايرة اسعاف للحالات المستعجله")
check("an all-soft attempt is told they heard the words correctly",
      any("سمعت" in s for s in allsoft["strengths"]), str(allsoft["strengths"]))

# ===========================================================================
print("\n4) word-boundary errors count once, not twice")
# ===========================================================================
merged = dg.grade("وننقل تراث الآباء والأجداد", "وننقل تراث الآباءوالأجداد")
check("a merge is a single error", len(merged["errors"]) == 1,
      f"{len(merged['errors'])} errors")
check("...tagged MG", merged["errors"] and merged["errors"][0]["tag"] == "MG")

split = dg.grade("وذلك طلبا للترويح وجلب المياه", "وذلك طلبا للترويح و جلب المياه")
check("a split is a single error", len(split["errors"]) == 1,
      f"{len(split['errors'])} errors")
check("...tagged SP", split["errors"] and split["errors"][0]["tag"] == "SP")

# the merge must not desynchronise the words that follow it
tail = dg.grade("وننقل تراث الآباء والأجداد اليوم", "وننقل تراث الآباءوالأجداد اليوم")
check("a merge does not mis-blame the following word",
      all("اليوم" not in (e.get("span") or "") for e in tail["errors"]))

missing = dg.grade("يخشى كثيرون أن تشكل تهديدا للبشر", "يخشى كثيرون أن تشكل تهديدا")
check("a missed word is tagged XM",
      any(e["tag"] == "XM" for e in missing["errors"]))

# ===========================================================================
print("\n5) scoring behaves sensibly")
# ===========================================================================
ref = "طائرة إسعاف للحالات المستعجلة"
perfect = dg.grade(ref, ref)["score"]
one = dg.grade(ref, "طايرة إسعاف للحالات المستعجلة")["score"]
three = dg.grade(ref, "طايرة اسعاف للحالات المستعجله")["score"]
blank = dg.grade(ref, "")["score"]
check("more errors never scores higher", perfect > one > three > blank,
      f"{perfect} > {one} > {three} > {blank}")
check("an empty answer scores 0", blank == 0.0, f"score={blank}")
check("a soft error costs less than losing the word",
      dg.grade("عناية", "عنايه")["score"] > dg.grade("عناية", "")["score"])
check("scores stay within 0..100",
      all(0.0 <= dg.grade(ref, s)["score"] <= 100.0
          for s in ["", ref, "كلمة", ref + " زائدة"]))

empty_ref = dg.grade("", "أي نص")
check("a missing reference does not crash", empty_ref["score"] == 0.0,
      empty_ref["summary"])

# ===========================================================================
print("\n6) the cards match the API contract")
# ===========================================================================
try:
    from schemas import EvaluationFeedback

    fb = dg.grade(ref, "طايرة اسعاف للحالات المستعجله")
    v = EvaluationFeedback(**{k: fb[k] for k in
                              ("summary", "strengths", "errors", "suggestions")})
    check("feedback validates as EvaluationFeedback", len(v.errors) == 3,
          f"{len(v.errors)} errors survived")
    check("the ARETA tag survives validation", v.errors[0].tag == "OH",
          str(v.errors[0].tag))
    check("the soft/hard class survives validation",
          v.errors[0].subcategory in ("soft", "hard"), str(v.errors[0].subcategory))
except Exception as exc:                                  # pragma: no cover
    check("feedback validates as EvaluationFeedback", False, repr(exc))

# ===========================================================================
print("\n7) the pure-Python fallback matches the C accelerator")
# ===========================================================================
# python-Levenshtein sits under the optional GEC section of requirements.txt,
# so dictation has to keep working without it — and produce the same answers.
src = (Path(__file__).resolve().parent / "dictation_grader.py").read_text(encoding="utf-8")
ns: dict = {}
exec(compile(src.replace("from Levenshtein import", "from _absent_module_ import"),
             "dictation_grader(fallback)", "exec"), ns)
check("fallback path is actually active", not ns["_HAVE_LEVENSHTEIN"])

PAIRS = [(ref, "طايرة اسعاف للحالات المستعجله"), (DIAC, PLAIN),
         ("وننقل تراث الآباء والأجداد", "وننقل تراث الآباءوالأجداد"),
         ("عناية", "عنياة"), ("سرير", "سريير"), ("الثلوج", "اثلوج")]
same_score = all(dg.grade(a, b)["score"] == ns["grade"](a, b)["score"] for a, b in PAIRS)
same_tags = all([e["tag"] for e in dg.grade(a, b)["errors"]] ==
                [e["tag"] for e in ns["grade"](a, b)["errors"]] for a, b in PAIRS)
check("fallback produces identical scores", same_score)
check("fallback produces identical tags", same_tags)

# ===========================================================================
print("\n8) the seeded references are themselves correct Arabic")
# ===========================================================================
# Five of the twenty-five shipped references had the definite article written
# with a hamza (ألتي for التي), which marked correct Arabic wrong.
seed_path = Path(__file__).resolve().parent / "seed_data" / "audio_dictation.json"
if seed_path.is_file():
    seed = json.load(open(seed_path, encoding="utf-8"))
    clips = [c for u in seed["units"] for c in u["clips"]]
    BAD = {"ألغذاء", "ألتي", "ألحكومة", "ألأيقونات", "الإنضمام"}
    offenders = [(c["slug"], w) for c in clips
                 for w in c["text_plain"].split() if w in BAD]
    check("no reference spells the definite article with a hamza",
          not offenders, str(offenders))

    try:
        from camel_tools.utils.dediac import dediac_ar
        mismatched = [c["slug"] for c in clips
                      if dediac_ar(c["text"]) != c["text_plain"]]
        check("text and text_plain agree for every clip",
              not mismatched, str(mismatched[:5]))
    except ImportError:
        print("  [skip] camel-tools not installed; text/text_plain check skipped")

    # and the grader gives a perfect student full marks on every real item
    perfect_all = [c["slug"] for c in clips
                   if dg.grade(c["text"], c["text_plain"])["score"] != 100.0]
    check("a perfect answer scores 100 on all 25 real items",
          not perfect_all, str(perfect_all))
else:
    print("  [skip] seed_data/audio_dictation.json not found")

# ===========================================================================
print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print("   -", f)
    sys.exit(1)
print("All checks passed.")
