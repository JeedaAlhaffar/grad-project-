# -*- coding: utf-8 -*-
"""
Grading a hand-written word or sentence (المستوى المتوسط — كلمات وجمل).

The intermediate level asks the student to WRITE a target word by hand. So the
grade is a two-step affair:

    strokes --[ADAB Conformer]--> recognised text --[compare to target]--> grade

The second step is the same problem the dictation grader already solves: a known
reference, a student rendering of it, and the question "which letters went
wrong, and what kind of mistake is that". So this module does not re-implement
alignment or error typing — it calls `dictation_grader`, which means a student
who writes ة for ه gets the identical Arabic error card whether they typed it in
an إملاء exercise or drew it here.

    beginner  (نسخ)   -> copying_grader   -> handwriting_model (letters, Keras)
    intermediate      -> THIS MODULE      -> word_handwriting_model (ADAB, torch)

THE HONESTY PROBLEM, AND WHAT IS DONE ABOUT IT
----------------------------------------------
Recognition is not perfect: the Conformer reports CER 1.65% / WER 6.01% on its
held-out ADAB test set. So a mismatch between the recognised text and the target
has two possible causes, and they deserve opposite responses:

    the student wrote it wrong      -> teach the spelling rule
    the model misread correct ink   -> say nothing, or the app lies to a child

Nothing in the strokes distinguishes these with certainty. What is available is
the model's own confidence, so this module refuses to assert an error when
confidence is below RECOGNITION_MIN_CONFIDENCE: the attempt is returned as
"couldn't read it clearly, try again" rather than as a wrong answer. That
deliberately trades a few missed corrections for not blaming a child for the
model's mistake. Tune the threshold against real drawings before trusting the
guided numbers — it is a policy dial, not a measured optimum.

WER 6.01% is also measured on ADAB's adult writers copying set phrases. Children
writing curriculum words are a different distribution, and the true error rate on
them is unknown until real attempts are logged.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

_log = logging.getLogger("word_grader")

Feedback = dict

# Below this mean per-frame confidence we decline to judge rather than risk
# blaming the student for a recognition failure. Env-overridable on purpose.
RECOGNITION_MIN_CONFIDENCE = float(
    os.environ.get("WORD_RECOGNITION_MIN_CONFIDENCE", "0.55"))

# A drawing with fewer points than this is not an attempt.
MIN_POINTS = 8


def _unreadable(target: str, reason: str) -> tuple[float, Feedback]:
    return 0.0, {
        "summary": "لم نتمكّن من قراءة ما كتبت. حاول مرة أخرى بخط أوضح.",
        "strengths": [],
        "errors": [{"message": reason, "correction": None, "type": "shape"}],
        "suggestions": [f'اكتب كلمة "{target}" داخل المساحة المخصّصة، وبخط واضح.'],
        "recognition": {"recognised": False, "target": target},
    }


def grade(target_text: str, submitted_drawing: list,
          use_model: bool = True) -> Optional[tuple[float, Feedback]]:
    """Grade one hand-written word/sentence against its target.

    Returns (score 0..100, feedback), or None when the model is unavailable so
    the caller can fall back — a missing checkpoint must never block a student.
    """
    target = (target_text or "").strip()
    if not target:
        return None

    strokes = submitted_drawing or []
    n_points = sum(len(s) for s in strokes if isinstance(s, (list, tuple)))
    if not strokes or n_points < MIN_POINTS:
        return _unreadable(target, "لا يوجد رسم كافٍ لتقييمه")

    if not use_model:
        return None

    import word_handwriting_model as wm

    if not wm.is_available():
        _log.warning("ADAB word model unavailable, missing: %s", wm.missing_files())
        return None

    # A target the 39-symbol alphabet cannot spell can never be graded here;
    # the seed scripts filter for this, so hitting it means bad content.
    bad = wm.unsupported_chars(target)
    if bad:
        _log.warning("target %r has characters outside the ADAB alphabet: %s",
                     target, "".join(sorted(bad)))
        return None

    try:
        rec = wm.recognise_drawing(strokes)
    except Exception as exc:
        _log.warning("recognition failed: %s", exc)
        return None
    if rec is None:
        return _unreadable(target, "تعذّرت قراءة الخط")

    recognised = (rec.get("text") or "").strip()
    confidence = float(rec.get("confidence") or 0.0)

    if not recognised:
        return _unreadable(target, "لم يتعرّف النظام على أي حرف")

    # --- decline to judge when the model itself is unsure -------------------
    if confidence < RECOGNITION_MIN_CONFIDENCE and recognised != target:
        score, fb = _unreadable(target, "الخط غير واضح بما يكفي للحكم عليه")
        fb["recognition"] = {"recognised": False, "target": target,
                             "read_as": recognised, "confidence": round(confidence, 3),
                             "declined": True}
        fb["suggestions"] = [f'أعد كتابة "{target}" بخط أوضح وأبطأ قليلاً.']
        return score, fb

    # --- same machinery the dictation grader uses --------------------------
    import dictation_grader

    result = dictation_grader.grade(target, recognised)

    feedback: Feedback = {
        "summary": result["summary"],
        "strengths": result["strengths"],
        "errors": result["errors"],
        "suggestions": result["suggestions"],
        "recognition": {
            "recognised": True,
            "target": target,
            "read_as": recognised,
            "confidence": round(confidence, 3),
            "declined": False,
        },
    }

    # Say plainly what the model read, so a wrong reading is visible to the
    # student and the teacher instead of surfacing as an unexplained error.
    if recognised != target:
        feedback["strengths"] = list(feedback["strengths"])
        feedback["suggestions"] = list(feedback["suggestions"]) + [
            f'قرأ النظام ما كتبته على أنه "{recognised}".'
        ]

    return float(result["score"]), feedback
