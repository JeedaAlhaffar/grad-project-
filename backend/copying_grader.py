# -*- coding: utf-8 -*-
"""
Copying (نسخ الحروف) grading — the three exercise modes of one letter.

A copying exercise shows the SAME letter three times, in increasing difficulty:

    bold_line    خط باهت      a light grey outline of the letter to trace over
    dotted_line  خط منقّط     dots to join
    free_draw    رسم حر        nothing — write the letter from memory

So the two questions worth asking are different per mode:

    "did you follow the line?"   -> geometry: how close the student's strokes
                                    are to `Letter.strokes` (the reference shape)
    "is it the right letter?"    -> the handwriting classifier
                                    (handwriting_model, 107 letter shapes)

free_draw has no line to follow, so it is graded purely by the classifier —
the model decides whether what the student wrote is the ع they were asked for.
The guided modes are mostly a tracing score, with the classifier still voting
so that a neat scribble along the guide can't pass as a letter.

    mode          recognition   trace
    bold_line        0.35        0.65      full guide -> mostly "did you follow it"
    dotted_line      0.55        0.45      partial guide
    free_draw        1.00        0.00      no guide  -> purely "is it the letter"

Honesty note: the recognition half is the trained, evaluated model (83.6% test
accuracy, see model/config.json). The trace half is a plain geometric measure
(symmetric Chamfer distance between the two normalised point clouds) — it is a
heuristic, not a validated model, which is why it never grades free_draw on its
own and why its constants are env-overridable. Tune TRACE_MAX_DISTANCE against
real student drawings before trusting the guided-mode numbers.

What the model actually does on our own shapes
----------------------------------------------
`validate_letter_mapping.py` feeds every reference letter in
`letters drawing/letters_normalized/` to the classifier: 63/92 land on the
mapped class top-1 (68%), below the 83.6% test accuracy because those files are
idealised vector outlines, not human ink. The failures are informative — they
are overwhelmingly the SAME letter one form away (ش medial -> B_sh, ي final ->
I_ya, ظ medial -> E_zha, all rank 2), which is why the grade is not a top-1
pass/fail: rank<=3 and same-letter-different-form both score above the pass
mark. Re-run that script after any retrain; a token-table error would show up
as one letter missing on ALL its forms, which is not what we see.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import letter_classes

_log = logging.getLogger("copying")

Feedback = dict

# --- tuning knobs -----------------------------------------------------------
# (recognition weight, trace weight) per copying mode. Must sum to 1.
MODE_WEIGHTS: dict[str, tuple[float, float]] = {
    "bold_line": (0.35, 0.65),
    "dotted_line": (0.55, 0.45),
    "free_draw": (1.00, 0.00),
}
_DEFAULT_WEIGHTS = MODE_WEIGHTS["free_draw"]

# Chamfer distance (on shapes normalised into a unit box) at which the trace
# score hits 0. A careful trace lands around 0.02-0.05; 0.25 is "nowhere near".
TRACE_MAX_DISTANCE = float(os.environ.get("COPYING_TRACE_MAX_DISTANCE", "0.25"))
# Points per stroke when building the comparison cloud. Higher = slower, finer.
TRACE_POINTS_PER_STROKE = int(os.environ.get("COPYING_TRACE_POINTS", "40"))

# A drawing with fewer than this many points is not an attempt.
MIN_POINTS = 4


# ===========================================================================
# Geometry — "did you follow the line?"
# ===========================================================================
def _point_cloud(strokes: list, per_stroke: int):
    """Strokes -> an (N, 2) cloud normalised into a unit box, aspect preserved.

    Every stroke is resampled to the same number of points so a long stroke
    does not outvote a short one, and the whole shape is scaled by its longest
    side — the student's canvas pixels and the reference's 0..1 coordinates
    then live in the same frame. Both sides are y-down, so no flip is needed.
    """
    import numpy as np

    from handwriting_preprocessing import resample_stroke

    pieces = []
    for s in strokes or []:
        pts = np.array(s, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] >= 2 and len(pts) >= 2:
            pieces.append(resample_stroke(pts[:, :2], per_stroke))
    if not pieces:
        return None

    cloud = np.vstack(pieces)
    lo = cloud.min(axis=0)
    scale = float(np.max(cloud.max(axis=0) - lo))
    if scale <= 1e-6:
        return None
    return (cloud - lo) / scale


def _chamfer(a, b) -> float:
    """Symmetric mean nearest-neighbour distance between two point clouds."""
    import numpy as np

    # (len(a), len(b)) pairwise distances — both clouds are a few hundred
    # points at most, so the dense matrix is cheap and exact.
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def trace_score(submitted: list, reference: list) -> Optional[float]:
    """0..100 for how closely `submitted` follows `reference`. None if N/A."""
    try:
        a = _point_cloud(submitted, TRACE_POINTS_PER_STROKE)
        b = _point_cloud(reference, TRACE_POINTS_PER_STROKE)
    except Exception as exc:                       # numpy/scipy missing
        _log.warning("trace scoring unavailable: %s: %s", type(exc).__name__, exc)
        return None
    if a is None or b is None:
        return None

    d = _chamfer(a, b)
    return round(max(0.0, min(100.0, 100.0 * (1.0 - d / TRACE_MAX_DISTANCE))), 1)


# ===========================================================================
# Recognition — "is it the right letter?"
# ===========================================================================
def letter_probability(probabilities: dict, arabic_letter: str) -> float:
    """Total probability the model put on this LETTER, in any of its forms.

    Measured on the reference shapes (validate_letter_mapping.py), the model's
    top-1 miss is usually the same letter in a neighbouring form — ش medial
    read as `B_sh`, ي final read as `I_ya`, ظ medial read as `E_zha`. The
    student in those cases wrote the right letter; only the form is arguable.
    Summing per letter separates that from "wrote a different letter", which
    is the mistake the exercise actually cares about.
    """
    total = 0.0
    for class_name, p in (probabilities or {}).items():
        parsed = letter_classes.parse_class(class_name)
        if parsed and parsed[0] == arabic_letter:
            total += p
    return total


def _recognition_score(probability: float, rank: int,
                       same_letter_top1: bool, letter_p: float) -> float:
    """Turn the classifier's view of the target into 0..100.

    Rank carries most of the signal — the model putting the asked-for shape
    first is the pass condition — and the probability grades within the band.
    The same-letter band sits deliberately just above the pass mark (50): a
    recognisable ع written in the wrong form is a partial success, not a fail.
    """
    if rank == 1:
        return round(75.0 + 25.0 * probability, 1)
    if rank <= 3:
        return round(58.0 + 17.0 * probability, 1)
    if same_letter_top1:
        return round(50.0 + 15.0 * letter_p, 1)
    if rank <= 10:
        return round(35.0 + 15.0 * probability, 1)
    return round(10.0 + 25.0 * probability, 1)


def _recognize(strokes: list, target_class: Optional[str]) -> Optional[dict]:
    """Run the classifier. None when it is off, unusable or unavailable."""
    import handwriting_model

    try:
        return handwriting_model.score_for_class(strokes, target_class, top_k=3)
    except handwriting_model.HandwritingUnavailable as exc:
        _log.warning("handwriting model unavailable: %s", exc)
        return None
    except Exception as exc:
        _log.warning("handwriting prediction failed: %s: %s",
                     type(exc).__name__, exc)
        return None


# ===========================================================================
# The grader
# ===========================================================================
def grade(letter, submitted_drawing: list, mode: str,
          use_model: bool = True) -> Optional[tuple[float, Feedback]]:
    """Grade one copying attempt.

    `letter` is a models.Letter row (arabic_letter, form, strokes, stroke_count).
    Returns (score 0..100, feedback dict), or None when neither half of the
    grade could be computed — the caller then falls back to the mock so a
    missing checkpoint never blocks a student.
    """
    strokes = submitted_drawing or []
    n_points = sum(len(s) for s in strokes if isinstance(s, (list, tuple)))
    expected_label = letter_classes.describe_letter(letter.arabic_letter, letter.form)

    if not strokes or n_points < MIN_POINTS:
        return 0.0, {
            "summary": "لم نتمكّن من قراءة الرسم. حاول كتابة الحرف مرة أخرى.",
            "strengths": [],
            "errors": [{"message": "لا يوجد رسم كافٍ لتقييمه",
                        "correction": None, "type": "shape"}],
            "suggestions": [f"ارسم {expected_label} داخل المربّع المخصّص."],
            "recognition": {"mode": mode, "recognized": False,
                            "expected": {"letter": letter.arabic_letter,
                                         "label": expected_label}},
        }

    target_class = letter_classes.class_for(letter.arabic_letter, letter.form)
    rec = _recognize(strokes, target_class) if use_model else None
    trace = trace_score(strokes, letter.strokes or [])

    # --- recognition half -------------------------------------------------
    rec_score: Optional[float] = None
    recognized = False
    letter_p: Optional[float] = None
    if rec is not None and rec.get("target_rank") is not None:
        letter_p = letter_probability(rec.get("probabilities"), letter.arabic_letter)
        top_parsed = letter_classes.parse_class(rec["predicted"])
        same_letter_top1 = bool(top_parsed and top_parsed[0] == letter.arabic_letter)
        rec_score = _recognition_score(
            rec["target_probability"], rec["target_rank"],
            same_letter_top1, letter_p,
        )
        recognized = rec["target_rank"] == 1
    elif rec is not None and target_class is None:
        # Shapes with no trained class (isolated غ) — the model ran but cannot
        # answer the question that was asked. Don't let it grade.
        _log.info("no trained class for %s/%s — shape-only grading",
                  letter.arabic_letter, getattr(letter.form, "value", letter.form))

    if rec_score is None and trace is None:
        return None                      # nothing to grade with

    # --- blend by mode ----------------------------------------------------
    w_rec, w_trace = MODE_WEIGHTS.get(mode, _DEFAULT_WEIGHTS)
    degraded = False
    if rec_score is None:
        # Classifier off, unavailable, or asked about an untrained shape. The
        # trace is the only signal left, so we use it — but free_draw is
        # DEFINED as "the classifier decides", so a trace-only free_draw grade
        # is not the same measurement. Flag it rather than pass it off as one.
        w_rec, w_trace = 0.0, 1.0
        degraded = True
    elif trace is None or w_trace == 0.0:  # free_draw, or no reference shape
        w_rec, w_trace = 1.0, 0.0
        degraded = trace is None and mode != "free_draw"

    score = round(w_rec * (rec_score or 0.0) + w_trace * (trace or 0.0), 1)

    feedback = _build_feedback(
        letter=letter, mode=mode, score=score, recognized=recognized,
        rec=rec, rec_score=rec_score, trace=trace, degraded=degraded,
        letter_p=letter_p, target_class=target_class,
        expected_label=expected_label,
    )
    return score, feedback


def _build_feedback(*, letter, mode: str, score: float, recognized: bool,
                    rec: Optional[dict], rec_score: Optional[float],
                    trace: Optional[float], degraded: bool,
                    letter_p: Optional[float], target_class: Optional[str],
                    expected_label: str) -> Feedback:
    strengths: list[str] = []
    errors: list[dict] = []
    suggestions: list[str] = []

    # --- what the model saw ----------------------------------------------
    predicted_label = None
    if rec is not None:
        predicted_label = letter_classes.describe_class(rec["predicted"])
        if recognized:
            strengths.append(f"تم التعرّف على {expected_label} بشكل صحيح")
        else:
            parsed = letter_classes.parse_class(rec["predicted"])
            if parsed and parsed[0] != letter.arabic_letter:
                # wrong letter entirely — the most useful thing we can say
                errors.append({
                    "message": f"الشكل المرسوم أقرب إلى {predicted_label} "
                               f"منه إلى {expected_label}",
                    "correction": f"المطلوب كتابة {expected_label}",
                    "span": letter.arabic_letter, "type": "shape",
                })
                suggestions.append(f"انظر إلى نموذج {expected_label} ثم أعد المحاولة.")
            elif parsed:
                # right letter, wrong form (ع في وسط الكلمة vs ع في آخرها)
                errors.append({
                    "message": f"الحرف صحيح لكن الشكل ليس {expected_label}؛ "
                               f"رسمتَ {predicted_label}",
                    "correction": f"المطلوب {expected_label}",
                    "span": letter.arabic_letter, "type": "shape",
                })
                suggestions.append("انتبه إلى موضع الحرف في الكلمة وشكله المناسب.")
            else:
                errors.append({
                    "message": f"لم نتعرّف بوضوح على {expected_label} في الرسم",
                    "correction": None, "span": None, "type": "shape",
                })

    # --- how well the guide was followed ----------------------------------
    if trace is not None and mode != "free_draw":
        if trace >= 80:
            strengths.append("اتّباع ممتاز لمسار الحرف")
        elif trace >= 60:
            strengths.append("اتّباع جيد لمسار الحرف")
        else:
            errors.append({
                "message": "الرسم بعيد عن المسار المرسوم لك",
                "correction": "حاول أن تمرّ فوق الخط تماماً",
                "span": None, "type": "shape",
            })
            suggestions.append(
                "تتبّع الخط الباهت ببطء" if mode == "bold_line"
                else "صِل بين النقاط بالترتيب ولا تتجاوزها"
            )

    if letter.stroke_count:
        suggestions.append(f"يُكتب هذا الحرف عادةً بـ {letter.stroke_count} حركة.")

    if score >= 90:
        summary = "ممتاز! كتابة صحيحة وواضحة"
    elif score >= 75:
        summary = "أحسنت! كتابة جيدة جداً"
    elif score >= 50:
        summary = "محاولة جيدة، واصل التدريب"
    else:
        summary = "أعد المحاولة، ركّز على شكل الحرف"

    if not strengths:
        strengths.append("بداية جيدة")
    if not suggestions:
        suggestions.append("تدرّب أكثر على شكل الحرف.")

    recognition: dict = {
        "mode": mode,
        "recognized": recognized,
        # True when the classifier did not contribute, so this grade is the
        # geometric fallback only. The app should not present it as letter
        # recognition, and free_draw in particular is not really being tested.
        "degraded": degraded,
        "expected": {
            "letter": letter.arabic_letter,
            "form": getattr(letter.form, "value", letter.form),
            "class": target_class,
            "label": expected_label,
        },
        "scores": {
            "recognition": rec_score,
            "trace": trace,
            "final": score,
        },
    }
    if rec is not None:
        recognition["predicted"] = {
            "class": rec["predicted"],
            "label": predicted_label,
            "confidence": round(rec["confidence"], 4),
        }
        recognition["target_probability"] = (
            round(rec["target_probability"], 4)
            if rec["target_probability"] is not None else None
        )
        # Probability mass on the right LETTER regardless of form — the number
        # to show when the student clearly wrote ع but the form is arguable.
        recognition["letter_probability"] = (
            round(letter_p, 4) if letter_p is not None else None
        )
        recognition["target_rank"] = rec["target_rank"]
        recognition["top_k"] = [
            {"class": c["class"],
             "label": letter_classes.describe_class(c["class"]),
             "confidence": round(c["confidence"], 4)}
            for c in rec["top_k"]
        ]

    return {
        "summary": summary,
        "strengths": strengths[:3],
        "errors": errors,
        "suggestions": suggestions[:3],
        "recognition": recognition,
    }


def warm_up() -> bool:
    """Preload the classifier (see main._warm_up_models)."""
    import handwriting_model

    return handwriting_model.warm_up()
