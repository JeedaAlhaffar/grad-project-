# -*- coding: utf-8 -*-
"""
AI service layer  ***  THE ACTUAL AI LOGIC LIVES HERE  ***

Every function in this module is the seam between the plain CRUD backend and
the machine-learning models the team will build (AraBERT spelling/grammar
error detection, stroke-based letter grading, placement scoring, exercise
generation, TTS).

Status:
  * evaluate_text / composition (تعبير) -> REAL pipeline: error detection (GEC)
        -> topic relevance (hybrid CrossEncoder) -> AES traits (see below)
  * evaluate_text / words (كلمات)       -> REAL error detection (GEC)
  * evaluate_text / dictation (إملاء)   -> REAL aligner + ARETA-tagged errors
        (dictation_grader.py; deterministic, no ML dependency)
  * evaluate_copying (نسخ الحروف)       -> REAL handwriting classifier +
        trace score, per copying mode (see copying_grader.py)
  * everything else                      -> DETERMINISTIC MOCK (settings.AI_MOCK)

Mocks let the Flutter app render realistic data. Replace the remaining bodies
marked `# TODO(ai)` with real model calls; routers and response shapes are stable.
"""
from __future__ import annotations

from typing import Optional

from config import settings
from models import Letter, WritingType, Level

# The response shape mirrors schemas.EvaluationFeedback.
Feedback = dict


class AINotImplemented(Exception):
    """Raised by AI entrypoints when AI_MOCK is off and no model is wired yet."""


# ===========================================================================
# 1) Copying (نسخ) — grade a hand-drawn letter against the reference strokes
# ===========================================================================
def evaluate_copying(
    letter: Letter, submitted_drawing: list, mode: str
) -> tuple[float, Feedback]:
    """Grade one letter drawing in one of the 3 copying modes.

    REAL pipeline (see copying_grader.py):
      free_draw    -> the handwriting classifier decides whether the strokes
                      are the letter/form the exercise asked for.
      bold_line    -> mostly "did you follow the guide" (geometric trace
      dotted_line     score against Letter.strokes), classifier still voting.

    Falls through to the mock below when the checkpoint is missing or
    HANDWRITING_ENABLED=0, so a student is never blocked by a model.
    """
    try:
        import copying_grader

        result = copying_grader.grade(
            letter, submitted_drawing, mode,
            use_model=settings.HANDWRITING_ENABLED,
        )
        if result is not None:
            return result
    except Exception as exc:            # TF/checkpoint/deps missing, bad input
        if not settings.AI_MOCK:
            raise AINotImplemented(f"evaluate_copying: {exc}")

    if not settings.AI_MOCK:
        raise AINotImplemented("evaluate_copying")

    drawn_strokes = len(submitted_drawing or [])
    expected = letter.stroke_count or 1
    ratio = min(drawn_strokes, expected) / max(drawn_strokes, expected, 1)
    score = round(60 + 40 * ratio, 1)

    feedback: Feedback = {
        "summary": "أحسنت! مستوى جيد جداً" if score >= 75 else "محاولة جيدة، واصل التدريب",
        "strengths": ["كتابة صحيحة للحرف", "شكل جيد"] if score >= 75 else ["بداية جيدة"],
        "errors": (
            [] if score >= 90 else
            [{"message": "شكل الحرف غير مكتمل", "correction": None, "type": "shape"}]
        ),
        "suggestions": ["حاول الكتابة بشكل أوضح", "تدرب أكثر على شكل الحرف"],
    }
    return score, feedback


# ===========================================================================
# 1b) Words (كلمات / جمل) — grade a hand-WRITTEN word against its target
# ===========================================================================
def evaluate_words_drawing(
    target_text: str, submitted_drawing: list
) -> tuple[float, Feedback]:
    """Grade a drawn word/sentence for the intermediate level.

    REAL pipeline (see word_grader.py): the ADAB Conformer reads the strokes
    into text, then the dictation grader compares that text to the target, so
    the student gets the same tagged Arabic error cards as everywhere else.

    This is NOT `evaluate_copying`. That grades one drawn LETTER with the Keras
    BiGRU+MHA classifier out of Jeeda/; this grades a whole WORD with the
    PyTorch Conformer out of adab_model/. The two models share nothing.

    Falls through to the mock when the checkpoint or the normalisation stats
    are missing, so a student is never blocked by a model.
    """
    try:
        import word_grader

        result = word_grader.grade(
            target_text, submitted_drawing,
            use_model=settings.WORD_MODEL_ENABLED,
        )
        if result is not None:
            return result
    except Exception as exc:            # torch/checkpoint/deps missing, bad input
        if not settings.AI_MOCK:
            raise AINotImplemented(f"evaluate_words_drawing: {exc}")

    if not settings.AI_MOCK:
        raise AINotImplemented("evaluate_words_drawing")

    strokes = submitted_drawing or []
    n_points = sum(len(s) for s in strokes if isinstance(s, (list, tuple)))
    score = 70.0 if n_points >= 8 else 0.0
    feedback: Feedback = {
        "summary": "أحسنت! واصل التدريب" if score else "لم نتمكّن من قراءة ما كتبت",
        "strengths": ["محاولة جيدة"] if score else [],
        "errors": [],
        "suggestions": [f'تدرّب على كتابة "{target_text}".'],
        "recognition": {"recognised": False, "target": target_text, "mock": True},
    }
    return score, feedback


# ===========================================================================
# 2) Text (كلمات / إملاء / تعبير) — scoring + error detection
# ===========================================================================
#   * composition (تعبير) -> the full three-stage pipeline (see
#     `_evaluate_composition`): GEC error detection, then topic relevance,
#     then the Hybrid AES traits -> 0..100 score + Arabic feedback card.
#   * words (كلمات)       -> GEC error detection only.
#   * dictation (إملاء)   -> diff the submission against the reference target.
def evaluate_text(
    writing_type: WritingType,
    submitted_text: str,
    reference: Optional[str] = None,      # dictation target / composition prompt
) -> tuple[float, Feedback]:
    text = (submitted_text or "").strip()

    # --- تعبير: detection + relevance + AES -------------------------------
    if writing_type == WritingType.composition:
        if settings.AES_ENABLED:
            try:
                return _evaluate_composition(text, reference)
            except Exception as exc:  # model file/deps missing, load error, etc.
                if not settings.AI_MOCK:
                    raise AINotImplemented(f"evaluate_text/composition: {exc}")
                # else fall through to the mock below
        elif not settings.AI_MOCK:
            raise AINotImplemented("evaluate_text/composition (AES disabled)")

    # --- إملاء and كلمات: diff against the known reference (deterministic) -
    # Both types give us the exact expected string, so both are graded the same
    # way: align the answer to the reference and price the differences. For
    # كلمات that reference is the word the student was asked to copy, which is
    # also what the hand-written branch (word_grader) compares against — so a
    # typed answer and a drawn one are judged by one standard.
    if writing_type in (WritingType.dictation, WritingType.words) and reference:
        return _evaluate_dictation(text, reference)

    # NOTE: كلمات deliberately does NOT route to GEC. A كلمات exercise is a
    # copying task — the student reproduces the word shown — so it is graded
    # against its target above, exactly like إملاء. GEC (correction + ARETA
    # tagging) belongs to the professional level, where تعبير is free writing
    # with no reference to diff against. A كلمات item with no target is bad
    # content, not a case to spell-check: it falls through to the mock, or
    # raises with AI_MOCK=0 so the gap is visible instead of silently scored.

    if not settings.AI_MOCK:
        raise AINotImplemented("evaluate_text")

    # --- words / fallback mock -------------------------------------------
    words = [w for w in text.split() if w]
    n = len(words)
    score = float(min(100, 55 + n * 4))

    errors = []
    if "الى" in words:
        errors.append({
            "message": "خطأ في كتابة همزة القطع",
            "correction": 'يجب كتابة "إلى" وليس "الى"',
            "span": "الى", "type": "spelling",
        })
    feedback: Feedback = {
        "summary": "أحسنت! مستوى جيد جداً" if score >= 75 else "محاولة جيدة",
        "strengths": [
            "استخدام علامات الترقيم بشكل صحيح",
            "الأسلوب واضح ومفهوم",
            "أفكار جيدة",
        ][: 3 if score >= 75 else 1],
        "errors": errors,
        "suggestions": [
            "حاول استخدام مرادفات أكثر تنوعاً",
            "يمكنك تطوير الجمل لتكون أكثر تفصيلاً",
        ],
    }
    return score, feedback


# ---------------------------------------------------------------------------
# 2a) shared: error detection (GEC + ARETA)
# ---------------------------------------------------------------------------
def _detect_errors(text: str, future=None) -> Optional[dict]:
    """Run / collect the GEC detection. None when it is off or unavailable."""
    if not settings.GEC_ENABLED or not text:
        return None
    import gec_service                      # lazy: backend runs without ML deps

    try:
        if future is None:
            future = gec_service.submit(text)
        return gec_service.resolve(future, timeout=settings.GEC_TIMEOUT)
    except Exception:
        return None


def _start_detection(text: str):
    """Kick the GEC pipeline off first so it runs while the scorers work."""
    if not settings.GEC_ENABLED or not text:
        return None
    try:
        import gec_service
        return gec_service.submit(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2b) composition (تعبير): detection -> relevance -> AES
# ---------------------------------------------------------------------------
# Arabic labels + short advice per trait. `relevance` now comes from the hybrid
# RelevanceScorer; the other six come from the Hybrid AES model.
_TRAIT_AR = {
    "relevance":    ("مدى الارتباط بالموضوع", "اربط أفكارك بشكل أوضح بموضوع التعبير"),
    "organization": ("تنظيم الأفكار",         "رتّب فقراتك: مقدمة ثم عرض ثم خاتمة"),
    "development":  ("تطوير الأفكار وإثراؤها", "وسّع أفكارك بالأمثلة والشرح والتفصيل"),
    "vocabulary":   ("الثروة اللغوية والمفردات", "استخدم مفردات أكثر تنوعاً ودقّة"),
    "style":        ("الأسلوب",               "نوّع في تراكيب الجمل ليصبح أسلوبك أجمل"),
    "mechanics":    ("الإملاء وعلامات الترقيم", "انتبه للإملاء وعلامات الترقيم"),
    "grammar":      ("القواعد النحوية",       "راجع القواعد النحوية والتراكيب اللغوية"),
}

# Sub-traits that make up the final grade in "traits" mode (each 0..4).
_SCORED_SUBTRAITS = [
    "relevance", "organization", "development",
    "vocabulary", "style", "mechanics", "grammar",
]


def _evaluate_composition(
    text: str, prompt: Optional[str] = None
) -> tuple[float, Feedback]:
    """Grade a تعبير essay.

    The pipeline, in this order (detection first — it also gates the grading):

      1. GEC + ARETA error detection      -> the real error cards, started
                                             first and run in parallel with 2/3
      2. RelevanceScorer(prompt, essay)   -> relevance 0..4  (+ the <20-word
                                             guardrail: rejected essays are
                                             returned unscored)
      3. Hybrid AES model                 -> the six remaining traits + holistic

    The old AES `relevance` head is not used anywhere (see aes_model.py).
    """
    if not text:
        raise ValueError("empty composition text")

    # --- stage 1: detection starts first, in the background ----------------
    gec_future = _start_detection(text)

    # --- stage 2: relevance + guardrail ------------------------------------
    relevance = _score_relevance(prompt, text)

    if relevance is not None and relevance.get("status") == "rejected":
        # Too short to grade: hand back the guardrail message (and whatever
        # errors the detector already found) without running the AES model.
        gec = _detect_errors(text, gec_future)
        feedback: Feedback = {
            "summary": relevance.get("message") or "النص قصير جداً.",
            "strengths": [],
            "errors": (gec or {}).get("errors", []),
            "suggestions": [
                f"اكتب موضوعاً لا يقل عن {relevance.get('min_words', 20)} كلمة.",
                "وسّع أفكارك بمقدمة وعرض وخاتمة.",
            ],
            "relevance": relevance,
            "corrected_text": (gec or {}).get("corrected"),
        }
        return 0.0, feedback

    # --- stage 3: the AES traits -------------------------------------------
    import aes_model  # imported lazily so the backend runs without ML deps

    traits = aes_model.predict_traits(text)          # holistic + 6 sub-traits
    if relevance is not None and relevance.get("score") is not None:
        traits["relevance"] = float(relevance["score"])

    # --- collect the detection result (it has been running all along) ------
    gec = _detect_errors(text, gec_future)
    errors: list[dict] = list((gec or {}).get("errors", []))

    score = _composition_score(traits)

    strengths: list[str] = []
    suggestions: list[str] = []

    # sub-traits are on a 0..4 scale: >=3 is a strength, <2 needs work.
    for trait, (label, advice) in _TRAIT_AR.items():
        if trait not in traits:
            continue
        val = traits[trait]
        if val >= 3.0:
            strengths.append(label)
        elif val < 2.0:
            suggestions.append(advice)
            # Only invent a generic chip when the detector gave us nothing real.
            if not errors and trait in ("mechanics", "grammar"):
                errors.append({
                    "message": ("ضعف في الإملاء أو علامات الترقيم"
                                if trait == "mechanics" else
                                "أخطاء في القواعد النحوية"),
                    "correction": None, "span": None,
                    "type": "spelling" if trait == "mechanics" else "grammar",
                })

    counts = (gec or {}).get("counts") or {}
    if counts.get("spelling"):
        suggestions.append("راجع الكلمات التي أخطأت في إملائها وأعد كتابتها.")
    if counts.get("grammar"):
        suggestions.append("انتبه للقواعد النحوية في الجمل التي صُحّحت لك.")

    if score >= 90:
        summary = "ممتاز! تعبير متكامل ومتماسك"
    elif score >= 75:
        summary = "أحسنت! مستوى جيد جداً"
    elif score >= 50:
        summary = "محاولة جيدة، وهناك مجال للتحسين"
    else:
        summary = "بداية جيدة، تحتاج إلى مزيد من التدريب"

    if not strengths:
        strengths.append("بذلت جهداً في كتابة الموضوع")
    if not suggestions:
        suggestions.append("واصل الكتابة للحفاظ على هذا المستوى الجيد")

    feedback: Feedback = {
        "summary": summary,
        "strengths": strengths[:4],
        "errors": errors[:settings.GEC_MAX_ERRORS],
        "suggestions": suggestions[:4],
        # full per-trait breakdown for a richer results screen (holistic /32,
        # the rest /4). Ignored by clients that don't need it.
        "traits": {k: round(v, 2) for k, v in traits.items()},
        # the hybrid relevance result, exactly as the scorer returned it
        "relevance": relevance,
        "corrected_text": (gec or {}).get("corrected"),
        "error_counts": counts or None,
    }
    return score, feedback


def _score_relevance(prompt: Optional[str], text: str) -> Optional[dict]:
    """The hybrid topic-relevance scorer. None when disabled/unavailable."""
    if not settings.RELEVANCE_ENABLED:
        return None
    try:
        import relevance_scorer
        return relevance_scorer.score_relevance(prompt or "", text)
    except Exception:
        # model folder missing / sentence-transformers not installed:
        # fall back to grading without the relevance trait.
        return None


def _composition_score(traits: dict[str, float]) -> float:
    """Turn the trait breakdown into the app's 0..100 score.

    "traits"   (default) — the mean of the available sub-traits (relevance
                           included), each 0..4. This is what lets the new
                           relevance actually move the grade.
    "holistic"           — the AES holistic head alone (0..32), i.e. the
                           previous behaviour.
    """
    if settings.COMPOSITION_SCORE_MODE == "holistic":
        holistic = traits.get("holistic", 0.0)
        return round(max(0.0, min(100.0, holistic / 32.0 * 100.0)), 1)

    values = [traits[t] for t in _SCORED_SUBTRAITS if t in traits]
    if not values:
        holistic = traits.get("holistic", 0.0)
        return round(max(0.0, min(100.0, holistic / 32.0 * 100.0)), 1)
    mean = sum(values) / len(values)
    return round(max(0.0, min(100.0, mean / 4.0 * 100.0)), 1)


# ---------------------------------------------------------------------------
# 2c) words (كلمات): detection only
# ---------------------------------------------------------------------------
def _evaluate_words(text: str, gec: dict) -> tuple[float, Feedback]:
    """Grade a words exercise: one wrong word costs its share of the mark."""
    words = [w for w in text.split() if w]
    # A words drill has no sentences — a "missing full stop" is not a mistake.
    errors = [e for e in gec.get("errors", []) if e.get("type") != "punctuation"]
    n_errors = len(errors)

    if words:
        ratio = max(0.0, 1.0 - n_errors / len(words))
    else:
        ratio = 0.0
    score = round(max(0.0, min(100.0, 100.0 * ratio)), 1)

    summary = ("ممتاز! كتابة صحيحة" if score >= 90 else
               "أحسنت، أخطاء قليلة" if score >= 75 else
               "محاولة جيدة، انتبه للأخطاء" if score >= 50 else
               "تحتاج إلى مزيد من التدريب")
    feedback: Feedback = {
        "summary": summary,
        "strengths": ["كتابة صحيحة لمعظم الكلمات"] if score >= 75 else [],
        "errors": errors[:settings.GEC_MAX_ERRORS],
        "suggestions": (["راجع الكلمات التي أخطأت فيها وأعد كتابتها"]
                        if errors else ["أحسنت، حافظ على هذا المستوى"]),
        "corrected_text": gec.get("corrected"),
        # Count what we actually show: the punctuation errors were dropped
        # above, so gec's own counts would claim a mistake with no card to
        # match it — and it would cost the student marks they can't see.
        "error_counts": {
            "spelling": sum(1 for e in errors if e.get("type") == "spelling"),
            "grammar": sum(1 for e in errors if e.get("type") == "grammar"),
            "punctuation": 0,
        },
    }
    return score, feedback


def _evaluate_dictation(text: str, reference: str) -> tuple[float, Feedback]:
    """Grade إملاء by aligning the student's text to the dictated reference.

    Delegates to `dictation_grader`, which replaces the word-by-word difflib
    comparison this used to do. The reasons, measured in
    `advanced_level_dictation_assessment.ipynb`:

      * difflib cannot represent a merged or split word, so `الآباء والأجداد`
        written as one token was charged as two errors and the words after it
        were mis-blamed. Word localisation on the weakest simulated students
        was 0.857 against 0.929 for the aligner used now — the old code failed
        hardest on the students who most need accurate feedback.
      * every error came back as "خطأ إملائي" with no type, so the app could
        not tell a student *why* a word was wrong. Errors now carry an ARETA
        tag (the same tagset `gec_service` emits) and a soft/hard class saying
        whether the student misheard the word or simply misspelt a sound they
        heard correctly.
      * a word with one wrong letter scored the same as leaving it blank.

    Signature and Feedback shape are unchanged, so no client change is needed.
    """
    from dictation_grader import DictationGrader, DICTATION_SEVERITY

    result = DictationGrader(severity=DICTATION_SEVERITY).grade(reference, text)
    feedback: Feedback = {
        "summary": result["summary"],
        "strengths": result["strengths"],
        "errors": result["errors"][:settings.GEC_MAX_ERRORS],
        "suggestions": result["suggestions"],
    }
    return result["score"], feedback


# ===========================================================================
# 3) Placement test — score the 5 answers and decide a level
# ===========================================================================
def score_placement(graded_answers: list[dict]) -> tuple[Level, float, float, dict]:
    """`graded_answers`: [{question_id, kind, answer, answer_key}]."""
    # TODO(ai): score each answer with the error-detection model and map the
    # aggregate to a level.
    if not settings.AI_MOCK:
        raise AINotImplemented("score_placement")

    per_q = []
    total = 0.0
    for a in graded_answers:
        ans = (a.get("answer") or "").strip()
        # crude mock: longer / non-empty answers score higher
        s = 100.0 if len(ans.split()) >= 4 else (60.0 if ans else 0.0)
        total += s
        per_q.append({"question_id": a.get("question_id"), "score": s})

    max_score = 100.0 * max(len(graded_answers), 1)
    pct = 100.0 * total / max_score if max_score else 0.0
    level = (
        Level.advanced if pct >= 80 else
        Level.intermediate if pct >= 50 else
        Level.beginner
    )
    return level, round(total, 1), max_score, {"per_question": per_q, "percent": round(pct, 1)}


# ===========================================================================
# 4) Custom exercise generation from a student's mistakes
# ===========================================================================
def generate_exercise(
    focus_skill: Optional[str],
    writing_type: WritingType,
    recent_errors: Optional[list[dict]] = None,
) -> dict:
    # TODO(ai): prompt an LLM (Claude) with the student's error profile to
    # generate a targeted exercise.
    if not settings.AI_MOCK:
        raise AINotImplemented("generate_exercise")

    skill = focus_skill or "همزة القطع"
    return {
        "title": "تمرين مخصص لك",
        "intro": (
            f"لاحظنا أنك تحتاج إلى مزيد من التدريب على {skill}. "
            "هذا التمرين مصمم خصيصاً لمساعدتك على تحسين هذه المهارة."
        ),
        "writing_type": writing_type.value,
        "content": {
            "prompt": f"أكمل الجمل التالية باستخدام {skill} الصحيحة:",
            "items": [
                "ذهب ___حمد ___لى المدرسة صباحاً.",
                "___ن الكتاب مفيد للطلاب.",
                "سمعت ___نّ الامتحان سيكون غداً.",
            ],
        },
    }


# ===========================================================================
# 5) Text-to-speech for dictation sentences
# ===========================================================================
def synthesize_tts(text: str) -> tuple[str, Optional[str]]:
    """Return (audio_url, audio_base64). One of them will be usable."""
    # TODO(ai): call a TTS engine (e.g. an Arabic TTS model / cloud TTS) and
    # store the audio, returning a real URL.
    if not settings.AI_MOCK:
        raise AINotImplemented("synthesize_tts")
    # Mock: a stable placeholder URL the frontend can wire the audio player to.
    from urllib.parse import quote
    return f"https://tts.placeholder.local/say?text={quote(text)}", None
