# -*- coding: utf-8 -*-
"""
=============================================================================
 AI ENDPOINTS  --  THESE NEED THE MACHINE-LEARNING LOGIC (IMPLEMENT LAST)
=============================================================================

These wrap `ai_service`, which is currently MOCKED (settings.AI_MOCK=1). They
are already usable by the Flutter team with realistic responses. The remaining
work is to replace the model bodies in `ai_service.py`:

  POST /ai/evaluate/text      full evaluation for words / dictation / composition
  POST /ai/detect/errors      spelling + grammar detection only (GEC + ARETA)
  POST /ai/relevance          topic relevance of an essay vs. its prompt (0..4)
  POST /ai/evaluate/copying   grade a hand-drawn letter vs. the reference strokes
  POST /ai/exercises/generate build a targeted exercise from the student's errors
  POST /ai/tts                text-to-speech for dictation sentences

(Placement scoring runs through POST /api/placement/submit, which also calls
 ai_service.score_placement.)

The rest of the backend — auth, content, progress, attempts flow — is complete
and does not depend on this router being finished.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

import ai_service
from deps import get_current_student, get_db
from models import Letter, Student
from schemas import (
    DetectErrorsIn, DetectErrorsOut, EvaluateCopyingIn, EvaluateTextIn,
    EvaluationFeedback, EvaluationOut, GeneratedExerciseOut, GenerateExerciseIn,
    RelevanceIn, RelevanceOut, TTSIn, TTSOut,
)
from services import is_passed, stars_from_score
from sqlalchemy.orm import Session

router = APIRouter(prefix="/ai", tags=["ai (needs ML logic)"])


def _wrap(score, feedback) -> EvaluationOut:
    return EvaluationOut(
        score=score,
        is_passed=is_passed(score),
        stars=stars_from_score(score),
        feedback=EvaluationFeedback(**feedback),
    )


@router.post("/evaluate/text", response_model=EvaluationOut)
def evaluate_text(body: EvaluateTextIn, _: Student = Depends(get_current_student)):
    try:
        score, feedback = ai_service.evaluate_text(
            body.writing_type, body.submitted_text, body.reference)
    except ai_service.AINotImplemented:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
                            "Text evaluation model not implemented yet.")
    return _wrap(score, feedback)


@router.post("/evaluate/copying", response_model=EvaluationOut)
def evaluate_copying(
    body: EvaluateCopyingIn,
    _: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    letter = db.get(Letter, body.letter_id)
    if letter is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Letter not found.")
    try:
        score, feedback = ai_service.evaluate_copying(
            letter, body.submitted_drawing, body.mode.value)
    except ai_service.AINotImplemented:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
                            "Copying evaluation model not implemented yet.")
    return _wrap(score, feedback)


@router.post("/detect/errors", response_model=DetectErrorsOut)
def detect_errors(body: DetectErrorsIn, _: Student = Depends(get_current_student)):
    """Spelling/grammar detection on its own (GEC correction + ARETA tags).

    This is the same detection stage that runs before the AES scoring; exposed
    separately so the app can show corrections without grading the text.
    """
    from config import settings

    if not settings.GEC_ENABLED:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
                            "Error detection is disabled (GEC_ENABLED=0).")
    import gec_service
    try:
        result = gec_service.detect(body.text, timeout=settings.GEC_TIMEOUT)
    except gec_service.GECUnavailable as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
                            f"Error-detection engine unavailable: {exc}")
    except TimeoutError:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT,
                            "Error detection took too long.")
    return DetectErrorsOut(text=body.text, **result)


@router.post("/relevance", response_model=RelevanceOut)
def score_relevance(body: RelevanceIn, _: Student = Depends(get_current_student)):
    """Topic relevance of an essay against its prompt (hybrid scorer, 0..4).

    Returns status="rejected" with an Arabic message for essays under the
    word guardrail — show that message instead of a grade.
    """
    from config import settings

    if not settings.RELEVANCE_ENABLED:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
                            "Relevance scoring is disabled (RELEVANCE_ENABLED=0).")
    import relevance_scorer
    try:
        result = relevance_scorer.score_relevance(body.prompt, body.text)
    except Exception as exc:      # model folder / sentence-transformers missing
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
                            f"Relevance model unavailable: {exc}")
    return RelevanceOut(**result)


@router.post("/exercises/generate", response_model=GeneratedExerciseOut)
def generate_exercise(
    body: GenerateExerciseIn, _: Student = Depends(get_current_student),
):
    from models import WritingType
    wt = body.writing_type or WritingType.composition
    try:
        data = ai_service.generate_exercise(body.focus_skill, wt)
    except ai_service.AINotImplemented:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
                            "Exercise generation not implemented yet.")
    return GeneratedExerciseOut(**data)


@router.post("/tts", response_model=TTSOut)
def text_to_speech(body: TTSIn, _: Student = Depends(get_current_student)):
    try:
        url, b64 = ai_service.synthesize_tts(body.text)
    except ai_service.AINotImplemented:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
                            "TTS not implemented yet.")
    return TTSOut(text=body.text, audio_url=url, audio_base64=b64)
