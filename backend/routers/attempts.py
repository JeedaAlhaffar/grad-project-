# -*- coding: utf-8 -*-
"""
Attempts: a student submits work for an exercise and gets it evaluated.

One submit endpoint serves all four writing types. The scoring itself is
delegated to `ai_service` (currently mocked); when AI_MOCK is off and the real
models are not wired yet, the attempt is stored as `evaluation_status="pending"`
and returned with HTTP 202 so the flow still works end-to-end.

A كلمات exercise can be answered two ways, so it has two evaluators: the
student may TYPE the word (graded as text, like إملاء) or WRITE it by hand on
the canvas (graded by the ADAB recogniser, see word_grader.py). Which one ran
is decided by whichever field the client sent.

Screens: the copying/dictation/composition/words exercise screens + the
evaluation result screen.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

import ai_service
from deps import get_current_student, get_db
from models import (
    Attempt, CopyingItem, Exercise, Student, WritingType,
)
from schemas import AttemptOut, AttemptSubmitIn
from services import is_passed, record_attempt_result, stars_from_score

router = APIRouter(tags=["attempts"])

MAX_SCORE = 100.0


def _words_target(exercise: Exercise) -> str:
    """The single word/sentence a كلمات exercise asks for.

    Both seeders (seed_intermediate_words.py, seed_khatt_sentences.py) store it
    as content={"words": [target]}; the title carries the same string, so it is
    a safe fallback for hand-made content.
    """
    words = (exercise.content or {}).get("words") or []
    return str(words[0]) if words else (exercise.title or "")


def _evaluate(exercise: Exercise, body: AttemptSubmitIn, db: Session):
    """Run the (AI) evaluator for this exercise/submission.

    Returns (score, feedback_dict) or (None, None) if AI is not available yet.
    """
    wt = exercise.writing_type
    try:
        if wt == WritingType.copying:
            item = db.get(CopyingItem, body.copying_item_id) if body.copying_item_id else None
            if item is None or item.exercise_id != exercise.id:
                raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                    "copying_item_id does not belong to this exercise.")
            return ai_service.evaluate_copying(
                item.letter, body.submitted_drawing or [],
                body.copying_mode.value if body.copying_mode else "free_draw",
            )
        elif wt == WritingType.words and body.submitted_drawing:
            # Hand-written word: the ADAB recogniser reads it, then the same
            # dictation grader compares the reading to the target.
            return ai_service.evaluate_words_drawing(
                _words_target(exercise), body.submitted_drawing)
        else:
            content = exercise.content or {}
            if wt == WritingType.composition:
                # the essay prompt — what the relevance scorer compares against
                reference = (exercise.instructions
                             or content.get("prompt")
                             or exercise.title)
            elif wt == WritingType.words:
                # A كلمات exercise asks the student to REPRODUCE the word shown
                # ("اكتب الكلمة الظاهرة أمامك"), so the only thing worth grading
                # is whether they wrote that word. Hand it the target, exactly
                # as the hand-written branch above does — otherwise the typed
                # answer is only spell-checked and any correctly-spelled text
                # scores full marks.
                reference = _words_target(exercise)
            else:
                # Dictation target. `text` is the fully diacritized script,
                # which the student cannot type on an ordinary keyboard, so
                # prefer `text_plain` — seeded precisely for grading. The
                # grader dediacritizes defensively too; this just hands it the
                # right field to begin with.
                reference = content.get("text_plain") or content.get("text")
            return ai_service.evaluate_text(wt, body.submitted_text or "", reference)
    except ai_service.AINotImplemented:
        return None, None


@router.post(
    "/exercises/{exercise_id}/attempts",
    response_model=AttemptOut, status_code=status.HTTP_201_CREATED,
)
def submit_attempt(
    exercise_id: int, body: AttemptSubmitIn, response: Response,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    ex = db.get(Exercise, exercise_id)
    if ex is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Exercise not found.")

    # --- validate the body against the exercise type ---
    if ex.writing_type == WritingType.copying:
        if body.copying_item_id is None or body.submitted_drawing is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "copying exercises need copying_item_id, copying_mode and submitted_drawing.")
    elif ex.writing_type == WritingType.words:
        # either answer form is valid — typed or hand-written
        if not body.submitted_drawing and not (
                body.submitted_text and body.submitted_text.strip()):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "words exercises need submitted_text (typed) "
                "or submitted_drawing (hand-written).")
    else:
        if not (body.submitted_text and body.submitted_text.strip()):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"{ex.writing_type.value} exercises need submitted_text.")

    score, feedback = _evaluate(ex, body, db)

    attempt = Attempt(
        student_id=student.id,
        exercise_id=ex.id,
        copying_item_id=body.copying_item_id,
        copying_mode=body.copying_mode,
        submitted_drawing=body.submitted_drawing,
        submitted_text=body.submitted_text,
        score=score,
        max_score=MAX_SCORE if score is not None else None,
        is_passed=is_passed(score) if score is not None else None,
        feedback=feedback,
    )
    db.add(attempt)

    if score is not None:
        record_attempt_result(db, student, ex.lesson_id, score)

    db.commit()
    db.refresh(attempt)

    out = AttemptOut.model_validate(attempt)
    out.stars = stars_from_score(score)
    if score is None:
        out.evaluation_status = "pending"
        response.status_code = status.HTTP_202_ACCEPTED
    return out


@router.get("/exercises/{exercise_id}/attempts", response_model=list[AttemptOut])
def list_attempts(
    exercise_id: int,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    rows = db.scalars(
        select(Attempt)
        .where(Attempt.exercise_id == exercise_id, Attempt.student_id == student.id)
        .order_by(Attempt.created_at.desc())
    ).all()
    return [_attempt_out(a) for a in rows]


@router.get("/attempts/{attempt_id}", response_model=AttemptOut)
def get_attempt(
    attempt_id: int,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    attempt = db.get(Attempt, attempt_id)
    if attempt is None or attempt.student_id != student.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Attempt not found.")
    return _attempt_out(attempt)


def _attempt_out(a: Attempt) -> AttemptOut:
    out = AttemptOut.model_validate(a)
    out.stars = stars_from_score(a.score)
    out.evaluation_status = "done" if a.score is not None else "pending"
    return out
