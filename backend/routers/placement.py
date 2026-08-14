# -*- coding: utf-8 -*-
"""
Placement test (اختبار تحديد المستوى).

Flow: fetch the 5 questions -> submit answers -> the (AI) scorer returns a
level, which is saved as the student's current_level. Screens: the 5 question
screens + the "your level is: ..." result screen.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

import ai_service
from deps import get_current_student, get_db
from models import PlacementQuestion, PlacementResult, Student
from schemas import (
    PlacementQuestionOut, PlacementResultOut, PlacementSubmitIn,
)

router = APIRouter(prefix="/placement", tags=["placement"])


@router.get("/questions", response_model=list[PlacementQuestionOut])
def placement_questions(
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    return db.scalars(
        select(PlacementQuestion)
        .where(PlacementQuestion.is_active.is_(True))
        .order_by(PlacementQuestion.sort_order, PlacementQuestion.id)
    ).all()


@router.post("/submit", response_model=PlacementResultOut)
def submit_placement(
    body: PlacementSubmitIn,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    q_by_id = {
        q.id: q for q in db.scalars(select(PlacementQuestion)).all()
    }
    graded_answers = []
    for a in body.answers:
        q = q_by_id.get(a.question_id)
        if q is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"Unknown question_id {a.question_id}.")
        graded_answers.append({
            "question_id": q.id, "kind": q.kind.value,
            "answer": a.answer, "answer_key": q.answer_key,
        })

    try:
        level, score, max_score, details = ai_service.score_placement(graded_answers)
    except ai_service.AINotImplemented:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "Placement scoring model is not available yet (set AI_MOCK=1 for mock scoring).")

    result = PlacementResult(
        student_id=student.id, level=level, score=score, max_score=max_score,
        answers=[a.model_dump() for a in body.answers], details=details,
    )
    db.add(result)

    student.current_level = level
    student.placement_done = True

    db.commit()
    db.refresh(result)
    return result


@router.get("/result", response_model=PlacementResultOut)
def latest_result(
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    result = db.scalar(
        select(PlacementResult)
        .where(PlacementResult.student_id == student.id)
        .order_by(PlacementResult.created_at.desc())
    )
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No placement result yet.")
    return result
