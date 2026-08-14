# -*- coding: utf-8 -*-
"""
Dashboard + per-lesson progress.

Screens: the Home/Dashboard (welcome, streak, progress %, totals, levels,
"continue where you left off") and the progress overlays on the map.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from deps import get_current_student, get_db
from models import Lesson, LessonProgress, ProgressStatus, Student
from routers.content import (
    _all_units, _build_unit, _level_summary, _level_unlocks, _progress_map,
    _units_by_level,
)
from schemas import (
    ContinueOut, DashboardOut, LessonProgressOut, StudentOut,
)
from services import LEVEL_ORDER, get_or_create_progress

router = APIRouter(tags=["progress"])


@router.get("/dashboard", response_model=DashboardOut)
def dashboard(
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    units = _all_units(db)
    grouped = _units_by_level(units)
    pmap = _progress_map(db, student.id)
    unlocks = _level_unlocks(grouped, pmap)

    levels = [_level_summary(lv, grouped[lv], pmap, unlocks[lv]) for lv in LEVEL_ORDER]
    total_lessons = sum(lv.lessons_count for lv in levels)
    completed_lessons = sum(lv.completed_lessons for lv in levels)
    progress_percent = (
        round(100 * completed_lessons / total_lessons) if total_lessons else 0
    )

    # "continue where you left off": first current lesson in level order
    continue_lesson = None
    for lv in LEVEL_ORDER:
        for unit in grouped[lv]:
            uo = _build_unit(unit, pmap, unlocks[lv])
            current = next((n for n in uo.lessons if n.is_current), None)
            if current is not None:
                continue_lesson = ContinueOut(
                    lesson_id=current.id, lesson_title=current.title,
                    unit_title=unit.title, level=lv,
                )
                break
        if continue_lesson is not None:
            break

    return DashboardOut(
        student=StudentOut.model_validate(student),
        streak_count=student.streak_count,
        progress_percent=progress_percent,
        total_lessons=total_lessons,
        completed_lessons=completed_lessons,
        levels=levels,
        continue_lesson=continue_lesson,
    )


@router.get("/progress", response_model=list[LessonProgressOut])
def my_progress(
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    return db.scalars(
        select(LessonProgress).where(LessonProgress.student_id == student.id)
    ).all()


@router.get("/progress/lessons/{lesson_id}", response_model=LessonProgressOut)
def lesson_progress(
    lesson_id: int,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    lp = db.scalar(
        select(LessonProgress).where(
            LessonProgress.student_id == student.id,
            LessonProgress.lesson_id == lesson_id,
        )
    )
    if lp is None:
        # report a not-started shell rather than 404 so the UI is simpler
        return LessonProgressOut(
            lesson_id=lesson_id, status=ProgressStatus.not_started,
            stars=0, score=None, started_at=None, completed_at=None,
        )
    return lp


@router.post("/progress/lessons/{lesson_id}/start", response_model=LessonProgressOut)
def start_lesson(
    lesson_id: int,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    if db.get(Lesson, lesson_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Lesson not found.")
    lp = get_or_create_progress(db, student.id, lesson_id)
    if lp.status == ProgressStatus.not_started:
        lp.status = ProgressStatus.in_progress
        lp.started_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(lp)
    return lp
