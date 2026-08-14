# -*- coding: utf-8 -*-
"""
Domain logic shared by the routers: level metadata & derivation, star/score
rules, streak updates, and lesson-completion recording.

Kept free of FastAPI so it is easy to unit-test.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from models import (
    Attempt, Lesson, LessonProgress, Level, ProgressStatus, Student, Unit,
    WritingType,
)

# ---------------------------------------------------------------------------
# Level metadata (Arabic copy taken from the Figma)
# ---------------------------------------------------------------------------
# Each level (track on the Levels screen) corresponds to one writing type.
LEVEL_META: dict[Level, dict] = {
    Level.beginner: {
        "title": "المستوى المبتدئ",
        "description": "تعلم نسخ الحروف وأساسيات الكتابة",
        "order": 0,
    },
    Level.intermediate: {
        "title": "المستوى المتوسط",
        "description": "تدرب على كتابة الكلمات",
        "order": 1,
    },
    Level.advanced: {
        "title": "المستوى المتقدم",
        "description": "أتقن الإملاء",
        "order": 2,
    },
    Level.professional: {
        "title": "المستوى الاحترافي",
        "description": "أتقن التعبير والكتابة الإبداعية",
        "order": 3,
    },
}

LEVEL_ORDER = [
    Level.beginner, Level.intermediate, Level.advanced, Level.professional,
]

# Canonical writing-type -> level mapping (the four Levels-screen tracks).
LEVEL_FOR_WRITING_TYPE: dict[WritingType, Level] = {
    WritingType.copying: Level.beginner,        # نسخ الحروف
    WritingType.words: Level.intermediate,      # الكلمات
    WritingType.dictation: Level.advanced,      # الإملاء
    WritingType.composition: Level.professional,  # التعبير
}


def level_for_writing_type(wt: Optional[WritingType]) -> Optional[Level]:
    """The level a given writing type belongs to (None if wt is None)."""
    return LEVEL_FOR_WRITING_TYPE.get(wt) if wt is not None else None


def level_from_grade(grade: Optional[int]) -> Level:
    """Last-resort fallback for units with no level column and no exercises."""
    return {2: Level.beginner, 3: Level.intermediate, 4: Level.advanced}.get(
        grade or 2, Level.beginner)


def _unit_dominant_writing_type(unit: Unit) -> Optional[WritingType]:
    """First writing type found across the unit's lessons, if any are loaded."""
    for lesson in unit.lessons:
        wt = dominant_writing_type(lesson)
        if wt is not None:
            return wt
    return None


def effective_level(unit: Unit) -> Level:
    """A unit's level.

    Precedence: explicit `level` column -> the level of the unit's dominant
    writing type -> book-grade fallback (for units with no exercises loaded).
    """
    if unit.level is not None:
        return unit.level
    by_type = level_for_writing_type(_unit_dominant_writing_type(unit))
    if by_type is not None:
        return by_type
    grade = unit.book.grade if unit.book is not None else None
    return level_from_grade(grade)


# ---------------------------------------------------------------------------
# Scoring / stars
# ---------------------------------------------------------------------------
def stars_from_score(score: Optional[float]) -> int:
    if score is None:
        return 0
    if score >= settings.STAR_3_MIN:
        return 3
    if score >= settings.STAR_2_MIN:
        return 2
    if score >= settings.STAR_1_MIN:
        return 1
    return 0


def is_passed(score: Optional[float]) -> bool:
    return score is not None and score >= settings.PASS_SCORE


def points_from_score(score: Optional[float]) -> int:
    """Simple XP rule: 1 point per score-point on a passed attempt."""
    return int(score) if is_passed(score) else 0


# ---------------------------------------------------------------------------
# Lesson helpers
# ---------------------------------------------------------------------------
def dominant_writing_type(lesson: Lesson) -> Optional[WritingType]:
    if not lesson.exercises:
        return None
    return lesson.exercises[0].writing_type


# ---------------------------------------------------------------------------
# Streak
# ---------------------------------------------------------------------------
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def update_streak(student: Student, now: Optional[datetime] = None) -> None:
    """Advance the daily learning streak based on the last active day."""
    now = now or _utcnow()
    today = now.date()
    last = student.last_active_on.date() if student.last_active_on else None

    if last == today:
        pass                                   # already counted today
    elif last == today - timedelta(days=1):
        student.streak_count += 1              # consecutive day
    else:
        student.streak_count = 1               # first day or streak broken

    student.longest_streak = max(student.longest_streak, student.streak_count)
    student.last_active_on = now


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------
def get_or_create_progress(
    db: Session, student_id: int, lesson_id: int
) -> LessonProgress:
    lp = db.scalar(
        select(LessonProgress).where(
            LessonProgress.student_id == student_id,
            LessonProgress.lesson_id == lesson_id,
        )
    )
    if lp is None:
        lp = LessonProgress(
            student_id=student_id, lesson_id=lesson_id,
            status=ProgressStatus.not_started,
        )
        db.add(lp)
        db.flush()
    return lp


def record_attempt_result(
    db: Session, student: Student, lesson_id: int, score: Optional[float],
    now: Optional[datetime] = None,
) -> LessonProgress:
    """Update lesson progress + streak + points after an evaluated attempt.

    A lesson counts as completed once any attempt in it passes; the best
    score/stars are kept.
    """
    now = now or _utcnow()
    lp = get_or_create_progress(db, student.id, lesson_id)

    if lp.started_at is None:
        lp.started_at = now
    if lp.status == ProgressStatus.not_started:
        lp.status = ProgressStatus.in_progress

    if score is not None and (lp.score is None or score > lp.score):
        lp.score = score
        lp.stars = stars_from_score(score)

    if is_passed(score) and lp.status != ProgressStatus.completed:
        lp.status = ProgressStatus.completed
        lp.completed_at = now
        student.total_points += points_from_score(score)

    update_streak(student, now)
    return lp
