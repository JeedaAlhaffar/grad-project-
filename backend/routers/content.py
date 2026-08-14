# -*- coding: utf-8 -*-
"""
Curriculum content browsing: levels -> units -> lessons -> exercises, plus the
letter shapes used by the copying screens.

Every lesson/unit/level is returned with the current student's progress
overlaid (status, stars, locked, is_current) so the Flutter map screens can
render directly from one response.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from deps import get_current_student, get_db
from models import (
    Exercise, Lesson, LessonProgress, Letter, LetterForm, Level,
    ProgressStatus, Student, Unit,
)
from schemas import (
    ExerciseOut, LessonDetailOut, LessonNode, LetterOut, LetterSummary,
    LevelDetailOut, LevelOut, UnitOut,
)
from services import (
    LEVEL_META, LEVEL_ORDER, dominant_writing_type, effective_level,
)

router = APIRouter(tags=["content"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _progress_map(db: Session, student_id: int) -> dict[int, LessonProgress]:
    rows = db.scalars(
        select(LessonProgress).where(LessonProgress.student_id == student_id)
    ).all()
    return {lp.lesson_id: lp for lp in rows}


def _all_units(db: Session) -> list[Unit]:
    return db.scalars(
        select(Unit)
        .where(Unit.is_published.is_(True))
        .options(
            selectinload(Unit.book),
            selectinload(Unit.lessons).selectinload(Lesson.exercises),
        )
        .order_by(Unit.sort_order, Unit.id)
    ).all()


def _build_lesson_node(
    lesson: Lesson, pmap: dict[int, LessonProgress], prev_completed: bool,
    unit_unlocked: bool,
) -> LessonNode:
    lp = pmap.get(lesson.id)
    status_ = lp.status if lp else ProgressStatus.not_started
    stars = lp.stars if lp else 0
    score = lp.score if lp else None
    completed = status_ == ProgressStatus.completed
    locked = not unit_unlocked or (not prev_completed and not completed)
    return LessonNode(
        id=lesson.id,
        title=lesson.title,
        description=lesson.description,
        sort_order=lesson.sort_order,
        writing_type=dominant_writing_type(lesson),
        status=status_,
        stars=stars,
        score=score,
        locked=locked,
        is_current=False,           # filled in by the caller
    )


def _build_unit(
    unit: Unit, pmap: dict[int, LessonProgress], unit_unlocked: bool = True,
) -> UnitOut:
    nodes: list[LessonNode] = []
    prev_completed = True           # first lesson is always reachable
    for lesson in unit.lessons:
        node = _build_lesson_node(lesson, pmap, prev_completed, unit_unlocked)
        nodes.append(node)
        prev_completed = node.status == ProgressStatus.completed

    # mark the first unlocked, not-completed lesson as the current one
    for node in nodes:
        if not node.locked and node.status != ProgressStatus.completed:
            node.is_current = True
            break

    completed = sum(1 for n in nodes if n.status == ProgressStatus.completed)
    total = len(nodes)
    return UnitOut(
        id=unit.id,
        book_id=unit.book_id,
        unit_number=unit.unit_number,
        level=effective_level(unit),
        title=unit.title,
        description=unit.description,
        sort_order=unit.sort_order,
        lessons_count=total,
        completed_lessons=completed,
        progress_percent=round(100 * completed / total) if total else 0,
        lessons=nodes,
    )


def _units_by_level(units: list[Unit]) -> dict[Level, list[Unit]]:
    grouped: dict[Level, list[Unit]] = {lv: [] for lv in LEVEL_ORDER}
    for u in units:
        grouped[effective_level(u)].append(u)
    return grouped


def _level_summary(
    level: Level, units: list[Unit], pmap: dict[int, LessonProgress],
    unlocked: bool,
) -> LevelOut:
    lessons_count = sum(len(u.lessons) for u in units)
    completed = sum(
        1 for u in units for l in u.lessons
        if (lp := pmap.get(l.id)) and lp.status == ProgressStatus.completed
    )
    meta = LEVEL_META[level]
    return LevelOut(
        level=level,
        title=meta["title"],
        description=meta["description"],
        units_count=len(units),
        lessons_count=lessons_count,
        completed_lessons=completed,
        progress_percent=round(100 * completed / lessons_count) if lessons_count else 0,
        is_unlocked=unlocked,
    )


def _level_unlocks(
    grouped: dict[Level, list[Unit]], pmap: dict[int, LessonProgress],
) -> dict[Level, bool]:
    """Beginner is always open; a level opens once the previous one has any
    completed lesson."""
    unlocks: dict[Level, bool] = {}
    prev_has_progress = True
    for level in LEVEL_ORDER:
        unlocks[level] = prev_has_progress
        prev_has_progress = any(
            (lp := pmap.get(l.id)) and lp.status == ProgressStatus.completed
            for u in grouped[level] for l in u.lessons
        )
    return unlocks


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------
@router.get("/levels", response_model=list[LevelOut])
def list_levels(
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    units = _all_units(db)
    grouped = _units_by_level(units)
    pmap = _progress_map(db, student.id)
    unlocks = _level_unlocks(grouped, pmap)
    return [
        _level_summary(lv, grouped[lv], pmap, unlocks[lv]) for lv in LEVEL_ORDER
    ]


@router.get("/levels/{level}", response_model=LevelDetailOut)
def get_level(
    level: Level,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    units = _all_units(db)
    grouped = _units_by_level(units)
    pmap = _progress_map(db, student.id)
    unlocks = _level_unlocks(grouped, pmap)
    summary = _level_summary(level, grouped[level], pmap, unlocks[level])
    return LevelDetailOut(
        **summary.model_dump(),
        units=[_build_unit(u, pmap, unlocks[level]) for u in grouped[level]],
    )


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
@router.get("/units/{unit_id}", response_model=UnitOut)
def get_unit(
    unit_id: int,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    unit = db.scalar(
        select(Unit)
        .where(Unit.id == unit_id)
        .options(
            selectinload(Unit.book),
            selectinload(Unit.lessons).selectinload(Lesson.exercises),
        )
    )
    if unit is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unit not found.")
    pmap = _progress_map(db, student.id)
    return _build_unit(unit, pmap, unit_unlocked=True)


# ---------------------------------------------------------------------------
# Lessons
# ---------------------------------------------------------------------------
@router.get("/lessons/{lesson_id}", response_model=LessonDetailOut)
def get_lesson(
    lesson_id: int,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    lesson = db.scalar(
        select(Lesson)
        .where(Lesson.id == lesson_id)
        .options(
            selectinload(Lesson.exercises).selectinload(Exercise.copying_items),
        )
    )
    if lesson is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Lesson not found.")
    lp = db.scalar(
        select(LessonProgress).where(
            LessonProgress.student_id == student.id,
            LessonProgress.lesson_id == lesson_id,
        )
    )
    return LessonDetailOut(
        id=lesson.id,
        unit_id=lesson.unit_id,
        title=lesson.title,
        description=lesson.description,
        sort_order=lesson.sort_order,
        status=lp.status if lp else ProgressStatus.not_started,
        stars=lp.stars if lp else 0,
        exercises=[ExerciseOut.model_validate(ex) for ex in lesson.exercises],
    )


# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------
@router.get("/exercises/{exercise_id}", response_model=ExerciseOut)
def get_exercise(
    exercise_id: int,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    ex = db.scalar(
        select(Exercise)
        .where(Exercise.id == exercise_id)
        .options(selectinload(Exercise.copying_items))
    )
    if ex is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Exercise not found.")
    return ex


# ---------------------------------------------------------------------------
# Letters (copying screens)
# ---------------------------------------------------------------------------
@router.get("/letters", response_model=list[LetterSummary])
def list_letters(
    form: LetterForm | None = Query(default=None),
    arabic_letter: str | None = Query(default=None),
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    stmt = select(Letter).order_by(Letter.code)
    if form is not None:
        stmt = stmt.where(Letter.form == form)
    if arabic_letter:
        stmt = stmt.where(Letter.arabic_letter == arabic_letter)
    return db.scalars(stmt).all()


@router.get("/letters/{letter_id}", response_model=LetterOut)
def get_letter(
    letter_id: int,
    student: Student = Depends(get_current_student), db: Session = Depends(get_db),
):
    letter = db.get(Letter, letter_id)
    if letter is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Letter not found.")
    return letter
