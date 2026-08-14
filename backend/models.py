# -*- coding: utf-8 -*-
"""
Database schema for the Arabic writing-teaching app.

SQLAlchemy 2.0 ORM models. Works on PostgreSQL and MySQL.

Content hierarchy (as described by the system owner)
----------------------------------------------------
    Unit  (has a title)
     |__ Lesson  (has a title)
          |__ Exercise  -- a "thing to write", one of 4 writing types:
                  copying (نسخ) | words (كلمات) |
                  composition (تعبير) | dictation (إملاء)

Copying type
------------
A copying Exercise contains CopyingItems; each item is one Letter.
Every letter is practised in 3 modes: bold line, dotted line, free draw.
The Letter table stores the normalized stroke data produced by
`letters drawing/normalize_letters.py` (points in 0.0-1.0, y-down).

Student side
------------
Student -> Attempt (one row per submission, any writing type)
Student -> LessonProgress (completion tracking)
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
)

# JSONB on PostgreSQL (indexable, faster), plain JSON on MySQL.
JSONType = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    pass


def _enum(py_enum):
    """Store the enum *value* (lowercase string) in the DB, not the name."""
    return SAEnum(py_enum, values_callable=lambda e: [m.value for m in e])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class WritingType(enum.Enum):
    copying = "copying"            # نسخ
    words = "words"                # كلمات
    composition = "composition"    # تعبير
    dictation = "dictation"        # إملاء


class Level(enum.Enum):
    """The four learning tracks shown on the Levels screen.

    Each track corresponds to one writing type (see services.LEVEL_FOR_WRITING_TYPE):
        beginner -> copying (نسخ)   intermediate -> words (كلمات)
        advanced -> dictation (إملاء)   professional -> composition (تعبير)
    """
    beginner = "beginner"          # المستوى المبتدئ    -- نسخ الحروف
    intermediate = "intermediate"  # المستوى المتوسط    -- الكلمات
    advanced = "advanced"          # المستوى المتقدم    -- الإملاء
    professional = "professional"  # المستوى الاحترافي  -- التعبير


class LetterForm(enum.Enum):
    isolated = "isolated"          # منفصلة
    initial = "initial"            # بداية
    medial = "medial"              # وسط
    final = "final"                # نهاية


class CopyingMode(enum.Enum):
    bold_line = "bold_line"        # following a bold line
    dotted_line = "dotted_line"    # following dotted lines
    free_draw = "free_draw"        # drawing the letter freehand


class ProgressStatus(enum.Enum):
    not_started = "not_started"
    in_progress = "in_progress"
    completed = "completed"


class PlacementKind(enum.Enum):
    """The kinds of prompts used in the placement test (اختبار تحديد المستوى)."""
    free_write = "free_write"      # اكتب جملة تعبر عن ...
    complete = "complete"          # أكمل الجملة التالية
    correct = "correct"            # صحح الأخطاء في الجملة التالية


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------
class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Content tables
# ---------------------------------------------------------------------------
class Book(Base, TimestampMixin):
    """A source textbook the curriculum content is seeded from.

    The reference series is "العربية بين يديك"; each book is one
    grade (2/3/4) and one part (1/2). A book contains units, and -- at the
    end of the book -- a master dictation vocabulary list (DictationWord
    rows with no unit).
    """
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(40), unique=True)   # e.g. "g3p1"
    title: Mapped[str] = mapped_column(String(300))
    series: Mapped[str | None] = mapped_column(String(120))
    grade: Mapped[int] = mapped_column(Integer)                  # 2, 3, 4
    part: Mapped[int] = mapped_column(Integer)                   # 1, 2
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    source_file: Mapped[str | None] = mapped_column(String(255))

    units: Mapped[list[Unit]] = relationship(
        back_populates="book", cascade="all, delete-orphan",
        order_by="Unit.sort_order")
    dictation_words: Mapped[list[DictationWord]] = relationship(
        back_populates="book", cascade="all, delete-orphan",
        order_by="DictationWord.sort_order")

    def __repr__(self) -> str:
        return f"<Book {self.id} {self.code!r}>"


class Unit(Base, TimestampMixin):
    """A unit of the curriculum. Has a title and a set of lessons."""
    __tablename__ = "units"

    id: Mapped[int] = mapped_column(primary_key=True)
    book_id: Mapped[int | None] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), index=True)
    unit_number: Mapped[int | None] = mapped_column(Integer)   # ordinal within book
    # Learning track shown on the Levels screen. When NULL the API derives it
    # from the parent book's grade (2->beginner, 3->intermediate, 4->advanced).
    level: Mapped[Level | None] = mapped_column(_enum(Level), index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    is_published: Mapped[bool] = mapped_column(Boolean, default=True)

    book: Mapped[Book | None] = relationship(back_populates="units")
    lessons: Mapped[list[Lesson]] = relationship(
        back_populates="unit", cascade="all, delete-orphan",
        order_by="Lesson.sort_order")
    dictation_words: Mapped[list[DictationWord]] = relationship(
        back_populates="unit", cascade="all, delete-orphan",
        order_by="DictationWord.sort_order")

    def __repr__(self) -> str:
        return f"<Unit {self.id} {self.title!r}>"


class Lesson(Base, TimestampMixin):
    """A lesson inside a unit. Has a title and a set of exercises."""
    __tablename__ = "lessons"

    id: Mapped[int] = mapped_column(primary_key=True)
    unit_id: Mapped[int] = mapped_column(
        ForeignKey("units.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    is_published: Mapped[bool] = mapped_column(Boolean, default=True)

    unit: Mapped[Unit] = relationship(back_populates="lessons")
    exercises: Mapped[list[Exercise]] = relationship(
        back_populates="lesson", cascade="all, delete-orphan",
        order_by="Exercise.sort_order")

    def __repr__(self) -> str:
        return f"<Lesson {self.id} {self.title!r}>"


class Letter(Base, TimestampMixin):
    """A single Arabic letter shape with its normalized stroke data.

    Strokes come from `letters drawing/letters_normalized/`: a list of
    strokes, each a list of [x, y] points with 0.0 <= x,y <= 1.0 (y-down).
    The same stroke data drives all 3 copying modes on the frontend.
    """
    __tablename__ = "letters"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(120), unique=True)   # e.g. "Ain_End"
    arabic_letter: Mapped[str] = mapped_column(String(8))         # e.g. "ع"
    form: Mapped[LetterForm] = mapped_column(_enum(LetterForm))

    strokes: Mapped[list] = mapped_column(JSONType)              # [[[x,y],...], ...]
    stroke_count: Mapped[int] = mapped_column(Integer, default=0)
    point_count: Mapped[int] = mapped_column(Integer, default=0)
    aspect_ratio: Mapped[float | None] = mapped_column(Float)

    source_format: Mapped[str | None] = mapped_column(String(4))   # A / B / C
    source_file: Mapped[str | None] = mapped_column(String(255))

    def __repr__(self) -> str:
        return f"<Letter {self.id} {self.code!r}>"


class Exercise(Base, TimestampMixin):
    """A "thing to write" inside a lesson, of one of the 4 writing types.

    For copying  -> letters live in `copying_items`; `content` may be null.
    For words    -> content = {"words": ["بيت", "ماء", ...]}
    For composition -> content = {"prompt": "...", "min_words": 20}
    For dictation   -> content = {"text": "النص المملى"}
    """
    __tablename__ = "exercises"

    id: Mapped[int] = mapped_column(primary_key=True)
    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("lessons.id", ondelete="CASCADE"), index=True)
    writing_type: Mapped[WritingType] = mapped_column(_enum(WritingType))
    title: Mapped[str | None] = mapped_column(String(200))
    instructions: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    page: Mapped[int | None] = mapped_column(Integer)        # page in the source book
    content: Mapped[dict | None] = mapped_column(JSONType)   # type-specific payload

    lesson: Mapped[Lesson] = relationship(back_populates="exercises")
    copying_items: Mapped[list[CopyingItem]] = relationship(
        back_populates="exercise", cascade="all, delete-orphan",
        order_by="CopyingItem.sort_order")

    def __repr__(self) -> str:
        return f"<Exercise {self.id} {self.writing_type.value}>"


class CopyingItem(Base):
    """One letter to copy inside a copying Exercise.

    Each item is practised in all 3 CopyingModes (bold / dotted / draw);
    the modes are not stored as rows -- they are tracked per Attempt.
    """
    __tablename__ = "copying_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    exercise_id: Mapped[int] = mapped_column(
        ForeignKey("exercises.id", ondelete="CASCADE"), index=True)
    letter_id: Mapped[int] = mapped_column(
        ForeignKey("letters.id", ondelete="RESTRICT"), index=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    exercise: Mapped[Exercise] = relationship(back_populates="copying_items")
    letter: Mapped[Letter] = relationship()

    def __repr__(self) -> str:
        return f"<CopyingItem {self.id} ex={self.exercise_id} letter={self.letter_id}>"


class DictationWord(Base):
    """One vocabulary entry from a book's dictation (إملاء) lists.

    Two scopes (faithful to the source books):
      * unit-level  -> the "الإملاء" list printed inside a unit; unit_id set.
      * book master -> the "مفردات للإملاء" appendix at the end of the book;
                       unit_id is NULL and is_master is True.
    An "entry" is one printed line, which may be more than one word
    (e.g. "أخبر المأمون", "لا بد"); it is stored verbatim.
    """
    __tablename__ = "dictation_words"

    id: Mapped[int] = mapped_column(primary_key=True)
    book_id: Mapped[int] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), index=True)
    unit_id: Mapped[int | None] = mapped_column(
        ForeignKey("units.id", ondelete="CASCADE"), index=True)
    word: Mapped[str] = mapped_column(String(255))
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    is_master: Mapped[bool] = mapped_column(Boolean, default=False)

    book: Mapped[Book] = relationship(back_populates="dictation_words")
    unit: Mapped[Unit | None] = relationship(back_populates="dictation_words")

    def __repr__(self) -> str:
        return f"<DictationWord {self.id} {self.word!r}>"


# ---------------------------------------------------------------------------
# Student tables
# ---------------------------------------------------------------------------
class Student(Base, TimestampMixin):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # --- authentication ---
    password_hash: Mapped[str | None] = mapped_column(String(255))

    # --- placement / current track (Levels screen) ---
    current_level: Mapped[Level | None] = mapped_column(_enum(Level))
    placement_done: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- gamification (Dashboard) ---
    streak_count: Mapped[int] = mapped_column(Integer, default=0)     # سلسلة التعلم اليومية
    longest_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_active_on: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_points: Mapped[int] = mapped_column(Integer, default=0)     # نقاط / XP

    attempts: Mapped[list[Attempt]] = relationship(
        back_populates="student", cascade="all, delete-orphan")
    progress: Mapped[list[LessonProgress]] = relationship(
        back_populates="student", cascade="all, delete-orphan")
    placement_results: Mapped[list[PlacementResult]] = relationship(
        back_populates="student", cascade="all, delete-orphan",
        order_by="PlacementResult.created_at.desc()")

    def __repr__(self) -> str:
        return f"<Student {self.id} {self.full_name!r}>"


class Attempt(Base):
    """One student submission for an exercise (any writing type).

    Copying  -> copying_item_id + copying_mode + submitted_drawing are set.
    Words / composition / dictation -> submitted_text is set.
    `score` / `feedback` are filled by the automated evaluation (AES).
    """
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), index=True)
    exercise_id: Mapped[int] = mapped_column(
        ForeignKey("exercises.id", ondelete="CASCADE"), index=True)

    # copying-only fields
    copying_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("copying_items.id", ondelete="SET NULL"))
    copying_mode: Mapped[CopyingMode | None] = mapped_column(_enum(CopyingMode))
    submitted_drawing: Mapped[list | None] = mapped_column(JSONType)  # student's points

    # text-writing-type fields
    submitted_text: Mapped[str | None] = mapped_column(Text)

    # evaluation result
    score: Mapped[float | None] = mapped_column(Float)
    max_score: Mapped[float | None] = mapped_column(Float)
    is_passed: Mapped[bool | None] = mapped_column(Boolean)
    feedback: Mapped[dict | None] = mapped_column(JSONType)  # detected errors, etc.

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())

    student: Mapped[Student] = relationship(back_populates="attempts")
    exercise: Mapped[Exercise] = relationship()
    copying_item: Mapped[CopyingItem | None] = relationship()

    def __repr__(self) -> str:
        return f"<Attempt {self.id} student={self.student_id} ex={self.exercise_id}>"


class LessonProgress(Base):
    """Per-student completion status for a lesson.

    Unit-level progress is derivable by aggregating its lessons.
    """
    __tablename__ = "lesson_progress"
    __table_args__ = (
        UniqueConstraint("student_id", "lesson_id", name="uq_student_lesson"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), index=True)
    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("lessons.id", ondelete="CASCADE"), index=True)
    status: Mapped[ProgressStatus] = mapped_column(
        _enum(ProgressStatus), default=ProgressStatus.not_started)
    score: Mapped[float | None] = mapped_column(Float)        # best score achieved
    stars: Mapped[int] = mapped_column(Integer, default=0)    # 0-3, shown on the map node
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    student: Mapped[Student] = relationship(back_populates="progress")
    lesson: Mapped[Lesson] = relationship()

    def __repr__(self) -> str:
        return f"<LessonProgress student={self.student_id} lesson={self.lesson_id} {self.status.value}>"


# ---------------------------------------------------------------------------
# Placement test tables (اختبار تحديد المستوى)
# ---------------------------------------------------------------------------
class PlacementQuestion(Base, TimestampMixin):
    """One question of the 5-question placement test.

    `answer_key` is optional guidance for the (AI) scorer -- e.g. the correct
    completion or the corrected sentence; it is never sent to the client.
    """
    __tablename__ = "placement_questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[PlacementKind] = mapped_column(_enum(PlacementKind))
    prompt: Mapped[str] = mapped_column(Text)                 # the question text
    given_text: Mapped[str | None] = mapped_column(Text)      # sentence to complete/correct
    answer_key: Mapped[str | None] = mapped_column(Text)      # scorer-only, not exposed
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    def __repr__(self) -> str:
        return f"<PlacementQuestion {self.id} {self.kind.value}>"


class PlacementResult(Base):
    """The outcome of one full placement-test submission for a student."""
    __tablename__ = "placement_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), index=True)
    level: Mapped[Level] = mapped_column(_enum(Level))
    score: Mapped[float | None] = mapped_column(Float)
    max_score: Mapped[float | None] = mapped_column(Float)
    answers: Mapped[list | None] = mapped_column(JSONType)     # [{question_id, answer, ...}]
    details: Mapped[dict | None] = mapped_column(JSONType)     # per-question AI breakdown

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())

    student: Mapped[Student] = relationship(back_populates="placement_results")

    def __repr__(self) -> str:
        return f"<PlacementResult student={self.student_id} {self.level.value}>"
