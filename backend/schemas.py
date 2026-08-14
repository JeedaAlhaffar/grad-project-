# -*- coding: utf-8 -*-
"""
Pydantic request/response schemas -- the JSON contract for the Flutter app.

Conventions the frontend can rely on
-------------------------------------
* All enums are serialised as their lowercase string value
  (e.g. writing_type = "copying", level = "beginner").
* Every list endpoint returns a bare JSON array unless a wrapper is documented.
* Timestamps are ISO-8601 UTC strings.
* Auth: send `Authorization: Bearer <access_token>` on protected endpoints.
* Errors follow FastAPI's shape: {"detail": "..."} with the right HTTP status.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from models import (
    CopyingMode, LetterForm, Level, PlacementKind, ProgressStatus, WritingType,
)


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ===========================================================================
# Generic
# ===========================================================================
class Message(BaseModel):
    message: str


# ===========================================================================
# Auth & profile
# ===========================================================================
class RegisterIn(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class ForgotPasswordIn(BaseModel):
    email: EmailStr


class ResetPasswordIn(BaseModel):
    email: EmailStr
    code: str                      # reset code delivered out-of-band (email/SMS)
    new_password: str = Field(min_length=6, max_length=128)


class UpdateMeIn(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=200)
    current_level: Optional[Level] = None


class StudentOut(ORMModel):
    id: int
    full_name: str
    email: Optional[str]
    current_level: Optional[Level]
    placement_done: bool
    streak_count: int
    longest_streak: int
    total_points: int
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    student: StudentOut


# ===========================================================================
# Content: letters
# ===========================================================================
class LetterOut(ORMModel):
    id: int
    code: str
    arabic_letter: str
    form: LetterForm
    stroke_count: int
    point_count: int
    aspect_ratio: Optional[float]
    strokes: list                  # [[[x, y], ...], ...] normalized 0..1, y-down


class LetterSummary(ORMModel):
    """Letters list item without the (heavy) stroke payload."""
    id: int
    code: str
    arabic_letter: str
    form: LetterForm
    stroke_count: int


# ===========================================================================
# Content: exercises
# ===========================================================================
class CopyingItemOut(ORMModel):
    id: int
    sort_order: int
    letter: LetterOut
    # every copying item is practised in all three modes
    modes: list[CopyingMode] = [
        CopyingMode.bold_line, CopyingMode.dotted_line, CopyingMode.free_draw,
    ]


class ExerciseOut(ORMModel):
    id: int
    lesson_id: int
    writing_type: WritingType
    title: Optional[str]
    instructions: Optional[str]
    sort_order: int
    page: Optional[int]
    # type-specific payload:
    #   words       -> {"words": ["..."]}
    #   composition -> {"prompt": "...", "min_words": 20, "elements": [...]}
    #   dictation   -> {"text": "...", "audio_url": "..."}
    content: Optional[dict]
    copying_items: list[CopyingItemOut] = []


# ===========================================================================
# Content: lessons / units / levels  (with per-student progress overlay)
# ===========================================================================
class LessonNode(ORMModel):
    """A lesson as drawn on the level-map/path screen."""
    id: int
    title: str
    description: Optional[str] = None
    sort_order: int
    writing_type: Optional[WritingType] = None   # dominant type in the lesson
    status: ProgressStatus = ProgressStatus.not_started
    stars: int = 0
    score: Optional[float] = None
    locked: bool = False
    is_current: bool = False


class LessonDetailOut(ORMModel):
    id: int
    unit_id: int
    title: str
    description: Optional[str]
    sort_order: int
    status: ProgressStatus = ProgressStatus.not_started
    stars: int = 0
    exercises: list[ExerciseOut] = []


class UnitOut(ORMModel):
    id: int
    book_id: Optional[int]
    unit_number: Optional[int]
    level: Level
    title: str
    description: Optional[str]
    sort_order: int
    lessons_count: int = 0
    completed_lessons: int = 0
    progress_percent: int = 0
    lessons: list[LessonNode] = []


class LevelOut(BaseModel):
    """Summary card on the Levels list / Dashboard."""
    level: Level
    title: str
    description: str
    units_count: int = 0
    lessons_count: int = 0
    completed_lessons: int = 0
    progress_percent: int = 0
    is_unlocked: bool = True


class LevelDetailOut(LevelOut):
    units: list[UnitOut] = []


# ===========================================================================
# Dashboard / progress
# ===========================================================================
class ContinueOut(BaseModel):
    lesson_id: int
    lesson_title: str
    unit_title: str
    level: Level


class DashboardOut(BaseModel):
    student: StudentOut
    streak_count: int
    progress_percent: int
    total_lessons: int
    completed_lessons: int
    levels: list[LevelOut]
    continue_lesson: Optional[ContinueOut] = None


class LessonProgressOut(ORMModel):
    lesson_id: int
    status: ProgressStatus
    stars: int
    score: Optional[float]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


# ===========================================================================
# Attempts / evaluation
# ===========================================================================
class CopyingAttemptIn(BaseModel):
    copying_item_id: int
    copying_mode: CopyingMode
    # student's strokes, same shape as Letter.strokes: [[[x, y], ...], ...]
    submitted_drawing: list


class TextAttemptIn(BaseModel):
    submitted_text: str = Field(min_length=1)


class AttemptSubmitIn(BaseModel):
    """One body for all writing types. Send the fields for the exercise's type:

      copying  -> copying_item_id, copying_mode, submitted_drawing
      words / composition / dictation -> submitted_text
    """
    # copying
    copying_item_id: Optional[int] = None
    copying_mode: Optional[CopyingMode] = None
    submitted_drawing: Optional[list] = None
    # text-based types
    submitted_text: Optional[str] = None


class ErrorItem(BaseModel):
    message: str                          # human-readable, Arabic
    correction: Optional[str] = None      # suggested fix
    span: Optional[str] = None            # the offending substring, if any
    type: Optional[str] = None            # "spelling" | "grammar" | "punctuation" | "shape"
    # From the GEC/ARETA detector: the fine-grained linguistic label,
    # e.g. tag="OH", category="spelling", subcategory="hamza_error".
    tag: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None


class RelevanceOut(BaseModel):
    """Result of the hybrid topic-relevance scorer (relevance_scorer.py).

    status: "success" | "rejected" (essay under the word guardrail) | "skipped"
    (no prompt to compare against). `score` is out of `max_score` (4.0).
    """
    status: str
    score: Optional[float] = None
    max_score: float = 4.0
    reason: Optional[str] = None
    message: Optional[str] = None          # Arabic message to show the student
    word_count: Optional[int] = None
    min_words: Optional[int] = None


class EvaluationFeedback(BaseModel):
    """The feedback card shown on the evaluation screen."""
    summary: Optional[str] = None
    strengths: list[str] = []
    errors: list[ErrorItem] = []
    suggestions: list[str] = []
    # composition (تعبير) only: per-trait breakdown. holistic is 0..32, the
    # sub-traits (relevance, organization, development, vocabulary, style,
    # mechanics, grammar) are each 0..4. `relevance` comes from the hybrid
    # scorer below — the AES model's own relevance head is not used.
    traits: Optional[dict[str, float]] = None
    # composition only: the raw relevance result (also carries the rejection
    # message when the essay is shorter than the guardrail).
    relevance: Optional[RelevanceOut] = None
    # text / words / composition: the GEC-corrected version of the submission
    # and the error tally per type.
    corrected_text: Optional[str] = None
    error_counts: Optional[dict[str, int]] = None
    # copying (نسخ) only: what the handwriting classifier saw. Carries the
    # expected letter/form and its class, the predicted class + top-3 with
    # confidences, and the recognition / trace / final sub-scores, so the
    # drawing screen can show "this looks like القاف, not الكاف" and the
    # three modes can be debugged independently. See copying_grader.py.
    recognition: Optional[dict[str, Any]] = None


class AttemptOut(ORMModel):
    id: int
    exercise_id: int
    copying_item_id: Optional[int]
    copying_mode: Optional[CopyingMode]
    submitted_text: Optional[str]
    score: Optional[float]
    max_score: Optional[float]
    is_passed: Optional[bool]
    stars: int = 0
    # "pending" until the AI evaluator has run; "done" once scored.
    evaluation_status: str = "done"
    feedback: Optional[EvaluationFeedback] = None
    created_at: datetime


# ===========================================================================
# Placement test
# ===========================================================================
class PlacementQuestionOut(ORMModel):
    id: int
    kind: PlacementKind
    prompt: str
    given_text: Optional[str]
    sort_order: int


class PlacementAnswerIn(BaseModel):
    question_id: int
    answer: str


class PlacementSubmitIn(BaseModel):
    answers: list[PlacementAnswerIn] = Field(min_length=1)


class PlacementResultOut(ORMModel):
    id: int
    level: Level
    score: Optional[float]
    max_score: Optional[float]
    details: Optional[dict]
    created_at: datetime


# ===========================================================================
# AI service payloads (used by the AI router; see routers/ai.py)
# ===========================================================================
class EvaluateTextIn(BaseModel):
    writing_type: WritingType
    submitted_text: str = Field(min_length=1)
    reference: Optional[str] = None          # dictation target / prompt


class EvaluateCopyingIn(BaseModel):
    letter_id: int
    submitted_drawing: list
    mode: CopyingMode = CopyingMode.free_draw


class DetectErrorsIn(BaseModel):
    """Raw spelling/grammar detection, without any scoring."""
    text: str = Field(min_length=1)


class DetectErrorsOut(BaseModel):
    text: str
    corrected: str                          # the GEC-corrected version
    errors: list[ErrorItem] = []
    total_errors: int = 0
    counts: dict[str, int] = {}
    tagged: bool = False                    # True when ARETA labelled the errors


class RelevanceIn(BaseModel):
    """Topic relevance of an essay against its prompt."""
    prompt: str
    text: str = Field(min_length=1)


class EvaluationOut(BaseModel):
    score: float
    max_score: float = 100.0
    is_passed: bool
    stars: int
    feedback: EvaluationFeedback


class GenerateExerciseIn(BaseModel):
    """Ask the AI to build a targeted exercise from a student's mistakes."""
    writing_type: Optional[WritingType] = None
    focus_skill: Optional[str] = None            # e.g. "همزة القطع"
    based_on_attempt_id: Optional[int] = None


class GeneratedExerciseOut(BaseModel):
    title: str
    intro: str
    writing_type: WritingType
    content: dict                                # same shape as ExerciseOut.content


class TTSIn(BaseModel):
    text: str = Field(min_length=1)


class TTSOut(BaseModel):
    text: str
    audio_url: str
    audio_base64: Optional[str] = None
