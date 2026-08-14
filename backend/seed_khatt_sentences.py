# -*- coding: utf-8 -*-
"""
Seed the sentences unit (جمل) of the intermediate level with 10 sentences taken
from OnlineKHATT line transcriptions.

Why these ten
-------------
OnlineKHATT `.txt` files are *lines* of a scanned page, not sentences — most of
them break mid-clause ("...ومن على" / " عباده بفضله..."). Out of 8,381 lines,
these survived four filters:

  1. every character is in the ADAB conformer's 39-class alphabet
  2. 3-5 words (the model was trained on 1-3 token labels; long lines are the
     KHATT regime, where CER is much worse than ADAB's reported 1.65%)
  3. does not open mid-clause (و / ف / على / التي / ...)
  4. every word is a whole token in AraBERT's vocabulary — this is what drops
     the transcription errors that riddle the raw data (وفتا for وقتا,
     النطريه for النظرية, الإقطاعيه for الإقطاعية)

then hand-picked for being meaningful on their own. `detectability` is the
product of the per-character recalls measured on OnlineKHATT SegmentedCharacters
(see khatt_words_content_builder.ipynb) — the probability of reading every
letter correctly. It falls off fast with length, which is exactly why this unit
stays at 3-4 words.

Run:
    DATABASE_URL=sqlite:///app.db python seed_khatt_sentences.py
    DATABASE_URL=sqlite:///app.db python seed_khatt_sentences.py --dry-run
"""
from __future__ import annotations

import sys

from sqlalchemy import select

from database import SessionLocal, init_db
from models import Exercise, Lesson, Level, Unit, WritingType

UNIT_TITLE = "جمل"
LESSON_TITLE = "جمل 1"
INSTRUCTION = "اكتب الجملة الظاهرة أمامك"

# (sentence, detectability) — ordered easiest first, so the lesson ramps up
SENTENCES = [
    ("له طابع خاص",          0.3754),
    ("تلعب دورا مهما",        0.3193),
    ("كتاب في التاريخ",       0.3151),
    ("رواية أخبار الجاهلية",  0.3141),
    ("استمر في الحكم",        0.3014),
    ("حماية كوكب الأرض",      0.2674),
    ("كل هذه الجبال",         0.2565),
    ("تظل إلى قيام الساعة",   0.2565),
    ("للدراسة في الصين",      0.2523),
    ("السعادة في الدنيا",     0.2427),
]


def main(dry_run: bool = False) -> None:
    init_db()
    with SessionLocal() as session:
        # --- idempotent: drop a previous run of this same unit ---------------
        existing = session.scalars(
            select(Unit).where(
                Unit.book_id.is_(None),
                Unit.level == Level.intermediate,
                Unit.title == UNIT_TITLE,
            )
        ).all()
        for u in existing:
            print(f"[replace] deleting existing unit {u.id} ({u.title})")
            if not dry_run:
                session.delete(u)
        if existing and not dry_run:
            session.flush()

        # --- unit number goes after the existing intermediate units ---------
        taken = session.scalars(
            select(Unit.unit_number).where(
                Unit.book_id.is_(None), Unit.level == Level.intermediate)
        ).all()
        unit_number = (max(taken) + 1) if taken else 1

        unit = Unit(
            book_id=None,
            unit_number=unit_number,
            level=Level.intermediate,
            title=UNIT_TITLE,
            description=f"{len(SENTENCES)} جمل قصيرة من مجموعة OnlineKHATT",
            sort_order=unit_number - 1,
            is_published=True,
        )
        lesson = Lesson(
            unit=unit,
            title=LESSON_TITLE,
            description="اكتب كل جملة كما تظهر أمامك",
            sort_order=0,
            is_published=True,
        )
        for i, (text, det) in enumerate(SENTENCES):
            lesson.exercises.append(Exercise(
                writing_type=WritingType.words,
                title=text,
                instructions=INSTRUCTION,
                sort_order=i,
                content={"words": [text]},
            ))
            print(f"  {i + 1:>2}. {text}   (detectability {det:.4f})")

        if dry_run:
            print("\n[dry-run] nothing written.")
            return

        session.add(unit)
        session.commit()
        print(f"\n✓ unit {unit.id} «{unit.title}» (unit_number={unit_number}) "
              f"+ lesson {lesson.id} + {len(SENTENCES)} exercises")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
