# -*- coding: utf-8 -*-
"""
Seed the composition (تعبير) and dictation (إملاء) tracks from the curated
`املاء  و تعبير محتوى/` folder.

Mapping (as requested by the app owner)
---------------------------------------
Composition (تعبير) -- the "professional" track:
    each تعبير docx           -> one Unit
      each الوحدة inside it    -> one Lesson
        each السؤال / prompt   -> one Exercise (composition)

Dictation (إملاء) -- word-based, the "intermediate" (words) track.
NOTE: sentence AUDIO dictation is the "advanced" track and is seeded
separately by seed_audio_dictation.py.
    each إملاء docx (a grade)  -> one Unit
      every ~10 words          -> one Lesson  ("مجموعة N")
        each word              -> one Exercise (dictation, content={"text": w})

Replaces the reference-book composition
---------------------------------------
This makes the 6 تعبير docx the single source of composition content, so it
deletes the reference-book units seeded by seed_reference_books.py (the
book-linked units). The Book rows and their master-dictation vocab are left
intact. Re-running this script is idempotent.

Run:
    python seed_imla_taabir.py
    python seed_imla_taabir.py --dry-run
    DATABASE_URL=sqlite:///verify.db python seed_imla_taabir.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from sqlalchemy import select

from database import SessionLocal, init_db
from models import Exercise, Lesson, Level, Unit, WritingType
from seed_reference_books import parse_book   # reuse the proven تعبير parser

# ---------------------------------------------------------------------------
# Source folder + file registry
# ---------------------------------------------------------------------------
CONTENT_DIR = Path(__file__).resolve().parent.parent / "املاء  و تعبير محتوى"

GRADE_NAME = {1: "الأول", 2: "الثاني", 3: "الثالث", 4: "الرابع"}
PART_NAME = {1: "الأول", 2: "الثاني"}

# (filename, grade, part)
COMPOSITION = [
    ("الطالب الثاني الجزء الاول تعبير.docx",        2, 1),
    ("الطالب_الثاني_الجزء_الثاني_تعبير (2).docx",   2, 2),
    ("الطالب الثالث الجزء  الاول تعبير.docx",       3, 1),
    ("الطالب_الثالث_الجزء_الثاني_تعبير.docx",       3, 2),
    ("الصف الرابع جزء أول تعبير.docx",              4, 1),
    ("الصف الرابع جزء تاني تعبير.docx",             4, 2),
]

# (filename, grade)
DICTATION = [
    ("ا1 املاء.docx", 1),
    ("2 املاء.docx",  2),
    ("ا3 املاء.docx", 3),
]

WORDS_PER_LESSON = 10

# Lines that are structural noise, not dictation words.
_IMLA_SKIP_PREFIXES = ("كتاب", "الوحدة", "الصفحة", "مفردات", "الإملاء",
                       "الاملاء", "الأملاء")


# ---------------------------------------------------------------------------
# إملاء parsing
# ---------------------------------------------------------------------------
def parse_imla_words(path: Path) -> list[str]:
    """Ordered, de-duplicated list of dictation words from a flat إملاء docx."""
    doc = Document(str(path))
    seen: set[str] = set()
    words: list[str] = []
    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        if text.startswith(_IMLA_SKIP_PREFIXES):
            continue
        if text in seen:
            continue
        seen.add(text)
        words.append(text)
    return words


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def _clear_previous(session) -> None:
    """Remove reference-book composition units + any prior run of this seed.

    - Units linked to a Book  -> the reference-book composition (being replaced).
    - Book-less professional / advanced units -> this script's own prior output.
    Cascades handle lessons, exercises and unit-scoped dictation words.
    """
    ref_units = session.scalars(
        select(Unit).where(Unit.book_id.is_not(None))
    ).all()
    # NOTE: word-based dictation lives at the *intermediate* (words) level;
    # the *advanced* level is owned by seed_audio_dictation.py (audio dictation),
    # so it is deliberately NOT cleared here.
    ours = session.scalars(
        select(Unit).where(
            Unit.book_id.is_(None),
            Unit.level.in_([Level.professional, Level.intermediate]),
        )
    ).all()
    for u in [*ref_units, *ours]:
        session.delete(u)
    session.flush()


def seed_composition(session) -> dict:
    n_units = n_lessons = n_ex = 0
    for idx, (fname, grade, part) in enumerate(COMPOSITION):
        path = CONTENT_DIR / fname
        if not path.exists():
            print(f"  !! missing تعبير file: {fname}")
            continue
        pb = parse_book(path, f"taabir_g{grade}p{part}", grade, part)

        title = f"التعبير — الصف {GRADE_NAME[grade]} (الجزء {PART_NAME[part]})"
        unit = Unit(
            book_id=None, unit_number=idx + 1, level=Level.professional,
            title=title, description=f"تدريبات الكتابة والتعبير — {fname}",
            sort_order=idx, is_published=True,
        )
        session.add(unit)
        n_units += 1

        l_idx = 0
        for pu in pb.units:
            if not pu.exercises:
                continue                        # skip الوحدة with no writing task
            lesson = Lesson(
                unit=unit, title=pu.title,
                description=None, sort_order=l_idx, is_published=True,
            )
            session.add(lesson)
            n_lessons += 1
            l_idx += 1
            for e_idx, pe in enumerate(pu.exercises):
                session.add(Exercise(
                    lesson=lesson,
                    writing_type=WritingType.composition,
                    title=None,
                    instructions=pe.prompt,
                    content={"elements": pe.elements} if pe.elements else None,
                    page=pe.page,
                    sort_order=e_idx,
                ))
                n_ex += 1
    return {"units": n_units, "lessons": n_lessons, "exercises": n_ex}


def seed_dictation(session) -> dict:
    n_units = n_lessons = n_ex = 0
    for idx, (fname, grade) in enumerate(DICTATION):
        path = CONTENT_DIR / fname
        if not path.exists():
            print(f"  !! missing إملاء file: {fname}")
            continue
        words = parse_imla_words(path)

        unit = Unit(
            book_id=None, unit_number=idx + 1, level=Level.intermediate,
            title=f"إملاء الصف {GRADE_NAME[grade]}",
            description=f"كلمات الإملاء — {len(words)} كلمة",
            sort_order=idx, is_published=True,
        )
        session.add(unit)
        n_units += 1

        for c_idx, chunk in enumerate(_chunks(words, WORDS_PER_LESSON)):
            lesson = Lesson(
                unit=unit, title=f"مجموعة {c_idx + 1}",
                description=None, sort_order=c_idx, is_published=True,
            )
            session.add(lesson)
            n_lessons += 1
            for w_idx, word in enumerate(chunk):
                session.add(Exercise(
                    lesson=lesson,
                    writing_type=WritingType.dictation,
                    title=word,
                    instructions="اكتب الكلمة التي تسمعها.",
                    content={"text": word, "audio_url": None},
                    sort_order=w_idx,
                ))
                n_ex += 1
    return {"units": n_units, "lessons": n_lessons, "exercises": n_ex}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    dry_run = "--dry-run" in sys.argv

    if not CONTENT_DIR.is_dir():
        print(f"!! not found: {CONTENT_DIR}")
        sys.exit(1)

    if dry_run:
        print("Composition (تعبير) — professional track:")
        for fname, grade, part in COMPOSITION:
            path = CONTENT_DIR / fname
            if not path.exists():
                print(f"  !! missing: {fname}"); continue
            pb = parse_book(path, "x", grade, part)
            uwith = [u for u in pb.units if u.exercises]
            nex = sum(len(u.exercises) for u in uwith)
            print(f"  الصف {GRADE_NAME[grade]} ج{part}: "
                  f"{len(uwith)} lessons, {nex} exercises  <- {fname}")
        print("\nDictation (إملاء) — advanced track:")
        for fname, grade in DICTATION:
            path = CONTENT_DIR / fname
            if not path.exists():
                print(f"  !! missing: {fname}"); continue
            words = parse_imla_words(path)
            nles = -(-len(words) // WORDS_PER_LESSON)
            print(f"  الصف {GRADE_NAME[grade]}: {len(words)} words -> "
                  f"{nles} lessons  <- {fname}")
        print("\n(dry run -- no database changes)")
        return

    init_db()
    session = SessionLocal()
    try:
        _clear_previous(session)
        comp = seed_composition(session)
        dic = seed_dictation(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    from database import engine
    print(f"\nSeeded تعبير + إملاء into {engine.url}")
    print(f"  professional (تعبير): {comp['units']} units, "
          f"{comp['lessons']} lessons, {comp['exercises']} exercises")
    print(f"  advanced (إملاء)    : {dic['units']} units, "
          f"{dic['lessons']} lessons, {dic['exercises']} exercises")


if __name__ == "__main__":
    main()
