# -*- coding: utf-8 -*-
"""
Seed the curriculum content from the six "العربية بين يديك" student books.

Each book is one grade (2/3/4) and one part (1/2). The parser walks the
docx paragraphs in document order and runs a small state machine that
recognises the recurring section markers:

    الوحدة ...            -> a new Unit
    الكتابة والتعبير      -> the writing/composition section of a unit
        السؤال            -> starts a new composition prompt (Exercise)
        الصفحة N          -> the source page of the upcoming/current prompt
        <list items>      -> "elements" helping the current prompt
    الإملاء               -> a unit-level dictation word list
    مفردات للإملاء        -> the book-level master dictation appendix

Mapping to the schema (models.py):

    Book
      Unit(unit_number, title)
        Lesson "الكتابة والتعبير"
          Exercise(composition, instructions=prompt, content={"elements":[...]}, page)
        DictationWord(unit-scoped)            # from the unit الإملاء list
      DictationWord(is_master, unit_id=NULL)  # from مفردات للإملاء

Run (creates tables if needed, idempotent per book code):

    python seed_reference_books.py

Point it at a throwaway db to inspect without touching your real one:

    DATABASE_URL=sqlite:///seed_test.db python seed_reference_books.py
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document

from database import SessionLocal, init_db
from models import (
    Book, Unit, Lesson, Exercise, DictationWord, WritingType,
)
from services import level_for_writing_type

# ---------------------------------------------------------------------------
# Book registry -- exact file names of the six reference books.
# ---------------------------------------------------------------------------
REF_DIR = Path(__file__).resolve().parent.parent / "refrence book"
SERIES = "العربية بين يديك"

BOOKS = [
    ("العربية_بين_يديك_كتاب_الطالب_الثاني_الجزء_الأول.docx",  "g2p1", 2, 1),
    ("العربية_بين_يديك_كتاب_الطالب_الثاني_الجزء_الثاني.docx", "g2p2", 2, 2),
    ("العربية_بين_يديك_كتاب_الطالب_الثالث_الجزء_الأول.docx",  "g3p1", 3, 1),
    ("العربية_بين_يديك_كتاب_الطالب_الثالث_الجزء_الثاني.docx", "g3p2", 3, 2),
    ("العربية_بين_يديك_كتاب_الطالب_الرابع_الجزء_الاول.docx",  "g4p1", 4, 1),
    ("العربية_بين_يديك_كتاب_الطالب_الرايع_الجزء_التاني.docx", "g4p2", 4, 2),
]

# Arabic ordinal -> unit number. Keys are normalised (see _norm_ordinal).
_ORDINALS = {
    "الاولى": 1, "الثانية": 2, "الثالثة": 3, "الرابعة": 4,
    "الخامسة": 5, "السادسة": 6, "السابعة": 7, "الثامنة": 8,
    "التاسعة": 9, "العاشرة": 10, "الحادية عشرة": 11, "الثانية عشرة": 12,
    "الثالثة عشرة": 13, "الرابعة عشرة": 14, "الخامسة عشرة": 15,
    "السادسة عشرة": 16,
}

# Lines that begin a fresh composition prompt even without a السؤال marker.
PROMPT_STARTERS = (
    "اكتب", "أكتب", "اعد قراءة", "أعد قراءة", "اعد قرايه",
    "رتب ", "صف ", "قارن", "اجب", "أجب", "حول ", "خمس طرف",
)

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


# ---------------------------------------------------------------------------
# Parsed (DB-free) representation
# ---------------------------------------------------------------------------
@dataclass
class ParsedExercise:
    prompt: str
    elements: list[str] = field(default_factory=list)
    page: int | None = None


@dataclass
class ParsedUnit:
    unit_number: int
    title: str
    exercises: list[ParsedExercise] = field(default_factory=list)
    dictation: list[str] = field(default_factory=list)


@dataclass
class ParsedBook:
    code: str
    grade: int
    part: int
    title: str
    source_file: str
    units: list[ParsedUnit] = field(default_factory=list)
    master_dictation: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Line classification helpers
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    """Collapse whitespace and the double-alef OCR typo (االسادسة -> السادسة)."""
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("اا", "ا")
    return s


def _strip_colon(s: str) -> str:
    return s.strip().rstrip(":").strip().rstrip(":").strip()


def is_book_title(line: str) -> bool:
    return line.startswith("كتاب الطالب")


def is_unit_header(line: str) -> bool:
    return _norm(line).startswith("الوحدة")


def parse_unit_number(line: str, fallback: int) -> int:
    rest = _norm(line)[len("الوحدة"):].strip()
    rest = _strip_colon(rest)
    return _ORDINALS.get(rest, fallback)


def is_writing_header(line: str) -> bool:
    n = _norm(line)
    # accepts الكتابة والتعبير and the الكتاية typo
    return n.startswith("الكت") and "والتعبير" in n


def is_master_header(line: str) -> bool:
    return _norm(line).startswith("مفردات")


def is_dictation_header(line: str) -> bool:
    n = _strip_colon(_norm(line))
    return n in ("الاملاء", "الإملاء", "الأملاء")


def is_page(line: str) -> int | None:
    n = _norm(line)
    if not n.startswith("الصفحة"):
        return None
    m = re.search(r"\d+", n.translate(_DIGITS))
    return int(m.group()) if m else None


def is_question(line: str) -> tuple[bool, str]:
    """Return (is_marker, trailing_prompt_text)."""
    n = line.strip()
    if not _norm(n).startswith("السؤال"):
        return False, ""
    body = n[n.find("السؤال") + len("السؤال"):]
    return True, _strip_colon(body)


def is_prompt_starter(line: str) -> bool:
    n = _norm(line)
    return any(n.startswith(p) for p in PROMPT_STARTERS)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_book(path: Path, code: str, grade: int, part: int) -> ParsedBook:
    book = ParsedBook(code=code, grade=grade, part=part,
                      title="", source_file=path.name)
    doc = Document(str(path))

    unit: ParsedUnit | None = None
    mode: str | None = None          # "writing" | "dictation" | "master"
    current_ex: ParsedExercise | None = None
    pending_page: int | None = None
    expect_prompt = False            # a السؤال marker is waiting for its body
    unit_counter = 0
    unit_offset = 8 if part == 2 else 0

    def new_exercise(prompt: str) -> None:
        nonlocal current_ex, pending_page, expect_prompt
        ex = ParsedExercise(prompt=prompt.strip(), page=pending_page)
        unit.exercises.append(ex)
        current_ex = ex
        pending_page = None
        expect_prompt = False

    for para in doc.paragraphs:
        is_list = "List" in (para.style.name or "")
        raw = para.text or ""
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue

            # --- structural markers (style-independent) -----------------
            if is_book_title(line) and not book.title:
                book.title = line
                continue

            if is_master_header(line):
                mode = "master"
                unit = None
                current_ex = None
                expect_prompt = False
                continue

            if is_unit_header(line):
                unit_counter += 1
                num = parse_unit_number(line, unit_offset + unit_counter)
                unit = ParsedUnit(unit_number=num, title=_norm(line))
                book.units.append(unit)
                mode = None
                current_ex = None
                expect_prompt = False
                pending_page = None
                continue

            if is_writing_header(line):
                mode = "writing"
                current_ex = None
                expect_prompt = False
                continue

            if is_dictation_header(line):
                mode = "dictation"
                current_ex = None
                expect_prompt = False
                continue

            pg = is_page(line)
            if pg is not None:
                pending_page = pg
                if current_ex is not None and current_ex.page is None:
                    current_ex.page = pg
                continue

            q, trailing = is_question(line)
            if q:
                if trailing:
                    new_exercise(trailing)
                else:
                    current_ex = None
                    expect_prompt = True
                continue

            # --- content lines ------------------------------------------
            if mode == "master":
                book.master_dictation.append(line)
            elif mode == "dictation":
                if unit is not None:
                    unit.dictation.append(line)
            elif mode == "writing" and unit is not None:
                if expect_prompt or is_prompt_starter(line) or current_ex is None:
                    new_exercise(line)
                else:
                    current_ex.elements.append(line)
            # any other stray content (e.g. before the first unit) is ignored

    return book


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def seed_book(session, pb: ParsedBook) -> dict:
    existing = session.query(Book).filter_by(code=pb.code).first()
    if existing is not None:
        session.delete(existing)
        session.flush()

    book = Book(
        code=pb.code, title=pb.title or pb.source_file, series=SERIES,
        grade=pb.grade, part=pb.part, sort_order=pb.grade * 10 + pb.part,
        source_file=pb.source_file,
    )
    session.add(book)

    n_ex = n_dict = 0
    for u_idx, pu in enumerate(pb.units):
        # Seeded units hold only composition (التعبير) exercises, so they belong
        # to the professional track. Units with no exercises keep level=NULL and
        # let effective_level() derive it.
        unit_level = (
            level_for_writing_type(WritingType.composition)
            if pu.exercises else None
        )
        unit = Unit(book=book, unit_number=pu.unit_number,
                    title=pu.title, sort_order=u_idx, level=unit_level)
        session.add(unit)

        if pu.exercises:
            lesson = Lesson(unit=unit, title="الكتابة والتعبير", sort_order=0)
            session.add(lesson)
            for e_idx, pe in enumerate(pu.exercises):
                session.add(Exercise(
                    lesson=lesson,
                    writing_type=WritingType.composition,
                    instructions=pe.prompt,
                    content={"elements": pe.elements} if pe.elements else None,
                    page=pe.page,
                    sort_order=e_idx,
                ))
                n_ex += 1

        for w_idx, word in enumerate(pu.dictation):
            session.add(DictationWord(
                book=book, unit=unit, word=word, sort_order=w_idx,
                is_master=False,
            ))
            n_dict += 1

    for w_idx, word in enumerate(pb.master_dictation):
        session.add(DictationWord(
            book=book, unit=None, word=word, sort_order=w_idx, is_master=True,
        ))

    return {
        "code": pb.code, "units": len(pb.units), "exercises": n_ex,
        "dictation": n_dict, "master": len(pb.master_dictation),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    init_db()

    found = [(REF_DIR / fn, code, g, p) for fn, code, g, p in BOOKS]
    missing = [str(p) for p, *_ in found if not p.exists()]
    if missing:
        print("WARNING: missing book files:")
        for m in missing:
            print("  -", m)

    session = SessionLocal()
    try:
        totals = []
        for path, code, grade, part in found:
            if not path.exists():
                continue
            pb = parse_book(path, code, grade, part)
            totals.append(seed_book(session, pb))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    from database import engine
    print(f"\nSeeded {len(totals)} book(s) into {engine.url}\n")
    hdr = f"{'code':<6} {'units':>6} {'exercises':>10} {'dictation':>10} {'master':>8}"
    print(hdr)
    print("-" * len(hdr))
    for t in totals:
        print(f"{t['code']:<6} {t['units']:>6} {t['exercises']:>10} "
              f"{t['dictation']:>10} {t['master']:>8}")
    tot = lambda k: sum(t[k] for t in totals)
    print("-" * len(hdr))
    print(f"{'TOTAL':<6} {tot('units'):>6} {tot('exercises'):>10} "
          f"{tot('dictation'):>10} {tot('master'):>8}")


if __name__ == "__main__":
    main()
