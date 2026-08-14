# -*- coding: utf-8 -*-
"""
Seed the intermediate (words) level from a generated content JSON.

Source: seed_data/intermediate_level_v2.json, built by
`intermediate_level_content_builder.ipynb` — 500 curriculum words, filtered to
the ADAB conformer's 39-class alphabet, ranked by measured per-character recall,
split into length bands and clustered into semantically coherent lessons.

The pre-existing units
----------------------
The three units already sitting on this level (إملاء الصف الأول/الثاني/الثالث,
1,890 exercises) are the *same curriculum words* mislabelled: they carry
`writing_type='dictation'` and the instruction «اكتب الكلمة التي تسمعها» while
`audio_url` is null for every one of them — there is no audio to listen to.
The units seeded here are the corrected form of that content.

By default they are LEFT ALONE, so the level ends up carrying both. Pass
`--replace-old` to delete them instead. Units this script regenerates (matching
titles) are always replaced, and the جمل unit from seed_khatt_sentences.py is
never touched.

Run:
    DATABASE_URL=sqlite:///app.db python seed_intermediate_words.py --dry-run
    DATABASE_URL=sqlite:///app.db python seed_intermediate_words.py
    DATABASE_URL=sqlite:///app.db python seed_intermediate_words.py --replace-old
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import select

from database import SessionLocal, init_db
from models import Exercise, Lesson, Level, Unit, WritingType

SRC = Path(__file__).resolve().parent / "seed_data" / "intermediate_level_v2.json"
SENTENCES_UNIT_TITLE = "جمل"          # owned by seed_khatt_sentences.py


def main(dry_run: bool = False, replace_old: bool = False) -> None:
    doc = json.loads(SRC.read_text(encoding="utf-8"))
    units_in = [u for u in doc["units"] if u["lessons"]]
    total = sum(len(l["exercises"]) for u in units_in for l in u["lessons"])
    print(f"source: {SRC.name}")
    print(f"  {len(units_in)} units · "
          f"{sum(len(u['lessons']) for u in units_in)} lessons · {total} words")
    print(f"  scoring: {doc['selection']['scoring']}\n")

    init_db()
    with SessionLocal() as session:
        new_titles = {u["title"] for u in units_in}
        existing = session.scalars(
            select(Unit).where(
                Unit.book_id.is_(None), Unit.level == Level.intermediate)
        ).all()

        for u in existing:
            if u.title == SENTENCES_UNIT_TITLE:
                print(f"[keep]    unit {u.id} «{u.title}» (sentences)")
                continue
            if u.title in new_titles:
                print(f"[replace] unit {u.id} «{u.title}» — regenerated below")
            elif not replace_old:
                print(f"[keep]    unit {u.id} «{u.title}» "
                      f"({len(u.lessons)} lessons) — pass --replace-old to remove")
                continue
            else:
                n = sum(len(l.exercises) for l in u.lessons)
                print(f"[delete]  unit {u.id} «{u.title}» "
                      f"({len(u.lessons)} lessons, {n} exercises)")
            if not dry_run:
                session.delete(u)
        if not dry_run:
            session.flush()

        for u_in in units_in:
            unit = Unit(
                book_id=None,
                unit_number=u_in["unit_number"],
                level=Level.intermediate,
                title=u_in["title"],
                description=u_in["description"],
                sort_order=u_in["sort_order"],
                is_published=u_in["is_published"],
            )
            for l_in in u_in["lessons"]:
                lesson = Lesson(
                    unit=unit,
                    title=l_in["title"],
                    description=l_in["description"],
                    sort_order=l_in["sort_order"],
                    is_published=l_in["is_published"],
                )
                for e_in in l_in["exercises"]:
                    lesson.exercises.append(Exercise(
                        writing_type=WritingType(e_in["writing_type"]),
                        title=e_in["title"],
                        instructions=e_in["instructions"],
                        sort_order=e_in["sort_order"],
                        content=e_in["content"],
                    ))
            print(f"[create]  «{unit.title}» — {len(u_in['lessons'])} lessons, "
                  f"{sum(len(l['exercises']) for l in u_in['lessons'])} words")
            if not dry_run:
                session.add(unit)

        if dry_run:
            print("\n[dry-run] nothing written.")
            return

        session.commit()

        # Renumber into a deliberate order. Without this the new units interleave
        # with the kept ones (both carry sort_order 0,1,2 from their own seeding):
        #   new word units, in the JSON's difficulty order
        #   -> any kept legacy units
        #   -> the sentences unit last
        units = session.scalars(
            select(Unit).where(Unit.book_id.is_(None),
                               Unit.level == Level.intermediate)
        ).all()
        by_title = {u.title: u for u in units}
        order = [by_title[u["title"]] for u in units_in if u["title"] in by_title]
        order += [u for u in units
                  if u.title not in new_titles and u.title != SENTENCES_UNIT_TITLE]
        order += [u for u in units if u.title == SENTENCES_UNIT_TITLE]

        for i, u in enumerate(order):
            u.unit_number, u.sort_order = i + 1, i
        session.commit()

        print(f"\n✓ intermediate level now has {len(units)} units:")
        for u in order:
            n = sum(len(l.exercises) for l in u.lessons)
            print(f"    {u.unit_number}. {u.title} — "
                  f"{len(u.lessons)} lessons, {n} exercises")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv, replace_old="--replace-old" in sys.argv)
