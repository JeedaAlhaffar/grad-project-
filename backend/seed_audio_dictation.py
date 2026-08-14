# -*- coding: utf-8 -*-
"""
Seed the advanced (إملاء) track with AUDIO dictation exercises.

Source
------
`seed_data/audio_dictation.json` — 25 normal MSA sentences selected from the
Arabic Speech Corpus (Nawar Halabi), ordered shortest -> longest. The matching
wav files live under `media/dictation/` and are served at `/media/dictation/*`
by the static mount in main.py.

Structure created
-----------------
    5 Units   (level = advanced, "إملاء صوتي — الوحدة N")   shortest ... longest
      1 Lesson each
        5 Exercises  (writing_type = dictation)
          content = {
            "text":       "<fully diacritized reference>",
            "text_plain": "<tashkeel stripped, for lenient grading>",
            "audio_url":  "/media/dictation/ara_XXXX.wav",
            "duration":   <seconds>,
          }

Idempotent & non-destructive
----------------------------
Only clears units this script created before (advanced, book-less, title starts
with the AUDIO_UNIT_PREFIX marker). It does NOT touch the word-based إملاء units
from seed_imla_taabir.py. Re-running replaces just this script's own content.

Run:
    python seed_audio_dictation.py
    python seed_audio_dictation.py --dry-run
    DATABASE_URL=sqlite:///verify.db python seed_audio_dictation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import select

from database import SessionLocal, init_db
from models import Exercise, Lesson, Level, Unit, WritingType

SEED_FILE = Path(__file__).resolve().parent / "seed_data" / "audio_dictation.json"
MEDIA_DIR = Path(__file__).resolve().parent / "media" / "dictation"

AUDIO_UNIT_PREFIX = "إملاء صوتي"      # marker used to find/replace our own units
INSTRUCTIONS = "استمع إلى المقطع الصوتي ثم اكتب الجملة التي سمعتها."


def _load() -> list[dict]:
    data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    return data["units"]


def _clear_previous(session) -> int:
    """Delete only this script's own prior units (safe to re-run)."""
    ours = session.scalars(
        select(Unit).where(
            Unit.book_id.is_(None),
            Unit.level == Level.advanced,
            Unit.title.like(f"{AUDIO_UNIT_PREFIX}%"),
        )
    ).all()
    for u in ours:
        session.delete(u)
    session.flush()
    return len(ours)


def seed(session) -> dict:
    n_units = n_lessons = n_ex = 0
    for grp in _load():
        u_no = grp["unit"]
        clips = grp["clips"]
        durs = [c["duration"] for c in clips]
        unit = Unit(
            book_id=None,
            unit_number=u_no,
            level=Level.advanced,
            title=f"{AUDIO_UNIT_PREFIX} — الوحدة {u_no}",
            description=f"إملاء صوتي ({min(durs):.1f}–{max(durs):.1f} ثانية)",
            sort_order=u_no,
            is_published=True,
        )
        session.add(unit)
        n_units += 1

        lesson = Lesson(
            unit=unit,
            title=f"مقاطع الوحدة {u_no}",
            description=None,
            sort_order=0,
            is_published=True,
        )
        session.add(lesson)
        n_lessons += 1

        for e_idx, c in enumerate(clips):
            session.add(Exercise(
                lesson=lesson,
                writing_type=WritingType.dictation,
                title=None,
                instructions=INSTRUCTIONS,
                content={
                    "text": c["text"],
                    "text_plain": c["text_plain"],
                    "audio_url": c["audio_url"],
                    "duration": c["duration"],
                },
                sort_order=e_idx,
            ))
            n_ex += 1
    return {"units": n_units, "lessons": n_lessons, "exercises": n_ex}


def _check_media() -> list[str]:
    missing = []
    for grp in _load():
        for c in grp["clips"]:
            if not (MEDIA_DIR / f"{c['slug']}.wav").exists():
                missing.append(c["slug"])
    return missing


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    dry_run = "--dry-run" in sys.argv

    if not SEED_FILE.exists():
        print(f"!! missing seed data: {SEED_FILE}")
        sys.exit(1)

    missing = _check_media()
    if missing:
        print(f"!! {len(missing)} audio files missing under {MEDIA_DIR}: "
              f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
        if not dry_run:
            sys.exit(1)

    if dry_run:
        for grp in _load():
            clips = grp["clips"]
            durs = [c["duration"] for c in clips]
            print(f"Unit {grp['unit']}  ({min(durs):.1f}–{max(durs):.1f}s), "
                  f"{len(clips)} exercises:")
            for c in clips:
                print(f"   {c['duration']:5.2f}s  {c['audio_url']}")
                print(f"          {c['text']}")
        print("\n(dry run — no database changes)")
        return

    init_db()
    session = SessionLocal()
    try:
        removed = _clear_previous(session)
        stats = seed(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    from database import engine
    print(f"\nSeeded audio إملاء into {engine.url}")
    if removed:
        print(f"  replaced {removed} previously-seeded audio unit(s)")
    print(f"  advanced (إملاء صوتي): {stats['units']} units, "
          f"{stats['lessons']} lessons, {stats['exercises']} exercises")


if __name__ == "__main__":
    main()
