# -*- coding: utf-8 -*-
"""
Seed the copying (نسخ الحروف) curriculum -- the beginner track.

Where the letter shapes come from
---------------------------------
Two folders of hand-drawn Arabic-letter strokes sit next to `backend/`:

    letters drawing/letters_normalized/   -- 0..1, y-down, unified format
                                             (output of normalize_letters.py)
    letters better version/               -- cleaner re-drawings of a few
                                             letters, still in the RAW inkml
                                             JSON format (y-up, needs
                                             normalizing here)

The filename tells us the letter and its form, e.g.
    B0006_2_1_001_Ain_S_points.json  -> ع, initial (Start)
    Haa_End_points.json              -> ح, final
    B0006_2_1_001_Ta_M_points.json   -> ت, medial   (better version wins)

For every (letter, form) we keep ONE source, preferring the "better version"
folder, then the curated plain files, then the B0006 set, then AHCR.

What gets seeded
----------------
The alphabet is grouped into 6 units exactly as the app owner specified:

    الوحدة الأولى   : ا د ذ ر ز و
    الوحدة الثانية  : ب ت ث ط ظ
    الوحدة الثالثة  : ج ح خ ع غ
    الوحدة الرابعة  : س ش ص ض
    الوحدة الخامسة : ك ل ن ف ق م
    الوحدة السادسة : ه ة ي

Mapping to the schema (models.py):

    Unit(level=beginner, "الوحدة ...")            # one per group
      Lesson("حرف ...")                            # one per LETTER
        Exercise(copying, "نسخ حرف ...")           # one per lesson
          CopyingItem -> Letter                    # one per SHAPE (form)

Every Letter row is also exposed on its own by GET /api/letters.

Run (idempotent -- safe to re-run; rebuilds letters + the 6 copying units):

    python seed_letters.py                 # against DATABASE_URL
    python seed_letters.py --dry-run       # print the catalog, touch no DB
    DATABASE_URL=sqlite:///seed_test.db python seed_letters.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from sqlalchemy import select

from database import SessionLocal, init_db
from models import (
    CopyingItem, Exercise, Lesson, Letter, LetterForm, Level, Unit, WritingType,
)

# ---------------------------------------------------------------------------
# Source folders
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = ROOT / "letters drawing" / "letters_normalized"
BETTER_DIR = ROOT / "letters better version"

MARGIN = 0.05                       # matches normalize_letters.py
CONTENT = 1.0 - 2 * MARGIN
ROUND = 5

# ---------------------------------------------------------------------------
# Filename -> (Arabic letter, form) resolver
# ---------------------------------------------------------------------------
F_ISO, F_INI, F_MED, F_FIN = (
    LetterForm.isolated, LetterForm.initial, LetterForm.medial, LetterForm.final,
)

FORM_TOKENS = {
    "end": F_FIN, "e": F_FIN,
    "mid": F_MED, "m": F_MED, "middle": F_MED, "midle": F_MED,
    "iso": F_ISO, "isolated": F_ISO, "isolate": F_ISO, "i": F_ISO,
    "start": F_INI, "s": F_INI,
}

# Transliteration token -> Arabic letter. Only tokens we actually see on disk.
LETTER_TOKENS = {
    "alef": "ا",
    "ba": "ب",
    "ta": "ت",
    "tha": "ث", "thaa": "ث",
    "gem": "ج", "jeem": "ج",
    "haa": "ح",
    "khaa": "خ",
    "dal": "د",
    "zal": "ذ",
    "raa": "ر",
    "zin": "ز",
    "seen": "س",
    "sheen": "ش", "shen": "ش",
    "saad": "ص",
    "daad": "ض",
    "too": "ط",
    "zah": "ظ",
    "ain": "ع",
    "ghain": "غ",
    "faa": "ف", "fa": "ف",
    "qaf": "ق",
    "kaf": "ك", "kaaf": "ك",
    "lam": "ل",
    "meem": "م",
    "noon": "ن",
    "ha": "ه",
    "ya": "ي",
    "waw": "و",
    # taa-marbuta (ة) is handled specially -- see resolve_core().
}


def clean_core(filename: str) -> str:
    """Strip extension, prefixes and dataset noise, leaving e.g. 'Ain_End'."""
    core = filename
    core = re.sub(r"\.(json|inkml)$", "", core, flags=re.I)
    core = re.sub(r"\s*\(\d+\)\s*$", "", core)      # trailing " (1)" / "(2)"
    core = re.sub(r"_?points?$", "", core, flags=re.I)
    core = re.sub(r"_poi$|_p$", "", core, flags=re.I)
    core = core.strip().strip("_").strip()
    core = re.sub(r"^B0006_2_1_001_", "", core)     # book-batch prefix
    core = re.sub(r"^AHCR_\d+_", "", core)          # AHCR prefix
    core = re.sub(r"_\d+$", "", core)               # AHCR trailing index (_63)
    core = re.sub(r"_?points?$", "", core, flags=re.I)   # again, after prefix strip
    return core.strip().strip("_").strip()


def resolve_core(core: str):
    """Return (arabic_letter, LetterForm) or None if unrecognised/skipped."""
    low = core.lower().strip()
    flat = re.sub(r"[ _]+", "_", low)

    # taa-marbuta (ة): 'E_TA' (final) and the 'TahM' family.
    if flat == "e_ta":
        return "ة", F_FIN
    if flat.startswith("tahm"):
        return "ة", (F_ISO if flat.endswith("_i") else F_FIN)

    tokens = re.split(r"[ _]+", low)
    letter = form = None
    for t in tokens:
        if letter is None and t in LETTER_TOKENS:
            letter = LETTER_TOKENS[t]
        elif form is None and t in FORM_TOKENS:
            form = FORM_TOKENS[t]
    if letter is None:
        return None                                 # e.g. 'AlefM' (alef maqsura)
    return letter, (form or F_ISO)


def source_priority(path: Path, is_better: bool) -> int:
    """Lower = preferred. better folder < curated plain < B0006 < AHCR."""
    if is_better:
        return 0
    name = path.name
    if name.startswith("AHCR_"):
        return 3
    if name.startswith("B0006_"):
        return 2
    return 1


# ---------------------------------------------------------------------------
# Reading / normalizing stroke data
# ---------------------------------------------------------------------------
def _normalize_raw(strokes):
    """Uniform-scale + center y-down strokes into the unit square.

    Same math as letters drawing/normalize_letters.py. Returns (strokes, info).
    """
    xs = [p[0] for st in strokes for p in st]
    ys = [p[1] for st in strokes for p in st]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w, h = max_x - min_x, max_y - min_y
    longer = max(w, h)
    if longer == 0:
        norm = [[[0.5, 0.5] for _ in st] for st in strokes]
        return norm, {"aspect_ratio": 1.0}
    scale = CONTENT / longer
    off_x = (1.0 - w * scale) / 2.0
    off_y = (1.0 - h * scale) / 2.0
    norm = [
        [[round((x - min_x) * scale + off_x, ROUND),
          round((y - min_y) * scale + off_y, ROUND)] for x, y in st]
        for st in strokes
    ]
    return norm, {"aspect_ratio": round(w / h, 4) if h else None}


def load_shape(path: Path, is_better: bool):
    """Return dict(strokes, stroke_count, point_count, aspect_ratio, source_format).

    Normalized folder -> already 0..1 y-down, read as-is.
    Better folder     -> RAW inkml JSON (format A, y-up); normalize here.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    if not is_better and "coordinate_system" in data:
        strokes = data["strokes"]
        return {
            "strokes": strokes,
            "stroke_count": data.get("n_strokes") or len(strokes),
            "point_count": data.get("n_points") or sum(len(s) for s in strokes),
            "aspect_ratio": data.get("aspect_ratio"),
            "source_format": (data.get("source_format") or "N")[:4],
        }

    # RAW format A: {"strokes":[{"raw":[[x,y,t],...]}, ...]}, y-up -> flip.
    raw_strokes = []
    for st in data.get("strokes", []):
        pts = st.get("smoothed") or st.get("raw") or []
        stroke = [[p[0], -p[1]] for p in pts if len(p) >= 2]
        if stroke:
            raw_strokes.append(stroke)
    if not raw_strokes:
        raise ValueError("no strokes")
    norm, info = _normalize_raw(raw_strokes)
    return {
        "strokes": norm,
        "stroke_count": len(norm),
        "point_count": sum(len(s) for s in norm),
        "aspect_ratio": info["aspect_ratio"],
        "source_format": "A",
    }


# ---------------------------------------------------------------------------
# Build the (letter, form) -> best source catalog
# ---------------------------------------------------------------------------
def build_catalog():
    """Return (catalog, unmapped).

    catalog[arabic_letter][LetterForm] = {path, is_better, priority}
    unmapped = list of filenames we could not classify.
    """
    catalog: dict[str, dict[LetterForm, dict]] = {}
    unmapped: list[str] = []

    sources = []
    if NORMALIZED_DIR.is_dir():
        sources += [(p, False) for p in NORMALIZED_DIR.glob("*.json")
                    if not p.name.startswith("_")]
    if BETTER_DIR.is_dir():
        sources += [(p, True) for p in BETTER_DIR.glob("*.json")]

    for path, is_better in sorted(sources, key=lambda s: s[0].name):
        resolved = resolve_core(clean_core(path.name))
        if resolved is None:
            unmapped.append(path.name)
            continue
        letter, form = resolved
        prio = source_priority(path, is_better)
        slot = catalog.setdefault(letter, {})
        current = slot.get(form)
        if current is None or prio < current["priority"]:
            slot[form] = {"path": path, "is_better": is_better, "priority": prio}

    return catalog, unmapped


# ---------------------------------------------------------------------------
# Curriculum definition (the app owner's grouping)
# ---------------------------------------------------------------------------
LETTER_NAMES = {
    "ا": "الألف", "ب": "الباء", "ت": "التاء", "ث": "الثاء", "ج": "الجيم",
    "ح": "الحاء", "خ": "الخاء", "د": "الدال", "ذ": "الذال", "ر": "الراء",
    "ز": "الزاي", "س": "السين", "ش": "الشين", "ص": "الصاد", "ض": "الضاد",
    "ط": "الطاء", "ظ": "الظاء", "ع": "العين", "غ": "الغين", "ف": "الفاء",
    "ق": "القاف", "ك": "الكاف", "ل": "اللام", "م": "الميم", "ن": "النون",
    "ه": "الهاء", "ة": "التاء المربوطة", "ي": "الياء", "و": "الواو",
}

# transliteration slug for the stable Letter.code (arabic -> latin)
LETTER_SLUGS = {
    "ا": "alef", "ب": "ba", "ت": "ta", "ث": "tha", "ج": "jeem", "ح": "haa",
    "خ": "khaa", "د": "dal", "ذ": "thal", "ر": "ra", "ز": "zay", "س": "seen",
    "ش": "sheen", "ص": "saad", "ض": "daad", "ط": "taa", "ظ": "zaa", "ع": "ain",
    "غ": "ghain", "ف": "fa", "ق": "qaf", "ك": "kaf", "ل": "lam", "م": "meem",
    "ن": "noon", "ه": "ha", "ة": "taa_marbuta", "ي": "ya", "و": "waw",
}

UNITS = [
    ("الوحدة الأولى", ["ا", "د", "ذ", "ر", "ز", "و"]),
    ("الوحدة الثانية", ["ب", "ت", "ث", "ط", "ظ"]),
    ("الوحدة الثالثة", ["ج", "ح", "خ", "ع", "غ"]),
    ("الوحدة الرابعة", ["س", "ش", "ص", "ض"]),
    ("الوحدة الخامسة", ["ك", "ل", "ن", "ف", "ق", "م"]),
    ("الوحدة السادسة", ["ه", "ة", "ي"]),
]

# teaching order of forms within a lesson
FORM_ORDER = [F_ISO, F_INI, F_MED, F_FIN]
FORM_LABEL = {
    F_ISO: "منفصلة", F_INI: "بداية", F_MED: "وسط", F_FIN: "نهاية",
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_catalog(catalog, unmapped):
    print("Letter shape catalog (per required letter):\n")
    all_letters = [c for _, letters in UNITS for c in letters]
    missing = []
    for u_title, letters in UNITS:
        print(f"  {u_title}")
        for c in letters:
            forms = catalog.get(c, {})
            if not forms:
                missing.append(c)
                print(f"    {c} {LETTER_NAMES[c]:<16} !! NO SHAPES FOUND")
                continue
            parts = []
            for f in FORM_ORDER:
                if f in forms:
                    tag = "*" if forms[f]["is_better"] else ""
                    parts.append(f"{FORM_LABEL[f]}{tag}")
            print(f"    {c} {LETTER_NAMES[c]:<16} {len(forms)} shapes: "
                  f"{'، '.join(parts)}")
        print()

    extra = sorted(set(catalog) - set(all_letters))
    if extra:
        print(f"Letters on disk but not in the curriculum (ignored): "
              f"{' '.join(extra)}")
    if unmapped:
        print(f"\nUnclassified files ({len(unmapped)}) -- skipped:")
        for n in unmapped:
            print(f"  - {n}")
    if missing:
        print(f"\n!! MISSING shapes for: {' '.join(missing)}")
    print(f"\n* = uses the 'letters better version' redraw.")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed(session, catalog):
    # --- wipe previously-seeded copying content, idempotently ---
    # Our copying units are the book-less beginner units; deleting them
    # cascades to lessons/exercises/copying_items and frees the letters.
    old_units = session.scalars(
        select(Unit).where(Unit.book_id.is_(None), Unit.level == Level.beginner)
    ).all()
    for u in old_units:
        session.delete(u)
    session.flush()
    session.query(Letter).delete()
    session.flush()

    # --- create Letter rows (one per shape) ---
    letters_by_key: dict[tuple[str, LetterForm], Letter] = {}
    n_letters = 0
    for arabic, forms in catalog.items():
        slug = LETTER_SLUGS.get(arabic)
        if slug is None:
            continue                                # not a curriculum letter
        for form, src in forms.items():
            shape = load_shape(src["path"], src["is_better"])
            letter = Letter(
                code=f"{slug}_{form.value}",
                arabic_letter=arabic,
                form=form,
                strokes=shape["strokes"],
                stroke_count=shape["stroke_count"],
                point_count=shape["point_count"],
                aspect_ratio=shape["aspect_ratio"],
                source_format=shape["source_format"],
                source_file=src["path"].name,
            )
            session.add(letter)
            letters_by_key[(arabic, form)] = letter
            n_letters += 1
    session.flush()

    # --- build the 6 units -> lesson per letter -> copying exercise -> items ---
    n_units = n_lessons = n_items = 0
    for u_idx, (u_title, letters) in enumerate(UNITS):
        present = ["، ".join(LETTER_NAMES[c] for c in letters if c in catalog)]
        unit = Unit(
            book_id=None,
            unit_number=u_idx + 1,
            level=Level.beginner,
            title=u_title,
            description="نسخ الحروف: " + " ".join(letters),
            sort_order=u_idx,
            is_published=True,
        )
        session.add(unit)
        n_units += 1

        for l_idx, arabic in enumerate(letters):
            forms = catalog.get(arabic, {})
            if not forms:
                continue                            # skip letters with no shapes
            name = LETTER_NAMES[arabic]
            lesson = Lesson(
                unit=unit,
                title=f"حرف {name}",
                description=f"نسخ حرف {name} ({arabic}) بجميع أشكاله",
                sort_order=l_idx,
                is_published=True,
            )
            session.add(lesson)
            n_lessons += 1

            exercise = Exercise(
                lesson=lesson,
                writing_type=WritingType.copying,
                title=f"نسخ حرف {name}",
                instructions=f"انسخ حرف {name} ({arabic}) في كل شكل من أشكاله.",
                sort_order=0,
                content=None,
            )
            session.add(exercise)

            sort = 0
            for form in FORM_ORDER:
                if form not in forms:
                    continue
                letter = letters_by_key[(arabic, form)]
                session.add(CopyingItem(
                    exercise=exercise, letter=letter, sort_order=sort,
                ))
                sort += 1
                n_items += 1

    return {
        "letters": n_letters, "units": n_units,
        "lessons": n_lessons, "copying_items": n_items,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    dry_run = "--dry-run" in sys.argv

    if not NORMALIZED_DIR.is_dir():
        print(f"!! not found: {NORMALIZED_DIR}")
        sys.exit(1)

    catalog, unmapped = build_catalog()
    print_catalog(catalog, unmapped)

    if dry_run:
        print("\n(dry run -- no database changes)")
        return

    init_db()
    session = SessionLocal()
    try:
        totals = seed(session, catalog)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    from database import engine
    print(f"\nSeeded copying content into {engine.url}")
    print(f"  letters (shapes) : {totals['letters']}")
    print(f"  units            : {totals['units']}")
    print(f"  lessons          : {totals['lessons']}")
    print(f"  copying items    : {totals['copying_items']}")


if __name__ == "__main__":
    main()
