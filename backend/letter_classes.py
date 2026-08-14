# -*- coding: utf-8 -*-
"""
Letter (Arabic char + form)  <->  handwriting-model class name.

The classifier in `handwriting_model.py` has 107 output classes named
`<PREFIX>_<token>`, e.g. `E_ay`, `I_ba`, `M_la`. The backend, on the other
hand, stores letters as (Letter.arabic_letter, Letter.form). This module is the
translation between the two — it is what lets the grader ask the question the
exercise actually asks: *"is this the medial ع the student was told to write?"*
rather than just *"what letter is this?"*.

Where the table comes from
--------------------------
NOT guessed. The tokens were recovered from the training notebook
(`letters drawing/FinalDataWithAug (4).ipynb`), whose dataset was laid out as

    FinalDataLetter/<LetterName>/<sample>_<CLASS>.inkml

so every extraction log line pairs a class token with its letter folder:
    aa->Alef  ay->Ain  ba->Ba  da->Dal  de->Daad  dh->zal  fa->fa  ha->Haa
    he->Ha    ja->Geem ka->qaf ke->kaf  kh->kha   la->Lam  ma->meem na->noon
    ra->ra    sa->saad se->seen sh->sheen ta->Ta  th->Tha  to->Taa  wa->waw
    ya->Ya    za->zay  zha->Zaa gh->Ghain

Cross-check that confirms it: the 107 classes contain no `B_`/`M_` form for
aa, da, dh, ra, za, wa, tee, ae, ee — exactly the letters that do not connect
to the following letter in Arabic script (ا د ذ ر ز و ة ...). A wrong token
table would not reproduce that pattern.

Known gaps (deliberately not faked)
-----------------------------------
* `I_gh` does not exist — the training set has no isolated غ. Isolated غ
  therefore has no target class; `class_for()` returns None and the grader
  falls back to shape-only scoring instead of failing a correct answer.
* `ae`, `ee`, `hh`, `al` are real classes we could not attribute with
  certainty (most likely أ / ى / ء / لا). They are left out on purpose: none
  of them is in the app's 6-unit alphabet, and a wrong guess here would mark
  correct letters wrong. If you identify them, add them to `TOKEN_FOR_LETTER`.
"""
from __future__ import annotations

from typing import Optional

# --- form -> class prefix ---------------------------------------------------
# B = Begin (initial), M = Middle (medial), E = End (final), I = Isolated.
PREFIX_FOR_FORM: dict[str, str] = {
    "initial": "B",
    "medial": "M",
    "final": "E",
    "isolated": "I",
}
FORM_FOR_PREFIX: dict[str, str] = {v: k for k, v in PREFIX_FOR_FORM.items()}

# --- Arabic letter -> class token (see provenance above) --------------------
TOKEN_FOR_LETTER: dict[str, str] = {
    "ا": "aa",     # Alef
    "ب": "ba",     # Ba
    "ت": "ta",     # Ta
    "ث": "th",     # Tha
    "ج": "ja",     # Geem
    "ح": "ha",     # Haa      (note: ha = ح, he = ه — easy to swap, don't)
    "خ": "kh",     # kha
    "د": "da",     # Dal
    "ذ": "dh",     # zal
    "ر": "ra",     # ra
    "ز": "za",     # zay
    "س": "se",     # seen     (note: se = س, sa = ص)
    "ش": "sh",     # sheen
    "ص": "sa",     # saad
    "ض": "de",     # Daad
    "ط": "to",     # Taa
    "ظ": "zha",    # Zaa
    "ع": "ay",     # Ain
    "غ": "gh",     # Ghain    (no isolated form in the training set)
    "ف": "fa",     # fa
    "ق": "ka",     # qaf      (note: ka = ق, ke = ك — easy to swap, don't)
    "ك": "ke",     # kaf
    "ل": "la",     # Lam
    "م": "ma",     # meem
    "ن": "na",     # noon
    "ه": "he",     # Ha
    "ة": "tee",    # Taa marbuta (final/isolated only, as in the class list)
    "و": "wa",     # waw
    "ي": "ya",     # Ya
}
LETTER_FOR_TOKEN: dict[str, str] = {v: k for k, v in TOKEN_FOR_LETTER.items()}

# Arabic names for the feedback card ("رسمتَ ما يشبه حرف القاف").
LETTER_NAME_AR: dict[str, str] = {
    "ا": "الألف", "ب": "الباء", "ت": "التاء", "ث": "الثاء", "ج": "الجيم",
    "ح": "الحاء", "خ": "الخاء", "د": "الدال", "ذ": "الذال", "ر": "الراء",
    "ز": "الزاي", "س": "السين", "ش": "الشين", "ص": "الصاد", "ض": "الضاد",
    "ط": "الطاء", "ظ": "الظاء", "ع": "العين", "غ": "الغين", "ف": "الفاء",
    "ق": "القاف", "ك": "الكاف", "ل": "اللام", "م": "الميم", "ن": "النون",
    "ه": "الهاء", "ة": "التاء المربوطة", "و": "الواو", "ي": "الياء",
}

FORM_NAME_AR: dict[str, str] = {
    "isolated": "منفصلة",
    "initial": "في أول الكلمة",
    "medial": "في وسط الكلمة",
    "final": "في آخر الكلمة",
}


# Letters that do not join to the letter that follows them, so they have no
# initial or medial shape at all — and correspondingly no B_/M_ class in the
# checkpoint. Naming one would invent a target the model can never predict.
NON_CONNECTING: frozenset[str] = frozenset("ا د ذ ر ز و ة".split())

# Shapes that exist in Arabic but were not in the training set. Kept explicit
# so `class_for` returns None instead of a class that isn't in label_map.npy.
UNTRAINED_CLASSES: frozenset[str] = frozenset({"I_gh"})   # isolated غ


def _form_value(form) -> str:
    """Accept a LetterForm enum, its .value, or a plain string."""
    return getattr(form, "value", form) or ""


def class_for(arabic_letter: str, form) -> Optional[str]:
    """(ع, medial) -> "M_ay". None when this shape has no trained class.

    Every string this returns is guaranteed to be one of the 107 classes in
    label_map.npy — the grader relies on that to tell "the student drew the
    wrong thing" apart from "we cannot judge this shape".
    """
    letter = (arabic_letter or "").strip()
    form_value = _form_value(form)

    token = TOKEN_FOR_LETTER.get(letter)
    prefix = PREFIX_FOR_FORM.get(form_value)
    if not token or not prefix:
        return None
    if letter in NON_CONNECTING and form_value in ("initial", "medial"):
        return None

    class_name = f"{prefix}_{token}"
    return None if class_name in UNTRAINED_CLASSES else class_name


def parse_class(class_name: str) -> Optional[tuple[str, str]]:
    """"M_ay" -> ("ع", "medial"). None for the unattributed classes."""
    if not class_name or "_" not in class_name:
        return None
    prefix, token = class_name.split("_", 1)
    letter = LETTER_FOR_TOKEN.get(token)
    form = FORM_FOR_PREFIX.get(prefix)
    if not letter or not form:
        return None
    return letter, form


def describe_class(class_name: str) -> str:
    """Arabic description of a predicted class, for the feedback card."""
    parsed = parse_class(class_name)
    if parsed is None:
        return class_name
    letter, form = parsed
    return f"{LETTER_NAME_AR.get(letter, letter)} {FORM_NAME_AR.get(form, '')}".strip()


def describe_letter(arabic_letter: str, form) -> str:
    """Arabic description of the letter the exercise asked for."""
    name = LETTER_NAME_AR.get((arabic_letter or "").strip(), arabic_letter)
    return f"{name} {FORM_NAME_AR.get(_form_value(form), '')}".strip()
