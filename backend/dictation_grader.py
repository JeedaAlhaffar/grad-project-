# -*- coding: utf-8 -*-
"""Reference-based grading for the dictation (إملاء) exercises.

The student hears a clip and writes what they heard; we hold the exact script.
Because the target text is known, this is not grammatical error correction — no
correction has to be inferred. The work is entirely in three places:

    align   which student word corresponds to which reference word
    type    what kind of mistake it is, in terms a teacher would use
    score   how much that mistake should cost

The method and the experiments behind it are in
`advanced_level_dictation_assessment.ipynb` at the project root. The short
version of why this replaced the previous `difflib` implementation:

  * difflib has no model for a merged or split word, so `الآباء والأجداد`
    written as one token was scored as two separate errors and mis-blamed the
    words after it. On simulated attempts from the weakest students, word
    localisation accuracy was 0.857 against 0.929 for the aligner here — the
    old behaviour failed hardest on the students who need feedback most.
  * every error was reported as "خطأ إملائي" with no type, so the feedback
    could not tell a student *why* a word was wrong.
  * a word with one wrong letter scored the same as a blank.

Errors are labelled with ARETA tags (Belkebir & Habash, CoNLL 2021), the same
tagset `gec_service.py` already uses, so the app renders them with no new UI.

No ML dependency: this module is pure Python plus an optional C accelerator.
"""
from __future__ import annotations

from collections import Counter

# ---------------------------------------------------------------------------
# Edit distance. python-Levenshtein is in requirements.txt but only under the
# optional GEC section, and dictation must keep working on a deployment that
# installs neither the models nor the ML stack.
# ---------------------------------------------------------------------------
try:
    from Levenshtein import distance as _distance, editops as _editops

    _HAVE_LEVENSHTEIN = True
except ImportError:  # pragma: no cover - exercised only on a minimal install
    _HAVE_LEVENSHTEIN = False

    def _distance(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def _editops(a: str, b: str):
        """Backtracked Levenshtein ops in the ('replace'|'delete'|'insert', i, j) form."""
        n, m = len(a), len(b)
        d = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                              d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
        ops, i, j = [], n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
                i, j = i - 1, j - 1
            elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
                ops.append(("replace", i - 1, j - 1)); i, j = i - 1, j - 1
            elif i > 0 and d[i][j] == d[i - 1][j] + 1:
                ops.append(("delete", i - 1, j)); i -= 1
            else:
                ops.append(("insert", i, j - 1)); j -= 1
        return ops[::-1]


# ---------------------------------------------------------------------------
# Arabic constants
# ---------------------------------------------------------------------------
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_YA = str.maketrans({"ى": "ي"})
_TA = str.maketrans({"ة": "ه"})
_DIACRITICS = "".join(chr(c) for c in range(0x064B, 0x0653)) + "ـٰ"

# A hamza error is not only أ<->إ: the commonest form is dropping the hamza and
# writing the bare carrier (طائرة -> طايرة). The seats ا و ي therefore count as
# one side of a hamza confusion, while requiring a true carrier on at least one
# side keeps أ->ع in perceptual confusion where it belongs.
_HAMZA_CARRIERS = set("أإآءؤئ")
_HAMZA_SEATS = set("أإآءؤئاوي")
_TA_SET = set("ةته")
_ALEF_SET = set("ىاي")

# ARETA tag -> Arabic label. Kept identical to gec_service._TAG_AR so the two
# detectors never describe the same mistake differently.
TAG_AR = {
    "OH": "همزة",
    "OT": "تاء مربوطة / مفتوحة",
    "OA": "ألف مقصورة / ياء",
    "ON": "نون / تنوين",
    "OC": "ترتيب الحروف",
    "OR": "حرف خاطئ",
    "OD": "حرف زائد",
    "OM": "حرف ناقص",
    "OO": "خطأ إملائي",
    "XT": "كلمة زائدة",
    "XM": "كلمة ناقصة",
    "MG": "كلمتان مدمجتان",
    "SP": "كلمة مفصولة خطأً",
}

# Errors the student could have avoided by knowing a spelling rule, as opposed
# to ones that mean they did not hear the word. The two need different
# remediation, and the split is surfaced as ErrorItem.subcategory.
_SOFT_TAGS = {"OH", "OT", "OA", "ON"}

# What each error class costs, as a fraction of one word. Soft orthographic
# slips are penalised less than losing the word altogether, which mirrors how
# إملاء is marked in class rather than how edit distance happens to fall out.
#
# These weights are a CURRICULUM decision, not a tuning constant: the notebook
# shows the ranking of scoring formulas changes completely depending on them.
# Override per deployment rather than editing the code.
DICTATION_SEVERITY = {
    "OH": 0.5, "OT": 0.5, "OA": 0.5, "ON": 0.4,        # soft, conventional
    "OM": 0.7, "OD": 0.7, "OC": 0.7, "OR": 0.8,        # letter-level
    "OO": 0.8, "XT": 0.7,
    "XM": 1.0, "MG": 0.9, "SP": 0.9,                   # structural
}

_MERGE_THRESHOLD = 0.72

_ADVICE = {
    "OH": "راجع قواعد الهمزة: همزة القطع وهمزة الوصل ومواضع كتابتها",
    "OT": "راجع الفرق بين التاء المربوطة والتاء المفتوحة والهاء",
    "OA": "راجع الفرق بين الألف المقصورة والألف والياء في آخر الكلمة",
    "ON": "راجع كتابة التنوين وألف التنوين",
    "OM": "انتبه إلى الحروف الناقصة، وخاصة اللام الشمسية وحروف المد",
    "OD": "انتبه إلى الحروف الزائدة",
    "OR": "أعد الاستماع بتركيز؛ بعض الحروف المتشابهة تحتاج تمييزاً دقيقاً",
    "OC": "انتبه إلى ترتيب الحروف داخل الكلمة",
    "XM": "استمع مرة أخرى وتأكد من كتابة كل الكلمات",
    "XT": "انتبه إلى الكلمات الزائدة",
    "MG": "انتبه إلى الفصل بين الكلمات",
    "SP": "انتبه إلى عدم فصل الكلمة الواحدة",
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def strip_diacritics(s: str) -> str:
    return "".join(c for c in s if c not in _DIACRITICS)


def normalize(s: str) -> str:
    """Aggressive normalisation, for MATCHING ONLY.

    This erases the hamza / ألف مقصورة / تاء مربوطة distinctions, which are
    precisely the errors we are here to detect. It must never decide whether a
    student was right — only which reference word a student word belongs to,
    so that مستويئت still pairs with مستويات instead of reading as a deletion
    plus an insertion. Every correctness judgement uses the surface forms.
    """
    return strip_diacritics(s).translate(_ALEF).translate(_YA).translate(_TA)


def _similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na and not nb:
        return 1.0
    return 1.0 - _distance(na, nb) / max(len(na), len(nb), 1)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
class _Edit:
    __slots__ = ("op", "ref", "stu", "ref_i", "stu_i")

    def __init__(self, op, ref, stu, ref_i, stu_i):
        self.op, self.ref, self.stu = op, ref, stu
        self.ref_i, self.stu_i = ref_i, stu_i


def _needleman_wunsch(ref, stu, gap=-0.55):
    """Word-level alignment scored on graded similarity.

    A binary match score would read a misspelt word as delete-plus-insert and
    lose the pairing; grading the score keeps مستويئت attached to مستويات.
    """
    n, m = len(ref), len(stu)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
        bt[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + (2.0 * _similarity(ref[i - 1], stu[j - 1]) - 1.0)
            up = dp[i - 1][j] + gap
            left = dp[i][j - 1] + gap
            best = max(diag, up, left)
            dp[i][j] = best
            bt[i][j] = 0 if best == diag else (1 if best == up else 2)

    out, i, j = [], n, m
    while i > 0 or j > 0:
        move = bt[i][j] if (i > 0 and j > 0) else (1 if j == 0 else 2)
        if move == 0:
            op = "equal" if ref[i - 1] == stu[j - 1] else "sub"
            out.append(_Edit(op, [ref[i - 1]], [stu[j - 1]], i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 1:
            out.append(_Edit("del", [ref[i - 1]], [], i - 1, j))
            i -= 1
        else:
            out.append(_Edit("ins", [], [stu[j - 1]], i, j - 1))
            j -= 1
    return out[::-1]


def align(ref: list[str], stu: list[str], threshold: float = _MERGE_THRESHOLD):
    """Align, then recover merged and split words as single errors.

    Operator order is easy to get wrong here. For `الآباء والأجداد` written as
    `الآباءوالأجداد` the aligner does NOT emit the intuitive
    substitute-then-delete: similarity favours pairing والأجداد with the merged
    token (0.57) over الآباء (0.43), so the real sequence is
    delete-then-substitute. The scan therefore works on whole runs of
    consecutive non-equal edits and tests concatenation in both directions
    rather than matching a fixed operator order.
    """
    edits = _needleman_wunsch(ref, stu)
    out, i = [], 0
    while i < len(edits):
        if edits[i].op == "equal":
            out.append(edits[i])
            i += 1
            continue
        j = i
        while j < len(edits) and edits[j].op != "equal":
            j += 1
        run = edits[i:j]
        run_ref = [w for e in run for w in e.ref]
        run_stu = [w for e in run for w in e.stu]
        ref_i = next((e.ref_i for e in run if e.ref), run[0].ref_i)
        stu_i = next((e.stu_i for e in run if e.stu), run[0].stu_i)

        if len(run_ref) > 1 and len(run_stu) == 1 and \
                _similarity("".join(run_ref), run_stu[0]) >= threshold:
            out.append(_Edit("merge", run_ref, run_stu, ref_i, stu_i))
        elif len(run_ref) == 1 and len(run_stu) > 1 and \
                _similarity(run_ref[0], "".join(run_stu)) >= threshold:
            out.append(_Edit("split", run_ref, run_stu, ref_i, stu_i))
        else:
            out.extend(run)
        i = j
    return out


# ---------------------------------------------------------------------------
# Error typing
# ---------------------------------------------------------------------------
def classify(ref_word: str, stu_word: str) -> str:
    """ARETA tag for one misspelt word, from its character-level edits."""
    if ref_word == stu_word:
        return ""

    # Transposition is checked directly, before any edit script is computed.
    # Reading it off the edit ops is fragile: swapping two letters costs 2
    # either way, so whether it appears as insert+delete or as two replaces is
    # a tie-break detail, and the C implementation of editops and a pure-Python
    # one do not resolve that tie the same way. Testing for the swap itself
    # makes the tag independent of which backend is installed.
    if len(ref_word) == len(stu_word) and sorted(ref_word) == sorted(stu_word):
        for k in range(len(ref_word) - 1):
            if ref_word[k] != ref_word[k + 1] and \
                    ref_word[:k] + ref_word[k + 1] + ref_word[k] + ref_word[k + 2:] == stu_word:
                return "OC"

    ops = _editops(ref_word, stu_word)
    tags = []
    for op, i, j in ops:
        rc = ref_word[i] if i < len(ref_word) else ""
        sc = stu_word[j] if j < len(stu_word) else ""
        last = i >= len(ref_word) - 1
        if op == "replace":
            if (rc in _HAMZA_CARRIERS and sc in _HAMZA_SEATS) or \
                    (rc in _HAMZA_SEATS and sc in _HAMZA_CARRIERS):
                tags.append("OH")
            elif last and rc in _TA_SET and sc in _TA_SET:
                tags.append("OT")
            elif last and rc in _ALEF_SET and sc in _ALEF_SET:
                tags.append("OA")
            else:
                tags.append("OR")
        elif op == "delete":
            tags.append("ON" if (last and rc == "ا") else "OM")
        else:
            tags.append("ON" if (j >= len(stu_word) - 1 and sc == "ا") else "OD")
    return tags[0] if tags else "OO"


def severity_class(tag: str) -> str:
    """"soft" = a spelling convention; "hard" = the word itself was not heard."""
    return "soft" if tag in _SOFT_TAGS else "hard"


# ---------------------------------------------------------------------------
# The grader
# ---------------------------------------------------------------------------
class DictationGrader:
    """Grade one dictation attempt against its reference script."""

    def __init__(self, severity: dict | None = None,
                 merge_threshold: float = _MERGE_THRESHOLD):
        self.severity = dict(severity or DICTATION_SEVERITY)
        self.merge_threshold = merge_threshold

    # -- public ------------------------------------------------------------
    def grade(self, reference: str, student: str) -> dict:
        # Both sides are dediacritized before anything else, and this is not a
        # nicety — it is the difference between the level working and not.
        #
        # `routers/attempts.py` passes `content["text"]`, which for the audio
        # exercises is the FULLY DIACRITIZED script (مِمَّا يَزِيدُ الْأَمْرَ تَعْقِيداً).
        # A student typing on any ordinary Arabic keyboard produces no tashkeel,
        # so a perfect answer used to match zero words and score 0/100 — every
        # one of the 25 audio dictation items was ungradeable.
        #
        # إملاء at this level tests orthography, not vowelling: the marks are
        # not dictated and cannot be typed. Stripping them from both sides makes
        # the grade depend on the letters, which is what is being assessed, and
        # keeps the grader correct whether the caller hands it `text` or
        # `text_plain`.
        ref_w = [w for w in strip_diacritics(reference or "").split() if w]
        stu_w = [w for w in strip_diacritics(student or "").split() if w]
        if not ref_w:
            return {"score": 0.0, "errors": [], "summary": "لا يوجد نص مرجعي",
                    "strengths": [], "suggestions": []}

        errors, penalty = [], 0.0
        for edit in align(ref_w, stu_w, self.merge_threshold):
            if edit.op == "equal":
                continue
            card, cost = self._card(edit)
            errors.append(card)
            penalty += cost

        score = round(100.0 * max(0.0, 1.0 - penalty / len(ref_w)), 1)
        return {
            "score": score,
            "errors": errors,
            "summary": self._summary(score),
            "strengths": self._strengths(score, errors),
            "suggestions": self._suggestions(errors),
        }

    # -- internals ---------------------------------------------------------
    def _card(self, e) -> tuple[dict, float]:
        if e.op == "sub":
            ref_w, stu_w = e.ref[0], e.stu[0]
            tag = classify(ref_w, stu_w) or "OO"
            return self._item(
                f'كتبت "{stu_w}" والصواب "{ref_w}"', ref_w, stu_w, tag, "spelling",
            ), self.severity.get(tag, 0.8)
        if e.op == "del":
            return self._item(
                f'نسيت كلمة "{e.ref[0]}"', e.ref[0], None, "XM", "spelling",
            ), self.severity["XM"]
        if e.op == "ins":
            return self._item(
                f'كلمة زائدة "{e.stu[0]}"', None, e.stu[0], "XT", "spelling",
            ), self.severity["XT"]
        if e.op == "merge":
            joined, separate = e.stu[0], " ".join(e.ref)
            return self._item(
                f'دمجت كلمتين: "{joined}" والصواب "{separate}"',
                separate, joined, "MG", "merge_split",
            ), self.severity["MG"]
        joined, separate = e.ref[0], " ".join(e.stu)
        return self._item(
            f'فصلت كلمة واحدة: "{separate}" والصواب "{joined}"',
            joined, separate, "SP", "merge_split",
        ), self.severity["SP"]

    @staticmethod
    def _item(message, correction, span, tag, category) -> dict:
        # Matches schemas.ErrorItem, including the optional tag/category/
        # subcategory fields gec_service already populates.
        return {
            "message": message,
            "correction": correction,
            "span": span,
            "type": "spelling",
            "tag": tag,
            "category": category,
            "subcategory": severity_class(tag),
        }

    @staticmethod
    def _summary(score: float) -> str:
        if score >= 90:
            return "ممتاز! إملاء صحيح"
        if score >= 75:
            return "أحسنت، أخطاء قليلة"
        if score >= 50:
            return "محاولة جيدة، انتبه للأخطاء"
        return "تحتاج إلى مزيد من التدريب على الإملاء"

    @staticmethod
    def _strengths(score: float, errors: list[dict]) -> list[str]:
        if not errors:
            return ["إملاء خالٍ من الأخطاء"]
        out = []
        if score >= 75:
            out.append("كتابة صحيحة لمعظم الكلمات")
        if all(e["subcategory"] == "soft" for e in errors):
            # Worth saying out loud: they heard every word correctly, and only
            # the writing conventions let them down.
            out.append("سمعت الكلمات بشكل صحيح، والخطأ في قواعد الكتابة فقط")
        return out

    @staticmethod
    def _suggestions(errors: list[dict]) -> list[str]:
        if not errors:
            return ["أحسنت، حافظ على هذا المستوى"]
        # Advise on the mistake actually made most often, not a generic tip.
        top = Counter(e["tag"] for e in errors).most_common(1)[0][0]
        advice = _ADVICE.get(top)
        out = [advice] if advice else []
        out.append("راجع الكلمات التي أخطأت فيها وأعد كتابتها")
        return out


_DEFAULT = DictationGrader()


def grade(reference: str, student: str) -> dict:
    """Module-level convenience wrapper around the default grader."""
    return _DEFAULT.grade(reference, student)
