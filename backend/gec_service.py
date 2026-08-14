# -*- coding: utf-8 -*-
"""
Arabic error detection (تصحيح الأخطاء) — GEC + ARETA wrapper.

This is the *detection* half of the AI stack and it runs BEFORE the AES
scoring: every text the student writes is first corrected and its errors are
extracted, and only then is the essay graded (see `ai_service`).

    student text
      ├─► GEC tagger (CAMeL text-editing: 3× nopnx + 1× pnx)  ─► corrected text
      └─► ARETA aligner (original vs corrected)               ─► tagged errors
                                                                (OH / XC / PM …)

The engine itself lives outside the backend, in `../arabic_gec_system`
(`gec_areta.py`, installed by the model team). We only:

  * import it lazily and keep the two BERT taggers loaded for the process,
  * serialise the ARETA step — it shells out to `annotate_err_type_ar.py` and
    writes fixed temp files inside its own folder, so two concurrent requests
    would corrupt each other,
  * bound it with a timeout and fall back to a word-level diff so a slow or
    missing ARETA install can never hang a student's request,
  * translate the ARETA tags into the Arabic error cards the app renders.

Latency notes (why it is structured this way):
  * models load once, on the first call / at server warm-up — not per request;
  * identical texts are served from a small LRU cache (re-submits, retries);
  * the caller runs this concurrently with the AES/relevance models, so the
    ARETA subprocess overlaps with the scoring instead of adding to it.
"""
from __future__ import annotations

import os

# Defensive: gec_areta -> gec.tag -> transformers.Trainer is what drags
# TensorFlow in (see config.py). Set here too, so importing gec_service on its
# own — a test script, a worker — still gets the lean process. gec_areta builds
# the ARETA subprocess env from os.environ, so this reaches the analyser too.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
from typing import Optional

_log = logging.getLogger("gec")

# Folder holding gec_areta.py + code/ + models/ (the unpacked arabic_gec_system).
_DEFAULT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "arabic_gec_system")
)
GEC_DIR = os.environ.get("GEC_DIR") or _DEFAULT_DIR

# Hard ceiling for one detection (correction + tagging), in seconds.
GEC_TIMEOUT = float(os.environ.get("GEC_TIMEOUT", "60"))
# Set GEC_ARETA=0 to skip the (slow) tag analyser and use the diff fallback.
USE_ARETA = os.environ.get("GEC_ARETA", "1") == "1"
MAX_ERRORS = int(os.environ.get("GEC_MAX_ERRORS", "25"))

_module = None
_load_lock = threading.Lock()
# One worker == the ARETA temp files are never touched by two requests at once.
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gec")
_cache: dict[str, dict] = {}
_CACHE_MAX = 128

# --- ARETA robustness -------------------------------------------------------
# The analyser runs in a *fresh* python subprocess that imports sklearn -> scipy.
# While the four checkpoints are loading, that import has been seen to die with
# MemoryError on a 16 GB box — a transient, timing-dependent failure on the
# first request. Retry it before dropping to the untagged diff.
ARETA_ATTEMPTS = int(os.environ.get("GEC_ARETA_ATTEMPTS", "2"))
ARETA_RETRY_WAIT = float(os.environ.get("GEC_ARETA_RETRY_WAIT", "1.5"))
# ...but if ARETA is genuinely broken (camel-tools data missing, bad install),
# retrying every request forever would be worse than the fallback. After this
# many consecutive failures it is declared unavailable for the process and the
# diff becomes the normal, cacheable path.
ARETA_GIVE_UP_AFTER = int(os.environ.get("GEC_ARETA_GIVE_UP_AFTER", "3"))
_areta_failures = 0
_areta_broken = False


def _areta_expected() -> bool:
    """True while a *tagged* result is still achievable.

    When this is true, an untagged result means "something went wrong", not
    "this is the configured behaviour" — and must therefore not be cached.
    """
    return USE_ARETA and not _areta_broken


# --- Arabic labels for the ARETA tags --------------------------------------
# (tag -> human readable Arabic name; the coarse category comes from gec_areta)
_TAG_AR = {
    "OH": "همزة",
    "OT": "تاء مربوطة / مفتوحة",
    "OA": "ألف مقصورة / ياء",
    "OW": "ألف فارقة",
    "ON": "نون / تنوين",
    "OS": "حرف مدّ ناقص",
    "OG": "حرف مدّ زائد",
    "OC": "ترتيب الحروف",
    "OR": "حرف خاطئ",
    "OD": "حرف زائد",
    "OM": "حرف ناقص",
    "OO": "خطأ إملائي",
    "MI": "تصريف الكلمة",
    "MT": "زمن الفعل",
    "MO": "خطأ صرفي",
    "XC": "علامة الإعراب",
    "XF": "التعريف والتنكير",
    "XG": "المطابقة في التذكير والتأنيث",
    "XN": "المطابقة في العدد",
    "XT": "كلمة زائدة",
    "XM": "كلمة ناقصة",
    "XO": "خطأ نحوي",
    "SW": "اختيار الكلمة",
    "SF": "أداة الربط",
    "SO": "خطأ في المعنى",
    "PC": "علامة ترقيم خاطئة",
    "PT": "علامة ترقيم زائدة",
    "PM": "علامة ترقيم ناقصة",
    "PO": "علامات الترقيم",
    "MG": "كلمتان مدمجتان",
    "SP": "كلمة مفصولة خطأً",
    "unk": "خطأ",
}

# tag -> coarse category (mirrors gec_areta.AR_TAG_CATEGORIES).
_TAG_CATEGORY = {
    "OH": "spelling", "OT": "spelling", "OA": "spelling", "OW": "spelling",
    "ON": "spelling", "OS": "spelling", "OG": "spelling", "OC": "spelling",
    "OR": "spelling", "OD": "spelling", "OM": "spelling", "OO": "spelling",
    "MI": "grammar", "MT": "grammar", "MO": "grammar",
    "XC": "grammar", "XF": "grammar", "XG": "grammar", "XN": "grammar",
    "XT": "grammar", "XM": "grammar", "XO": "grammar",
    "SW": "grammar", "SF": "grammar", "SO": "grammar",
    "PC": "punctuation", "PT": "punctuation", "PM": "punctuation",
    "PO": "punctuation",
    "MG": "merge_split", "SP": "merge_split",
}

# gec_areta category -> the `type` the Flutter error chip understands.
_TYPE_OF_CATEGORY = {
    "spelling": "spelling",
    "grammar": "grammar",
    "punctuation": "punctuation",
    "merge_split": "spelling",
    "unknown": "spelling",
}


def _resolve_tag(tag: str) -> tuple[str, str]:
    """(Arabic label, category) for a tag — ARETA also emits compounds ("MI+OH")."""
    parts = [p for p in (tag or "").split("+") if p]
    known = [p for p in parts if p in _TAG_CATEGORY]
    if not known:
        return _TAG_AR["unk"], "unknown"
    label = " + ".join(_TAG_AR.get(p, p) for p in known)
    return label, _TAG_CATEGORY[known[0]]

_PUNCT = set(".،,;:؛!؟?\"'()[]{}-—…")
_PUNCT_RE = re.compile(r"([.,،؛;:!؟?«»\"'()\[\]{}])")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,،؛;:!؟?»)\]}])")
UNK = "[UNK]"


def _normalize_spacing(text: str) -> str:
    """Detach punctuation from words, exactly like the GEC tagger does.

    Without this the student's "الحالي." and the model's "الحالي ." look like a
    spelling error plus a missing full stop — pure tokenisation noise. Both
    sides are normalised, so only real edits survive the comparison.
    """
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(r" \1 ", text)).strip()


def _prettify(text: str) -> str:
    """Re-attach punctuation for display ("الحالي ." -> "الحالي.")."""
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text).strip()


def _restore_unk(original: str, corrected: str) -> str:
    """Put back words the tagger's vocabulary swallowed.

    Words the BertTokenizer doesn't know (e.g. "سيفاً") come back as [UNK];
    showing that to a student as "the correct spelling" would be nonsense, so
    every [UNK] is replaced by the word it was aligned with (or dropped).
    """
    if UNK not in corrected:
        return corrected
    import difflib

    a, b = original.split(), corrected.split()
    out = list(b)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if op == "equal":
            continue
        for k in range(j1, j2):
            if out[k] != UNK:
                continue
            src = i1 + (k - j1)
            out[k] = a[src] if src < i2 else ""
    return " ".join(w for w in out if w)


class GECUnavailable(Exception):
    """The GEC engine could not be imported / loaded."""


# ===========================================================================
# engine loading
# ===========================================================================
def _ensure_areta_importable() -> None:
    """Make ARETA's `scripts/` a real package.

    It ships without `__init__.py`, so it is only a *namespace* package — and a
    regular `scripts` package installed in site-packages (casanova ships one)
    silently wins the import, and the analyser subprocess dies with
    "No module named 'scripts.annotation'". Adding the empty init files makes
    the local package win by path order. Idempotent, safe to run on any box.
    """
    root = os.path.join(GEC_DIR, "code", "areta", "scripts")
    if not os.path.isdir(root):
        return
    try:
        for dirpath, _dirnames, _files in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            init = os.path.join(dirpath, "__init__.py")
            if not os.path.exists(init):
                with open(init, "w", encoding="utf-8"):
                    pass
    except OSError:
        pass          # read-only deployment: the diff fallback still works


def _load():
    """Import `gec_areta` from GEC_DIR once and keep it for the process."""
    global _module
    if _module is None:
        with _load_lock:
            if _module is None:
                if not os.path.isfile(os.path.join(GEC_DIR, "gec_areta.py")):
                    raise GECUnavailable(
                        f"gec_areta.py not found in {GEC_DIR}. Unpack "
                        f"arabic_gec_system next to the backend or set GEC_DIR."
                    )
                _ensure_areta_importable()
                if GEC_DIR not in sys.path:
                    sys.path.insert(0, GEC_DIR)
                try:
                    import gec_areta  # noqa: WPS433 (deliberate lazy import)
                except Exception as exc:            # missing torch / camel-tools
                    raise GECUnavailable(str(exc)) from exc
                _module = gec_areta
    return _module


# ===========================================================================
# error building
# ===========================================================================
def _message_for(tag: str, original: str, corrected: str, label: str = "") -> str:
    label = label or _TAG_AR.get(tag, _TAG_AR["unk"])
    orig = (original or "").strip()
    corr = (corrected or "").strip()
    if orig and corr and orig != corr:
        return f'{label}: كتبت "{orig}" والصواب "{corr}"'
    if corr and not orig:
        return f'{label}: ينقص "{corr}"'
    if orig and not corr:
        return f'{label}: "{orig}" زائدة'
    return label


def _to_error_item(raw: dict) -> dict:
    """ARETA row -> the app's ErrorItem shape (schemas.ErrorItem)."""
    tag = raw.get("tag") or "unk"
    label, category = _resolve_tag(tag)
    if category == "unknown":
        # keep whatever gec_areta itself managed to say
        category = raw.get("category") or "unknown"
    # ARETA works on the tokenised text, so a missing full stop comes back as
    # "البيت ." — re-attach the punctuation before it reaches the error card.
    original = _prettify(raw.get("original") or "")
    corrected = _prettify(raw.get("corrected") or "")
    return {
        "message": _message_for(tag, original, corrected, label),
        "correction": corrected.strip() or None,
        "span": original.strip() or None,
        "type": _TYPE_OF_CATEGORY.get(category, "spelling"),
        # extra detail for a richer results screen (ignored by older clients)
        "tag": tag,
        "category": category,
        "subcategory": raw.get("subcategory"),
    }


def _diff_errors(original_text: str, corrected_text: str) -> list[dict]:
    """Fallback: word-level diff when ARETA is unavailable or too slow.

    Same error cards, minus the fine-grained linguistic tag.
    """
    import difflib

    a = original_text.split()
    b = corrected_text.split()
    errors: list[dict] = []

    def _type_of(word: str) -> str:
        return "punctuation" if word and all(c in _PUNCT for c in word) else "spelling"

    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                src = a[i1 + k] if i1 + k < i2 else ""
                dst = b[j1 + k] if j1 + k < j2 else ""
                errors.append({
                    "message": _message_for("unk", src, dst),
                    "correction": dst or None, "span": src or None,
                    "type": _type_of(src or dst),
                    "tag": None, "category": "unknown", "subcategory": None,
                })
        elif op == "delete":
            for k in range(i1, i2):
                errors.append({
                    "message": f'"{a[k]}" زائدة',
                    "correction": None, "span": a[k], "type": _type_of(a[k]),
                    "tag": None, "category": "unknown", "subcategory": None,
                })
        elif op == "insert":
            for k in range(j1, j2):
                errors.append({
                    "message": f'ينقص "{b[k]}"',
                    "correction": b[k], "span": None, "type": _type_of(b[k]),
                    "tag": None, "category": "unknown", "subcategory": None,
                })
    return errors


def _counts(errors: list[dict]) -> dict[str, int]:
    out = {"spelling": 0, "grammar": 0, "punctuation": 0}
    for e in errors:
        t = e.get("type") or "spelling"
        out[t] = out.get(t, 0) + 1
    return out


# ===========================================================================
# public API
# ===========================================================================
def _analyze_with_retry(gec, source: str, corrected: str) -> Optional[list]:
    """Run the ARETA analyser, retrying a transient failure.

    Returns the raw rows, or None when every attempt failed (the caller then
    uses the diff). Failures are LOGGED — swallowing them silently once cost a
    debugging session: a MemoryError inside the subprocess looked exactly like
    "ARETA produced no tags".
    """
    global _areta_failures, _areta_broken

    last_exc: Optional[BaseException] = None
    for attempt in range(1, ARETA_ATTEMPTS + 1):
        try:
            raw = gec.analyze_errors(source, corrected)
            _areta_failures = 0
            return raw
        except Exception as exc:   # subprocess died / camel-tools data missing
            last_exc = exc
            _log.warning("ARETA attempt %d/%d failed: %s: %s",
                         attempt, ARETA_ATTEMPTS, type(exc).__name__, exc)
            if attempt < ARETA_ATTEMPTS:
                time.sleep(ARETA_RETRY_WAIT)

    _areta_failures += 1
    if _areta_failures >= ARETA_GIVE_UP_AFTER and not _areta_broken:
        _areta_broken = True
        _log.error("ARETA failed %d times in a row — using the word diff for "
                   "the rest of this process. Last error: %s: %s",
                   _areta_failures, type(last_exc).__name__, last_exc)
    return None


def _detect_sync(text: str) -> dict:
    """Correct + analyse. Runs on the single GEC worker thread."""
    gec = _load()

    # Compare like for like: both sides tokenised the way the tagger writes.
    source = _normalize_spacing(text)
    corrected = _restore_unk(source, gec.correct_text(source))

    errors: list[dict] = []
    tagged = False
    if _areta_expected():
        raw_errors = _analyze_with_retry(gec, source, corrected)
        if raw_errors is not None:
            errors = [_to_error_item(e) for e in raw_errors]
            tagged = True
    if not tagged:
        errors = _diff_errors(source, corrected)

    # Never show a card that says "the correct form is [UNK]".
    errors = [
        e for e in errors
        if UNK not in (e.get("correction") or "") and UNK not in (e.get("span") or "")
    ]

    return {
        "corrected": _prettify(corrected),
        "errors": errors[:MAX_ERRORS],
        "total_errors": len(errors),
        "counts": _counts(errors),
        "tagged": tagged,
    }


def detect(text: str, timeout: Optional[float] = None) -> dict:
    """Detect the errors in `text`.

    Returns {"corrected", "errors", "total_errors", "counts", "tagged"}.
    Raises GECUnavailable if the engine cannot run, TimeoutError if it takes
    longer than `timeout` (default GEC_TIMEOUT) — the caller decides whether
    to degrade or fail.
    """
    text = (text or "").strip()
    if not text:
        return {"corrected": "", "errors": [], "total_errors": 0,
                "counts": _counts([]), "tagged": False}

    cached = _cache.get(text)
    if cached is not None:
        return dict(cached, errors=list(cached["errors"]))

    future = submit(text)
    result = future.result(timeout=timeout if timeout is not None else GEC_TIMEOUT)
    return result


def submit(text: str):
    """Queue a detection and return a Future — lets the caller run the AES
    models while the GEC/ARETA pipeline works in parallel.

    Resolve it with `resolve(future)`, which applies the timeout and caches.
    """
    text = (text or "").strip()

    def _run() -> dict:
        cached = _cache.get(text)
        if cached is not None:
            return cached
        result = _detect_sync(text)
        # Only cache a result we trust. An untagged result while ARETA is still
        # expected means the analyser failed *this time* — caching it pins
        # tag-less error cards to this text for the life of the process, so one
        # transient MemoryError at start-up silently degrades every re-submit
        # of that essay. Let the next request try again instead.
        if result.get("tagged") or not _areta_expected():
            if len(_cache) >= _CACHE_MAX:
                _cache.clear()
            _cache[text] = result
        return result

    return _pool.submit(_run)


def resolve(future, timeout: Optional[float] = None) -> Optional[dict]:
    """Wait for a `submit()` future. Returns None instead of raising when the
    engine is unavailable, failed or ran past the timeout (the request then
    simply carries on without error cards)."""
    try:
        result = future.result(timeout=timeout if timeout is not None else GEC_TIMEOUT)
    except Exception:       # timeout, GECUnavailable, model/subprocess failure
        return None
    return dict(result, errors=list(result["errors"]))


def warm_up() -> bool:
    """Preload the two taggers so the first student request isn't slow.

    Also fires one throw-away ARETA run. The analyser is a subprocess that
    imports sklearn/scipy, so a broken install (or missing camel-tools data)
    used to stay invisible until a student's essay came back with untagged
    error cards. Doing it here reports it at boot instead. It is deliberately
    not fatal: correction still works without the tags.
    """
    try:
        gec = _load()
        gec._ensure_models_loaded()
    except Exception:
        return False

    if _areta_expected():
        if _analyze_with_retry(gec, "انا اكلت", "أنا أكلت") is None:
            _log.warning("GEC warm-up: taggers are up but ARETA tagging is not "
                         "working — error cards will have no linguistic tag.")
    return True
