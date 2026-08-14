# -*- coding: utf-8 -*-
"""Regression test: a transient ARETA failure must not be cached.

    python test_gec_recovery.py

Background — the bug this pins down:
ARETA runs in a fresh python subprocess that imports sklearn -> scipy. While the
four checkpoints were loading, that import died with MemoryError on the very
first request. `gec_service` swallowed the exception, fell back to the untagged
word diff, and then *cached* that degraded result — so one transient OOM at
start-up left that essay permanently without linguistic tags, while every other
text tagged fine. It cost 2 failing checks in the API suite and a long hunt
through subprocess stderr, because nothing was logged.

Uses a fake engine, so no model is loaded and this runs in about a second.
"""
import logging
import os
import sys

os.environ.setdefault("GEC_ARETA_RETRY_WAIT", "0")     # keep the test fast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
logging.basicConfig(level=logging.WARNING, format="   log> %(levelname)s %(message)s")

import gec_service

FAILURES = {"left": 0}
ROWS = [{"original": "انا", "corrected": "أنا", "tag": "OH",
         "category": "spelling", "subcategory": "hamza_error"}]


class FakeGec:
    """Stands in for gec_areta: correction always works, ARETA fails N times."""

    def correct_text(self, text):
        return text.replace("انا", "أنا")

    def analyze_errors(self, source, corrected):
        if FAILURES["left"] > 0:
            FAILURES["left"] -= 1
            raise MemoryError("simulated: subprocess died importing scipy")
        return list(ROWS)

    def _ensure_models_loaded(self):
        pass


gec_service._load = lambda: FakeGec()

FAILS = 0


def check(name, cond, detail=""):
    global FAILS
    if not cond:
        FAILS += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def reset(attempts=2, give_up=3):
    gec_service._cache.clear()
    gec_service._areta_failures = 0
    gec_service._areta_broken = False
    gec_service.ARETA_ATTEMPTS = attempts
    gec_service.ARETA_GIVE_UP_AFTER = give_up
    gec_service.ARETA_RETRY_WAIT = 0.0


TEXT = "انا اكلت"

# --- 1. a one-off failure is absorbed by the retry -------------------------
print("\n### retry absorbs a one-off failure")
reset()
FAILURES["left"] = 1                      # attempt 1 dies, attempt 2 works
r = gec_service.detect(TEXT)
check("tagged despite the failure", r["tagged"] is True, f"tagged={r['tagged']}")
check("tag present", [e["tag"] for e in r["errors"]] == ["OH"],
      f"tags={[e['tag'] for e in r['errors']]}")

# --- 2. THE REGRESSION: a failed detection must not be cached --------------
print("\n### a failed detection is not cached (the actual bug)")
reset()
FAILURES["left"] = 2                      # both attempts of request #1 die
r1 = gec_service.detect(TEXT)
check("request 1 degrades to the diff", r1["tagged"] is False, f"tagged={r1['tagged']}")
check("request 1 still returns error cards", len(r1["errors"]) >= 1,
      f"{len(r1['errors'])} cards")
check("degraded result was NOT cached", TEXT not in gec_service._cache,
      f"cache keys={list(gec_service._cache)}")

r2 = gec_service.detect(TEXT)             # ARETA healthy again
check("request 2 RECOVERS with tags", r2["tagged"] is True, f"tagged={r2['tagged']}")
check("request 2 has the real tag", [e["tag"] for e in r2["errors"]] == ["OH"],
      f"tags={[e['tag'] for e in r2['errors']]}")
check("good result IS cached", TEXT in gec_service._cache)

# --- 3. a genuinely broken ARETA gives up instead of retrying forever ------
print("\n### permanently broken ARETA stops retrying forever")
reset(attempts=1, give_up=3)
FAILURES["left"] = 999
for i in range(3):
    gec_service.detect(TEXT + " " * i)    # 3 distinct texts -> 3 failures
check("declared broken after the give-up threshold", gec_service._areta_broken is True,
      f"failures={gec_service._areta_failures}")
before = FAILURES["left"]
r = gec_service.detect("نص جديد تماما")
check("no further ARETA attempts once broken", FAILURES["left"] == before,
      f"attempts consumed={before - FAILURES['left']}")
check("fallback result is cached again", "نص جديد تماما" in gec_service._cache)
check("still returns usable cards", r["tagged"] is False and "corrected" in r)

print("\n" + "=" * 55)
print("  ALL PASS" if FAILS == 0 else f"  {FAILS} CHECK(S) FAILED")
print("=" * 55)
sys.exit(1 if FAILS else 0)
