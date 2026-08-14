# -*- coding: utf-8 -*-
"""
Seed the 5 placement-test questions (اختبار تحديد المستوى) shown in the Figma.

Idempotent: clears existing questions and re-inserts.

Run:
    python seed_placement.py
"""
from __future__ import annotations

import sys

from database import SessionLocal, init_db
from models import PlacementKind, PlacementQuestion

QUESTIONS = [
    dict(kind=PlacementKind.free_write, sort_order=0, weight=1.0,
         prompt="اكتب جملة تعبر عن شعورك اليوم", given_text=None, answer_key=None),
    dict(kind=PlacementKind.complete, sort_order=1, weight=1.0,
         prompt="أكمل الجملة التالية:", given_text="العلم نور والجهل ...",
         answer_key="العلم نور والجهل ظلام"),
    dict(kind=PlacementKind.complete, sort_order=2, weight=1.0,
         prompt="أكمل الجملة التالية:", given_text="من جدّ ... ومن زرع حصد",
         answer_key="من جدّ وجد ومن زرع حصد"),
    dict(kind=PlacementKind.correct, sort_order=3, weight=1.0,
         prompt="صحح الأخطاء في الجملة التالية:", given_text="الطالب يدرس دروسة بجد",
         answer_key="الطالب يدرس دروسه بجد"),
    dict(kind=PlacementKind.free_write, sort_order=4, weight=1.0,
         prompt="اكتب فقرة قصيرة (٢-٣ جمل) تصف فيها مدرستك", given_text=None,
         answer_key=None),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    init_db()
    db = SessionLocal()
    try:
        db.query(PlacementQuestion).delete()
        for q in QUESTIONS:
            db.add(PlacementQuestion(is_active=True, **q))
        db.commit()
        print(f"Seeded {len(QUESTIONS)} placement questions.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
