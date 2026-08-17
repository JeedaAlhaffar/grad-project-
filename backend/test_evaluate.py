# -*- coding: utf-8 -*-
"""Manual smoke test of the real evaluation pipeline.

    python test_evaluate.py

Runs, for compositions: GEC error detection -> hybrid relevance -> AES traits.
Nothing is mocked (AI_MOCK=0), so a missing model/dependency fails loudly.
"""
import os
import sys

os.environ["AI_MOCK"] = "0"   # force real path, no fallback masking

import json
import time

import ai_service
from models import WritingType

# The Windows console is cp1252; make Arabic printable.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROMPT = "اكتب موضوعاً تعبر فيه عن رأيك في وسائل التواصل الاجتماعي وأثرها على حياتنا."

weak = "التواصل الاجتماعي مفيد ولكن له أضرار. يجب أن نستخدمه بحذر. لا تضييع الوقت عليه. وشكراً."
strong = ("تعد وسائل التواصل الاجتماعي سيفاً ذا حدين في عصرنا الحالي. من الناحية الإيجابية، "
          "تتيح لنا التواصل الفوري مع الأصدقاء وتبادل المعرفة بسرعة مذهلة. ومن الناحية السلبية، "
          "قد تؤدي إلى تضييع الوقت وإهمال العلاقات الأسرية. لذا، يقع على عاتقنا مسؤولية تنظيم وقتنا "
          "واستغلال هذه المنصات في تطوير مهاراتنا وتوسيع آفاقنا العلمية، بدلاً من الانغماس في المحتوى "
          "السطحي الذي لا يسمن ولا يغني من جوع.")
# off-topic on purpose: the hybrid relevance scorer should punish it
off_topic = ("تعتبر السياحة من أهم مصادر الدخل القومي للعديد من الدول، فهي تساهم في توفير فرص عمل "
             "للشباب وتنشيط الحركة التجارية في الأسواق. ويجب على الحكومات الاهتمام بالمعالم الأثرية "
             "وتطوير البنية التحتية من فنادق ومواصلات لجذب السياح من مختلف أنحاء العالم.")
# under the 20-word guardrail: must come back rejected, unscored
too_short = "وسائل التواصل مفيدة جداً للطلاب."


def show(label: str, score, fb) -> None:
    print(f"\n===== {label} -> score {score}/100 =====")
    print(json.dumps(fb, ensure_ascii=False, indent=2))


for label, t in [("WEAK", weak), ("STRONG", strong),
                 ("OFF-TOPIC", off_topic), ("TOO SHORT", too_short)]:
    t0 = time.time()
    score, fb = ai_service.evaluate_text(WritingType.composition, t, PROMPT)
    show(f"{label} composition [{time.time() - t0:.1f}s]", score, fb)

# words: a copying task — graded against the word the exercise asked for, not
# spell-checked. GEC is the professional level's tool; كلمات never uses it.
score, fb = ai_service.evaluate_text(
    WritingType.words, "التفاحه", reference="التفاحة")
show("WORDS", score, fb)

# dictation diff
ref = "ذهب الطالب إلى المدرسة"
sub = "ذهب الطالب الى المدرسه"
score, fb = ai_service.evaluate_text(WritingType.dictation, sub, ref)
show("DICTATION", score, fb)
