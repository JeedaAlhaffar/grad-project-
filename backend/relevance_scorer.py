# -*- coding: utf-8 -*-
"""
Topic relevance (مدى الارتباط بالموضوع) — the hybrid scorer.

This REPLACES the `relevance` trait that used to come out of the Hybrid AES
model (`aes_model.py`); that output is now dropped entirely (see
`aes_model.DROPPED_TRAITS`). Relevance is judged here and only here.

The scorer is the notebook's final `RelevanceScorer` class, unchanged in
behaviour:

    prompt + essay
      ├─► CrossEncoder (Smart_Relevance_Model_Final)  ─► bert_score  [0..1]
      └─► TF-IDF cosine over 4+ letter words          ─► tfidf_score [0..1]
                                   │
                                   ▼
              tfidf < 0.12  ->  penalty = (tfidf/0.12)**3   (off-topic killer)
              final = bert_score * penalty * 4.0            (0..4)

Two guards:
  * essays shorter than `MIN_WORDS` (20) are rejected outright — the caller
    gets {"status": "rejected", ...} and shows the message to the student;
  * an empty/missing prompt means there is nothing to be relevant *to*, so we
    report {"status": "skipped"} instead of inventing a score.

`get_score()` returns a plain JSON-ready dict, exactly as specified by the
model author. The heavy CrossEncoder is loaded once per process (lazily, under
a lock) and reused by every request.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

# The trained CrossEncoder folder. Lives next to the backend by default
# (../Smart_Relevance_Model_Final), overridable with RELEVANCE_MODEL_DIR.
_DEFAULT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Smart_Relevance_Model_Final")
)
MODEL_DIR = os.environ.get("RELEVANCE_MODEL_DIR") or _DEFAULT_DIR

MIN_WORDS = int(os.environ.get("RELEVANCE_MIN_WORDS", "20"))
MAX_SCORE = 4.0            # the UI grades relevance out of 4
SAFE_THRESHOLD = 0.12      # TF-IDF floor below which the essay gets punished

REJECT_MESSAGE_AR = "النص قصير جداً (أقل من {n} كلمة)."

_scorer: Optional["RelevanceScorer"] = None
_lock = threading.Lock()


class RelevanceScorer:
    """CrossEncoder + TF-IDF hybrid. Build once, call `get_score` per essay."""

    def __init__(self, model_path: str = MODEL_DIR, min_words: int = MIN_WORDS):
        from sentence_transformers import CrossEncoder

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Relevance model folder not found: {model_path}. Copy "
                f"Smart_Relevance_Model_Final next to the backend or set "
                f"RELEVANCE_MODEL_DIR."
            )
        # Loaded once when the server boots (see main.py warm-up).
        self.model = CrossEncoder(model_path)
        self.min_words = min_words

    # -- the contract agreed with the model author: dict in, dict out --------
    def get_score(self, prompt_text: str, student_essay: str) -> dict:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        essay = (student_essay or "").strip()
        prompt = (prompt_text or "").strip()

        # 1) pre-validation guard -------------------------------------------
        word_count = len(essay.split())
        if word_count < self.min_words:
            return {
                "status": "rejected",
                "reason": "short_text",
                "message": REJECT_MESSAGE_AR.format(n=self.min_words),
                "score": 0.0,
                "max_score": MAX_SCORE,
                "word_count": word_count,
                "min_words": self.min_words,
            }

        # No prompt -> relevance is undefined, don't guess.
        if not prompt:
            return {
                "status": "skipped",
                "reason": "no_prompt",
                "message": "لا يوجد نص للسؤال لقياس الارتباط به.",
                "score": None,
                "max_score": MAX_SCORE,
            }

        # 2) the smart judge (CrossEncoder) ---------------------------------
        raw_bert = self.model.predict([(prompt, essay)])[0]
        bert_score = min(max(float(raw_bert), 0.0), 1.0)

        # 3) the lexical sniper (TF-IDF over content words) ------------------
        vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w{4,}\b")
        try:
            vectors = vectorizer.fit_transform([prompt, essay])
            tfidf_score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        except ValueError:
            # no usable tokens on one of the two sides
            tfidf_score = 0.0

        # 4) calibration + off-topic penalty --------------------------------
        if tfidf_score < SAFE_THRESHOLD:
            penalty_multiplier = (tfidf_score / SAFE_THRESHOLD) ** 3
        else:
            penalty_multiplier = 1.0

        # 5) final score out of 4 -------------------------------------------
        final_ratio = bert_score * penalty_multiplier
        final_score = round(final_ratio * MAX_SCORE, 2)

        return {
            "status": "success",
            "score": final_score,
            "max_score": MAX_SCORE,
        }


def get_scorer() -> RelevanceScorer:
    """Lazily build (once) and return the shared scorer."""
    global _scorer
    if _scorer is None:
        with _lock:
            if _scorer is None:
                _scorer = RelevanceScorer()
    return _scorer


def score_relevance(prompt_text: str, student_essay: str) -> dict:
    """Convenience wrapper: (prompt, essay) -> JSON-ready relevance dict."""
    return get_scorer().get_score(prompt_text, student_essay)


def warm_up() -> bool:
    """Preload the CrossEncoder so the first student request isn't slow."""
    try:
        get_scorer()
        return True
    except Exception:
        return False
