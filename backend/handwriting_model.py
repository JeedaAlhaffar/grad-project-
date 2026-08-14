# -*- coding: utf-8 -*-
"""
Arabic handwriting recognition (نسخ الحروف) — BiGRU + multi-head-attention.

This is the *letter* half of the AI stack: it takes the strokes the student
drew on the Flutter canvas and says which of the 107 letter shapes they are.
It is what turns "the student drew something" into "the student drew a medial
ع", which is the only way to grade a copying exercise honestly.

    student strokes  ->  handwriting_preprocessing (verbatim training pipeline)
                     ->  BiGRU + MHA  ->  107-way softmax  ->  E_ay / I_ba / ...

The checkpoint lives outside the backend, in `../beginer level copy letters/model`:
    bgru_mha_final_model.h5   label_map.npy   config.json
(point HANDWRITING_MODEL_DIR somewhere else if you move it).

*** A NOTE ON TENSORFLOW ***
Everything else in this backend is PyTorch, and config.py goes out of its way
to keep TensorFlow out of the process (USE_TF=0) because the four GEC/AES/
relevance checkpoints already want ~3 GB. This model is a Keras `.h5`, so TF is
genuinely required here — but the import is LAZY and only happens when a
copying exercise is actually graded (or at warm-up when HANDWRITING_ENABLED=1).
On a 16 GB box, run this together with the text models only after checking RSS;
`HANDWRITING_ENABLED=0` turns it off and the grader falls back to shape-only
scoring. Converting the checkpoint to TFLite/ONNX would remove the dependency.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

_log = logging.getLogger("handwriting")

# Folder holding bgru_mha_final_model.h5 + label_map.npy + config.json.
_DEFAULT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "beginer level copy letters", "model")
)
MODEL_DIR = os.environ.get("HANDWRITING_MODEL_DIR") or _DEFAULT_DIR

MODEL_PATH = os.path.join(MODEL_DIR, "bgru_mha_final_model.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.npy")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Defaults only used when config.json is missing; the real values come from it.
_FALLBACK_MAX_LEN = 85
_FALLBACK_N_FEATURES = 6

_engine: Optional["_Engine"] = None
_load_lock = threading.Lock()
# Keras predict() on one graph from several request threads is not worth the
# risk (and one student drawing is ~1 ms of compute) — serialise it.
_predict_lock = threading.Lock()


class HandwritingUnavailable(Exception):
    """Raised when the checkpoint or TensorFlow is not available."""


def _compat_objects() -> dict:
    """Custom objects that let the Keras-2 `.h5` load under Keras 3.

    The notebook saved the self-attention layer with `score_mode="dot"`. When
    Keras 3 deserializes that config it resolves the string through its object
    registry and hands `Attention.__init__` the FUNCTION `keras.layers.dot`,
    which the constructor then rejects:

        Invalid value for argument score_mode. Expected one of {'dot',
        'concat'}. Received: score_mode=<function dot at ...>

    Without this the checkpoint simply never loads on Keras >= 3 and every
    copying grade silently degrades to the trace score. Turning the callable
    back into its name is enough; the weights are unaffected.
    """
    import tensorflow as tf

    class _CompatAttention(tf.keras.layers.Attention):
        @staticmethod
        def _fix(config: dict) -> dict:
            score_mode = config.get("score_mode")
            if callable(score_mode):
                config = dict(config)
                config["score_mode"] = getattr(score_mode, "__name__", "dot")
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**cls._fix(dict(config)))

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **self._fix(kwargs))

    return {"Attention": _CompatAttention}


class _Engine:
    """The loaded checkpoint + its label map. One per process."""

    def __init__(self) -> None:
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError as exc:                      # no TF on this box
            raise HandwritingUnavailable(
                f"TensorFlow/numpy not installed: {exc}. "
                f"pip install tensorflow, or set HANDWRITING_ENABLED=0."
            ) from exc

        if not os.path.isfile(MODEL_PATH):
            raise HandwritingUnavailable(
                f"{MODEL_PATH} not found. Put bgru_mha_final_model.h5, "
                f"label_map.npy and config.json there, or set "
                f"HANDWRITING_MODEL_DIR."
            )
        if not os.path.isfile(LABEL_MAP_PATH):
            raise HandwritingUnavailable(f"{LABEL_MAP_PATH} not found.")

        _log.info("loading handwriting model from %s", MODEL_PATH)
        self._np = np
        self.model = tf.keras.models.load_model(
            MODEL_PATH, compile=False, custom_objects=_compat_objects())

        label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
        self.idx_to_class = {int(v): k for k, v in label_map.items()}
        self.class_to_idx = {k: int(v) for k, v in label_map.items()}

        self.max_len = _FALLBACK_MAX_LEN
        self.n_features = _FALLBACK_N_FEATURES
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            self.max_len = int(cfg.get("max_len", self.max_len))
            self.n_features = int(cfg.get("n_features", self.n_features))

        _log.info("handwriting model ready: classes=%d max_len=%d n_features=%d",
                  len(self.idx_to_class), self.max_len, self.n_features)

    def probabilities(self, strokes: list):
        """strokes -> the raw 107-way softmax vector. None if unusable."""
        from handwriting_preprocessing import preprocess_strokes

        x = preprocess_strokes(strokes, max_len=self.max_len,
                               n_features=self.n_features)
        if x is None:
            return None
        with _predict_lock:
            return self.model.predict(x, verbose=0)[0]


def get_engine() -> _Engine:
    """Load the checkpoint once, on first use."""
    global _engine
    if _engine is None:
        with _load_lock:
            if _engine is None:
                _engine = _Engine()
    return _engine


def is_available() -> bool:
    """True when a prediction could be served (files present, TF importable)."""
    try:
        get_engine()
        return True
    except Exception:
        return False


def predict(strokes: list, top_k: int = 3) -> Optional[dict]:
    """Classify a drawing.

    Returns
        {"top_k": [{"class": "M_ay", "confidence": 0.92}, ...],
         "probabilities": {class_name: p, ...}}
    or None when the strokes do not contain a usable drawing.
    """
    engine = get_engine()
    probs = engine.probabilities(strokes)
    if probs is None:
        return None

    np = engine._np
    order = np.argsort(probs)[::-1][:max(1, top_k)]
    return {
        "top_k": [
            {"class": engine.idx_to_class.get(int(i), "unknown"),
             "confidence": float(probs[i])}
            for i in order
        ],
        "probabilities": {
            engine.idx_to_class.get(int(i), f"idx_{i}"): float(p)
            for i, p in enumerate(probs)
        },
    }


def score_for_class(strokes: list, target_class: Optional[str],
                    top_k: int = 3) -> Optional[dict]:
    """Classify, then report how the *target* class did.

    This is the question a copying exercise asks — not "what is this?" but "is
    this the shape I asked for?". Returns None when the drawing is unusable.

        {"target": "M_ay", "target_probability": 0.83, "target_rank": 1,
         "predicted": "M_ay", "confidence": 0.83, "top_k": [...]}
    """
    result = predict(strokes, top_k=top_k)
    if result is None:
        return None

    probs = result["probabilities"]
    top = result["top_k"][0]

    target_p: Optional[float] = None
    target_rank: Optional[int] = None
    if target_class and target_class in probs:
        target_p = probs[target_class]
        # rank = how many classes the model preferred over the target, + 1.
        target_rank = 1 + sum(1 for p in probs.values() if p > target_p)

    return {
        "target": target_class,
        "target_probability": target_p,
        "target_rank": target_rank,
        "predicted": top["class"],
        "confidence": top["confidence"],
        "top_k": result["top_k"],
        # The full distribution — the grader sums it per Arabic letter to tell
        # "wrote a different letter" apart from "wrote the right letter in a
        # neighbouring form", which the model confuses far more often.
        "probabilities": probs,
    }


def warm_up() -> bool:
    """Preload the checkpoint so the first student drawing isn't slow."""
    try:
        get_engine()
        return True
    except Exception as exc:
        _log.warning("handwriting warm-up failed: %s: %s", type(exc).__name__, exc)
        return False
