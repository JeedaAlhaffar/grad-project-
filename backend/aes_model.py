# -*- coding: utf-8 -*-
"""
Arabic Automated Essay Scoring (AES) — inference engine.

This is the production twin of the training/eval notebooks
(`ai/cleaaaan_full_tuning.ipynb`). It rebuilds the exact same
`TrueHybridMultiTraitModel` (AraBERT + 5 hand-crafted statistical features)
and runs the *identical* preprocessing pipeline so a composition (تعبير)
submitted through the API is scored the same way it was during training:

    raw text
      ├─► extract_raw_features(text)  ─►  StandardScaler (scaler.pkl)  ─┐
      └─► AraBERT tokenizer  ─►  BERT  ─►  [CLS] (768-d)  ───────────────┤
                                                                        ▼
                                              regressor ─► 8 trait logits
                                              logits * 32, clamp [0,32]

Traits (order matters — must match training):
    holistic (0–32) + relevance, organization, development, vocabulary,
    style, mechanics, grammar (each 0–4).

IMPORTANT — relevance is NOT reported by this model any more. Topic relevance
is now produced by the dedicated hybrid scorer in `relevance_scorer.py`
(CrossEncoder + TF-IDF, trained on the same data, Pearson 0.86). The head still
predicts a relevance value because the checkpoint's architecture is fixed, but
it is dropped from `predict_traits()` output (`DROPPED_TRAITS`) so nothing in
the backend can accidentally use the old, weaker signal.

The model, tokenizer and scaler are loaded once, lazily, and cached. Loading is
fully offline: the tokenizer comes from the saved `tokenizer.json` and the BERT
weights come entirely from the checkpoint (`clean_true_hybrid_model_final.pth`),
so no Hugging Face download is required at request time.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

# Model artifacts live in ../clean_True_Hybrid_AES_Model relative to backend/,
# overridable with AES_MODEL_DIR.
_DEFAULT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "clean_True_Hybrid_AES_Model")
)
MODEL_DIR = os.environ.get("AES_MODEL_DIR", _DEFAULT_DIR)

# Base encoder the model was fine-tuned from (used only to build the config /
# architecture; the actual weights come from the checkpoint).
BERT_NAME = os.environ.get("AES_BERT_NAME", "aubmindlab/bert-base-arabertv02-twitter")
WEIGHTS_FILE = "clean_true_hybrid_model_final.pth"
TOKENIZER_FILE = "tokenizer.json"
SCALER_FILE = "scaler.pkl"

MAX_LEN = 256
SCORE_SCALE = 32.0          # training divided every label by 32, so we multiply back
NUM_TRAITS = 8
NUM_MANUAL_FEATURES = 5

# The 8 output positions of the checkpoint — the order is fixed by training.
TRAITS = [
    "holistic", "relevance", "organization", "development",
    "vocabulary", "style", "mechanics", "grammar",
]

# Heads whose predictions we deliberately throw away. `relevance` is now the
# job of relevance_scorer.RelevanceScorer (hybrid CrossEncoder + TF-IDF).
DROPPED_TRAITS = {"relevance"}

# What callers actually get back: holistic (0..32) + 6 sub-traits (0..4).
REPORTED_TRAITS = [t for t in TRAITS if t not in DROPPED_TRAITS]

# Global singleton, guarded by a lock so concurrent requests load only once.
_engine: Optional["_AESEngine"] = None
_lock = threading.Lock()


def extract_raw_features(text: str) -> list[float]:
    """The 5 statistical features — byte-for-byte identical to the notebook."""
    words = str(text).split()
    num_words = len(words)
    unique_words = len(set(words))
    num_chars = len(str(text))
    num_sentences = (
        str(text).count(".") + str(text).count("!") + str(text).count("؟")
    )
    diversity = unique_words / (num_words + 1e-5)
    return [num_words, unique_words, num_chars, num_sentences, diversity]


class _AESEngine:
    """Holds the loaded model/tokenizer/scaler and runs a single prediction."""

    def __init__(self, model_dir: str = MODEL_DIR):
        import torch
        import torch.nn as nn
        import joblib
        from transformers import PreTrainedTokenizerFast, AutoModel, AutoConfig
        from transformers import BertConfig

        self.torch = torch

        weights_path = os.path.join(model_dir, WEIGHTS_FILE)
        tok_path = os.path.join(model_dir, TOKENIZER_FILE)
        scaler_path = os.path.join(model_dir, SCALER_FILE)
        for p in (weights_path, tok_path, scaler_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"AES artifact missing: {p}. Set AES_MODEL_DIR to the folder "
                    f"holding {WEIGHTS_FILE}, {TOKENIZER_FILE} and {SCALER_FILE}."
                )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- tokenizer: load the saved fast tokenizer (fully offline) ----------
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_path,
            unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
            cls_token="[CLS]", mask_token="[MASK]",
        )

        # --- architecture ------------------------------------------------------
        # Build the BERT config without downloading pretrained weights (the
        # checkpoint carries them). Prefer the real config; fall back to the
        # known arabertv02 hyper-params if there is no network / cache.
        try:
            bert_cfg = AutoConfig.from_pretrained(BERT_NAME)
        except Exception:
            bert_cfg = BertConfig(
                vocab_size=self.tokenizer.vocab_size or 64000,
                hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                intermediate_size=3072, max_position_embeddings=512,
                type_vocab_size=2,
            )

        class TrueHybridMultiTraitModel(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.bert = AutoModel.from_config(cfg)
                self.regressor = nn.Sequential(
                    nn.Linear(768 + NUM_MANUAL_FEATURES, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, NUM_TRAITS),
                )

            def forward(self, input_ids, attention_mask, manual_features):
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
                combined = torch.cat((cls, manual_features), dim=1)
                return self.regressor(combined)

        self.model = TrueHybridMultiTraitModel(bert_cfg).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        # strict=True: the checkpoint must match the architecture exactly.
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        # --- feature scaler ----------------------------------------------------
        self.scaler = joblib.load(scaler_path)

    def predict(self, text: str) -> dict[str, float]:
        """Return {trait: score} — holistic on 0–32, sub-traits on 0–4.

        The `relevance` head is dropped here (see DROPPED_TRAITS).
        """
        torch = self.torch
        feats = self.scaler.transform([extract_raw_features(text)])
        manual = torch.tensor(feats, dtype=torch.float).to(self.device)
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt",
        )
        with torch.no_grad():
            preds = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
                manual,
            )
            final = torch.clamp(preds * SCORE_SCALE, 0, SCORE_SCALE)
        values = final.cpu().numpy()[0].tolist()
        return {
            trait: float(v)
            for trait, v in zip(TRAITS, values)
            if trait not in DROPPED_TRAITS
        }


def get_engine() -> "_AESEngine":
    """Lazily build (once) and return the shared AES engine."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = _AESEngine()
    return _engine


def predict_traits(text: str) -> dict[str, float]:
    """Convenience wrapper: raw text → {trait: score} (no `relevance`)."""
    return get_engine().predict(text)


def warm_up() -> bool:
    """Preload the checkpoint so the first student request isn't slow."""
    try:
        get_engine()
        return True
    except Exception:
        return False
