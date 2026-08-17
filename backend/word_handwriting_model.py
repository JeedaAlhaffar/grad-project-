# -*- coding: utf-8 -*-
"""
ADAB Conformer — online recognition of a hand-written Arabic WORD (المستوى المتوسط).

    student strokes -> word_handwriting_preprocessing (46 allograph features)
                    -> Conformer + CTC -> "الحكومة"

This is the *word* half of the AI stack, and it is a different model from the
beginner one in every respect:

                    beginner (نسخ الحروف)      intermediate (كلمات/جمل)
    file            handwriting_model.py       this file
    checkpoint      Jeeda/model/*.h5           adab_model/best_model_adab.pth
    framework       Keras / TensorFlow         PyTorch / torchaudio
    architecture    BiGRU + multi-head attn    Conformer (6 layers) + CTC
    input           6 features/point, T=85     46 features/allograph, variable T
    output          107 letter shapes          39-symbol alphabet, free length
    task            "which letter shape is     "what word did they write"
                     this drawing"

Nothing in `Jeeda/` is read or written by this module — that delivery belongs
to the beginner classifier alone.

Reported accuracy (adab_model/adab_conformer_training.ipynb, ADAB sets 1-3,
writer-based 70/15/15 split, held-out test set used exactly once):
    CER 1.65%   WER 6.01%

*** THE NORMALISATION FILE IS REQUIRED ***
Training normalised every feature with per-dimension statistics taken over the
44,420-sample training cache:

    feats = (feats - mean) / std          # feat_norm_adab.pt

Those 46+46 numbers are NOT recoverable from the checkpoint — the first layer is
a plain Linear(46, 256) with no input normalisation of its own — and the std
spans six orders of magnitude across dimensions (1e-6 for the constant feature,
~20 for the widest). Feeding raw features would put the model far outside its
training distribution and produce confident nonsense, so this module REFUSES to
run without the file rather than degrade silently.

    adab_model/feat_norm_adab.pt      {'mean': tensor(46), 'std': tensor(46)}

On the training machine it is written to `./feature_cache_adab/feat_norm_adab.pt`
by Section 5 of the notebook. Copy it next to the checkpoint. If it is lost, it
can be regenerated from the ADAB training split with:

    feat_stack = torch.cat([f for f, l in train_samples], dim=0)
    norm = {'mean': feat_stack.mean(0), 'std': feat_stack.std(0) + 1e-6}
"""
from __future__ import annotations

import logging
import math
import os
import threading
from typing import Optional

import numpy as np

_log = logging.getLogger("word_handwriting")

# Folder holding best_model_adab.pth + feat_norm_adab.pt.
_DEFAULT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "adab_model")
)
MODEL_DIR = os.environ.get("WORD_MODEL_DIR") or _DEFAULT_DIR

MODEL_PATH = os.path.join(MODEL_DIR, "best_model_adab.pth")
NORM_PATH = os.path.join(MODEL_DIR, "feat_norm_adab.pt")

# --- the alphabet, verbatim from the notebook (index 0 is the CTC blank) -----
ARABIC_CHARS = [
    "blank", " ", "أ", "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز",
    "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه",
    "و", "ي", "ة", "ى", "ء", "ئ", "ؤ", "لا", "إ", "آ",
]
IDX_TO_CHAR = {i: c for i, c in enumerate(ARABIC_CHARS)}
CHAR_TO_IDX = {c: i for i, c in enumerate(ARABIC_CHARS)}
NUM_CLASSES = len(ARABIC_CHARS)          # 39
BLANK = 0

# Everything the model can spell. Content outside this set can never be
# recognised, which is why the seed scripts filter the curriculum against it.
SUPPORTED_CHARS = frozenset("".join(ARABIC_CHARS[1:]))

_engine: Optional["_Engine"] = None
_load_lock = threading.Lock()
_predict_lock = threading.Lock()


class WordModelUnavailable(Exception):
    """Raised when the checkpoint or the normalisation stats are missing."""


def unsupported_chars(text: str) -> set[str]:
    """Characters of `text` the 39-symbol alphabet cannot represent."""
    return {c for c in (text or "") if c not in SUPPORTED_CHARS}


def _import_conformer():
    """Get torchaudio's Conformer without requiring torchaudio to import.

    `import torchaudio` runs a native extension (libtorchaudio.pyd) that provides
    audio I/O and DSP ops. None of that is used here — Conformer is plain PyTorch,
    importing only `typing` and `torch` — but a broken or mismatched native build
    makes the whole package unimportable and takes the model down with it.

    So: try the normal import, and if the package cannot load, read
    torchaudio/models/conformer.py directly as a standalone module.
    """
    try:
        from torchaudio.models import Conformer
        return Conformer
    except Exception as exc:                      # OSError, ImportError, ...
        _log.warning("torchaudio would not import (%s); loading Conformer "
                     "directly from its source file", exc)

    import importlib.util

    import torch
    src = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)),
                       "torchaudio", "models", "conformer.py")
    if not os.path.isfile(src):
        raise WordModelUnavailable(
            "torchaudio is unusable and its conformer.py was not found at "
            f"{src} — install a torchaudio matching torch {torch.__version__}")

    spec = importlib.util.spec_from_file_location("_adab_conformer", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Conformer


def _build_module():
    """Define the architecture exactly as trained. Imports torch lazily."""
    import torch
    import torch.nn as nn

    Conformer = _import_conformer()

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=1000, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float()
                                 * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return self.dropout(x + self.pe[:, :x.size(1), :])

    class ConformerModel(nn.Module):
        def __init__(self, input_dim=46, num_heads=4, head_dim=64, num_layers=6,
                     ffn_dim=512, conv_kernel_size=15, dense_dim=128,
                     num_classes=NUM_CLASSES, dropout=0.3):
            super().__init__()
            self.d_model = num_heads * head_dim
            self.input_projection = nn.Linear(input_dim, self.d_model)
            self.pos_encoding = PositionalEncoding(self.d_model, dropout=dropout)
            self.conformer = Conformer(
                input_dim=self.d_model, num_heads=num_heads, ffn_dim=ffn_dim,
                num_layers=num_layers, depthwise_conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            self.dense_projection = nn.Sequential(
                nn.Linear(self.d_model, dense_dim), nn.ReLU(), nn.Dropout(dropout))
            self.ctc_output = nn.Linear(dense_dim, num_classes)

        def forward(self, x, lengths):
            h = self.pos_encoding(self.input_projection(x))
            h, _ = self.conformer(h, lengths)
            return self.ctc_output(self.dense_projection(h))

    return ConformerModel


class _Engine:
    def __init__(self):
        import torch

        if not os.path.isfile(MODEL_PATH):
            raise WordModelUnavailable(f"checkpoint not found: {MODEL_PATH}")
        if not os.path.isfile(NORM_PATH):
            raise WordModelUnavailable(
                f"normalisation stats not found: {NORM_PATH} — the model cannot "
                "be run without them (see this module's docstring)"
            )

        self.torch = torch
        self.device = torch.device("cpu")     # one drawing is milliseconds

        norm = torch.load(NORM_PATH, map_location="cpu", weights_only=False)
        mean, std = norm["mean"], norm["std"]
        if mean.numel() != 46 or std.numel() != 46:
            raise WordModelUnavailable(
                f"feat_norm_adab.pt has {mean.numel()} dims, expected 46")
        self.mean = mean.float().view(1, -1)
        self.std = std.float().view(1, -1)

        ConformerModel = _build_module()
        self.model = ConformerModel(num_classes=NUM_CLASSES, dropout=0.4)
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)

        _log.info("ADAB word model ready: classes=%d dir=%s", NUM_CLASSES, MODEL_DIR)

    def recognise(self, feats: np.ndarray, top_k: int = 1) -> dict:
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(feats)).float()
        x = (x - self.mean) / self.std               # the training normalisation
        x = x.unsqueeze(0).to(self.device)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=self.device)

        with _predict_lock, torch.no_grad():
            logits = self.model(x, lengths)          # (1, T, 39)

        log_probs = logits.log_softmax(2)[0]         # (T, 39)
        text = _ctc_greedy_decode(logits, lengths, IDX_TO_CHAR)[0]

        # Mean per-frame confidence of the argmax path — a rough but honest
        # signal for "the model was unsure", used only to soften feedback.
        conf = float(log_probs.max(dim=1).values.mean().exp().item())
        return {"text": text, "confidence": conf, "frames": int(feats.shape[0])}


def _ctc_greedy_decode(logits, input_lengths, idx_to_char) -> list[str]:
    """Verbatim from the notebook: collapse repeats, drop blanks."""
    import torch

    arg_maxes = torch.argmax(logits, dim=2)
    decoded = []
    for i in range(arg_maxes.size(0)):
        seq = arg_maxes[i, :input_lengths[i]].tolist()
        cleaned, prev = [], None
        for c in seq:
            if c != prev and c != BLANK:
                cleaned.append(c)
            prev = c
        decoded.append("".join(idx_to_char[c] for c in cleaned if c in idx_to_char))
    return decoded


def get_engine() -> _Engine:
    global _engine
    if _engine is None:
        with _load_lock:
            if _engine is None:
                _engine = _Engine()
    return _engine


def is_available() -> bool:
    """True when both files are present — cheap, does not import torch."""
    return os.path.isfile(MODEL_PATH) and os.path.isfile(NORM_PATH)


def missing_files() -> list[str]:
    return [p for p in (MODEL_PATH, NORM_PATH) if not os.path.isfile(p)]


def recognise_drawing(submitted_drawing: list) -> Optional[dict]:
    """Canvas strokes -> {'text', 'confidence', 'frames'}, or None if unreadable."""
    import word_handwriting_preprocessing as prep

    feats = prep.drawing_to_features(submitted_drawing)
    if feats is None:
        return None
    return get_engine().recognise(feats)


def warm_up() -> bool:
    try:
        get_engine()
        return True
    except Exception as exc:
        _log.warning("ADAB word model unavailable: %s", exc)
        return False
