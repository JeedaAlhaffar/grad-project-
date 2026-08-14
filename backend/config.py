# -*- coding: utf-8 -*-
"""
Central configuration for the Arabic writing-teaching API.

Everything is overridable with environment variables so the Flutter team can
point the app at a dev/staging/prod backend without touching code.
"""
import os

# --- keep TensorFlow out of the process -----------------------------------
# `gec/tag.py` imports transformers' Trainer, which makes transformers probe
# for TensorFlow and import it (~2500 modules, ~0.4 GB RSS, ~20 s of start-up)
# even though every model here is pure PyTorch. On a 16 GB laptop that is the
# difference between the four checkpoints fitting and an out-of-memory crash.
# Must be set before transformers is imported anywhere — config is imported
# first by both main.py and ai_service.py, and all model imports are lazy.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class Settings:
    # --- app ---
    APP_NAME: str = os.environ.get("APP_NAME", "Arabic Writing App API")
    API_PREFIX: str = os.environ.get("API_PREFIX", "/api")
    DEBUG: bool = os.environ.get("DEBUG", "1") == "1"

    # --- auth (JWT) ---
    # CHANGE THIS in production (export JWT_SECRET=...).
    JWT_SECRET: str = os.environ.get("JWT_SECRET", "dev-secret-change-me")
    JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "43200"))  # 30 days

    # --- CORS (Flutter web / local dev) ---
    # Comma-separated list, or "*" for all.
    CORS_ORIGINS: str = os.environ.get("CORS_ORIGINS", "*")

    # --- AI ---
    # When 1, the AI service returns deterministic MOCK evaluations so the
    # Flutter team can build against realistic data before the models are wired.
    # Set AI_MOCK=0 once the real AES/AraBERT logic is implemented.
    AI_MOCK: bool = os.environ.get("AI_MOCK", "1") == "1"

    # Composition (تعبير) scoring uses the real Hybrid AES model (AraBERT + stats)
    # whenever it can be loaded. Set AES_ENABLED=0 to force the mock even for
    # composition (e.g. on a machine without torch/transformers installed).
    AES_ENABLED: bool = os.environ.get("AES_ENABLED", "1") == "1"
    # Folder holding clean_true_hybrid_model_final.pth, tokenizer.json, scaler.pkl.
    # Defaults (in aes_model.py) to ../clean_True_Hybrid_AES_Model.
    AES_MODEL_DIR: str = os.environ.get("AES_MODEL_DIR", "")

    # How the 0..100 composition grade is computed:
    #   "traits"   -> mean of the 7 sub-traits (relevance from the new hybrid
    #                 scorer + the 6 AES traits). Default.
    #   "holistic" -> the AES holistic head only (the pre-relevance behaviour).
    COMPOSITION_SCORE_MODE: str = os.environ.get("COMPOSITION_SCORE_MODE", "traits")

    # --- AI: topic relevance (مدى الارتباط بالموضوع) ---
    # Hybrid CrossEncoder + TF-IDF scorer (relevance_scorer.py). This REPLACES
    # the relevance output of the AES model, which is now ignored entirely.
    RELEVANCE_ENABLED: bool = os.environ.get("RELEVANCE_ENABLED", "1") == "1"
    # Folder of the trained CrossEncoder; defaults (in relevance_scorer.py) to
    # ../Smart_Relevance_Model_Final.
    RELEVANCE_MODEL_DIR: str = os.environ.get("RELEVANCE_MODEL_DIR", "")
    # Essays shorter than this are rejected before scoring (academic guardrail).
    RELEVANCE_MIN_WORDS: int = int(os.environ.get("RELEVANCE_MIN_WORDS", "20"))

    # --- AI: copying (نسخ الحروف) — handwriting recognition ---
    # BiGRU + multi-head-attention classifier over the 107 Arabic letter
    # shapes; grades the three copying modes (see copying_grader.py).
    # NOTE: this is the one Keras/TensorFlow model in the stack — the import is
    # lazy, but it still costs ~0.5 GB RSS on top of the PyTorch checkpoints.
    # Set HANDWRITING_ENABLED=0 on a tight box: grading then falls back to the
    # geometric trace score alone (and to the mock for free_draw).
    HANDWRITING_ENABLED: bool = os.environ.get("HANDWRITING_ENABLED", "1") == "1"
    # Folder with bgru_mha_final_model.h5, label_map.npy, config.json; defaults
    # (in handwriting_model.py) to ../beginer level copy letters/model.
    HANDWRITING_MODEL_DIR: str = os.environ.get("HANDWRITING_MODEL_DIR", "")

    # --- AI: error detection (GEC + ARETA) ---
    # Correction + tagged error extraction, from ../arabic_gec_system.
    GEC_ENABLED: bool = os.environ.get("GEC_ENABLED", "1") == "1"
    GEC_DIR: str = os.environ.get("GEC_DIR", "")
    # Hard ceiling per request; past it the response goes out without the
    # error cards instead of making the student wait.
    GEC_TIMEOUT: float = float(os.environ.get("GEC_TIMEOUT", "60"))
    GEC_MAX_ERRORS: int = int(os.environ.get("GEC_MAX_ERRORS", "25"))

    # Load the ML models when the server boots instead of on the first student
    # request (set AI_WARM_UP=0 for a fast dev start-up).
    AI_WARM_UP: bool = os.environ.get("AI_WARM_UP", "1") == "1"

    # --- scoring thresholds (score is 0..100) ---
    PASS_SCORE: float = float(os.environ.get("PASS_SCORE", "50"))
    STAR_1_MIN: float = float(os.environ.get("STAR_1_MIN", "50"))
    STAR_2_MIN: float = float(os.environ.get("STAR_2_MIN", "75"))
    STAR_3_MIN: float = float(os.environ.get("STAR_3_MIN", "90"))

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
