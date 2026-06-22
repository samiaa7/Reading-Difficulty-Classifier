from __future__ import annotations
import logging
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_pipeline import extract_features, load_scaler, FEATURE_NAMES
from train_sklearn import load_rf_model, predict_difficulty, DIFFICULTY_LABELS
from train_pytorch import load_nn_model, predict_emotional_load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion-Aware Reading Classifier", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rf_model  = None
nn_model  = None
scaler    = None
models_ready = False


@app.on_event("startup")
async def startup() -> None:
    global rf_model, nn_model, scaler, models_ready
    try:
        logger.info("Loading models …")
        scaler   = load_scaler()
        rf_model = load_rf_model()
        nn_model = load_nn_model(input_dim=len(FEATURE_NAMES))
        models_ready = True
        logger.info("All models loaded.")
    except FileNotFoundError as e:
        logger.error("Model file not found: %s — run the training scripts first.", e)
    except Exception as e:
        logger.error("Model loading failed: %s", e)

class TextRequest(BaseModel):
    text: str


EMOTIONAL_LABELS = [
    {"label": "Calm",        "color": "#22C55E", "description": "Text is accessible and unlikely to cause frustration."},
    {"label": "Engaged",     "color": "#3B82F6", "description": "Mildly challenging — keeps the reader engaged."},
    {"label": "Strained",    "color": "#F59E0B", "description": "Cognitively demanding — may cause fatigue for struggling readers."},
    {"label": "Frustrated",  "color": "#E24B4A", "description": "High cognitive load — likely to cause frustration in dyslexic readers."},
]


def emotional_label(score: float) -> dict:
    if score < 0.25:  return EMOTIONAL_LABELS[0]
    if score < 0.50:  return EMOTIONAL_LABELS[1]
    if score < 0.75:  return EMOTIONAL_LABELS[2]
    return EMOTIONAL_LABELS[3]


@app.get("/health")
async def health():
    return {"status": "ok", "models_ready": models_ready}


@app.post("/classify")
async def classify(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if len(text) > 8000:
        raise HTTPException(status_code=400, detail="Text too long (max 8000 chars).")

    # ── Feature extraction ──
    raw_features = extract_features(text)
    X_raw = np.array([[raw_features[f] for f in FEATURE_NAMES]], dtype=np.float32)

    if not models_ready:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run train_sklearn.py and train_pytorch.py first."
        )
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    diff_class, diff_conf, diff_probs = predict_difficulty(rf_model, X_scaled)

    emo_score = predict_emotional_load(nn_model, X_scaled)
    emo_info  = emotional_label(emo_score)

    feature_groups = {
        "Readability": ["flesch_score", "fk_grade", "avg_word_len", "avg_sent_len", "syllable_density"],
        "Vocabulary":  ["rare_word_pct", "type_token_ratio", "long_word_pct"],
        "Syntax":      ["noun_ratio", "verb_ratio", "adj_ratio", "passive_voice_count"],
        "Cognitive":   ["negation_count", "punct_density", "clause_density"],
    }

    feature_ranges = {
        "flesch_score": (0, 100),  "fk_grade": (0, 18),
        "avg_word_len": (2, 12),   "avg_sent_len": (3, 40),
        "syllable_density": (1, 4), "rare_word_pct": (0, 1),
        "type_token_ratio": (0, 1), "long_word_pct": (0, 1),
        "noun_ratio": (0, 0.5),    "verb_ratio": (0, 0.4),
        "adj_ratio": (0, 0.3),     "passive_voice_count": (0, 5),
        "negation_count": (0, 0.2), "punct_density": (0, 0.1),
        "clause_density": (0, 0.3),
    }

    features_normalised = {}
    for fname, val in raw_features.items():
        lo, hi = feature_ranges.get(fname, (0, 1))
        features_normalised[fname] = float(np.clip((val - lo) / max(hi - lo, 1e-6), 0, 1))

    return {
        "difficulty": {
            "label":        DIFFICULTY_LABELS[diff_class],
            "class_index":  diff_class,
            "confidence":   round(diff_conf, 3),
            "probabilities": {
                DIFFICULTY_LABELS[i]: round(float(p), 3)
                for i, p in enumerate(diff_probs)
            },
        },
        "emotional_load": {
            "score":       round(emo_score, 3),
            "label":       emo_info["label"],
            "color":       emo_info["color"],
            "description": emo_info["description"],
        },
        "features_raw":        {k: round(float(v), 4) for k, v in raw_features.items()},
        "features_normalised": features_normalised,
        "feature_groups":      feature_groups,
    }
