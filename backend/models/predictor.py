import numpy as np
import pandas as pd
from .trainer import load_model

_model = None
_scaler = None
_features = None


def _ensure_loaded():
    global _model, _scaler, _features
    if _model is None:
        _model, _scaler, _features = load_model()
    return _model is not None


def reload_model():
    """Reload the model after training completes."""
    global _model, _scaler, _features
    _model, _scaler, _features = load_model()
    return _model is not None


SIGNAL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
SIGNAL_COLOR = {0: "#ef4444", 1: "#f59e0b", 2: "#22c55e"}


def _align_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align the feature columns the model expects within df.
    Missing columns (e.g. 1d_* MTF) are filled with 0.
    Extra columns are dropped.
    """
    missing = [f for f in _features if f not in df.columns]
    if missing:
        for col in missing:
            df[col] = 0.0
    return df[_features]


def predict(df: pd.DataFrame) -> dict:
    if not _ensure_loaded():
        return {"signal": "NO_MODEL", "confidence": 0.0, "probabilities": {}}

    try:
        row = _align_features(df.copy()).iloc[[-1]]
        X = row.fillna(0).replace([np.inf, -np.inf], 0).values
        X_sc = _scaler.transform(X)
        proba = _model.predict_proba(X_sc)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        return {
            "signal": SIGNAL_MAP[pred_class],
            "confidence": round(confidence, 4),
            "probabilities": {
                "sell": round(float(proba[0]), 4),
                "hold": round(float(proba[1]), 4),
                "buy": round(float(proba[2]), 4),
            },
            "color": SIGNAL_COLOR[pred_class],
        }
    except Exception as e:
        return {"signal": "NO_MODEL", "confidence": 0.0,
                "probabilities": {}, "error": str(e)}


def predict_batch(df: pd.DataFrame) -> list[dict]:
    if not _ensure_loaded():
        return []

    try:
        X_df = _align_features(df.copy()).fillna(0).replace([np.inf, -np.inf], 0)
        X_sc = _scaler.transform(X_df.values)
        probas = _model.predict_proba(X_sc)

        results = []
        for i, proba in enumerate(probas):
            pred_class = int(np.argmax(proba))
            results.append({
                "index": str(df.index[i]),
                "signal": SIGNAL_MAP[pred_class],
                "confidence": round(float(proba[pred_class]), 4),
                "probabilities": {
                    "sell": round(float(proba[0]), 4),
                    "hold": round(float(proba[1]), 4),
                    "buy": round(float(proba[2]), 4),
                },
                "color": SIGNAL_COLOR[pred_class],
            })
        return results
    except Exception as e:
        print(f"predict_batch hatası: {e}")
        return []
