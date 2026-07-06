"""
Hybrid ML Model: XGBoost + LightGBM + RandomForest Ensemble
============================================================
Time-series-safe training with walk-forward cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(exist_ok=True)

SCALER_PATH   = MODEL_DIR / "scaler.pkl"
MODEL_PATH    = MODEL_DIR / "ensemble_model.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"
METRICS_PATH  = MODEL_DIR / "metrics.pkl"


def build_ensemble() -> VotingClassifier:
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_samples=30,
        class_weight="balanced",   # balance out HOLD class dominance
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        voting="soft",
        weights=[2, 2, 1],  # XGB + LGBM weighted more heavily
    )


def walk_forward_train(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> tuple:
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scaler = RobustScaler()
    all_reports = []
    X_arr, y_arr = X.values, y.values

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc   = scaler.transform(X_val)

        model = build_ensemble()
        model.fit(X_train_sc, y_train)

        preds = model.predict(X_val_sc)
        acc   = accuracy_score(y_val, preds)
        f1    = f1_score(y_val, preds, average="macro")
        report = classification_report(
            y_val, preds, target_names=["sell", "hold", "buy"], zero_division=0
        )
        all_reports.append({"fold": fold + 1, "accuracy": acc, "f1": f1, "report": report})
        print(f"Fold {fold+1} | Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
        print(report)

    # Final model using all data
    X_sc = scaler.fit_transform(X_arr)
    final_model = build_ensemble()
    final_model.fit(X_sc, y_arr)

    return final_model, scaler, all_reports


def train_and_save(X: pd.DataFrame, y: pd.Series) -> dict:
    print(f"Eğitim başlıyor: {len(X)} örnek, {X.shape[1]} özellik")
    print(f"Sınıf dağılımı:\n{y.value_counts().to_dict()}")

    model, scaler, reports = walk_forward_train(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    avg_acc = float(np.mean([r["accuracy"] for r in reports]))
    avg_f1  = float(np.mean([r["f1"] for r in reports]))

    metrics = {
        "average_accuracy": avg_acc,
        "average_f1": avg_f1,
        "n_features": X.shape[1],
        "n_samples": len(X),
        "folds": reports,
    }
    joblib.dump(metrics, METRICS_PATH)

    print(f"\nOrtalama Accuracy: {avg_acc:.4f} | Ortalama F1: {avg_f1:.4f}")
    return metrics


def load_model():
    if not MODEL_PATH.exists():
        return None, None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(FEATURES_PATH)


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return joblib.load(METRICS_PATH)
    return {}


def is_model_trained() -> bool:
    return MODEL_PATH.exists()
