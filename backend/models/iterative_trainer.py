"""
Iterative Training Engine
======================
Tries different parameter combinations until the target accuracy is reached.
On each iteration:
  1. Fetches more / different data
  2. Tries different horizon & threshold
  3. Applies feature selection
  4. Stops once accuracy >= target
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, StackingClassifier
import joblib
from pathlib import Path
import time

from data.collector import fetch_full_history, SYMBOLS, enrich_with_futures_data
from data.processor import (
    add_technical_indicators, create_labels,
    create_labels_triple_barrier, prepare_features,
)
from models.trainer import MODEL_PATH, SCALER_PATH, FEATURES_PATH, METRICS_PATH

# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid (tries a different combination on each iteration)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRID = [
    # ── 4h Sweet Spot Grid (proven: 4h + horizon=3 works best) ─────────────
    # Simulation analysis finding: Triple Barrier / Stacking didn't help.
    # Focus: fine-tuning threshold_mult, horizon, depth, lr, top_n.

    # A — Proven baseline (64.21% reference point)
    {"symbols": ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"],
     "start": "2020-01-01", "tf": "4h", "horizon": 3, "threshold_mult": 0.50,
     "n_estimators": 1200, "depth": 7, "lr": 0.015, "top_n": 80},

    # B — Tight threshold (0.35x ATR → clear BUY/SELL, little HOLD)
    {"symbols": ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"],
     "start": "2020-01-01", "tf": "4h", "horizon": 3, "threshold_mult": 0.35,
     "n_estimators": 1200, "depth": 7, "lr": 0.015, "top_n": 80},

    # C — Wide threshold (0.65x ATR → reliable but fewer signals)
    {"symbols": ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"],
     "start": "2020-01-01", "tf": "4h", "horizon": 3, "threshold_mult": 0.65,
     "n_estimators": 1200, "depth": 7, "lr": 0.015, "top_n": 80},

    # D — Short horizon=2 (8-hour horizon, faster signal)
    {"symbols": ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"],
     "start": "2020-01-01", "tf": "4h", "horizon": 2, "threshold_mult": 0.45,
     "n_estimators": 1200, "depth": 7, "lr": 0.015, "top_n": 80},

    # E — Long horizon=4 (16-hour horizon, letting the trend settle)
    {"symbols": ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"],
     "start": "2020-01-01", "tf": "4h", "horizon": 4, "threshold_mult": 0.50,
     "n_estimators": 1200, "depth": 7, "lr": 0.015, "top_n": 80},

    # F — Deep learning (2000 trees, low LR → less overfit)
    {"symbols": ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"],
     "start": "2020-01-01", "tf": "4h", "horizon": 3, "threshold_mult": 0.50,
     "n_estimators": 2000, "depth": 6, "lr": 0.008, "top_n": 80},

    # G — More features (top_n=120, broader knowledge base)
    {"symbols": ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"],
     "start": "2020-01-01", "tf": "4h", "horizon": 3, "threshold_mult": 0.50,
     "n_estimators": 1200, "depth": 7, "lr": 0.015, "top_n": 120},

    # H — Start from 2019 (bear + bull + crash + recovery → multi-regime data)
    {"symbols": ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"],
     "start": "2019-01-01", "tf": "4h", "horizon": 3, "threshold_mult": 0.50,
     "n_estimators": 1500, "depth": 7, "lr": 0.012, "top_n": 80},

    # I — 3 symbols, highest quality data (noise reduction)
    {"symbols": ["BTC/USDT","ETH/USDT","SOL/USDT"],
     "start": "2020-01-01", "tf": "4h", "horizon": 3, "threshold_mult": 0.45,
     "n_estimators": 1500, "depth": 7, "lr": 0.012, "top_n": 80},

    # J — Best combination (2019 + threshold=0.40 + depth=8 + n=1800)
    {"symbols": ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"],
     "start": "2019-01-01", "tf": "4h", "horizon": 3, "threshold_mult": 0.40,
     "n_estimators": 1800, "depth": 8, "lr": 0.010, "top_n": 100},
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection
# ─────────────────────────────────────────────────────────────────────────────

def select_top_features(X: pd.DataFrame, y: pd.Series, top_n: int = 80) -> list[str]:
    """Select the most informative features using mutual information."""
    print(f"  Feature seçimi: {X.shape[1]} → {top_n}")
    X_filled = X.fillna(0)
    mi_scores = mutual_info_classif(X_filled, y, random_state=42, n_jobs=-1)
    mi_series  = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    top = mi_series.head(top_n).index.tolist()
    print(f"  Top 5 feature: {top[:5]}")
    return top


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

class _BalancedXGB(XGBClassifier):
    """XGBClassifier wrapper: automatically computes sample_weight inside fit().
    Provides balanced training without requiring VotingClassifier metadata routing."""
    def fit(self, X, y, **kwargs):
        sw = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sw, **kwargs)


def build_stacking_model(p: dict) -> StackingClassifier:
    """StackingClassifier — more adaptive than VotingClassifier."""
    xgb = XGBClassifier(
        n_estimators=p["n_estimators"],
        max_depth=max(p["depth"] - 1, 3),
        learning_rate=p["lr"],
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=10,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=p["n_estimators"],
        max_depth=max(p["depth"] - 1, 3),
        learning_rate=p["lr"],
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_samples=50,
        class_weight="balanced",
        num_leaves=31,
        min_split_gain=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=min(p["n_estimators"] // 2, 400),
        max_depth=p["depth"],
        min_samples_leaf=30,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    meta = LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
    )
    return StackingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=3,
        n_jobs=1,
    )


def build_model(p: dict) -> VotingClassifier:
    xgb = _BalancedXGB(
        n_estimators=p["n_estimators"],
        max_depth=p["depth"],
        learning_rate=p["lr"],
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["depth"],
        learning_rate=p["lr"],
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=min(p["n_estimators"] // 2, 400),
        max_depth=p["depth"] + 1,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        voting="soft",
        weights=[2, 2, 1],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single iteration
# ─────────────────────────────────────────────────────────────────────────────

def run_iteration(
    p: dict,
    status_callback=None,
    n_splits: int = 5,
) -> dict:

    def log(msg: str):
        print(msg)
        if status_callback:
            status_callback(msg)

    use_tb      = p.get("use_triple_barrier", False)
    use_futures = p.get("use_futures_data", False)
    use_stack   = p.get("use_stacking", False)

    mode = "TripleBarrier" if use_tb else "ATR-threshold"
    log(f"📥 Veri çekiliyor: {p['symbols']} | {p['start']} | {p['tf']} | {mode}")
    all_dfs = []
    for sym in p["symbols"]:
        log(f"  ↳ {sym}...")
        try:
            df = fetch_full_history(sym, p["tf"], p["start"])
            if df.empty:
                continue

            # Futures data integration (optional)
            if use_futures:
                try:
                    df = enrich_with_futures_data(df, sym, p["tf"])
                except Exception as fe:
                    log(f"    ⚠️ Futures data hatası: {fe}")

            df = add_technical_indicators(df)

            if use_tb:
                df = create_labels_triple_barrier(
                    df,
                    horizon=p["horizon"],
                    tp_multiplier=p.get("tb_tp", 1.5),
                    sl_multiplier=p.get("tb_sl", 1.0),
                )
            else:
                df = create_labels(
                    df,
                    horizon=p["horizon"],
                    use_dynamic_threshold=True,
                    threshold_mult=p.get("threshold_mult", 1.0),
                )
            if len(df) > 100:
                all_dfs.append(df)
        except Exception as e:
            log(f"  ⚠️ {sym} hatası: {e}")

    if not all_dfs:
        return {"error": "Veri alınamadı"}

    combined = pd.concat(all_dfs).sort_index()
    log(f"✅ Toplam veri: {len(combined)} satır")

    X_full, y_full = prepare_features(combined)

    # Feature selection — top_n value from PARAM_GRID, otherwise default 80
    top_n = p.get("top_n", 100 if use_tb else 80)
    selected = select_top_features(X_full, y_full, top_n=top_n)
    X = X_full[selected]

    # Walk-forward cross-validation
    # Purged gap: leave a gap equal to the triple barrier horizon (prevents data leakage)
    embargo = p["horizon"] if use_tb else 0
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = RobustScaler()
    fold_accs, fold_f1s = [], []

    X_arr, y_arr = X.values, y_full.values

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
        # Purging: embargo between end of training and start of validation
        if embargo > 0:
            train_end = train_idx[-1]
            val_start = val_idx[0]
            purge_end = min(train_end + embargo, val_start)
            train_idx = train_idx[train_idx <= train_end - embargo]
            if len(train_idx) < 100:
                continue

        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        model = build_stacking_model(p) if use_stack else build_model(p)
        model.fit(X_tr_sc, y_tr)

        preds = model.predict(X_val_sc)
        acc = accuracy_score(y_val, preds)
        f1  = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        log(f"  Fold {fold+1}/{n_splits} | Acc: {acc:.4f} | F1: {f1:.4f}")

    if not fold_accs:
        return {"error": "Hiç fold tamamlanamadı"}

    avg_acc = float(np.mean(fold_accs))
    avg_f1  = float(np.mean(fold_f1s))
    log(f"🎯 Ortalama Accuracy: {avg_acc:.4f} | F1: {avg_f1:.4f}")

    # Final model (all data)
    X_sc = scaler.fit_transform(X_arr)
    final = build_stacking_model(p) if use_stack else build_model(p)
    final.fit(X_sc, y_arr)

    # Save
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(selected, FEATURES_PATH)

    metrics = {
        "average_accuracy": avg_acc,
        "average_f1": avg_f1,
        "n_features": len(selected),
        "n_samples": len(X),
        "params": p,
        "fold_accs": fold_accs,
    }
    joblib.dump(metrics, METRICS_PATH)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main loop: reach the target accuracy
# ─────────────────────────────────────────────────────────────────────────────

def train_until_target(
    target_accuracy: float = 0.80,
    status_callback=None,
    max_iterations: int = None,
) -> dict:
    def log(msg: str):
        print(msg)
        if status_callback:
            status_callback(msg)

    best_acc   = 0.0
    best_metrics = {}
    max_iter = max_iterations or len(PARAM_GRID)

    for i, params in enumerate(PARAM_GRID[:max_iter]):
        log(f"\n{'='*55}")
        log(f"İTERASYON {i+1}/{max_iter} | Hedef: {target_accuracy:.0%}")
        log(f"{'='*55}")

        result = run_iteration(params, status_callback=status_callback)

        if "error" in result:
            log(f"❌ Hata: {result['error']}")
            continue

        acc = result["average_accuracy"]
        if acc > best_acc:
            best_acc     = acc
            best_metrics = result
            log(f"⭐ Yeni en iyi: {best_acc:.4f}")

        if acc >= target_accuracy:
            log(f"\n🏆 HEDEF ULAŞILDI! Accuracy: {acc:.4f} >= {target_accuracy:.4f}")
            log("✅ Model kaydedildi. Auto-trading başlatılıyor...")
            return {**result, "target_reached": True, "iteration": i + 1}

        log(f"📈 Devam: {acc:.4f} < {target_accuracy:.4f}. Sonraki parametre seti...")
        time.sleep(2)

    log(f"\n⚠️ {max_iter} iterasyon tamamlandı.")
    log(f"En iyi accuracy: {best_acc:.4f}")
    return {**best_metrics, "target_reached": best_acc >= target_accuracy}
