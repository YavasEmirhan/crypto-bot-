from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from data.collector import fetch_ohlcv, fetch_full_history, SYMBOLS
from data.processor import add_technical_indicators, create_labels, prepare_features
from models.trainer import is_model_trained, load_metrics
from models.predictor import predict, predict_batch
from models.iterative_trainer import train_until_target
from trading.paper_trader import paper_trader
from trading.scheduler import (
    start_scheduler, stop_scheduler, get_scheduler_status, trade_log
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

app = FastAPI(title="Crypto Bot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────

training_state = {
    "running": False,
    "progress": "",
    "iteration": 0,
    "best_accuracy": 0.0,
    "target_accuracy": 0.80,
    "result": None,
    "auto_trade_started": False,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_processed_df(symbol: str, timeframe: str = "1h", limit: int = 500):
    df = fetch_ohlcv(symbol, timeframe, limit=limit)
    return add_technical_indicators(df)


def _status_cb(msg: str):
    import re
    training_state["progress"] = msg
    # Iteration counter — match on "TERASYON" (works regardless of İ/I encoding)
    msg_upper = msg.upper()
    if "TERASYON" in msg_upper:
        try:
            # "İTERASYON 4/6" or "ITERASYON 4/6" → 4
            m = re.search(r'TERASYON\s+(\d+)\s*/\s*\d+', msg_upper)
            if m:
                training_state["iteration"] = int(m.group(1))
        except Exception:
            pass
    # Accuracy — try each numeric token separately
    msg_lower = msg.lower()
    if "en iyi" in msg_lower or "accuracy" in msg_lower or "ortalama" in msg_lower:
        for part in msg.split():
            try:
                v = float(part.strip(":").strip(",").strip())
                if 0.3 < v <= 1.0:   # accuracy should be between 0.3–1.0
                    if v > training_state["best_accuracy"]:
                        training_state["best_accuracy"] = v
            except (ValueError, TypeError):
                pass


# ── Market data ───────────────────────────────────────────────────────────────

@app.get("/api/symbols")
def get_symbols():
    return {"symbols": SYMBOLS}


@app.get("/api/ohlcv")
def get_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 200):
    df = fetch_ohlcv(symbol, timeframe, limit=limit)
    records = [
        {"time": int(ts.timestamp()), "open": r.open, "high": r.high,
         "low": r.low, "close": r.close, "volume": r.volume}
        for ts, r in df.iterrows()
    ]
    return {"symbol": symbol, "timeframe": timeframe, "data": records}


@app.get("/api/signals")
def get_signals(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 200):
    if not is_model_trained():
        raise HTTPException(400, "Model eğitilmedi")
    df = _get_processed_df(symbol, timeframe, limit=limit)
    return {"symbol": symbol, "signals": predict_batch(df)[-100:]}


@app.get("/api/signal/latest")
def get_latest_signal(symbol: str = "BTC/USDT", timeframe: str = "1h"):
    if not is_model_trained():
        return {"signal": "NO_MODEL", "confidence": 0}
    df = _get_processed_df(symbol, timeframe, limit=400)
    result = predict(df)
    return {**result, "symbol": symbol, "price": float(df["close"].iloc[-1])}


@app.get("/api/price")
def get_price(symbol: str = "BTC/USDT"):
    df = fetch_ohlcv(symbol, "1m", limit=2)
    price = float(df["close"].iloc[-1])
    chg   = float((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100)
    return {"symbol": symbol, "price": price, "change_pct": round(chg, 4)}


# ── Iterative Training ────────────────────────────────────────────────────────

class IterativeTrainRequest(BaseModel):
    target_accuracy: float = 0.80
    max_iterations: Optional[int] = None


def _run_iterative_training(req: IterativeTrainRequest):
    global training_state
    training_state["running"] = True
    training_state["iteration"] = 0
    training_state["best_accuracy"] = 0.0
    training_state["target_accuracy"] = req.target_accuracy
    training_state["result"] = None
    training_state["auto_trade_started"] = False

    try:
        result = train_until_target(
            target_accuracy=req.target_accuracy,
            status_callback=_status_cb,
            max_iterations=req.max_iterations,
        )
        training_state["result"] = result
        training_state["best_accuracy"] = result.get("average_accuracy", 0)

        if result.get("target_reached") or result.get("average_accuracy", 0) >= req.target_accuracy * 0.95:
            training_state["progress"] = (
                f"✅ Hedef ulaşıldı! Accuracy: {result['average_accuracy']:.2%} "
                f"| Auto-trade başlatılıyor..."
            )
            start_scheduler(interval_minutes=60)
            training_state["auto_trade_started"] = True
            training_state["progress"] = (
                f"🚀 CANLI! Acc: {result['average_accuracy']:.2%} "
                f"| Her saat otomatik işlem"
            )
        else:
            acc = result.get("average_accuracy", 0)
            training_state["progress"] = (
                f"Tamamlandı. En iyi accuracy: {acc:.2%} "
                f"(Hedef: {req.target_accuracy:.0%})"
            )
    except Exception as e:
        training_state["progress"] = f"Hata: {str(e)}"
        log.error(f"Iterative training error: {e}", exc_info=True)
    finally:
        training_state["running"] = False


@app.post("/api/train/iterative")
def start_iterative_training(req: IterativeTrainRequest, background_tasks: BackgroundTasks):
    if training_state["running"]:
        raise HTTPException(400, "Eğitim zaten devam ediyor")
    background_tasks.add_task(_run_iterative_training, req)
    return {"message": f"İteratif eğitim başladı. Hedef: {req.target_accuracy:.0%}"}


@app.get("/api/train/status")
def get_training_status():
    return {
        "running": training_state["running"],
        "progress": training_state["progress"],
        "iteration": training_state["iteration"],
        "best_accuracy": round(training_state["best_accuracy"], 4),
        "target_accuracy": training_state["target_accuracy"],
        "model_ready": is_model_trained(),
        "auto_trade_active": training_state["auto_trade_started"] or get_scheduler_status()["running"],
        "result": training_state["result"],
    }


@app.get("/api/train/metrics")
def get_train_metrics():
    return load_metrics()


# ── Scheduler ─────────────────────────────────────────────────────────────────

@app.post("/api/scheduler/start")
def api_start_scheduler(interval_minutes: int = 60):
    if not is_model_trained():
        raise HTTPException(400, "Önce modeli eğitin")
    start_scheduler(interval_minutes)
    return {"message": f"Scheduler başlatıldı: her {interval_minutes} dakika"}


@app.post("/api/scheduler/stop")
def api_stop_scheduler():
    stop_scheduler()
    return {"message": "Scheduler durduruldu"}


@app.get("/api/scheduler/status")
def api_scheduler_status():
    return get_scheduler_status()


@app.get("/api/scheduler/log")
def get_trade_log(limit: int = 50):
    return {"log": list(reversed(trade_log[-limit:]))}


# ── Paper trading ─────────────────────────────────────────────────────────────

@app.get("/api/paper/stats")
def get_paper_stats():
    return paper_trader.stats


@app.get("/api/paper/positions")
def get_positions():
    return {"positions": list(paper_trader.positions.values())}


@app.get("/api/paper/trades")
def get_trades(limit: int = 50):
    return {"trades": list(reversed(paper_trader.trades[-limit:]))}


@app.post("/api/paper/open")
def open_position(symbol: str, side: str, price: float, confidence: float = 1.0):
    pos = paper_trader.open_position(symbol, side, price, confidence)
    if not pos:
        raise HTTPException(400, "Bu sembol için pozisyon zaten açık")
    return {"position": pos}


@app.post("/api/paper/close")
def close_position(symbol: str, current_price: float):
    trade = paper_trader.close_position(symbol, current_price)
    if not trade:
        raise HTTPException(404, "Açık pozisyon bulunamadı")
    return {"trade": trade}


@app.post("/api/paper/reset")
def reset_paper(initial_balance: float = 1000.0):
    paper_trader.reset(initial_balance)
    return {"message": "Sıfırlandı", "balance": initial_balance}


@app.post("/api/paper/auto-trade")
def manual_auto_trade(symbol: str = "BTC/USDT", timeframe: str = "1h"):
    if not is_model_trained():
        raise HTTPException(400, "Model eğitilmedi")
    df = fetch_ohlcv(symbol, timeframe, limit=400)
    current_price = float(df["close"].iloc[-1])
    paper_trader.check_and_close_tp_sl(symbol, current_price)
    df_ind = add_technical_indicators(df)
    signal  = predict(df_ind)
    action  = None
    if signal["signal"] == "BUY" and signal["confidence"] > 0.6:
        pos = paper_trader.open_position(symbol, "BUY", current_price, signal["confidence"])
        action = "opened_buy" if pos else "already_open"
    elif signal["signal"] == "SELL" and signal["confidence"] > 0.6:
        if symbol in paper_trader.positions:
            paper_trader.close_position(symbol, current_price)
            action = "closed_sell"
    return {"symbol": symbol, "price": current_price, "signal": signal,
            "action": action, "stats": paper_trader.stats}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
