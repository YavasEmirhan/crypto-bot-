"""
Continuous Auto-Trade Scheduler
============================
Uses APScheduler to generate signals for all symbols every X minutes,
and updates the paper trader.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timezone
import logging

from data.collector import fetch_ohlcv
from data.processor import add_technical_indicators
from models.predictor import predict
from models.trainer import is_model_trained
from trading.paper_trader import paper_trader

log = logging.getLogger("scheduler")

TRADE_SYMBOLS   = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TIMEFRAME       = "1h"
MIN_CONFIDENCE  = 0.60   # signal threshold

scheduler = BackgroundScheduler(timezone="UTC")

# Live status log
trade_log: list[dict] = []
MAX_LOG  = 200


def _log_action(symbol: str, action: str, price: float, signal: dict):
    entry = {
        "ts":       datetime.now(timezone.utc).isoformat(),
        "symbol":   symbol,
        "action":   action,
        "price":    price,
        "signal":   signal.get("signal"),
        "conf":     signal.get("confidence"),
    }
    trade_log.append(entry)
    if len(trade_log) > MAX_LOG:
        trade_log.pop(0)
    log.info(f"[{action}] {symbol} @ {price:.2f} | {signal.get('signal')} {signal.get('confidence'):.2f}")


def run_auto_trade_cycle():
    """Runs each time the schedule triggers."""
    if not is_model_trained():
        return

    for symbol in TRADE_SYMBOLS:
        try:
            # Price + indicators
            df = fetch_ohlcv(symbol, TIMEFRAME, limit=400)
            current_price = float(df["close"].iloc[-1])

            # TP/SL check (existing positions)
            closed = paper_trader.check_and_close_tp_sl(symbol, current_price)
            if closed:
                _log_action(symbol, "TP/SL_CLOSE", current_price, {"signal": closed["close_reason"], "confidence": 1.0})

            # Generate signal
            df_ind = add_technical_indicators(df)
            signal = predict(df_ind)

            # Trade decision
            if signal["signal"] == "BUY" and signal["confidence"] >= MIN_CONFIDENCE:
                if symbol not in paper_trader.positions:
                    pos = paper_trader.open_position(
                        symbol, "BUY", current_price, signal["confidence"]
                    )
                    if pos:
                        _log_action(symbol, "OPEN_BUY", current_price, signal)

            elif signal["signal"] == "SELL" and signal["confidence"] >= MIN_CONFIDENCE:
                if symbol in paper_trader.positions:
                    paper_trader.close_position(symbol, current_price)
                    _log_action(symbol, "CLOSE_SELL", current_price, signal)

        except Exception as e:
            log.error(f"Auto-trade hatası {symbol}: {e}")


def start_scheduler(interval_minutes: int = 60):
    if scheduler.running:
        return
    scheduler.add_job(
        run_auto_trade_cycle,
        trigger=IntervalTrigger(minutes=interval_minutes),
        id="auto_trade",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc),  # run immediately the first time
    )
    scheduler.start()
    log.info(f"✅ Auto-trade scheduler başlatıldı: her {interval_minutes} dakika")


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
        log.info("Scheduler durduruldu.")


def get_scheduler_status() -> dict:
    jobs = []
    if scheduler.running:
        for job in scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "next_run": str(job.next_run_time),
            })
    return {
        "running": scheduler.running,
        "jobs": jobs,
        "recent_actions": trade_log[-20:],
    }
