from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STATE_FILE = Path(__file__).parent.parent / "paper_trading_state.json"


class PaperTrader:
    def __init__(
        self,
        initial_balance: float = 10_000.0,
        risk_per_trade: float = 0.02,
        take_profit_pct: float = 0.04,
        stop_loss_pct: float = 0.02,
    ):
        self.risk_per_trade = risk_per_trade
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self._state = self._load_state(initial_balance)

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_state(self, initial_balance: float) -> dict:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
        return {
            "balance": initial_balance,
            "initial_balance": initial_balance,
            "positions": {},
            "trades": [],
            "total_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
        }

    def _save(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    # ── public properties ─────────────────────────────────────────────────────

    @property
    def balance(self) -> float:
        return self._state["balance"]

    @property
    def positions(self) -> dict:
        return self._state["positions"]

    @property
    def trades(self) -> list:
        return self._state["trades"]

    @property
    def stats(self) -> dict:
        total_trades = self._state["win_count"] + self._state["loss_count"]
        win_rate = (
            self._state["win_count"] / total_trades if total_trades > 0 else 0.0
        )
        return {
            "balance": round(self._state["balance"], 2),
            "initial_balance": self._state["initial_balance"],
            "total_pnl": round(self._state["total_pnl"], 2),
            "total_pnl_pct": round(
                (self._state["total_pnl"] / self._state["initial_balance"]) * 100, 2
            ),
            "win_count": self._state["win_count"],
            "loss_count": self._state["loss_count"],
            "total_trades": total_trades,
            "win_rate": round(win_rate * 100, 2),
            "open_positions": len(self._state["positions"]),
        }

    # ── trading actions ───────────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        signal_confidence: float = 1.0,
    ) -> Optional[dict]:
        if symbol in self._state["positions"]:
            return None

        position_size = self._state["balance"] * self.risk_per_trade * signal_confidence
        quantity = position_size / price

        if side == "BUY":
            take_profit = price * (1 + self.take_profit_pct)
            stop_loss = price * (1 - self.stop_loss_pct)
        else:
            take_profit = price * (1 - self.take_profit_pct)
            stop_loss = price * (1 + self.stop_loss_pct)

        position = {
            "id": str(uuid.uuid4())[:8],
            "symbol": symbol,
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "position_size": position_size,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "confidence": signal_confidence,
        }

        self._state["positions"][symbol] = position
        self._state["balance"] -= position_size
        self._save()
        return position

    def close_position(self, symbol: str, current_price: float) -> Optional[dict]:
        pos = self._state["positions"].get(symbol)
        if not pos:
            return None

        if pos["side"] == "BUY":
            pnl = (current_price - pos["entry_price"]) * pos["quantity"]
        else:
            pnl = (pos["entry_price"] - current_price) * pos["quantity"]

        self._state["balance"] += pos["position_size"] + pnl
        self._state["total_pnl"] += pnl

        if pnl > 0:
            self._state["win_count"] += 1
        else:
            self._state["loss_count"] += 1

        trade_record = {
            **pos,
            "exit_price": current_price,
            "pnl": round(pnl, 4),
            "pnl_pct": round((pnl / pos["position_size"]) * 100, 2),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._state["trades"].append(trade_record)
        del self._state["positions"][symbol]
        self._save()
        return trade_record

    def check_and_close_tp_sl(self, symbol: str, current_price: float) -> Optional[dict]:
        pos = self._state["positions"].get(symbol)
        if not pos:
            return None

        hit_tp = (
            (pos["side"] == "BUY" and current_price >= pos["take_profit"]) or
            (pos["side"] == "SELL" and current_price <= pos["take_profit"])
        )
        hit_sl = (
            (pos["side"] == "BUY" and current_price <= pos["stop_loss"]) or
            (pos["side"] == "SELL" and current_price >= pos["stop_loss"])
        )

        if hit_tp or hit_sl:
            result = self.close_position(symbol, current_price)
            if result:
                result["close_reason"] = "TP" if hit_tp else "SL"
            return result
        return None

    def reset(self, initial_balance: float = 10_000.0):
        self._state = {
            "balance": initial_balance,
            "initial_balance": initial_balance,
            "positions": {},
            "trades": [],
            "total_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
        }
        self._save()


paper_trader = PaperTrader()
