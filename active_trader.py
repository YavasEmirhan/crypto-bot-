#!/usr/bin/env python3
"""
ACTIVE Paper Trader — Dynamic TP/SL + Trailing Stop + Signal Tracking
Capital: $1,000 | 8 symbols | 5min decision loop
"""
import json, time, sys
from datetime import datetime, timezone
import requests

BACKEND   = "http://localhost:8000"
CAPITAL   = 1000.0
POS_PCT   = 0.15     # 15% per position → max 6 positions ($150 × 6 = $900)
TP_PCT    = 0.008    # 0.8% TP — reachable within 30-60min
SL_PCT    = 0.005    # 0.5% SL — quick exit
TRAIL_ACT = 0.003    # trailing activates at 0.3% profit
TRAIL_PCT = 0.002    # trail distance 0.2% (from peak)
CONF_OPEN = 0.65     # min confidence level for entry (0.70 was too strict, went 19+ hours without a signal)
CONF_FLIP = 0.68     # force exit if opposite signal exceeds this threshold
CHECK_SEC = 60       # price check interval
DECIDE_N  = 5        # how many checks between new signal evaluations (5min)
MAX_OPEN  = 4        # limit on simultaneously open positions (correlation risk control)
LOG_FILE  = "/tmp/active_trader.json"
TEXT_LOG  = "/tmp/active_trader.log"

SYMBOLS = [
    "BTC/USDT","ETH/USDT","SOL/USDT",
    "XRP/USDT","BNB/USDT","DOGE/USDT","ADA/USDT","AVAX/USDT"
]

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(TEXT_LOG, "a") as f:
        f.write(line + "\n")

def save():
    with open(LOG_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def get_price(sym):
    r = requests.get(f"{BACKEND}/api/price?symbol={sym}", timeout=8)
    return r.json()["price"]

def get_signal(sym):
    r = requests.get(f"{BACKEND}/api/signal/latest?symbol={sym}", timeout=8)
    return r.json()

def pct(v): return f"{v:+.2f}%"

# ─── STARTUP (resume from existing state if present, otherwise start fresh) ────────────
import os
resumed = False
if os.path.exists(LOG_FILE):
    try:
        with open(LOG_FILE) as f:
            state = json.load(f)
        resumed = True
    except Exception:
        state = None
else:
    state = None

if state is None:
    open(TEXT_LOG, "w").close()
    state = {
        "started": datetime.now(timezone.utc).isoformat(),
        "capital": CAPITAL,
        "balance": CAPITAL,
        "positions": {},   # sym → pos dict
        "closed_trades": [],
        "decisions": [],
        "checks": [],
    }
    save()

log("=" * 65)
log(f"  AKTİF TRADER {'DEVAM EDİYOR' if resumed else 'BAŞLADI'}")
if resumed:
    log(f"  Mevcut bakiye: ${state['balance']:,.2f} | Açık pozisyon: {len(state['positions'])} | Kapanan işlem: {len(state['closed_trades'])}")
log(f"  Sermaye: ${CAPITAL:,.2f} | TP: %{TP_PCT*100:.1f} | SL: %{SL_PCT*100:.1f} | Trailing: %{TRAIL_PCT*100:.1f}")
log(f"  Min güven (giriş): %{CONF_OPEN*100:.0f} | Sinyal flip (çıkış): %{CONF_FLIP*100:.0f}")
log("=" * 65)

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────
def open_pos(sym, side, price, conf, reason=""):
    if sym in state["positions"]:
        return
    if len(state["positions"]) >= MAX_OPEN:
        log(f"  ⏸️  {sym}: maks pozisyon sayısına ulaşıldı ({MAX_OPEN}), giriş yok")
        return
    avail = state["balance"]
    pos_usd = min(CAPITAL * POS_PCT, avail * 0.95)
    if pos_usd < 10:
        log(f"  ⚠️  {sym}: yeterli bakiye yok (${avail:.2f})")
        return
    qty = pos_usd / price
    if side == "BUY":
        tp = price * (1 + TP_PCT)
        sl = price * (1 - SL_PCT)
    else:
        tp = price * (1 - TP_PCT)
        sl = price * (1 + SL_PCT)

    state["balance"] -= pos_usd
    pos = {
        "symbol": sym, "side": side,
        "entry": price, "qty": qty, "pos_usd": pos_usd,
        "tp": tp, "sl": sl, "conf": conf,
        "peak_pnl_pct": 0.0,
        "trailing_active": False,
        "trail_sl": sl,
        "opened_at": ts(), "status": "OPEN",
        "reason": reason,
    }
    state["positions"][sym] = pos
    dir_lbl = "LONG 📈" if side == "BUY" else "SHORT 📉"
    log(f"\n  ✅ GİRİŞ: {sym} {dir_lbl} @ ${price:,.4f}")
    log(f"     Büyüklük: ${pos_usd:.2f} | Güven: {conf:.1%} | Neden: {reason}")
    log(f"     TP: ${tp:,.4f} | SL: ${sl:,.4f} | Trail aktif > %{TRAIL_ACT*100:.1f}")
    state["decisions"].append({"ts": ts(), "action": "OPEN", "sym": sym, "side": side,
                                "price": price, "conf": conf, "reason": reason})
    save()

def close_pos(sym, price, reason):
    pos = state["positions"].get(sym)
    if not pos or pos["status"] != "OPEN":
        return None
    if pos["side"] == "BUY":
        pnl = (price - pos["entry"]) * pos["qty"]
    else:
        pnl = (pos["entry"] - price) * pos["qty"]
    pnl_pct = pnl / pos["pos_usd"] * 100
    state["balance"] += pos["pos_usd"] + pnl
    pos.update({"status": "CLOSED", "exit": price,
                "pnl": round(pnl, 4), "pnl_pct": round(pnl_pct, 2),
                "closed_at": ts(), "close_reason": reason})
    state["closed_trades"].append(dict(pos))
    del state["positions"][sym]
    icon = "✅" if pnl > 0 else "❌"
    log(f"\n  {icon} ÇIKIŞ: {sym} @ ${price:,.4f} | PNL: ${pnl:+.4f} ({pct(pnl_pct)}) | {reason}")
    state["decisions"].append({"ts": ts(), "action": "CLOSE", "sym": sym,
                                "price": price, "pnl": pnl, "reason": reason})
    save()
    return pnl

# ─── INITIAL ENTRIES ─────────────────────────────────────────────────────────────
log("\n  📡 İlk sinyal taraması...")
for sym in SYMBOLS:
    try:
        sig = get_signal(sym)
        signal, conf, price = sig["signal"], sig["confidence"], sig["price"]
        if signal == "SELL" and conf >= CONF_OPEN:
            open_pos(sym, "SELL", price, conf, f"Açılış taraması SELL {conf:.0%}")
        elif signal == "BUY" and conf >= CONF_OPEN:
            open_pos(sym, "BUY", price, conf, f"Açılış taraması BUY {conf:.0%}")
        else:
            log(f"  ⏸️  {sym}: {signal} {conf:.1%} → giriş yok (düşük güven veya HOLD)")
    except Exception as e:
        log(f"  ⚠️  {sym}: {e}")

log(f"\n  Açık pozisyon: {len(state['positions'])} | Bakiye: ${state['balance']:.2f}")
log("─" * 65)

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
check_num = 0
start = time.time()

while True:
    time.sleep(CHECK_SEC)
    check_num += 1

    total_unrealized = 0.0
    status_lines = []

    # 1. Update existing positions (TP/SL/Trailing)
    for sym, pos in list(state["positions"].items()):
        if pos["status"] != "OPEN":
            continue
        try:
            cur = get_price(sym)
        except:
            continue

        if pos["side"] == "BUY":
            pnl     = (cur - pos["entry"]) * pos["qty"]
            hit_tp  = cur >= pos["tp"]
            hit_sl  = cur <= pos["trail_sl"]
        else:
            pnl     = (pos["entry"] - cur) * pos["qty"]
            hit_tp  = cur <= pos["tp"]
            hit_sl  = cur >= pos["trail_sl"]

        pnl_pct = pnl / pos["pos_usd"] * 100
        total_unrealized += pnl

        # Trailing stop update
        if pnl_pct >= TRAIL_ACT * 100 and not pos["trailing_active"]:
            pos["trailing_active"] = True
            if pos["side"] == "BUY":
                pos["trail_sl"] = cur * (1 - TRAIL_PCT)
            else:
                pos["trail_sl"] = cur * (1 + TRAIL_PCT)
            log(f"  🔔 {sym}: Trailing AKTIF @ ${cur:,.4f} | Yeni SL: ${pos['trail_sl']:,.4f}")
            save()
        elif pos["trailing_active"]:
            if pos["side"] == "BUY":
                new_trail = cur * (1 - TRAIL_PCT)
                if new_trail > pos["trail_sl"]:
                    pos["trail_sl"] = new_trail
                    save()
            else:
                new_trail = cur * (1 + TRAIL_PCT)
                if new_trail < pos["trail_sl"]:
                    pos["trail_sl"] = new_trail
                    save()

        if hit_tp:
            close_pos(sym, cur, "TP ✅")
        elif hit_sl:
            reason = "Trailing SL 🔒" if pos["trailing_active"] else "SL ❌"
            close_pos(sym, cur, reason)
        else:
            icon = "🟢" if pnl >= 0 else "🔴"
            trail_mark = " [trail]" if pos["trailing_active"] else ""
            status_lines.append(
                f"  {icon} {sym}: ${cur:,.4f} | PNL: ${pnl:+.2f} ({pct(pnl_pct)}){trail_mark}"
            )

    # 2. Every 5 checks, scan signals → new opportunity?
    if check_num % DECIDE_N == 0:
        for sym in SYMBOLS:
            if sym in state["positions"]:
                continue  # already open
            try:
                sig = get_signal(sym)
                signal, conf, price = sig["signal"], sig["confidence"], sig["price"]
                if signal == "SELL" and conf >= CONF_OPEN:
                    open_pos(sym, "SELL", price, conf, f"Tarama #{check_num} SELL {conf:.0%}")
                elif signal == "BUY" and conf >= CONF_OPEN:
                    open_pos(sym, "BUY", price, conf, f"Tarama #{check_num} BUY {conf:.0%}")
            except:
                pass

        # Signal flip check on open positions
        for sym, pos in list(state["positions"].items()):
            if pos["status"] != "OPEN":
                continue
            try:
                sig = get_signal(sym)
                signal, conf = sig["signal"], sig["confidence"]
                cur = sig["price"]
                if pos["side"] == "SELL" and signal == "BUY" and conf >= CONF_FLIP:
                    close_pos(sym, cur, f"Sinyal flip → BUY {conf:.0%} 🔄")
                elif pos["side"] == "BUY" and signal == "SELL" and conf >= CONF_FLIP:
                    close_pos(sym, cur, f"Sinyal flip → SELL {conf:.0%} 🔄")
            except:
                pass

    # 3. Report every 15 checks (15min)
    if check_num % 15 == 0:
        elapsed = (time.time() - start) / 60
        realized = sum(t.get("pnl", 0) for t in state["closed_trades"])
        open_margin = sum(p["pos_usd"] for p in state["positions"].values() if p.get("status") == "OPEN")
        equity = state["balance"] + open_margin + total_unrealized
        net = equity - CAPITAL
        wins = sum(1 for t in state["closed_trades"] if t.get("pnl", 0) > 0)
        losses = sum(1 for t in state["closed_trades"] if t.get("pnl", 0) <= 0)

        log(f"\n  ─── RAPOR +{elapsed:.0f}dk | Check #{check_num} ───────────────────")
        for line in status_lines:
            log(line)
        icon_net = "🟢" if net >= 0 else "🔴"
        log(f"  {icon_net} Equity: ${equity:,.2f} | Net PNL: ${net:+.2f} ({pct(net/CAPITAL*100)})")
        log(f"     Realized: ${realized:+.2f} | Unrealized: ${total_unrealized:+.2f}")
        log(f"     Kazanan: {wins} | Kaybeden: {losses} | Açık: {len(state['positions'])}")
        log("  " + "─" * 60)

        check_entry = {
            "check": check_num, "elapsed_min": round(elapsed, 1),
            "equity": round(equity, 2), "net_pnl": round(net, 4),
            "realized": round(realized, 4), "unrealized": round(total_unrealized, 4),
            "open_count": len(state["positions"]), "wins": wins, "losses": losses,
        }
        state["checks"].append(check_entry)
        save()
