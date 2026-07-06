#!/usr/bin/env python3
"""
3-Hour Transparent Paper Trading Simulation
Capital: $1,000 | BTC/ETH/SOL | Live model signal + real OKX price
"""
import json, time, sys, os
from datetime import datetime, timezone, timedelta
import requests

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BACKEND        = "http://localhost:8000"
SYMBOLS        = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
CAPITAL        = 1000.0
POS_PCT        = 0.30    # 30% of capital per position → max 3 × $300
TP_PCT         = 0.020   # 2% take profit
SL_PCT         = 0.012   # 1.2% stop loss  (risk:reward ≈ 1.7:1)
DURATION_SEC   = 3 * 3600
LOG_FILE       = "/tmp/sim3h.json"
TEXT_LOG       = "/tmp/sim3h.log"
CHECK_SEC      = 60      # price check (seconds)
REPORT_EVERY   = 15      # every Nth check gets a detailed report

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

def pct_str(v):
    return f"{v:+.2f}%"

def get_price(symbol: str) -> float:
    r = requests.get(f"{BACKEND}/api/price?symbol={symbol}", timeout=8)
    return r.json()["price"]

def get_signal(symbol: str) -> dict:
    r = requests.get(f"{BACKEND}/api/signal/latest?symbol={symbol}", timeout=8)
    return r.json()

def log_line(msg: str):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(TEXT_LOG, "a") as f:
        f.write(line + "\n")

def save_state():
    with open(LOG_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

# ─── INITIAL STATE ─────────────────────────────────────────────────────────────
now_utc  = datetime.now(timezone.utc)
end_utc  = now_utc + timedelta(hours=3)

state = {
    "start_time":    now_utc.isoformat(),
    "end_time":      end_utc.isoformat(),
    "config": {
        "capital": CAPITAL,
        "pos_pct": POS_PCT,
        "tp_pct":  TP_PCT,
        "sl_pct":  SL_PCT,
    },
    "balance":        CAPITAL,
    "positions":      {},
    "closed_trades":  [],
    "checks":         [],
}

# Clear the log file
open(TEXT_LOG, "w").close()

# ─── OPENING ───────────────────────────────────────────────────────────────────
log_line("=" * 60)
log_line("  3-SAATLİK PAPER TRADING SİMÜLASYONU BAŞLADI")
log_line(f"  Sermaye: ${CAPITAL:,.2f}  |  Bitiş: {end_utc.strftime('%H:%M UTC')}")
log_line(f"  Pozisyon: %{POS_PCT*100:.0f} (${CAPITAL*POS_PCT:.0f}) × 3 sembol")
log_line(f"  TP: %{TP_PCT*100:.1f}  |  SL: %{SL_PCT*100:.1f}")
log_line("=" * 60)

for sym in SYMBOLS:
    try:
        sig   = get_signal(sym)
        price = sig["price"]
        signal= sig["signal"]
        conf  = sig["confidence"]
        probs = sig.get("probabilities", {})

        if signal in ("BUY", "SELL"):
            pos_usd = CAPITAL * POS_PCT
            qty     = pos_usd / price
            if signal == "BUY":
                tp  = price * (1 + TP_PCT)
                sl  = price * (1 - SL_PCT)
                dir_lbl = "LONG 📈"
            else:
                tp  = price * (1 - TP_PCT)
                sl  = price * (1 + SL_PCT)
                dir_lbl = "SHORT 📉"

            state["positions"][sym] = {
                "symbol": sym, "side": signal, "dir": dir_lbl,
                "entry": price, "qty": qty, "pos_usd": pos_usd,
                "tp": tp, "sl": sl, "conf": conf,
                "probs": probs,
                "opened_at": ts(), "status": "OPEN",
                "pnl": 0.0,
            }
            state["balance"] -= pos_usd

            log_line(f"  ✅ POZİSYON AÇILDI: {sym} {dir_lbl}")
            log_line(f"     Giriş: ${price:,.4f}  |  Miktar: {qty:.6f}")
            log_line(f"     Büyüklük: ${pos_usd:.2f}  |  Güven: {conf:.1%}")
            log_line(f"     TP: ${tp:,.4f} ({pct_str(TP_PCT*100)})  |  SL: ${sl:,.4f} ({pct_str(-SL_PCT*100)})")
            log_line(f"     Olasılıklar: SELL {probs.get('sell',0):.1%} | HOLD {probs.get('hold',0):.1%} | BUY {probs.get('buy',0):.1%}")
        else:
            log_line(f"  ⏸️  {sym}: HOLD sinyali → pozisyon açılmadı")
    except Exception as e:
        log_line(f"  ❌ {sym}: sinyal alınamadı → {e}")

open_syms = [s for s,p in state["positions"].items() if p["status"] == "OPEN"]
log_line(f"\n  Açık pozisyon: {len(open_syms)}/{len(SYMBOLS)}")
log_line(f"  Kullanılan: ${CAPITAL - state['balance']:.2f}  |  Serbest: ${state['balance']:.2f}")
log_line("─" * 60)
save_state()

# ─── MONITORING LOOP ─────────────────────────────────────────────────────────
start_ts   = time.time()
check_num  = 0

while time.time() - start_ts < DURATION_SEC:
    time.sleep(CHECK_SEC)
    check_num += 1
    elapsed   = time.time() - start_ts
    remaining = DURATION_SEC - elapsed

    events          = []
    status_lines    = []
    total_unrealized = 0.0

    for sym, pos in list(state["positions"].items()):
        if pos["status"] != "OPEN":
            continue
        try:
            cur = get_price(sym)
            if pos["side"] == "BUY":
                pnl   = (cur - pos["entry"]) * pos["qty"]
                hit_tp = cur >= pos["tp"]
                hit_sl = cur <= pos["sl"]
            else:
                pnl   = (pos["entry"] - cur) * pos["qty"]
                hit_tp = cur <= pos["tp"]
                hit_sl = cur >= pos["sl"]

            pnl_pct = pnl / pos["pos_usd"] * 100
            total_unrealized += pnl

            if hit_tp or hit_sl:
                reason = "TP ✅" if hit_tp else "SL ❌"
                pos.update({
                    "status": "CLOSED", "exit": cur,
                    "pnl": round(pnl, 4), "pnl_pct": round(pnl_pct, 2),
                    "closed_at": ts(), "reason": reason,
                })
                state["balance"] += pos["pos_usd"] + pnl
                state["closed_trades"].append(dict(pos))
                total_unrealized -= pnl  # now realized
                events.append({
                    "sym": sym, "reason": reason,
                    "exit": cur, "pnl": pnl, "pnl_pct": pnl_pct,
                })
            else:
                icon = "🟢" if pnl >= 0 else "🔴"
                move_pct = (cur - pos["entry"]) / pos["entry"] * 100
                status_lines.append({
                    "sym": sym, "icon": icon,
                    "cur": cur, "pnl": pnl, "pnl_pct": pnl_pct,
                    "move_pct": move_pct,
                    "tp_dist": abs(pos["tp"] - cur) / cur * 100,
                    "sl_dist": abs(pos["sl"] - cur) / cur * 100,
                })
        except Exception as e:
            status_lines.append({"sym": sym, "error": str(e)})

    # Realized PNL
    realized   = sum(t.get("pnl", 0) for t in state["closed_trades"])
    open_margin = sum(p["pos_usd"] for p in state["positions"].values() if p.get("status") == "OPEN")
    equity     = state["balance"] + open_margin + total_unrealized
    net_pnl    = equity - CAPITAL

    check_entry = {
        "check":          check_num,
        "elapsed_min":    round(elapsed / 60, 1),
        "remaining_min":  round(remaining / 60, 1),
        "equity":         round(equity, 2),
        "net_pnl":        round(net_pnl, 4),
        "realized_pnl":   round(realized, 4),
        "unrealized_pnl": round(total_unrealized, 4),
        "events":         events,
    }
    state["checks"].append(check_entry)
    save_state()

    # Report immediately if there's an event
    if events:
        log_line("\n  ═══ OLAY ═══")
        for ev in events:
            icon = "✅" if "TP" in ev["reason"] else "❌"
            log_line(f"  {icon} {ev['sym']} KAPANDI → {ev['reason']}")
            log_line(f"     Çıkış: ${ev['exit']:,.4f}  |  PNL: ${ev['pnl']:+.4f} ({pct_str(ev['pnl_pct'])})")

    # Detailed table every REPORT_EVERY checks
    if check_num % REPORT_EVERY == 0 or events:
        mins_e = int(elapsed // 60)
        mins_l = int(remaining // 60)
        log_line(f"\n  ─── RAPOR +{mins_e}dk (kalan {mins_l}dk) | Check #{check_num} ─────────────")
        for s in status_lines:
            if "error" in s:
                log_line(f"  ⚠️  {s['sym']}: {s['error']}")
            else:
                log_line(f"  {s['icon']} {s['sym']}: ${s['cur']:,.4f} "
                         f"| PNL: ${s['pnl']:+.2f} ({pct_str(s['pnl_pct'])}) "
                         f"| Fiyat hareketi: {pct_str(s['move_pct'])}")
                log_line(f"        TP'ye kalan: %{s['tp_dist']:.3f}  |  SL'ye kalan: %{s['sl_dist']:.3f}")
        icon_net = "🟢" if net_pnl >= 0 else "🔴"
        log_line(f"  {icon_net} TOPLAM EQUİTY: ${equity:,.2f}  |  Net PNL: ${net_pnl:+.2f} ({pct_str(net_pnl/CAPITAL*100)})")
        log_line(f"     Realized: ${realized:+.2f}  |  Unrealized: ${total_unrealized:+.2f}")
        log_line("  " + "─" * 55)

# ─── 3 HOURS ELAPSED: close open positions ────────────────────────────────────
log_line("\n" + "=" * 60)
log_line("  3 SAAT DOLDU — ZAMAN KAPAT")
log_line("=" * 60)

for sym, pos in list(state["positions"].items()):
    if pos["status"] == "OPEN":
        try:
            cur = get_price(sym)
        except:
            cur = pos["entry"]
        if pos["side"] == "BUY":
            pnl = (cur - pos["entry"]) * pos["qty"]
        else:
            pnl = (pos["entry"] - cur) * pos["qty"]
        pnl_pct = pnl / pos["pos_usd"] * 100
        state["balance"] += pos["pos_usd"] + pnl
        pos.update({
            "status": "CLOSED", "exit": cur,
            "pnl": round(pnl, 4), "pnl_pct": round(pnl_pct, 2),
            "closed_at": ts(), "reason": "TIME_CLOSE",
        })
        state["closed_trades"].append(dict(pos))
        icon = "🟢" if pnl >= 0 else "🔴"
        log_line(f"  {icon} {sym}: Zaman kapandı @ ${cur:,.4f}  |  PNL: ${pnl:+.4f} ({pct_str(pnl_pct)})")

# ─── CLOSING REPORT ───────────────────────────────────────────────────────────
realized  = sum(t.get("pnl", 0) for t in state["closed_trades"])
final_eq  = state["balance"]
net_pnl   = final_eq - CAPITAL
wins      = sum(1 for t in state["closed_trades"] if t.get("pnl", 0) > 0)
losses    = sum(1 for t in state["closed_trades"] if t.get("pnl", 0) <= 0)
win_rate  = wins / max(len(state["closed_trades"]), 1) * 100

state["final_equity"] = final_eq
state["net_pnl"]      = net_pnl
save_state()

log_line("\n  ══════════════════════════════════════════════")
log_line("  KAPANIŞ RAPORU")
log_line("  ══════════════════════════════════════════════")
log_line(f"  Başlangıç sermayesi:  ${CAPITAL:,.2f}")
log_line(f"  Bitiş equity:         ${final_eq:,.2f}")
log_line(f"  Net PNL:              ${net_pnl:+.2f}  ({pct_str(net_pnl/CAPITAL*100)})")
log_line(f"  Kazanan işlem:        {wins}")
log_line(f"  Kaybeden işlem:       {losses}")
log_line(f"  Kazanma oranı:        %{win_rate:.1f}")
log_line("\n  İşlem detayları:")
for t in state["closed_trades"]:
    icon = "✅" if t.get("pnl", 0) > 0 else "❌"
    log_line(f"    {icon} {t['symbol']} {t['dir']}")
    log_line(f"       Giriş: ${t['entry']:,.4f}  →  Çıkış: ${t['exit']:,.4f}")
    log_line(f"       PNL: ${t['pnl']:+.4f} ({pct_str(t['pnl_pct'])})  |  Neden: {t['reason']}")
log_line(f"\n  JSON log: {LOG_FILE}")
log_line(f"  Text log: {TEXT_LOG}")
log_line("  ══════════════════════════════════════════════")
