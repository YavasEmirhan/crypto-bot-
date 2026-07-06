"use client";

import { useEffect, useState, useCallback } from "react";
import { api, Candle, Signal, TradeStats, Position, Trade, TrainStatus, SchedulerStatus } from "@/lib/api";
import dynamic from "next/dynamic";
import StatCard from "@/components/StatCard";
import SignalBadge from "@/components/SignalBadge";
import TradeTable from "@/components/TradeTable";

const CandlestickChart = dynamic(() => import("@/components/CandlestickChart"), { ssr: false });

const SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"];
const TIMEFRAMES = ["1h", "4h", "1d"];

export default function Dashboard() {
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [timeframe, setTimeframe] = useState("1h");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [latestSignal, setLatestSignal] = useState<(Signal & { price: number }) | null>(null);
  const [stats, setStats] = useState<TradeStats | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [schedulerStatus, setSchedulerStatus] = useState<SchedulerStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<"chart" | "paper" | "train">("train");
  const [backendUp, setBackendUp] = useState<boolean | null>(null);

  const loadAll = useCallback(async () => {
    try {
      const [candleRes, trainRes, statsRes, posRes, tradeRes, schedRes] = await Promise.all([
        api.ohlcv(symbol, timeframe, 200),
        api.trainStatus(),
        api.paperStats(),
        api.paperPositions(),
        api.paperTrades(50),
        api.schedulerStatus(),
      ]);
      setCandles(candleRes.data);
      setTrainStatus(trainRes);
      setStats(statsRes);
      setPositions(posRes.positions);
      setTrades(tradeRes.trades);
      setSchedulerStatus(schedRes);
      setBackendUp(true);

      if (trainRes.model_ready) {
        const [sigRes, latestRes] = await Promise.all([
          api.signals(symbol, timeframe).catch(() => ({ signals: [] as Signal[] })),
          api.latestSignal(symbol, timeframe).catch(() => null),
        ]);
        setSignals(sigRes.signals);
        if (latestRes) setLatestSignal(latestRes as Signal & { price: number });
      }
    } catch {
      setBackendUp(false);
    }
  }, [symbol, timeframe]);

  useEffect(() => {
    loadAll();
    const iv = setInterval(loadAll, 15_000);
    return () => clearInterval(iv);
  }, [loadAll]);

  const handleStartTraining = async () => {
    setLoading(true);
    try { await api.startIterativeTraining(0.80); setTab("train"); }
    finally { setLoading(false); }
  };

  const handleStartScheduler = async () => {
    setLoading(true);
    try { await api.startScheduler(60); await loadAll(); }
    finally { setLoading(false); }
  };

  const handleStopScheduler = async () => { await api.stopScheduler(); await loadAll(); };

  const handleReset = async () => {
    if (!confirm("Paper trading sıfırlansın mı? (Bakiye: $1,000)")) return;
    await api.resetPaper(1000); await loadAll();
  };

  const handleManualTrade = async () => {
    setLoading(true);
    try { await api.autoTrade(symbol, timeframe); await loadAll(); }
    finally { setLoading(false); }
  };

  const pnlPos = (stats?.total_pnl ?? 0) >= 0;
  const isLive = schedulerStatus?.running || trainStatus?.auto_trade_active;
  const accPct = trainStatus?.best_accuracy ? (trainStatus.best_accuracy * 100).toFixed(1) : null;
  const targetPct = trainStatus?.target_accuracy ? (trainStatus.target_accuracy * 100).toFixed(0) : "80";
  const accNum  = parseFloat(accPct ?? "0");

  return (
    <div style={{ minHeight: "100vh", background: "#0a0e1a", padding: "16px 20px" }}>

      {/* HEADER */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 34, height: 34, borderRadius: 9,
            background: "linear-gradient(135deg,#3b82f6,#8b5cf6)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 17, fontWeight: 800, color: "white" }}>₿</div>
          <div>
            <div style={{ fontSize: 17, fontWeight: 800, color: "#e2e8f0" }}>Crypto Bot</div>
            <div style={{ fontSize: 10, color: "#6b7280" }}>OKX · Hibrit TA+ML · 168 özellik</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {accPct && (
            <div style={{ display: "flex", alignItems: "center", gap: 6,
              background: "#111827", border: "1px solid #1f2937", borderRadius: 8, padding: "4px 10px" }}>
              <span style={{ fontSize: 11, color: "#6b7280" }}>Accuracy</span>
              <span style={{ fontSize: 13, fontWeight: 700,
                color: accNum >= 80 ? "#22c55e" : accNum >= 65 ? "#f59e0b" : "#ef4444" }}>
                {accPct}%
              </span>
              <span style={{ fontSize: 10, color: "#6b7280" }}>/ {targetPct}%</span>
            </div>
          )}
          <div style={{ display: "flex", alignItems: "center", gap: 6,
            background: isLive ? "#14532d" : "#111827",
            border: `1px solid ${isLive ? "#16a34a" : "#1f2937"}`,
            borderRadius: 8, padding: "4px 10px" }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%",
              background: isLive ? "#22c55e" : backendUp ? "#f59e0b" : "#ef4444",
              boxShadow: isLive ? "0 0 6px #22c55e" : "none" }}/>
            <span style={{ fontSize: 11, fontWeight: 600,
              color: isLive ? "#4ade80" : backendUp ? "#fbbf24" : "#ef4444" }}>
              {isLive ? "CANLI" : backendUp ? "Hazır" : "Bağlantı yok"}
            </span>
          </div>
        </div>
      </div>

      {/* SYMBOL / TIMEFRAME */}
      <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: 4 }}>
          {SYMBOLS.map(s => (
            <button key={s} className={`btn ${s === symbol ? "btn-primary" : "btn-gray"}`}
              style={{ fontSize: 12, padding: "5px 10px" }} onClick={() => setSymbol(s)}>
              {s.replace("/USDT", "")}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {TIMEFRAMES.map(tf => (
            <button key={tf} className={`btn ${tf === timeframe ? "btn-primary" : "btn-gray"}`}
              style={{ fontSize: 12, padding: "5px 10px" }} onClick={() => setTimeframe(tf)}>
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* TABS */}
      <div style={{ display: "flex", gap: 0, marginBottom: 16, borderBottom: "1px solid #1f2937" }}>
        {(["train", "chart", "paper"] as const).map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: "9px 18px", fontSize: 13, fontWeight: 600,
            background: "none", border: "none", cursor: "pointer",
            color: tab === t ? "#3b82f6" : "#6b7280",
            borderBottom: tab === t ? "2px solid #3b82f6" : "2px solid transparent",
            marginBottom: -1,
          }}>
            {t === "train" ? "🧠 Eğitim" : t === "chart" ? "📈 Grafik" : "💼 Paper Trading"}
          </button>
        ))}
      </div>

      {/* ══ TRAINING ══ */}
      {tab === "train" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14, maxWidth: 700 }}>
          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
              <span style={{ fontSize: 14, fontWeight: 700 }}>Model Doğruluğu</span>
              <span style={{ fontSize: 13, color: "#6b7280" }}>Hedef: <b style={{ color: "#e2e8f0" }}>{targetPct}%</b></span>
            </div>
            <div style={{ background: "#0a0e1a", borderRadius: 8, height: 10, overflow: "hidden", marginBottom: 8 }}>
              <div style={{
                height: "100%", borderRadius: 8, transition: "width 0.5s ease",
                width: `${Math.min((trainStatus?.best_accuracy ?? 0) * 100, 100)}%`,
                background: accNum >= 80
                  ? "linear-gradient(90deg,#16a34a,#22c55e)"
                  : accNum >= 65
                  ? "linear-gradient(90deg,#b45309,#f59e0b)"
                  : "linear-gradient(90deg,#991b1b,#ef4444)",
              }}/>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
              <span style={{ color: "#6b7280" }}>
                {trainStatus?.running
                  ? `İterasyon ${trainStatus.iteration} devam ediyor...`
                  : trainStatus?.model_ready ? "Model hazır" : "Henüz eğitilmedi"}
              </span>
              <span style={{ fontWeight: 700, fontSize: 16,
                color: accNum >= 80 ? "#22c55e" : accNum >= 65 ? "#f59e0b" : "#e2e8f0" }}>
                {accPct ? `${accPct}%` : "—"}
              </span>
            </div>
          </div>

          {trainStatus?.progress && (
            <div className="card" style={{ background: "#0d1117", padding: "12px 16px" }}>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>Son Durum</div>
              <div style={{ fontSize: 13, color: "#a3e635", fontFamily: "monospace",
                lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
                {trainStatus.progress}
              </div>
            </div>
          )}

          {!trainStatus?.running && (
            <button className="btn btn-primary"
              style={{ padding: "14px 0", fontSize: 15, width: "100%" }}
              onClick={handleStartTraining} disabled={loading}>
              {trainStatus?.model_ready
                ? "🔄 Yeniden Eğit (Doğruluğu Arttır)"
                : "🚀 İteratif Eğitimi Başlat (Hedef: %80)"}
            </button>
          )}

          {trainStatus?.running && (
            <div className="card" style={{ textAlign: "center", padding: 20 }}>
              <div style={{ fontSize: 13, color: "#f59e0b", marginBottom: 8 }}>⏳ Eğitim devam ediyor...</div>
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                OKX'ten veri çekiliyor → İndikatörler → Ensemble model
              </div>
            </div>
          )}

          <div className="card">
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Auto-Trade Scheduler</div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%",
                background: schedulerStatus?.running ? "#22c55e" : "#6b7280",
                boxShadow: schedulerStatus?.running ? "0 0 6px #22c55e" : "none" }}/>
              <span style={{ fontSize: 13, color: schedulerStatus?.running ? "#4ade80" : "#6b7280" }}>
                {schedulerStatus?.running ? "Aktif — Her 60 dakika çalışıyor" : "Durdurulmuş"}
              </span>
              {schedulerStatus?.jobs[0] && (
                <span style={{ fontSize: 11, color: "#6b7280" }}>
                  Sonraki: {new Date(schedulerStatus.jobs[0].next_run).toLocaleTimeString("tr-TR")}
                </span>
              )}
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              {!schedulerStatus?.running ? (
                <button className="btn btn-green"
                  onClick={handleStartScheduler} disabled={!trainStatus?.model_ready || loading}
                  style={{ flex: 1 }}>▶ Başlat</button>
              ) : (
                <button className="btn btn-red" onClick={handleStopScheduler} style={{ flex: 1 }}>
                  ⏹ Durdur
                </button>
              )}
            </div>
          </div>

          {(schedulerStatus?.recent_actions?.length ?? 0) > 0 && (
            <div className="card" style={{ padding: 0 }}>
              <div style={{ padding: "12px 16px", borderBottom: "1px solid #1f2937", fontSize: 13, fontWeight: 600 }}>
                Son Otomatik İşlemler
              </div>
              <div style={{ maxHeight: 220, overflowY: "auto" }}>
                {(schedulerStatus?.recent_actions ?? []).slice(0, 10).map((a: {ts: string; symbol: string; action: string; price: number; signal: string; conf: number}, i: number) => (
                  <div key={i} style={{ display: "flex", gap: 12, padding: "8px 16px",
                    borderBottom: "1px solid #1a2234", fontSize: 12, alignItems: "center" }}>
                    <span style={{ color: "#6b7280", minWidth: 60 }}>
                      {new Date(a.ts).toLocaleTimeString("tr-TR")}
                    </span>
                    <span style={{ fontWeight: 600, minWidth: 70 }}>{a.symbol}</span>
                    <span style={{ color: a.action?.includes("BUY") ? "#22c55e" : "#ef4444",
                      minWidth: 100, fontWeight: 600 }}>{a.action}</span>
                    <span style={{ color: "#9ca3af" }}>${a.price?.toFixed(2)}</span>
                    <span style={{ color: "#6b7280" }}>
                      {a.signal} {a.conf ? `${(a.conf * 100).toFixed(0)}%` : ""}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card" style={{ fontSize: 12, color: "#9ca3af", lineHeight: 1.9 }}>
            <div style={{ fontWeight: 600, color: "#e2e8f0", marginBottom: 8 }}>Nasıl çalışır?</div>
            <ol style={{ paddingLeft: 16, margin: 0 }}>
              <li>OKX'ten 2021'den bugüne <b style={{ color: "#3b82f6" }}>3+ yıl veri</b> çekilir</li>
              <li><b style={{ color: "#3b82f6" }}>168 teknik özellik</b> hesaplanır (Ichimoku, MTF, Candlestick…)</li>
              <li>XGBoost + LightGBM + RF <b style={{ color: "#3b82f6" }}>ensemble</b> eğitilir</li>
              <li>Walk-forward CV ile <b style={{ color: "#3b82f6" }}>doğruluk ölçülür</b></li>
              <li>%80 altıysa farklı parametrelerle <b style={{ color: "#3b82f6" }}>yeniden eğitilir</b></li>
              <li>Hedef tutturulunca <b style={{ color: "#22c55e" }}>her saat auto-trade</b> başlar</li>
            </ol>
          </div>
        </div>
      )}

      {/* ══ CHART ══ */}
      {tab === "chart" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {latestSignal ? (
            <div className="card" style={{ display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap" }}>
              <div>
                <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>Son Sinyal · {symbol}</div>
                <SignalBadge signal={latestSignal.signal} confidence={latestSignal.confidence} size="lg" />
              </div>
              <div>
                <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>Fiyat</div>
                <div style={{ fontSize: 22, fontWeight: 700 }}>${latestSignal.price?.toLocaleString("tr-TR")}</div>
              </div>
              {latestSignal.probabilities && (
                <div style={{ display: "flex", gap: 20 }}>
                  {([["SATIŞ", latestSignal.probabilities.sell, "#ef4444"],
                     ["BEKLE", latestSignal.probabilities.hold, "#f59e0b"],
                     ["ALIM",  latestSignal.probabilities.buy,  "#22c55e"]] as [string, number, string][]).map(([l, v, c]) => (
                    <div key={l}>
                      <div style={{ fontSize: 10, color: "#6b7280" }}>{l}</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color: c }}>{(v * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="card" style={{ color: "#6b7280", fontSize: 13, textAlign: "center", padding: 24 }}>
              {trainStatus?.model_ready ? "Sinyal yükleniyor..." : "Model eğitilmedi — Eğitim sekmesinden başlatın"}
            </div>
          )}
          <div className="card" style={{ padding: 0, overflow: "hidden", borderRadius: 12 }}>
            <CandlestickChart candles={candles} signals={signals} height={460} />
          </div>
        </div>
      )}

      {/* ══ PAPER TRADING ══ */}
      {tab === "paper" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(160px,1fr))", gap: 10 }}>
            <StatCard label="Bakiye" value={`$${(stats?.balance ?? 1000).toLocaleString("tr-TR", { maximumFractionDigits: 2 })}`} />
            <StatCard label="Toplam PnL"
              value={`${pnlPos && (stats?.total_pnl ?? 0) !== 0 ? "+" : ""}$${(stats?.total_pnl ?? 0).toFixed(2)}`}
              sub={`${pnlPos && (stats?.total_pnl ?? 0) !== 0 ? "+" : ""}${(stats?.total_pnl_pct ?? 0).toFixed(2)}%`}
              positive={pnlPos && (stats?.total_pnl ?? 0) > 0} negative={!pnlPos && (stats?.total_pnl ?? 0) < 0} />
            <StatCard label="Kazanma Oranı" value={`${(stats?.win_rate ?? 0).toFixed(1)}%`}
              sub={`${stats?.win_count ?? 0}K / ${stats?.loss_count ?? 0}K`} />
            <StatCard label="Toplam İşlem" value={stats?.total_trades ?? 0} />
            <StatCard label="Açık Pozisyon" value={stats?.open_positions ?? 0} />
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="btn btn-green" onClick={handleManualTrade}
              disabled={loading || !trainStatus?.model_ready}>⚡ Manuel İşlem</button>
            {!schedulerStatus?.running ? (
              <button className="btn btn-primary" onClick={handleStartScheduler}
                disabled={!trainStatus?.model_ready}>▶ Auto-Trade Başlat</button>
            ) : (
              <button className="btn btn-red" onClick={handleStopScheduler}>⏹ Durdur</button>
            )}
            <button className="btn btn-gray" onClick={handleReset}>Sıfırla ($1,000)</button>
          </div>

          {positions.length > 0 && (
            <div>
              <div style={{ fontSize: 13, color: "#9ca3af", marginBottom: 8, fontWeight: 600 }}>
                Açık Pozisyonlar ({positions.length})
              </div>
              {positions.map(p => (
                <div key={p.id} className="card" style={{ display: "flex", gap: 16, alignItems: "center",
                  flexWrap: "wrap", marginBottom: 8 }}>
                  <span style={{ fontWeight: 700 }}>{p.symbol}</span>
                  <SignalBadge signal={p.side} />
                  <span style={{ fontSize: 12, color: "#9ca3af" }}>
                    Giriş: <b style={{ color: "#e2e8f0" }}>${p.entry_price.toFixed(2)}</b>
                  </span>
                  <span style={{ fontSize: 12, color: "#22c55e" }}>TP: ${p.take_profit.toFixed(2)}</span>
                  <span style={{ fontSize: 12, color: "#ef4444" }}>SL: ${p.stop_loss.toFixed(2)}</span>
                  <span style={{ fontSize: 11, color: "#6b7280" }}>
                    Güven: {(p.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          )}

          <div>
            <div style={{ fontSize: 13, color: "#9ca3af", marginBottom: 8, fontWeight: 600 }}>İşlem Geçmişi</div>
            <TradeTable trades={trades} />
          </div>
        </div>
      )}
    </div>
  );
}
