import { Trade } from "@/lib/api";
import SignalBadge from "./SignalBadge";

interface Props {
  trades: Trade[];
}

export default function TradeTable({ trades }: Props) {
  if (!trades.length) {
    return (
      <div className="card" style={{ color: "#6b7280", textAlign: "center", padding: 32 }}>
        Henüz kapalı işlem yok
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: 0, overflow: "hidden" }}>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #1f2937" }}>
              {["Sembol", "Yön", "Giriş", "Çıkış", "PnL", "PnL %", "Güven", "Tarih"].map((h) => (
                <th
                  key={h}
                  style={{ padding: "10px 14px", textAlign: "left", color: "#6b7280", fontWeight: 600 }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => (
              <tr
                key={t.id + i}
                style={{ borderBottom: "1px solid #1f2937" }}
              >
                <td style={{ padding: "10px 14px", fontWeight: 600 }}>{t.symbol}</td>
                <td style={{ padding: "10px 14px" }}>
                  <SignalBadge signal={t.side} />
                </td>
                <td style={{ padding: "10px 14px" }}>{t.entry_price.toFixed(2)}</td>
                <td style={{ padding: "10px 14px" }}>{t.exit_price?.toFixed(2) ?? "—"}</td>
                <td style={{ padding: "10px 14px", color: t.pnl >= 0 ? "#22c55e" : "#ef4444", fontWeight: 600 }}>
                  {t.pnl >= 0 ? "+" : ""}{t.pnl.toFixed(2)} $
                </td>
                <td style={{ padding: "10px 14px", color: t.pnl_pct >= 0 ? "#22c55e" : "#ef4444" }}>
                  {t.pnl_pct >= 0 ? "+" : ""}{t.pnl_pct.toFixed(2)}%
                </td>
                <td style={{ padding: "10px 14px", color: "#9ca3af" }}>
                  {(t.confidence * 100).toFixed(0)}%
                </td>
                <td style={{ padding: "10px 14px", color: "#6b7280" }}>
                  {new Date(t.closed_at).toLocaleString("tr-TR")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
