interface Props {
  label: string;
  value: string | number;
  sub?: string;
  positive?: boolean;
  negative?: boolean;
}

export default function StatCard({ label, value, sub, positive, negative }: Props) {
  let valueColor = "#e2e8f0";
  if (positive) valueColor = "#22c55e";
  if (negative) valueColor = "#ef4444";

  return (
    <div className="card flex flex-col gap-1">
      <span style={{ fontSize: 12, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.08em" }}>
        {label}
      </span>
      <span style={{ fontSize: 22, fontWeight: 700, color: valueColor }}>{value}</span>
      {sub && <span style={{ fontSize: 12, color: "#9ca3af" }}>{sub}</span>}
    </div>
  );
}
