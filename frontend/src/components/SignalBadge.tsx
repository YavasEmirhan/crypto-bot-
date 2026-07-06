interface Props {
  signal: string;
  confidence?: number;
  size?: "sm" | "lg";
}

export default function SignalBadge({ signal, confidence, size = "sm" }: Props) {
  const cls = `tag tag-${signal.toLowerCase()}`;
  const label = size === "lg"
    ? `${signal}${confidence ? ` · ${(confidence * 100).toFixed(1)}%` : ""}`
    : signal;

  return (
    <span className={cls} style={size === "lg" ? { fontSize: 15, padding: "4px 14px" } : {}}>
      {label}
    </span>
  );
}
