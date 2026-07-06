const BASE = "http://localhost:8000/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export type Candle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type Signal = {
  index: string;
  signal: "BUY" | "SELL" | "HOLD" | "NO_MODEL";
  confidence: number;
  probabilities: { sell: number; hold: number; buy: number };
  color: string;
};

export type TradeStats = {
  balance: number;
  initial_balance: number;
  total_pnl: number;
  total_pnl_pct: number;
  win_count: number;
  loss_count: number;
  total_trades: number;
  win_rate: number;
  open_positions: number;
};

export type Position = {
  id: string;
  symbol: string;
  side: string;
  entry_price: number;
  quantity: number;
  position_size: number;
  take_profit: number;
  stop_loss: number;
  opened_at: string;
  confidence: number;
};

export type Trade = Position & {
  exit_price: number;
  pnl: number;
  pnl_pct: number;
  closed_at: string;
};

export type TrainStatus = {
  running: boolean;
  progress: string;
  iteration: number;
  best_accuracy: number;
  target_accuracy: number;
  model_ready: boolean;
  auto_trade_active: boolean;
  result: { average_accuracy: number; average_f1?: number } | null;
};

export type SchedulerStatus = {
  running: boolean;
  jobs: { id: string; next_run: string }[];
  recent_actions: {
    ts: string; symbol: string; action: string;
    price: number; signal: string; conf: number;
  }[];
};

export const api = {
  symbols: () => get<{ symbols: string[] }>("/symbols"),
  ohlcv: (symbol: string, tf: string, limit = 200) =>
    get<{ data: Candle[] }>(`/ohlcv?symbol=${encodeURIComponent(symbol)}&timeframe=${tf}&limit=${limit}`),
  latestSignal: (symbol: string, tf: string) =>
    get<Signal & { price: number; symbol: string }>(`/signal/latest?symbol=${encodeURIComponent(symbol)}&timeframe=${tf}`),
  signals: (symbol: string, tf: string) =>
    get<{ signals: Signal[] }>(`/signals?symbol=${encodeURIComponent(symbol)}&timeframe=${tf}`),
  price: (symbol: string) =>
    get<{ price: number; change_pct: number }>(`/price?symbol=${encodeURIComponent(symbol)}`),
  trainStatus: () => get<TrainStatus>("/train/status"),
  startIterativeTraining: (target: number) =>
    post<{ message: string }>("/train/iterative", { target_accuracy: target }),
  paperStats: () => get<TradeStats>("/paper/stats"),
  paperPositions: () => get<{ positions: Position[] }>("/paper/positions"),
  paperTrades: (limit = 50) => get<{ trades: Trade[] }>(`/paper/trades?limit=${limit}`),
  autoTrade: (symbol: string, tf: string) =>
    post<unknown>(`/paper/auto-trade?symbol=${encodeURIComponent(symbol)}&timeframe=${tf}`),
  resetPaper: (balance = 1000) => post<unknown>(`/paper/reset?initial_balance=${balance}`),
  schedulerStatus: () => get<SchedulerStatus>("/scheduler/status"),
  startScheduler: (minutes = 60) => post<unknown>(`/scheduler/start?interval_minutes=${minutes}`),
  stopScheduler: () => post<unknown>("/scheduler/stop"),
  tradeLog: (limit = 30) => get<{ log: unknown[] }>(`/scheduler/log?limit=${limit}`),
};
