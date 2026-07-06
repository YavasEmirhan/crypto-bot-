import ccxt
import pandas as pd
import time
import requests
from datetime import datetime, timezone
from typing import Optional

EXCHANGE = ccxt.okx({"enableRateLimit": True})

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT",
]

TIMEFRAMES = ["1h", "4h", "1d"]


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 1000,
    since: Optional[int] = None,
) -> pd.DataFrame:
    raw = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df


def fetch_full_history(
    symbol: str,
    timeframe: str = "1h",
    start_date: str = "2022-01-01",
) -> pd.DataFrame:
    since_ms = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    all_candles: list[pd.DataFrame] = []

    print(f"Fetching {symbol} {timeframe} from {start_date}...")
    while True:
        df = fetch_ohlcv(symbol, timeframe, limit=300, since=since_ms)
        if df.empty:
            break
        all_candles.append(df)
        last_ts = int(df.index[-1].timestamp() * 1000)
        if last_ts <= since_ms:
            break
        since_ms = last_ts + 1
        time.sleep(EXCHANGE.rateLimit / 1000)

        if len(all_candles) % 10 == 0:
            print(f"  Collected {sum(len(c) for c in all_candles)} candles...")

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if since_ms >= now_ms:
            break

    if not all_candles:
        return pd.DataFrame()

    result = pd.concat(all_candles)
    result = result[~result.index.duplicated(keep="last")]
    result.sort_index(inplace=True)
    print(f"  Total: {len(result)} candles")
    return result


def fetch_all_symbols_history(
    timeframe: str = "1h",
    start_date: str = "2022-01-01",
) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        try:
            df = fetch_full_history(symbol, timeframe, start_date)
            if not df.empty:
                datasets[symbol] = df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# FUTURES MARKET DATA (Funding Rate + Open Interest)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_funding_rate(symbol: str, limit: int = 500) -> pd.DataFrame:
    """OKX perpetual funding rate — 8-hour data."""
    try:
        perp = symbol.replace("/USDT", "/USDT:USDT")
        data = EXCHANGE.fetch_funding_rate_history(perp, limit=limit)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["fundingRate"]].rename(
            columns={"fundingRate": "funding_rate"}
        )
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception as e:
        print(f"  Funding rate fetch hatası ({symbol}): {e}")
        return pd.DataFrame()


def fetch_open_interest(symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
    """OKX open interest history — hourly."""
    try:
        perp = symbol.replace("/USDT", "/USDT:USDT")
        data = EXCHANGE.fetch_open_interest_history(perp, timeframe, limit=limit)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        col = "openInterestAmount" if "openInterestAmount" in df.columns else "openInterest"
        df = df.set_index("timestamp")[[col]].rename(
            columns={col: "open_interest"}
        )
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception as e:
        print(f"  Open Interest fetch hatası ({symbol}): {e}")
        return pd.DataFrame()


def fetch_fear_greed_index(limit: int = 365) -> pd.DataFrame:
    """
    Alternative.me Fear & Greed Index — free API.
    0-25: Extreme Fear, 25-45: Fear, 45-55: Neutral,
    55-75: Greed, 75-100: Extreme Greed
    """
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype(int), unit="s", utc=True
        )
        df = df.set_index("timestamp")[["value"]].rename(
            columns={"value": "fng_value"}
        )
        df["fng_value"] = df["fng_value"].astype(float) / 100.0  # normalize 0-1
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception as e:
        print(f"  Fear & Greed fetch hatası: {e}")
        return pd.DataFrame()


def enrich_with_futures_data(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Enrich the OHLCV DataFrame with funding rate + open interest + fear & greed.
    Missing values are filled with 0 (handled by the model's feature_align).
    """
    # Funding rate (8h → hourly forward-fill)
    fr_df = fetch_funding_rate(symbol, limit=500)
    if not fr_df.empty:
        fr_reindexed = fr_df.reindex(df.index, method="ffill")
        df["funding_rate"] = fr_reindexed["funding_rate"]
        print(f"  ✓ Funding rate: {fr_df.shape[0]} bar")

    # Open Interest
    oi_df = fetch_open_interest(symbol, timeframe, limit=500)
    if not oi_df.empty:
        oi_reindexed = oi_df.reindex(df.index, method="ffill")
        df["open_interest"] = oi_reindexed["open_interest"]
        print(f"  ✓ Open Interest: {oi_df.shape[0]} bar")

    # Fear & Greed Index (daily → hourly forward-fill)
    fng_df = fetch_fear_greed_index(limit=365)
    if not fng_df.empty:
        fng_reindexed = fng_df.reindex(df.index, method="ffill")
        df["fng_value"] = fng_reindexed["fng_value"]
        print(f"  ✓ Fear & Greed: {fng_df.shape[0]} gün")

    return df
