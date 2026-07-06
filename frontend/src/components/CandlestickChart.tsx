"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
  createSeriesMarkers,
  ColorType,
  IChartApi,
} from "lightweight-charts";
import type { ISeriesApi, CandlestickData, Time } from "lightweight-charts";
import { Candle, Signal } from "@/lib/api";

interface Props {
  candles: Candle[];
  signals?: Signal[];
  height?: number;
}

export default function CandlestickChart({ candles, signals, height = 420 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "#111827" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      crosshair: {
        vertLine: { color: "#4b5563" },
        horzLine: { color: "#4b5563" },
      },
      rightPriceScale: { borderColor: "#1f2937" },
      timeScale: { borderColor: "#1f2937", timeVisible: true },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    const resize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
      chart.remove();
    };
  }, [height]);

  useEffect(() => {
    if (!candleSeriesRef.current || !candles.length) return;
    const data: CandlestickData[] = candles.map((c) => ({
      time: c.time as Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    candleSeriesRef.current.setData(data);

    if (signals && signals.length) {
      const markers = signals
        .filter((s) => s.signal === "BUY" || s.signal === "SELL")
        .map((s) => ({
          time: Math.floor(new Date(s.index).getTime() / 1000) as Time,
          position: s.signal === "BUY" ? ("belowBar" as const) : ("aboveBar" as const),
          color: s.signal === "BUY" ? "#22c55e" : "#ef4444",
          shape: s.signal === "BUY" ? ("arrowUp" as const) : ("arrowDown" as const),
          text: `${s.signal} ${(s.confidence * 100).toFixed(0)}%`,
          size: 1,
        }));
      createSeriesMarkers(candleSeriesRef.current, markers);
    }

    chartRef.current?.timeScale().fitContent();
  }, [candles, signals]);

  return <div ref={containerRef} style={{ width: "100%", height }} />;
}
