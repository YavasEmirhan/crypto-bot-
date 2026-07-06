#!/bin/bash

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Clean up old processes
echo "==> Eski süreçler temizleniyor..."
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :3000 | xargs kill -9 2>/dev/null || true
sleep 1

echo "==> Crypto Bot başlatılıyor..."

# Backend
echo "[1/2] FastAPI backend başlatılıyor (:8000)..."
cd "$ROOT/backend"
source venv/bin/activate
python main.py > /tmp/cryptobot_backend.log 2>&1 &
BACKEND_PID=$!

# Wait until the backend is ready (max 15s)
echo "    Backend bekleniyor..."
for i in $(seq 1 15); do
  if curl -s http://localhost:8000/api/symbols > /dev/null 2>&1; then
    echo "    Backend hazır! (${i}sn)"
    break
  fi
  sleep 1
done

# Frontend
echo "[2/2] Next.js frontend başlatılıyor (:3000)..."
cd "$ROOT/frontend"
npm run dev > /tmp/cryptobot_frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait until the frontend is ready (max 20s)
echo "    Frontend bekleniyor..."
for i in $(seq 1 20); do
  if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "    Frontend hazır! (${i}sn)"
    break
  fi
  sleep 1
done

echo ""
echo "╔══════════════════════════════════════╗"
echo "║        Crypto Bot Çalışıyor!         ║"
echo "╠══════════════════════════════════════╣"
echo "║  Dashboard : http://localhost:3000   ║"
echo "║  API Docs  : http://localhost:8000/docs ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Durdurmak için Ctrl+C"
echo "Loglar: /tmp/cryptobot_backend.log"

# Open the browser
open http://localhost:3000 2>/dev/null || true

cleanup() {
  echo "Durduruluyor..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait $BACKEND_PID $FRONTEND_PID
