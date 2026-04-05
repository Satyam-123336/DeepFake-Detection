#!/bin/bash

# Linux/Mac startup script for React + FastAPI stack

echo ""
echo "========================================"
echo " DeepFake Detection - React + FastAPI"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    exit 1
fi

# Check if Node is installed
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed"
    echo ""
    echo "Please download and install Node.js from:"
    echo "https://nodejs.org/"
    echo ""
    exit 1
fi

echo "[INFO] Python version:"
python3 --version

echo "[INFO] Node version:"
node --version

echo ""
echo "[INFO] Installing/updating backend dependencies..."
pip3 install -q fastapi 'uvicorn[standard]' python-multipart aiofiles plotly pandas requests

echo "[INFO] Installing/updating frontend dependencies..."
cd frontend || exit
npm install > /dev/null 2>&1
cd .. || exit

echo ""
echo "========================================"
echo " Two terminals will open:"
echo " 1. Backend API Server (port 8000)"
echo " 2. Frontend Dev Server (port 3000)"
echo "========================================"
echo ""
echo "[INFO] Starting services..."
echo ""

# Create uploads directory if not exists
mkdir -p data/uploads

# Start backend in background
python3 api_server.py &
BACKEND_PID=$!
echo "[OK] Backend started (PID: $BACKEND_PID)"

# Wait a bit for backend to start
sleep 2

# Start frontend in background
cd frontend || exit
npm run dev &
FRONTEND_PID=$!
cd .. || exit
echo "[OK] Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "========================================"
echo " Services Starting:"
echo "========================================"
echo " Backend:  http://localhost:8000"
echo " API Docs: http://localhost:8000/docs"
echo " Frontend: http://localhost:3000"
echo ""
echo " Press Ctrl+C to stop both services"
echo "========================================"
echo ""

# Wait for Ctrl+C and cleanup
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Keep script running
wait
