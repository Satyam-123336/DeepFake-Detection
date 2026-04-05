@echo off
REM Windows startup script for React + FastAPI stack

echo.
echo ========================================
echo  DeepFake Detection - React + FastAPI
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo.
    echo Please download and install Node.js from:
    echo https://nodejs.org/
    echo.
    pause
    exit /b 1
)

echo [INFO] Python version:
python --version

echo [INFO] Node version:
node --version

echo.
echo [INFO] Installing/updating backend dependencies...
pip install -q fastapi uvicorn[standard] python-multipart aiofiles plotly pandas requests

echo [INFO] Installing/updating frontend dependencies...
cd frontend
call npm install >nul 2>&1
cd ..

echo.
echo ========================================
echo  Two terminals will open:
echo  1. Backend API Server (port 8000)
echo  2. Frontend Dev Server (port 3000)
echo ========================================
echo.
echo [INFO] Starting services...
echo.

REM Create uploads directory if not exists  
if not exist "data\uploads" (
    mkdir data\uploads
)

REM Start backend in new window
start "DeepFake API Server" python api_server.py
echo [OK] Backend started in new window

REM Wait a bit for backend to start
timeout /t 2 /nobreak

REM Start frontend in new window
start "DeepFake React Dev Server" cmd /k "cd frontend && npm run dev"
echo [OK] Frontend started in new window

echo.
echo ========================================
echo  Services Starting:
echo ========================================
echo  Backend:  http://localhost:8000
echo  API Docs: http://localhost:8000/docs
echo  Frontend: http://localhost:3000
echo.
echo  Press Ctrl+C in any window to stop
echo ========================================
echo.

pause
