@echo off
REM Windows startup script for Streamlit enhanced app

echo.
echo ========================================
echo  DeepFake Detection - Streamlit App
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Streamlit not found. Installing...
    pip install streamlit
)

REM Create uploads directory if not exists
if not exist "data\uploads" (
    mkdir data\uploads
    echo [INFO] Created data\uploads directory
)

echo.
echo [INFO] Starting Streamlit app...
echo [INFO] Opening http://localhost:8501 in browser...
echo.

REM Run streamlit
python -m streamlit run streamlit_app.py --logger.level=info

pause
