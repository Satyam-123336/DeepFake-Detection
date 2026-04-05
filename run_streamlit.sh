#!/bin/bash

# Linux/Mac startup script for Streamlit enhanced app

echo ""
echo "========================================"
echo " DeepFake Detection - Streamlit App"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    exit 1
fi

# Check if streamlit is installed
if ! python3 -m streamlit --version &> /dev/null; then
    echo "[WARNING] Streamlit not found. Installing..."
    pip3 install streamlit
fi

# Create uploads directory if not exists
mkdir -p data/uploads

echo "[INFO] Python version:"
python3 --version

echo ""
echo "[INFO] Starting Streamlit app..."
echo "[INFO] Opening http://localhost:8501 in browser..."
echo ""

# Run streamlit
python3 -m streamlit run streamlit_app.py --logger.level=info
