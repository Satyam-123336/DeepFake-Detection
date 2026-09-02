FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (ffmpeg and libgl) for OpenCV & Librosa
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port and run the API
EXPOSE 10000
CMD uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-10000}
