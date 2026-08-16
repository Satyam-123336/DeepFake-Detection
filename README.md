# RealityGuard AI: A Multi-Signal and Explainable Deepfake Detection Engine

> **🏆 Research Paper Accepted!**
> "RealityGuard AI: A Multi-Signal and Explainable Deepfake Detection Engine" has been accepted for presentation at the **AITA 2026 International Conference** and will be published in **Springer Lecture Notes in Networks and Systems (Scopus-indexed)**.

This project implements a complete, end-to-end multi-signal deepfake detection pipeline, combining visual artifact analysis, behavioral inconsistencies (blink rate, lip-sync), and forensic watermarking detection.

## Key Features

1. **Robust Face Detection**: Utilizes industry-standard **MTCNN** (`facenet-pytorch`) for accurate face extraction and bounding box alignment across varied angles and lighting conditions.
2. **Visual Artifact CNN**: A lightweight PyTorch CNN trained to detect visual synthesis artifacts, unnatural textures, and blending boundaries on extracted faces.
3. **Behavioral Analysis**: Tracks eye-blink rhythms (via Eye Aspect Ratio) and lip-sync mismatches to detect deepfakes that visually pass but fail temporal biological tests.
4. **Scoring Engine**: An intelligent heuristic risk assessment engine that combines all module scores into a final weighted confidence score, dynamically escalating suspicious visual anomalies that are corroborated by other signals.
5. **Modern Dashboard UI**: A premium React/NextJS frontend offering seamless Dark/Light modes, sleek forensic visualizers, and detailed explainable metrics.

## Quick Start

### 1. Install Python Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install facenet-pytorch  # Required for MTCNN face detection
```

### 2. Start the Backend API (FastAPI)
The modern web dashboard requires the FastAPI backend to be running:
```bash
python api_server.py
```
This will start the API server on `http://localhost:8000`.

### 3. Start the Frontend (Node.js)
In a separate terminal, start the React dev server:
```bash
cd frontend
npm install
npm run dev
```
Access the premium RealityGuard AI dashboard at `http://localhost:3000`.

## Legacy/Alternative UI (Streamlit)

For quick local testing without the full React frontend, you can still launch the legacy Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

## Dataset Gathering & Training Workflow

To train the Phase 4 Visual CNN from scratch:

1. **Organize Raw Data**: Place videos in `data/raw/real/` and `data/raw/fake/`.
   Alternatively, ingest FaceForensics++ and DFDC datasets automatically:
   ```bash
   python main.py --mode gather-datasets --ffpp-root path/to/FaceForensics++ --dfdc-root path/to/dfdc --raw-output-dir data/raw
   ```

2. **Preprocess and Extract Faces**:
   ```bash
   python main.py --mode prepare-dataset --raw-dir data/raw --processed-dir data/processed
   ```

3. **Train the CNN**:
   ```bash
   python main.py --mode train-cnn --train-csv data/splits/train_faces.csv --val-csv data/splits/val_faces.csv
   ```

4. **Evaluate Model Weights**:
   ```bash
   python main.py --mode evaluate-cnn --test-csv data/splits/test_faces.csv --weights-path models/cnn/weights/lightweight_artifact_cnn.pt
   ```

## Project Health Check

Run a quick status audit to check your dataset inventory, pipeline readiness, and split coverage:
```bash
python main.py --mode project-status
```

## Notes

- **Multi-Signal Integration**: Watermark detection, NLP transcription, and lipsync metrics are modular. The pipeline safely degrades if specific streams (like audio) are unavailable in a video.
- **Hardware Acceleration**: The MTCNN face detector and CNN inference will automatically use CUDA if available, providing a massive speedup on Nvidia GPUs.