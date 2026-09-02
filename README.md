---
title: RealityGuard AI — DeepFake Detection
emoji: 🔍
colorFrom: teal
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: streamlit_app.py
pinned: true
license: mit
tags:
  - deepfake-detection
  - computer-vision
  - pytorch
  - mtcnn
  - cnn
  - lip-sync
  - blink-detection
  - forensic-analysis
  - video-analysis
short_description: Multi-signal explainable deepfake detection engine (AITA 2026 / Springer)
---

# RealityGuard AI: A Multi-Signal and Explainable Deepfake Detection Engine

> **🏆 Research Paper Accepted!**
> "RealityGuard AI: A Multi-Signal and Explainable Deepfake Detection Engine" has been accepted for presentation at the **AITA 2026 International Conference** and will be published in **Springer Lecture Notes in Networks and Systems (Scopus-indexed)**.

This project implements a complete, end-to-end multi-signal deepfake detection pipeline, combining visual artifact analysis, behavioral inconsistencies (blink rate, lip-sync), and forensic watermarking detection.

## Key Features

1. **Robust Face Detection**: Utilizes industry-standard **MTCNN** (`facenet-pytorch`) for accurate face extraction and bounding box alignment across varied angles and lighting conditions.
2. **Visual Artifact CNN**: A lightweight PyTorch CNN trained to detect visual synthesis artifacts, unnatural textures, and blending boundaries on extracted faces.
3. **Behavioral Analysis**: Tracks eye-blink rhythms (via Eye Aspect Ratio) and lip-sync mismatches to detect deepfakes that visually pass but fail temporal biological tests.
4. **Scoring Engine**: An intelligent heuristic risk assessment engine that combines all module scores into a final weighted confidence score, dynamically escalating suspicious visual anomalies that are corroborated by other signals.
5. **Modern Dashboard UI**: A premium Streamlit frontend offering interactive forensic visualizations, radar charts, gauge views, and detailed explainable metrics.

## Quick Start (Local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Modules

| Module | Signal | Weight |
|--------|--------|--------|
| 👁️ Blink Behavior | Eye Aspect Ratio (EAR) | 20% |
| 🎬 Lip-Sync | Audio-visual correlation | 25% |
| 📊 Visual Artifacts | CNN + heuristics | 35% |
| 🔍 Watermark/Trace | Metadata + frame patterns | 10% |
| 🎤 Speech Pattern | NLP suspicion scoring | 10% |