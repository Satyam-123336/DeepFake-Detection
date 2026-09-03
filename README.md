# RealityGuard AI: A Multi-Signal and Explainable Deepfake Detection Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22282866.svg)](https://doi.org/10.5281/zenodo.22282866) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-orange)](https://huggingface.co/Satysam-26/RealityGuardAI)

> **🏆 Research Paper Accepted!**
> "RealityGuard AI: A Multi-Signal and Explainable Deepfake Detection Engine" has been accepted for presentation at the **AITA 2026 International Conference** and will be published in **Springer Lecture Notes in Networks and Systems (Scopus-indexed)**.

👉 **[Download the Pre-Trained CNN Weights from our Hugging Face Model Repository](https://huggingface.co/Satysam-26/RealityGuardAI)**

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

## Citation

If you use this architecture or model in your research, please cite it using the following DOI:

```bibtex
@software{realityguardai_2026,
  author       = {Satyam},
  title        = {RealityGuardAI: A Multi-Signal and Explainable Deepfake Detection Engine},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22282866},
  url          = {https://doi.org/10.5281/zenodo.22282866}
}
```
