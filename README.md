# DeepFake Detection

This project currently implements Phases 1-4 and includes early scaffolding for Phases 5-6.

## Covered Phases

1. Foundation and setup
2. Preprocessing and key frame extraction
3. Behavioral analysis
4. Visual artifact detection with a lightweight CNN

Scaffolded (in progress):

5. NLP and forensic integration
6. Scoring engine and explainable UI

Roadmap tracking for all 7 phases is available in `PROJECT_ROADMAP.md`.

## Phase Mapping

- `src/preprocessing/`: video ingestion, audio extraction, key frame extraction, face and landmark processing
- `src/behavioral/blink/`: eye-blink rhythm analysis using Eye Aspect Ratio
- `src/behavioral/lipsync/`: audio and mouth-motion alignment checks
- `src/visual/`: artifact heuristics and CNN inference utilities
- `src/data/`: dataset discovery, split generation, and CNN dataset loading
- `models/cnn/`: lightweight CNN architecture and training entry points

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py --mode analyze-video --video path/to/video.mp4
```

## Dataset Preparation Workflow

Phase 1 to Phase 4 training data preparation expects the following raw layout:

- `data/raw/real/`
- `data/raw/fake/`

Then run:

```bash
python main.py --mode prepare-dataset --raw-dir data/raw --processed-dir data/processed
```

## Dataset Gathering (FaceForensics++ And DFDC)

Use local copies of these datasets and ingest them into your standardized raw layout:

- `data/raw/real/`
- `data/raw/fake/`

Ingest FaceForensics++ and DFDC in one step:

```bash
python main.py --mode gather-datasets --ffpp-root path/to/FaceForensics++ --dfdc-root path/to/dfdc --raw-output-dir data/raw
```

You can also ingest custom folders directly:

```bash
python main.py --mode gather-datasets --custom-real-dir path/to/real_videos --custom-fake-dir path/to/fake_videos --raw-output-dir data/raw
```

Tip for first experiments:

```bash
python main.py --mode gather-datasets --ffpp-root path/to/FaceForensics++ --dfdc-root path/to/dfdc --max-real 200 --max-fake 200
```

This will:

1. Scan real and fake videos into a video manifest
2. Build train, validation, and test video splits
3. Preprocess every split into key frames, audio, and face crops
4. Create face-level manifests for CNN training

After gathering datasets, run preparation:

```bash
python main.py --mode prepare-dataset --raw-dir data/raw --processed-dir data/processed
```

Train the Phase 4 CNN with:

```bash
python main.py --mode train-cnn --train-csv data/splits/train_faces.csv --val-csv data/splits/val_faces.csv
```

Evaluate it with:

```bash
python main.py --mode evaluate-cnn --test-csv data/splits/test_faces.csv --weights-path models/cnn/weights/lightweight_artifact_cnn.pt
```

## Project Health Check

Run a quick roadmap and dataset status audit:

```bash
python main.py --mode project-status
```

This reports phase coverage, dataset inventory, and split/face manifest counts.

## Explainable UI (Streamlit)

Launch the dashboard:

```bash
streamlit run streamlit_app.py
```

The dashboard supports video upload, runs the integrated pipeline, and displays confidence, risk level, module scores, and explanations.

## Dataset Layout

- `data/raw/real/`
- `data/raw/fake/`
- `data/processed/frames/`
- `data/processed/audio/`
- `data/processed/faces/`
- `data/processed/landmarks/`

## Notes

- Watermark/NLP and scoring/UI are currently scaffolded for practical iteration, and can be upgraded with production STT/NLP backends.
- The current scaffold favors modularity and explainability over an end-to-end black box.
- Landmark-based blink and mouth tracking depend on a compatible facial landmark backend. If the installed MediaPipe package does not expose the legacy face mesh API, the pipeline degrades safely and still supports preprocessing plus visual artifact analysis.