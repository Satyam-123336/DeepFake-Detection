# DeepFake Detection Project: Full Details So Far (Easy Language)

## 1. Project Goal
This project checks whether a video is likely real or likely deepfake.

It does not rely on just one signal. It combines multiple checks:
- Face and frame quality checks
- Eye blink behavior checks
- Lip movement vs audio timing checks
- CNN-based visual fake probability
- Watermark or synthetic trace checks
- Basic speech/NLP suspicion checks

This gives a more explainable result instead of a black-box yes/no output.

## 2. What Is Already Built
According to the roadmap and project docs, these phases are implemented:

- Phase 1: Foundation and setup (implemented, with metadata schema support)
- Phase 2: Preprocessing and key-frame extraction (complete)
- Phase 3: Behavioral analysis (complete)
- Phase 4: Visual artifact + lightweight CNN (complete)
- Phase 5: NLP + forensic integration (implemented in initial/proxy form)
- Phase 6: Scoring engine + explainable UI (implemented initial version)
- Phase 7: Testing and optimization (in progress, with many tests already present)

## 3. Core Workflow in Simple Terms
When a video is analyzed, this is what happens:

1. The system reads video metadata and extracts audio.
2. It selects key frames from the video.
3. It detects face regions and landmarks from frames.
4. It calculates behavioral signals:
   - Blink patterns
   - Lip-sync consistency with audio
5. It calculates visual signals:
   - Lighting asymmetry
   - Sharpness and texture artifact scores
   - CNN fake probability from face crops
6. It runs forensic and NLP modules:
   - Watermark/synthetic trace heuristics
   - Proxy transcription + NLP suspicion score
7. It combines all module scores with weighted scoring logic.
8. It returns:
   - Confidence score
   - Risk level (low/medium/high)
   - Module-level scores
   - Human-readable reasons

## 4. Interfaces You Can Use
The project supports multiple ways to run:

### 4.1 Command Line (CLI)
Main modes currently available:
- analyze-video
- gather-datasets
- prepare-dataset
- train-cnn
- evaluate-cnn
- project-status

### 4.2 Streamlit UI
A rich explainable dashboard is available with:
- Upload and analyze flow
- Risk verdict and confidence
- Radar/bar/gauge visualizations
- Module-wise reasons
- Technical metadata views
- Cache/system stats sidebar

### 4.3 FastAPI + React Frontend
Production-style API + TypeScript frontend are included:
- Async job-based analysis
- Sync analysis endpoint
- Job tracking and cancellation
- Cache stats and cleanup endpoints
- WebSocket progress updates

## 5. Dataset and Training Pipeline Status
The project already includes a full data workflow:

- Ingest from FaceForensics++ and DFDC
- Ingest custom real/fake folders
- Create video manifests
- Build train/val/test splits
- Preprocess splits into frames/audio/faces
- Build face-level manifests
- Train and evaluate a lightweight CNN model

This means end-to-end data preparation and model training flow is available.

## 6. Scoring Logic (Easy Explanation)
The final decision uses weighted module scores:

- Blink: 0.20
- Lip-sync: 0.25
- Visual: 0.35
- Watermark: 0.10
- NLP: 0.10

Why this is useful:
- Visual artifacts have highest impact.
- Behavioral checks still matter strongly.
- Forensic/NLP act as supporting evidence.

The system also has escalation rules so strong suspicious evidence is not hidden by averaging.

Risk thresholds:
- High: confidence >= 0.70
- Medium: confidence >= 0.45 and < 0.70
- Low: confidence < 0.45

## 7. Important Fixes Already Done (Methodology Audit)
The audit document reports major correctness fixes already applied:

1. CNN semantic bug fixed
- The system now uses fake class probability correctly.

2. CNN weight file path fallback fixed
- It can resolve canonical and demo weight paths.

3. Lip-sync timestamp alignment fixed
- Audio sampling is aligned to mouth-signal timestamps.

4. Missing-signal scoring sensitivity reduced
- Weak evidence no longer causes aggressive false suspicion.

These fixes improved consistency with methodology and reduced false alarms.

## 8. Testing Status
The repository has strong test coverage for major modules, including:
- Dataset ingestion and manifest tests
- Split builder tests
- Blink detector tests
- Artifact feature tests
- CNN dataset tests
- Scoring engine tests
- Full-pipeline schema tests
- End-to-end smoke test
- Phase 5 module tests
- Project status tests

The documentation says current suite is passing at the time of the audit, and test expansion is ongoing.

## 9. Deployment and Demo Readiness
Current deployment options documented:
- Local Streamlit app
- Local FastAPI + React development stack
- Dockerized run option
- Cloud deployment suggestions

Demo runbook is available with curated real/fake sample order for presentation.

## 10. Current Limitations (Open Work)
Based on roadmap and docs, key open items are:
- Replace proxy transcription with production STT backend
- Improve NLP model depth on real transcripts
- Add richer threshold calibration using validation metrics
- Continue performance optimization and caching strategy
- Complete final methodology/limitations publication and broader end-to-end validation

## 11. Business/Practical Value Already Achieved
This project already delivers:
- Explainable multi-signal deepfake risk analysis
- Practical UI for non-technical users
- API for integration with other systems
- Data ingestion/training path for future model improvements
- Modular architecture for safe iteration phase by phase

## 12. Recommended Next Steps (Simple Priority)
1. Productionize Phase 5:
- Integrate Whisper or equivalent STT
- Upgrade NLP suspicion logic with stronger text features

2. Calibration and validation:
- Run labeled batch evaluation
- Tune thresholds from real metrics (not only heuristics)

3. Hardening for production:
- Add stronger async monitoring/logging
- Expand E2E tests with edge-case clips

4. Documentation finalization:
- Freeze a versioned methodology report
- Add known-failure-mode section for users

## 13. Source References Used
- [README.md](README.md)
- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)
- [reports/METHODOLOGY_AUDIT_AND_FIXES.md](reports/METHODOLOGY_AUDIT_AND_FIXES.md)
- [FRONTEND_DEPLOYMENT.md](FRONTEND_DEPLOYMENT.md)
- [main.py](main.py)
- [api_server.py](api_server.py)
- [streamlit_app.py](streamlit_app.py)
- [src/pipeline/run_full_pipeline.py](src/pipeline/run_full_pipeline.py)
- [src/scoring/engine.py](src/scoring/engine.py)
- [src/pipeline/project_status.py](src/pipeline/project_status.py)
- [test videos/DEMO_RUNBOOK.md](test%20videos/DEMO_RUNBOOK.md)
- [tests/test_end_to_end_pipeline_smoke.py](tests/test_end_to_end_pipeline_smoke.py)
- [tests/test_scoring_engine.py](tests/test_scoring_engine.py)
- [tests/test_run_full_pipeline_schema.py](tests/test_run_full_pipeline_schema.py)

## 14. Exact Detection Process (What Exactly Happens)

This section explains each detection module exactly as implemented in code.

### 14.1 Preprocessing Stage
1. Read video metadata using OpenCV:
- FPS
- frame count
- width/height
- duration

2. Create a unique artifact folder from video path hash.

3. Extract audio from video to WAV (16 kHz) using MoviePy.

4. Extract key frames every 0.2 seconds (dense sampling for behavior analysis).

5. Save frame-to-time mapping JSON for downstream stages.

### 14.2 Face Detection Stage
For each candidate frame:
1. Try MediaPipe face detection first.
2. If MediaPipe is unavailable or fails, fallback to OpenCV Haar cascade detector.
3. If no face is detected, visual module for that frame returns empty.

### 14.3 Landmark Extraction Stage (for behavior)
1. Use MediaPipe Face Mesh for 468 landmarks (static image mode).
2. If frame is very large, it downsizes before mesh inference for stability.
3. If full-frame mesh fails, it tries Haar-face crop + mesh on crop.
4. If still no landmarks, that frame is skipped in behavioral timelines.

### 14.4 Blink Detection (EAR-based)
For each valid landmark frame:
1. Compute Eye Aspect Ratio (EAR) for left and right eyes.
2. Use average EAR timeline.
3. A blink event is detected when EAR stays below threshold 0.21 for at least 2 consecutive sampled frames.

EAR formula:
$$
EAR = \frac{\lVert p_2-p_6 \rVert + \lVert p_3-p_5 \rVert}{2\lVert p_1-p_4 \rVert}
$$

Then blink windows are summarized into features (count and irregularity).

### 14.5 Lip-Sync Detection
1. Build mouth openness signal from upper/lower lip landmarks.
2. Extract audio RMS envelope from WAV.
3. Interpolate audio envelope exactly at mouth timestamps.
4. Compute cross-correlation between normalized audio signal and mouth signal.
5. Return:
- best time offset (seconds)
- correlation score
- average absolute error

If audio or valid signals are missing, module returns fallback values.

### 14.6 Visual Artifact Detection
On detected face crop:
1. Compute sharpness score: Laplacian variance.
2. Compute texture score: grayscale standard deviation.
3. Compute brightness variance.
4. Compute lighting asymmetry by comparing left/right mirrored halves.

This gives handcrafted forensic-style artifact signals.

### 14.7 CNN Fake Probability Detection
1. Load lightweight CNN weights.
2. Resize face crop to 128 x 128.
3. Run forward pass and softmax.
4. Use class-1 probability as fake probability.

CNN fake probability:
$$
P_{fake} = \text{softmax}(logits)_1
$$

Prediction rule:
- fake if $P_{fake} \ge$ threshold (default 0.5 or calibration file value)
- real otherwise

Returned values:
- predicted class
- confidence
- fake probability

### 14.8 Watermark / Synthetic Trace Detection
Two heuristic channels are used:
1. Metadata signature score:
- checks filename tokens like deepfake, faceswap, synth, fake, etc.

2. Frame overlay pattern score:
- runs Canny edges on top strip, center strip, bottom strip
- measures edge-density delta
- captures suspicious high-contrast overlay-like patterns

Final watermark confidence is weighted combination:
$$
0.65 \times metadata\_score + 0.35 \times overlay\_score
$$

### 14.9 NLP Suspicion Detection
Transcription process:
1. Try Whisper tiny model (if installed).
2. If Whisper is unavailable, fallback to energy-proxy speech segmentation.

Suspicion scoring checks:
- very low speech segment count for long clips
- near-continuous voiced activity
- unusually uniform segment durations
- low lexical diversity (when real text exists)
- elevated consecutive word repetition
- very flat punctuation cadence

Outputs:
- NLP suspicion score (0 to 1)
- reasons list

### 14.10 Final Decision Engine
Module scores are combined using weights:
- Blink 0.20
- Lip-sync 0.25
- Visual 0.35
- Watermark 0.10
- NLP 0.10

Base confidence is weighted sum, then escalation rules are applied for strong corroborated evidence.

Risk mapping:
- High: confidence >= 0.70
- Medium: confidence >= 0.45 and < 0.70
- Low: confidence < 0.45

## 15. Libraries Used (Exact Stack)

### 15.1 Core Python / ML / Signal Libraries
- numpy
- opencv-python
- mediapipe
- librosa
- soundfile
- moviepy
- PyYAML
- torch
- torchaudio
- Pillow
- pytest

### 15.2 App / API / Visualization Libraries
- streamlit
- fastapi
- uvicorn[standard]
- python-multipart
- aiofiles
- plotly
- pandas
- requests

### 15.3 Frontend Libraries (React App)
- react
- react-dom
- axios
- recharts
- lucide-react
- typescript

Frontend tooling/dev:
- vite
- @vitejs/plugin-react
- tailwindcss
- postcss
- autoprefixer
- @types/react
- @types/react-dom
- @types/node

### 15.4 Python Standard Library Modules Also Used
- pathlib
- dataclasses
- hashlib
- json
- argparse
- uuid
- datetime
- asyncio
- typing
- contextlib
- re
- math
- importlib

### 15.5 Optional/Conditional Libraries
- whisper (OpenAI Whisper): used only if installed; otherwise transcription falls back to energy-proxy mode.

## 16. Where Each Detection Is Implemented
- Pipeline orchestration: [src/pipeline/run_full_pipeline.py](src/pipeline/run_full_pipeline.py)
- Preprocessing: [src/pipeline/run_preprocessing.py](src/pipeline/run_preprocessing.py)
- Behavioral analysis: [src/pipeline/run_behavioral.py](src/pipeline/run_behavioral.py)
- Visual analysis: [src/pipeline/run_visual.py](src/pipeline/run_visual.py)
- Blink detector: [src/behavioral/blink/blink_detector.py](src/behavioral/blink/blink_detector.py)
- Lip-sync analyzer: [src/behavioral/lipsync/sync_analyzer.py](src/behavioral/lipsync/sync_analyzer.py)
- Face detector: [src/preprocessing/face_detector.py](src/preprocessing/face_detector.py)
- Landmark extractor: [src/preprocessing/landmark_extractor.py](src/preprocessing/landmark_extractor.py)
- Artifact features: [src/visual/artifact_features.py](src/visual/artifact_features.py)
- CNN inference: [models/cnn/infer.py](models/cnn/infer.py)
- Watermark detector: [src/forensic/watermark_detector.py](src/forensic/watermark_detector.py)
- Transcription: [src/nlp/transcription.py](src/nlp/transcription.py)
- NLP suspicion: [src/nlp/suspicion.py](src/nlp/suspicion.py)
- Scoring engine: [src/scoring/engine.py](src/scoring/engine.py)
- Backend dependencies list: [requirements.txt](requirements.txt)
- Frontend dependencies list: [frontend/package.json](frontend/package.json)
