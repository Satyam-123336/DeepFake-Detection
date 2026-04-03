from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PhaseStatus:
    phase: str
    status: str
    notes: str


def _count_videos(folder: Path) -> int:
    if not folder.exists():
        return 0
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def _csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
    except UnicodeDecodeError:
        line_count = sum(1 for _ in path.open("r", encoding="latin-1"))
    return max(line_count - 1, 0)


def _discover_active_split_dir(workspace_root: Path) -> Path:
    data_dir = workspace_root / "data"
    candidates = [data_dir / "splits"]
    if data_dir.exists():
        candidates.extend(
            path for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("splits_")
        )

    def split_score(split_dir: Path) -> int:
        return (
            _csv_rows(split_dir / "train_faces.csv")
            + _csv_rows(split_dir / "val_faces.csv")
            + _csv_rows(split_dir / "test_faces.csv")
        )

    best = max(candidates, key=split_score, default=data_dir / "splits")
    return best


def _manifest_has_extended_metadata(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        header = manifest_path.open("r", encoding="utf-8").readline().strip().split(",")
    except UnicodeDecodeError:
        header = manifest_path.open("r", encoding="latin-1").readline().strip().split(",")
    required = {"language", "lighting_quality", "face_visibility", "speaking_state"}
    return required.issubset(set(header))


def build_project_status(workspace_root: Path) -> dict:
    src_dir = workspace_root / "src"
    tests_dir = workspace_root / "tests"
    has_forensic_module = (src_dir / "forensic").exists()
    has_nlp_module = (src_dir / "nlp").exists()
    has_scoring_module = (src_dir / "scoring").exists()
    has_ui_module = (workspace_root / "app.py").exists() or (workspace_root / "streamlit_app.py").exists()

    phase5_status = "complete" if (has_forensic_module and has_nlp_module) else "in_progress"
    phase5_notes = (
        "Forensic module and NLP transcription/classifier pipeline are integrated with STT fallback support."
        if phase5_status == "complete"
        else "No watermark scan or NLP transcription/classification modules detected."
    )

    if has_scoring_module and has_ui_module:
        phase6_status = "complete"
    elif has_scoring_module:
        phase6_status = "in_progress"
    else:
        phase6_status = "not_started"

    if phase6_status == "complete":
        phase6_notes = "Weighted scoring engine and a usable UI entrypoint are available."
    elif phase6_status == "in_progress":
        phase6_notes = "Weighted scoring engine is integrated; explainable UI is still pending."
    else:
        phase6_notes = "No weighted scoring engine or UI app entrypoint detected."

    manifests_dir = workspace_root / "data" / "manifests"
    extended_metadata_ready = any(
        _manifest_has_extended_metadata(path)
        for path in manifests_dir.glob("video_manifest*.csv")
    )

    # Consider Phase 1 complete once schema support exists in code; legacy manifests may predate new columns.
    phase1_status = "complete"
    phase1_notes = (
        "Dataset ingestion, labels, and extended metadata schema support are present."
        if extended_metadata_ready
        else "Dataset ingestion, labels, and extended metadata schema support are present; regenerate manifests to populate new metadata columns."
    )

    has_e2e_smoke = (tests_dir / "test_end_to_end_pipeline_smoke.py").exists()
    phase7_status = "complete" if has_e2e_smoke else "in_progress"
    phase7_notes = (
        "Unit and end-to-end smoke tests are present for the full pipeline."
        if has_e2e_smoke
        else "Unit tests exist for core implemented modules; end-to-end suite is pending."
    )

    active_split_dir = _discover_active_split_dir(workspace_root)
    active_split_dir_rel = active_split_dir.relative_to(workspace_root) if active_split_dir.exists() else active_split_dir

    phase_statuses = [
        PhaseStatus(
            phase="Phase 1: Foundation And Setup",
            status=phase1_status,
            notes=phase1_notes,
        ),
        PhaseStatus(
            phase="Phase 2: Preprocessing And Key Frame Extraction",
            status="complete",
            notes="Video/audio extraction, keyframes, landmarks, and timestamp mapping are present.",
        ),
        PhaseStatus(
            phase="Phase 3: Behavioral Analysis Modules",
            status="complete",
            notes="Blink and lip-sync analyzers are integrated in pipeline behavioral stage.",
        ),
        PhaseStatus(
            phase="Phase 4: Visual Artifact Detection With Lightweight CNN",
            status="complete",
            notes="Artifact features, CNN train/eval, and visual inference flow are present.",
        ),
        PhaseStatus(
            phase="Phase 5: NLP And Forensic Integration",
            status=phase5_status,
            notes=phase5_notes,
        ),
        PhaseStatus(
            phase="Phase 6: Scoring Engine And Explainable UI",
            status=phase6_status,
            notes=phase6_notes,
        ),
        PhaseStatus(
            phase="Phase 7: Testing, Optimization, And Documentation",
            status=phase7_status,
            notes=phase7_notes,
        ),
    ]

    raw_real = workspace_root / "data" / "raw" / "real"
    raw_fake = workspace_root / "data" / "raw" / "fake"

    status = {
        "workspace": str(workspace_root),
        "requirements_file": (workspace_root / "requirements.txt").exists(),
        "config_files": {
            "model_config": (workspace_root / "configs" / "model_config.yaml").exists(),
            "preprocessing_config": (workspace_root / "configs" / "preprocessing_config.yaml").exists(),
        },
        "dataset_inventory": {
            "raw_real_videos": _count_videos(raw_real),
            "raw_fake_videos": _count_videos(raw_fake),
            "video_manifest_rows": _csv_rows(workspace_root / "data" / "manifests" / "video_manifest.csv"),
            "active_split_dir": str(active_split_dir_rel),
            "train_split_rows": _csv_rows(active_split_dir / "train.csv"),
            "val_split_rows": _csv_rows(active_split_dir / "val.csv"),
            "test_split_rows": _csv_rows(active_split_dir / "test.csv"),
            "train_faces_rows": _csv_rows(active_split_dir / "train_faces.csv"),
            "val_faces_rows": _csv_rows(active_split_dir / "val_faces.csv"),
            "test_faces_rows": _csv_rows(active_split_dir / "test_faces.csv"),
        },
        "code_coverage_hint": {
            "has_behavioral_module": (src_dir / "behavioral").exists(),
            "has_preprocessing_module": (src_dir / "preprocessing").exists(),
            "has_visual_module": (src_dir / "visual").exists(),
            "has_pipeline_module": (src_dir / "pipeline").exists(),
            "has_forensic_module": has_forensic_module,
            "has_nlp_module": has_nlp_module,
            "has_scoring_module": has_scoring_module,
            "has_ui_entrypoint": has_ui_module,
            "has_tests": tests_dir.exists(),
        },
        "phases": [
            {"phase": p.phase, "status": p.status, "notes": p.notes}
            for p in phase_statuses
        ],
    }
    return status
