from pathlib import Path

from src.pipeline.project_status import build_project_status


def test_project_status_reports_phase6_complete_with_ui(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "behavioral").mkdir(parents=True)
    (src / "preprocessing").mkdir(parents=True)
    (src / "visual").mkdir(parents=True)
    (src / "pipeline").mkdir(parents=True)
    (src / "forensic").mkdir(parents=True)
    (src / "nlp").mkdir(parents=True)
    (src / "scoring").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "streamlit_app.py").write_text("", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest>=8.0\n", encoding="utf-8")
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "model_config.yaml").write_text("{}", encoding="utf-8")
    (tmp_path / "configs" / "preprocessing_config.yaml").write_text("{}", encoding="utf-8")

    status = build_project_status(tmp_path)
    phase6 = [p for p in status["phases"] if p["phase"].startswith("Phase 6")][0]

    assert phase6["status"] == "complete"
    assert status["code_coverage_hint"]["has_ui_entrypoint"] is True
