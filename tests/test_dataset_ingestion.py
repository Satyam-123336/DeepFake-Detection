import json
from pathlib import Path

from src.data.dataset_ingestion import ingest_dfdc, ingest_labeled_directory


def test_ingest_labeled_directory_copies_videos(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_real"
    source_dir.mkdir()
    (source_dir / "r1.mp4").write_bytes(b"video")
    (source_dir / "r2.mp4").write_bytes(b"video")

    raw_dir = tmp_path / "raw"
    stats = ingest_labeled_directory(source_dir=source_dir, label_name="real", raw_dir=raw_dir, max_videos=1)

    assert stats.real_count == 1
    assert len(list((raw_dir / "real").glob("*.mp4"))) == 1


def test_ingest_dfdc_reads_metadata_labels(tmp_path: Path) -> None:
    dfdc_part = tmp_path / "dfdc_train_part_00"
    dfdc_part.mkdir(parents=True)

    real_video = dfdc_part / "real_a.mp4"
    fake_video = dfdc_part / "fake_a.mp4"
    real_video.write_bytes(b"real")
    fake_video.write_bytes(b"fake")

    metadata = {
        "real_a.mp4": {"label": "REAL"},
        "fake_a.mp4": {"label": "FAKE"},
    }
    (dfdc_part / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    raw_dir = tmp_path / "raw"
    stats = ingest_dfdc(dfdc_root=tmp_path, raw_dir=raw_dir)

    assert stats.real_count == 1
    assert stats.fake_count == 1
    assert len(list((raw_dir / "real").glob("*.mp4"))) == 1
    assert len(list((raw_dir / "fake").glob("*.mp4"))) == 1