from pathlib import Path

from src.data.dataset_manifest import scan_raw_video_dataset, write_video_records_csv, read_video_records_csv


def test_dataset_manifest_scan_and_roundtrip(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    (real_dir / "a.mp4").write_bytes(b"0")
    (fake_dir / "b.mp4").write_bytes(b"1")

    records = scan_raw_video_dataset(tmp_path)
    assert len(records) == 2

    manifest_path = tmp_path / "manifest.csv"
    write_video_records_csv(manifest_path, records)
    loaded = read_video_records_csv(manifest_path)
    assert len(loaded) == 2
    assert {r.label_name for r in loaded} == {"real", "fake"}


def test_dataset_manifest_read_backward_compatible_without_extended_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "legacy_manifest.csv"
    manifest_path.write_text(
        "video_path,label,label_name\n"
        "C:/tmp/a.mp4,0,real\n",
        encoding="utf-8",
    )

    loaded = read_video_records_csv(manifest_path)
    assert len(loaded) == 1
    record = loaded[0]
    assert record.language == "unknown"
    assert record.lighting_quality == "unknown"
    assert record.face_visibility == "unknown"
    assert record.speaking_state == "unknown"
