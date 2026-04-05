from src.data.dataset_manifest import VideoRecord
from src.data.split_builder import build_splits


def test_split_builder_preserves_all_records() -> None:
    records = [
        VideoRecord(video_path=f"real_{index}.mp4", label=0, label_name="real")
        for index in range(10)
    ] + [
        VideoRecord(video_path=f"fake_{index}.mp4", label=1, label_name="fake")
        for index in range(10)
    ]

    splits = build_splits(records, seed=7)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(records)