from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from src.data.dataset_manifest import VideoRecord
from src.utils.io import ensure_dir


def _partition_count(total: int, ratio: float) -> int:
    return int(total * ratio)


def build_splits(
    records: list[VideoRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[VideoRecord]]:
    grouped: dict[int, list[VideoRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for label_records in grouped.values():
        rng.shuffle(label_records)
        total = len(label_records)
        train_count = _partition_count(total, train_ratio)
        val_count = _partition_count(total, val_ratio)

        splits["train"].extend(label_records[:train_count])
        splits["val"].extend(label_records[train_count:train_count + val_count])
        splits["test"].extend(label_records[train_count + val_count:])

    return splits


def build_split_files(records: list[VideoRecord], output_dir: Path, seed: int = 42) -> dict[str, Path]:
    ensure_dir(output_dir)
    split_records = build_splits(records, seed=seed)
    split_paths: dict[str, Path] = {}

    for split_name, rows in split_records.items():
        output_path = output_dir / f"{split_name}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video_path", "label", "label_name"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "video_path": row.video_path,
                        "label": row.label,
                        "label_name": row.label_name,
                    }
                )
        split_paths[split_name] = output_path

    return split_paths