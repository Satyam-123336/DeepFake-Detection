from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deepfake detection workflow through Phase 4.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["analyze-video", "gather-datasets", "prepare-dataset", "train-cnn", "evaluate-cnn", "project-status"],
        help="Workflow mode to run.",
    )
    parser.add_argument("--video", help="Path to an input video file.")
    parser.add_argument(
        "--output-dir",
        "--processed-dir",
        dest="output_dir",
        default="data/processed",
        help="Directory for intermediate artifacts.",
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Raw dataset directory containing real/ and fake/.")
    parser.add_argument("--manifest-path", default="data/manifests/video_manifest.csv", help="Video manifest output path.")
    parser.add_argument("--split-dir", default="data/splits", help="Directory for train/val/test manifests.")
    parser.add_argument("--train-csv", default="data/splits/train_faces.csv", help="Face-level training CSV.")
    parser.add_argument("--val-csv", default="data/splits/val_faces.csv", help="Face-level validation CSV.")
    parser.add_argument("--test-csv", default="data/splits/test_faces.csv", help="Face-level test CSV.")
    parser.add_argument("--weights-path", default="models/cnn/weights/lightweight_artifact_cnn.pt", help="CNN weights path.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=128, help="Square image size for the CNN.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument("--max-videos", type=int, default=None, help="Optional limit for dataset preparation runs.")
    parser.add_argument("--raw-output-dir", default="data/raw", help="Destination raw directory with real/ and fake/.")
    parser.add_argument("--ffpp-root", default=None, help="Local FaceForensics++ root directory.")
    parser.add_argument("--dfdc-root", default=None, help="Local DFDC root directory containing metadata.json files.")
    parser.add_argument("--custom-real-dir", default=None, help="Custom local directory of real videos to ingest.")
    parser.add_argument("--custom-fake-dir", default=None, help="Custom local directory of fake videos to ingest.")
    parser.add_argument("--max-real", type=int, default=None, help="Optional maximum real videos to ingest per source.")
    parser.add_argument("--max-fake", type=int, default=None, help="Optional maximum fake videos to ingest per source.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "analyze-video":
        from src.pipeline.run_full_pipeline import run_phase_four_pipeline

        if not args.video:
            raise ValueError("--video is required when --mode analyze-video is used.")
        result = run_phase_four_pipeline(Path(args.video), Path(args.output_dir))
        print(result.to_dict())
        return

    if args.mode == "prepare-dataset":
        from src.data.dataset_manifest import scan_raw_video_dataset, write_video_records_csv
        from src.data.split_builder import build_split_files
        from src.pipeline.run_dataset_preprocessing import preprocess_manifest_to_faces
        from src.utils.io import ensure_dir

        records = scan_raw_video_dataset(Path(args.raw_dir), max_videos=args.max_videos)
        manifest_path = Path(args.manifest_path)
        write_video_records_csv(manifest_path, records)
        split_paths = build_split_files(records, Path(args.split_dir), seed=args.seed)
        status_path = Path("reports/prepare_dataset_status.log")
        ensure_dir(status_path.parent)
        status_path.write_text("{\n  \"stage\": \"initialized\"\n}\n", encoding="utf-8")
        print({"status_log": str(status_path.resolve())})

        summaries = {}
        for split_name, split_path in split_paths.items():
            face_manifest_path = Path(args.split_dir) / f"{split_name}_faces.csv"
            summaries[split_name] = preprocess_manifest_to_faces(
                manifest_path=split_path,
                processed_dir=Path(args.output_dir),
                face_manifest_path=face_manifest_path,
                split_name=split_name,
                status_path=status_path,
            )

        print(
            {
                "video_manifest": str(manifest_path),
                "splits": {name: str(path) for name, path in split_paths.items()},
                "face_summaries": summaries,
            }
        )
        return

    if args.mode == "gather-datasets":
        from src.data.dataset_ingestion import ingest_dfdc, ingest_faceforensicspp, ingest_labeled_directory

        raw_output_dir = Path(args.raw_output_dir)
        summaries = []

        if args.ffpp_root:
            summary = ingest_faceforensicspp(
                ffpp_root=Path(args.ffpp_root),
                raw_dir=raw_output_dir,
                max_real=args.max_real,
                max_fake=args.max_fake,
            )
            summaries.append(asdict(summary))

        if args.dfdc_root:
            summary = ingest_dfdc(
                dfdc_root=Path(args.dfdc_root),
                raw_dir=raw_output_dir,
                max_real=args.max_real,
                max_fake=args.max_fake,
            )
            summaries.append(asdict(summary))

        if args.custom_real_dir:
            summary = ingest_labeled_directory(
                source_dir=Path(args.custom_real_dir),
                label_name="real",
                raw_dir=raw_output_dir,
                max_videos=args.max_real,
                prefix="custom_real",
            )
            summaries.append(asdict(summary))

        if args.custom_fake_dir:
            summary = ingest_labeled_directory(
                source_dir=Path(args.custom_fake_dir),
                label_name="fake",
                raw_dir=raw_output_dir,
                max_videos=args.max_fake,
                prefix="custom_fake",
            )
            summaries.append(asdict(summary))

        if not summaries:
            raise ValueError(
                "No dataset source provided. Use --ffpp-root, --dfdc-root, --custom-real-dir, or --custom-fake-dir."
            )

        print({"raw_output_dir": str(raw_output_dir), "summaries": summaries})
        return

    if args.mode == "train-cnn":
        from models.cnn.train import train_from_csv

        result = train_from_csv(
            train_csv=Path(args.train_csv),
            validation_csv=Path(args.val_csv),
            output_path=Path(args.weights_path),
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
        print(result)
        return

    if args.mode == "evaluate-cnn":
        from models.cnn.train import evaluate_from_csv

        result = evaluate_from_csv(
            dataset_csv=Path(args.test_csv),
            weights_path=Path(args.weights_path),
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
        print(result)
        return

    if args.mode == "project-status":
        from src.pipeline.project_status import build_project_status

        print(build_project_status(Path(".")))
        return


if __name__ == "__main__":
    main()