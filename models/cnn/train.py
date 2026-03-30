from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.cnn.architecture import LightweightArtifactCNN
from src.data.cnn_dataset import FaceImageDataset


@dataclass(slots=True)
class TrainingResult:
    weights_path: str
    train_size: int
    validation_size: int
    final_train_loss: float
    final_validation_loss: float | None
    final_validation_accuracy: float | None


def _evaluate_model(model: LightweightArtifactCNN, data_loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_samples += len(labels)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def _collect_fake_probabilities(model: LightweightArtifactCNN, data_loader: DataLoader) -> tuple[list[float], list[int]]:
    model.eval()
    probabilities: list[float] = []
    labels_out: list[int] = []
    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            fake_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            probabilities.extend(float(item) for item in fake_probs.cpu().tolist())
            labels_out.extend(int(item) for item in labels.cpu().tolist())
    return probabilities, labels_out


def _threshold_accuracy(probabilities: list[float], labels: list[int], threshold: float) -> float:
    if not labels:
        return 0.0
    predicted = [1 if prob >= threshold else 0 for prob in probabilities]
    correct = sum(int(p == y) for p, y in zip(predicted, labels))
    return correct / len(labels)


def _find_best_threshold(probabilities: list[float], labels: list[int]) -> tuple[float, float]:
    if not labels:
        return 0.5, 0.0

    candidates = {0.5}
    candidates.update(min(max(prob, 0.0), 1.0) for prob in probabilities)

    best_threshold = 0.5
    best_accuracy = _threshold_accuracy(probabilities, labels, best_threshold)
    for threshold in sorted(candidates):
        accuracy = _threshold_accuracy(probabilities, labels, threshold)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold, best_accuracy


def _calibration_path(weights_path: Path) -> Path:
    return weights_path.with_suffix(".calibration.json")


def _write_calibration(weights_path: Path, threshold: float, validation_accuracy: float) -> Path:
    calibration_path = _calibration_path(weights_path)
    payload = {
        "fake_probability_threshold": float(threshold),
        "validation_accuracy_at_threshold": float(validation_accuracy),
    }
    calibration_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return calibration_path


def _read_calibration_threshold(weights_path: Path) -> float:
    calibration_path = _calibration_path(weights_path)
    if not calibration_path.exists():
        return 0.5
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        threshold = float(payload.get("fake_probability_threshold", 0.5))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, threshold))


def train_model(train_loader: DataLoader, validation_loader: DataLoader | None, output_path: Path, epochs: int = 10) -> Path:
    model = LightweightArtifactCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * len(labels)
            train_samples += len(labels)

        train_loss = train_loss_total / train_samples if train_samples else 0.0

        validation_loss = None
        validation_accuracy = None
        if validation_loader is not None:
            validation_loss, validation_accuracy = _evaluate_model(model, validation_loader, criterion)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        elapsed = time.perf_counter() - epoch_started
        if validation_loss is None:
            print(
                f"epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | elapsed={elapsed:.1f}s | checkpoint={output_path}",
                flush=True,
            )
        else:
            print(
                f"epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_loss={validation_loss:.4f} | val_acc={validation_accuracy:.4f} | elapsed={elapsed:.1f}s | checkpoint={output_path}",
                flush=True,
            )

    return output_path


def train_from_csv(
    train_csv: Path,
    validation_csv: Path | None,
    output_path: Path,
    epochs: int = 10,
    batch_size: int = 16,
    image_size: int = 128,
) -> dict:
    train_dataset = FaceImageDataset(train_csv, image_size=image_size)
    validation_dataset = FaceImageDataset(validation_csv, image_size=image_size) if validation_csv and validation_csv.exists() else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if validation_dataset else None

    model = LightweightArtifactCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    label_counts = Counter(record.label for record in train_dataset.records)
    total = max(sum(label_counts.values()), 1)
    count_0 = max(label_counts.get(0, 0), 1)
    count_1 = max(label_counts.get(1, 0), 1)
    class_weights = torch.tensor([total / (2.0 * count_0), total / (2.0 * count_1)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if output_path.exists():
        model.load_state_dict(torch.load(output_path, map_location="cpu"))
        print(f"resume checkpoint | path={output_path}", flush=True)
    print(
        f"train class distribution | real(0)={label_counts.get(0, 0)} fake(1)={label_counts.get(1, 0)} | class_weights={[round(float(w), 4) for w in class_weights]}",
        flush=True,
    )
    total_batches = len(train_loader)
    print(
        f"training setup | samples={len(train_dataset)} | batch_size={batch_size} | total_batches={total_batches} | image_size={image_size}",
        flush=True,
    )
    final_train_loss = 0.0
    best_metric = float("inf")
    best_validation_loss = None
    best_validation_accuracy = None
    best_threshold = 0.5
    best_validation_accuracy_at_threshold = None

    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        log_interval = max(min(total_batches // 10, 25), 1)
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * len(labels)
            train_samples += len(labels)
            if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches:
                batch_elapsed = time.perf_counter() - epoch_started
                running_loss = train_loss_total / train_samples if train_samples else 0.0
                print(
                    f"epoch {epoch + 1}/{epochs} | batch {batch_idx}/{total_batches} | running_loss={running_loss:.4f} | elapsed={batch_elapsed:.1f}s",
                    flush=True,
                )
        final_train_loss = train_loss_total / train_samples if train_samples else 0.0

        validation_loss = None
        validation_accuracy = None
        if validation_loader is not None:
            validation_loss, validation_accuracy = _evaluate_model(model, validation_loader, criterion)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        metric = validation_loss if validation_loss is not None else final_train_loss
        if metric <= best_metric:
            best_metric = metric
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), output_path)
        elapsed = time.perf_counter() - epoch_started
        if validation_loss is None:
            print(
                f"epoch {epoch + 1}/{epochs} | train_loss={final_train_loss:.4f} | elapsed={elapsed:.1f}s | checkpoint={output_path}",
                flush=True,
            )
        else:
            print(
                f"epoch {epoch + 1}/{epochs} | train_loss={final_train_loss:.4f} | val_loss={validation_loss:.4f} | val_acc={validation_accuracy:.4f} | elapsed={elapsed:.1f}s | checkpoint={output_path}",
                flush=True,
            )

    result = TrainingResult(
        weights_path=str(output_path),
        train_size=len(train_dataset),
        validation_size=len(validation_dataset) if validation_dataset else 0,
        final_train_loss=final_train_loss,
        final_validation_loss=best_validation_loss,
        final_validation_accuracy=best_validation_accuracy,
    )

    if validation_loader is not None and output_path.exists():
        best_model = LightweightArtifactCNN()
        best_model.load_state_dict(torch.load(output_path, map_location="cpu"))
        probabilities, labels = _collect_fake_probabilities(best_model, validation_loader)
        best_threshold, best_validation_accuracy_at_threshold = _find_best_threshold(probabilities, labels)
        calibration_path = _write_calibration(output_path, best_threshold, best_validation_accuracy_at_threshold)
        print(
            f"calibration | threshold={best_threshold:.4f} | val_acc_at_threshold={best_validation_accuracy_at_threshold:.4f} | path={calibration_path}",
            flush=True,
        )

    payload = asdict(result)
    payload["fake_probability_threshold"] = best_threshold
    payload["validation_accuracy_at_threshold"] = best_validation_accuracy_at_threshold
    return payload


def evaluate_from_csv(dataset_csv: Path, weights_path: Path, batch_size: int = 16, image_size: int = 128) -> dict:
    from models.cnn.infer import load_model

    dataset = FaceImageDataset(dataset_csv, image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = load_model(weights_path)
    criterion = nn.CrossEntropyLoss()
    loss, accuracy = _evaluate_model(model, data_loader, criterion)
    probabilities, labels = _collect_fake_probabilities(model, data_loader)
    threshold = _read_calibration_threshold(weights_path)
    threshold_accuracy = _threshold_accuracy(probabilities, labels, threshold)
    return {
        "dataset_size": len(dataset),
        "weights_path": str(weights_path),
        "loss": loss,
        "accuracy": threshold_accuracy,
        "argmax_accuracy": accuracy,
        "fake_probability_threshold": threshold,
    }