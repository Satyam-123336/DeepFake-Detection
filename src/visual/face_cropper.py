from __future__ import annotations

from pathlib import Path

import cv2

from src.preprocessing.face_detector import FaceBox
from src.utils.io import ensure_dir


def crop_face(image_path: Path, face_box: FaceBox, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    x1 = max(face_box.x, 0)
    y1 = max(face_box.y, 0)
    x2 = max(face_box.x + face_box.width, x1 + 1)
    y2 = max(face_box.y + face_box.height, y1 + 1)
    cropped = image[y1:y2, x1:x2]

    output_path = output_dir / f"{image_path.stem}_face.jpg"
    cv2.imwrite(str(output_path), cropped)
    return output_path