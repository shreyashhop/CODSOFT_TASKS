"""utils.py - drawing and small helper functions shared across the app."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detector import FaceBox


def draw_detections(
    frame_bgr: np.ndarray,
    boxes: List[FaceBox],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 220, 60),
) -> np.ndarray:
    """Draw bounding boxes (and optional recognition labels) on a copy of frame."""
    out = frame_bgr.copy()
    for i, box in enumerate(boxes):
        cv2.rectangle(out, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 2)
        text = ""
        if labels and i < len(labels):
            text = labels[i]
        elif box.confidence < 1.0:
            text = f"{box.confidence:.2f}"
        if text:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(box.y - 10, th + 4)
            cv2.rectangle(out, (box.x, ty - th - 6), (box.x + tw + 6, ty + 2), color, -1)
            cv2.putText(
                out, text, (box.x + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
    return out


def crop_box(frame_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
    y1, y2, x1, x2 = box.as_slice()
    return frame_bgr[y1:y2, x1:x2]
