"""
detector.py
-----------
Face detection backends.

Two interchangeable detectors are provided behind one interface:

1. HaarCascadeDetector  - classic Viola-Jones detector. Ships INSIDE
   opencv-python, works fully offline, fast on CPU, but less accurate
   on profile faces / poor lighting / small faces.

2. DnnFaceDetector       - OpenCV's deep-learning SSD face detector
   (ResNet-10 backbone, res10_300x300_ssd_iter_140000). Much more
   robust to pose, occlusion and lighting, and still runs comfortably
   on CPU. Requires two small model files (see download_models.sh)
   since they are not bundled with OpenCV.

Both return detections in the same format:
    List[Tuple[x, y, w, h, confidence]]
so the rest of the application never needs to know which backend
is active.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

Detection = Tuple[int, int, int, int, float]  # x, y, w, h, confidence


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float

    def as_slice(self):
        """Return (y1, y2, x1, x2) for cropping a numpy image."""
        return self.y, self.y + self.h, self.x, self.x + self.w


class BaseDetector:
    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        raise NotImplementedError


class HaarCascadeDetector(BaseDetector):
    """
    Fast, dependency-free detector built into opencv-python.
    Good default / fallback when no internet access is available
    to download DNN model weights.
    """

    def __init__(
        self,
        cascade_name: str = "haarcascade_frontalface_default.xml",
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ):
        cascade_path = os.path.join(cv2.data.haarcascades, cascade_name)
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")
        self.classifier = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves detection in uneven lighting
        rects = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        # Haar cascades don't produce a confidence score, so we use 1.0
        return [FaceBox(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in rects]


class DnnFaceDetector(BaseDetector):
    """
    OpenCV DNN SSD face detector (Caffe model).
    Needs model files downloaded once via download_models.sh:
        models/deploy.prototxt
        models/res10_300x300_ssd_iter_140000.caffemodel
    """

    def __init__(
        self,
        prototxt_path: str = "models/deploy.prototxt",
        model_path: str = "models/res10_300x300_ssd_iter_140000.caffemodel",
        confidence_threshold: float = 0.5,
    ):
        if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
            raise FileNotFoundError(
                "DNN model files not found. Run `bash download_models.sh` first, "
                f"expected:\n  {prototxt_path}\n  {model_path}"
            )
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(FaceBox(x1, y1, x2 - x1, y2 - y1, confidence))
        return boxes


def build_detector(method: str = "haar", **kwargs) -> BaseDetector:
    """
    Factory. method: "haar" | "dnn"
    Falls back to Haar automatically if DNN weights are missing,
    so the app always runs offline.
    """
    method = method.lower()
    if method == "dnn":
        try:
            return DnnFaceDetector(**kwargs)
        except FileNotFoundError as e:
            print(f"[warn] {e}\n[warn] Falling back to Haar cascade detector.")
            return HaarCascadeDetector()
    return HaarCascadeDetector(**kwargs)
