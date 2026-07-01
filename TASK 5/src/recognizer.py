"""
recognizer.py
-------------
Face recognition backends.

1. LBPHRecognizer   - Local Binary Patterns Histogram recognizer.
   Ships with opencv-contrib-python (cv2.face module), trains in
   seconds on CPU, fully offline. Good baseline / small-dataset
   recognizer.

2. ArcFaceRecognizer - Deep embedding recognizer. Runs a pre-trained
   ArcFace ONNX model (e.g. buffalo_l / w600k_r50 from InsightFace)
   through OpenCV's DNN module to get a 512-d embedding per face,
   then compares embeddings with cosine similarity. Much stronger
   accuracy than LBPH, especially at scale, but needs the ONNX
   weights downloaded once (see download_models.sh) and onnxruntime
   or cv2.dnn support for the model op-set.

Both expose the same interface:
    train(dataset_dir)               -> build a gallery from labeled face crops
    predict(face_crop) -> (name, score)
    save(path) / load(path)
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 1. LBPH - classic, fully offline
# --------------------------------------------------------------------------
class LBPHRecognizer:
    def __init__(self, image_size: Tuple[int, int] = (200, 200)):
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "cv2.face module not found. Install opencv-contrib-python "
                "(pip install opencv-contrib-python)."
            )
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.image_size = image_size
        self.label_to_name: Dict[int, str] = {}
        self._trained = False

    @staticmethod
    def _preprocess(face_bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return cv2.resize(gray, size)

    def train(self, dataset_dir: str) -> None:
        """
        dataset_dir/
            person_a/ img1.jpg img2.jpg ...
            person_b/ img1.jpg ...
        Each image should be a already-cropped face (or a photo where a
        Haar/DNN detector can find exactly one clear face).
        """
        faces, labels = [], []
        self.label_to_name = {}
        next_label = 0

        people = sorted(
            d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
        )
        if not people:
            raise ValueError(f"No person subfolders found in {dataset_dir}")

        detector_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

        for person in people:
            person_dir = os.path.join(dataset_dir, person)
            label = next_label
            next_label += 1
            self.label_to_name[label] = person

            img_count = 0
            for fname in os.listdir(person_dir):
                fpath = os.path.join(person_dir, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue

                # If the image isn't already a tight face crop, try to find
                # and crop the largest face in it.
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                if len(rects) > 0:
                    x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
                    img = img[y : y + h, x : x + w]

                faces.append(self._preprocess(img, self.image_size))
                labels.append(label)
                img_count += 1

            if img_count == 0:
                print(f"[warn] no usable images found for '{person}', skipping")

        if not faces:
            raise ValueError("No training faces collected from dataset_dir.")

        self.model.train(faces, np.array(labels))
        self._trained = True
        print(f"[info] Trained LBPH on {len(faces)} images across {len(people)} people.")

    def predict(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Returns (name, confidence). NOTE: for LBPH, LOWER confidence means
        a BETTER match (it's a distance, not a similarity score).
        """
        if not self._trained:
            raise RuntimeError("Call train() or load() before predict().")
        face = self._preprocess(face_bgr, self.image_size)
        label, distance = self.model.predict(face)
        name = self.label_to_name.get(label, "unknown")
        return name, float(distance)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.write(path + ".yml")
        with open(path + ".labels.json", "w") as f:
            json.dump(self.label_to_name, f)

    def load(self, path: str) -> None:
        self.model.read(path + ".yml")
        with open(path + ".labels.json") as f:
            raw = json.load(f)
        self.label_to_name = {int(k): v for k, v in raw.items()}
        self._trained = True


# --------------------------------------------------------------------------
# 2. ArcFace-style embedding recognizer (optional, higher accuracy)
# --------------------------------------------------------------------------
class ArcFaceRecognizer:
    """
    Uses a pre-trained ArcFace ONNX model to extract a 512-d embedding
    per face, then does nearest-neighbor matching via cosine similarity
    against an enrolled gallery. This mirrors how production face-ID
    systems work (embedding + metric learning) far more closely than
    LBPH, and generalizes much better to unseen lighting/pose.

    Get weights (one-time, requires internet on your own machine):
        see download_models.sh -> models/arcface.onnx
    """

    def __init__(self, model_path: str = "models/arcface.onnx", similarity_threshold: float = 0.45):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace ONNX model not found at {model_path}. "
                "Run `bash download_models.sh` on a machine with internet access."
            )
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.similarity_threshold = similarity_threshold
        self.gallery: Dict[str, List[np.ndarray]] = {}

    @staticmethod
    def _align_and_preprocess(face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(face, axis=0)

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        blob = self._align_and_preprocess(face_bgr)
        self.net.setInput(blob)
        emb = self.net.forward().flatten()
        return emb / (np.linalg.norm(emb) + 1e-10)

    def train(self, dataset_dir: str) -> None:
        """Build an embedding gallery: {person_name: [embeddings]}."""
        self.gallery = {}
        detector_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        people = sorted(
            d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
        )
        for person in people:
            person_dir = os.path.join(dataset_dir, person)
            embeddings = []
            for fname in os.listdir(person_dir):
                img = cv2.imread(os.path.join(person_dir, fname))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                if len(rects) > 0:
                    x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
                    img = img[y : y + h, x : x + w]
                embeddings.append(self.embed(img))
            if embeddings:
                self.gallery[person] = embeddings
        print(f"[info] Built ArcFace gallery for {len(self.gallery)} people.")

    def predict(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        """Returns (name, similarity). HIGHER similarity is a better match."""
        query = self.embed(face_bgr)
        best_name, best_score = "unknown", -1.0
        for name, embeddings in self.gallery.items():
            for emb in embeddings:
                score = float(np.dot(query, emb))
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_score < self.similarity_threshold:
            return "unknown", best_score
        return best_name, best_score

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + ".gallery.pkl", "wb") as f:
            pickle.dump(self.gallery, f)

    def load(self, path: str) -> None:
        with open(path + ".gallery.pkl", "rb") as f:
            self.gallery = pickle.load(f)


def build_recognizer(method: str = "lbph", **kwargs):
    method = method.lower()
    if method == "arcface":
        return ArcFaceRecognizer(**kwargs)
    return LBPHRecognizer(**kwargs)
