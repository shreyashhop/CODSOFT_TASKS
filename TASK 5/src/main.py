"""
main.py
-------
Command-line entry point for the face detection & recognition app.

Commands
--------
detect      Detect faces in an image, a video file, or the webcam.
enroll      Capture face images of a person from webcam/photos into dataset/.
train       Train a recognizer (LBPH or ArcFace) on dataset/.
recognize   Detect + recognize faces in an image, video file, or webcam.

Examples
--------
python -m src.main detect --source photo.jpg
python -m src.main detect --source 0                      # webcam
python -m src.main enroll --name alice --source 0 --count 20
python -m src.main train --dataset dataset --method lbph --out models/gallery
python -m src.main recognize --source photo.jpg --method lbph --model models/gallery
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import build_detector
from src.recognizer import build_recognizer
from src.utils import crop_box, draw_detections


def _open_source(source: str):
    """source is either an image/video file path or a webcam index like '0'."""
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    if source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        return None  # signal: still image
    return cv2.VideoCapture(source)


def cmd_detect(args):
    detector = build_detector(args.method)

    if args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        frame = cv2.imread(args.source)
        if frame is None:
            raise FileNotFoundError(args.source)
        boxes = detector.detect(frame)
        out = draw_detections(frame, boxes)
        out_path = args.output or "detected.jpg"
        cv2.imwrite(out_path, out)
        print(f"[info] {len(boxes)} face(s) found. Saved -> {out_path}")
        return

    cap = _open_source(args.source)
    _run_stream(cap, lambda frame: _detect_frame(frame, detector), args.output)


def _detect_frame(frame, detector):
    boxes = detector.detect(frame)
    return draw_detections(frame, boxes)


def cmd_enroll(args):
    os.makedirs(os.path.join(args.dataset, args.name), exist_ok=True)
    detector = build_detector(args.method)
    cap = _open_source(args.source)

    if cap is None:  # single still image supplied
        frame = cv2.imread(args.source)
        boxes = detector.detect(frame)
        for i, box in enumerate(boxes):
            crop = crop_box(frame, box)
            path = os.path.join(args.dataset, args.name, f"{args.name}_{i:03d}.jpg")
            cv2.imwrite(path, crop)
        print(f"[info] Saved {len(boxes)} face crop(s) to {args.dataset}/{args.name}")
        return

    saved = 0
    print(f"[info] Capturing up to {args.count} face images for '{args.name}'. Press 'q' to stop early.")
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detector.detect(frame)
        display = draw_detections(frame, boxes)
        cv2.imshow("enroll - press q to quit", display)

        if boxes:
            box = max(boxes, key=lambda b: b.w * b.h)
            crop = crop_box(frame, box)
            path = os.path.join(args.dataset, args.name, f"{args.name}_{saved:03d}.jpg")
            cv2.imwrite(path, crop)
            saved += 1
            time.sleep(0.2)  # small delay so consecutive frames aren't near-duplicates

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[info] Saved {saved} images to {args.dataset}/{args.name}")


def cmd_train(args):
    recognizer = build_recognizer(args.method)
    recognizer.train(args.dataset)
    recognizer.save(args.out)
    print(f"[info] Model saved with prefix: {args.out}")


def cmd_recognize(args):
    detector = build_detector(args.detector)
    recognizer = build_recognizer(args.method)
    recognizer.load(args.model)

    # LBPH: lower distance = better match, so threshold is a max-distance.
    # ArcFace: higher similarity = better match, threshold is a min-similarity
    # (already applied inside ArcFaceRecognizer.predict).
    def label_for(name, score):
        if args.method == "lbph":
            if score > args.lbph_max_distance:
                return "unknown"
            return f"{name} ({score:.0f})"
        return f"{name} ({score:.2f})"

    def process(frame):
        boxes = detector.detect(frame)
        labels = []
        for box in boxes:
            crop = crop_box(frame, box)
            if crop.size == 0:
                labels.append("unknown")
                continue
            name, score = recognizer.predict(crop)
            labels.append(label_for(name, score))
        return draw_detections(frame, boxes, labels)

    if args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        frame = cv2.imread(args.source)
        if frame is None:
            raise FileNotFoundError(args.source)
        out = process(frame)
        out_path = args.output or "recognized.jpg"
        cv2.imwrite(out_path, out)
        print(f"[info] Saved -> {out_path}")
        return

    cap = _open_source(args.source)
    _run_stream(cap, process, args.output)


def _run_stream(cap, process_fn, output_path):
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = process_fn(frame)

        if output_path:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
            writer.write(out)
        else:
            cv2.imshow("press q to quit", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def build_parser():
    p = argparse.ArgumentParser(description="Face detection & recognition CLI")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="Detect faces in image/video/webcam")
    d.add_argument("--source", required=True, help="image/video path or webcam index e.g. 0")
    d.add_argument("--method", default="haar", choices=["haar", "dnn"])
    d.add_argument("--output", default=None, help="output image/video path (omit for live window)")
    d.set_defaults(func=cmd_detect)

    e = sub.add_parser("enroll", help="Capture labeled face images into the dataset")
    e.add_argument("--name", required=True)
    e.add_argument("--source", required=True, help="webcam index or a photo path")
    e.add_argument("--dataset", default="dataset")
    e.add_argument("--count", type=int, default=20)
    e.add_argument("--method", default="haar", choices=["haar", "dnn"])
    e.set_defaults(func=cmd_enroll)

    t = sub.add_parser("train", help="Train a recognizer on dataset/")
    t.add_argument("--dataset", default="dataset")
    t.add_argument("--method", default="lbph", choices=["lbph", "arcface"])
    t.add_argument("--out", default="models/gallery")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("recognize", help="Detect + recognize faces")
    r.add_argument("--source", required=True)
    r.add_argument("--model", default="models/gallery")
    r.add_argument("--method", default="lbph", choices=["lbph", "arcface"])
    r.add_argument("--detector", default="haar", choices=["haar", "dnn"])
    r.add_argument("--output", default=None)
    r.add_argument("--lbph-max-distance", type=float, default=70.0,
                    help="LBPH distance above this is labeled 'unknown' (lower=stricter)")
    r.set_defaults(func=cmd_recognize)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
