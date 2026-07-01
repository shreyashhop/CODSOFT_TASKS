# Face Detection & Recognition App

A CLI application for detecting and recognizing faces in images, video files,
or a live webcam feed, with two swappable backends at each stage:

| Stage       | Offline default          | Upgrade option                          |
|-------------|---------------------------|------------------------------------------|
| Detection   | Haar cascade (Viola-Jones)| OpenCV DNN SSD face detector (ResNet-10) |
| Recognition | LBPH (Local Binary Patterns)| ArcFace-style ONNX embeddings + cosine similarity |

The defaults require **no downloads** — they ship inside `opencv-contrib-python`.
The upgrade options are more accurate (especially for pose/lighting variation
and larger galleries of people) but need ~35MB of model weights fetched once
via `download_models.sh` on a machine with internet access.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Detect faces

```bash
# In a photo
python -m src.main detect --source photo.jpg --output out.jpg

# From your webcam (live window, press 'q' to quit)
python -m src.main detect --source 0

# In a video file, saving the annotated result
python -m src.main detect --source input.mp4 --output out.mp4 --method dnn
```

`--method haar` (default) or `--method dnn` (needs downloaded weights, see below).

## 3. Enroll people for recognition

Capture face crops for each person you want the recognizer to know:

```bash
# from webcam - captures 20 face crops as you look at the camera
python -m src.main enroll --name alice --source 0 --count 20

# or from an existing folder of a person's photos - just call it once per photo
python -m src.main enroll --name bob --source bob_photo1.jpg
python -m src.main enroll --name bob --source bob_photo2.jpg
```

This populates `dataset/<name>/*.jpg`. You can also just manually drop
cropped or uncropped face photos into `dataset/<name>/` yourself — training
will auto-crop the largest face it finds in each image.

## 4. Train the recognizer

```bash
# Fast, offline, good for small galleries (a handful of people)
python -m src.main train --dataset dataset --method lbph --out models/gallery

# More accurate, needs models/arcface.onnx (see download_models.sh)
python -m src.main train --dataset dataset --method arcface --out models/gallery
```

## 5. Recognize faces

```bash
python -m src.main recognize --source photo.jpg --method lbph --model models/gallery

python -m src.main recognize --source 0 --method arcface --model models/gallery --detector dnn
```

Unrecognized faces are labeled `unknown`. Tune sensitivity with:
- LBPH: `--lbph-max-distance` (lower = stricter; default 70)
- ArcFace: edit `similarity_threshold` in `ArcFaceRecognizer.__init__` (higher = stricter; default 0.45)

## 6. (Optional) Get the higher-accuracy models

```bash
bash download_models.sh
```

This fetches:
- `models/deploy.prototxt` + `models/res10_300x300_ssd_iter_140000.caffemodel` — DNN face detector
- `models/arcface.onnx` — ArcFace embedding model (InsightFace project; check its license for commercial use)

## Project layout

```
face_recognition_app/
├── requirements.txt
├── download_models.sh
├── dataset/                 # your enrolled people go here: dataset/<name>/*.jpg
├── models/                  # trained recognizer + downloaded weights land here
└── src/
    ├── detector.py          # HaarCascadeDetector, DnnFaceDetector
    ├── recognizer.py        # LBPHRecognizer, ArcFaceRecognizer
    ├── utils.py              # drawing / cropping helpers
    └── main.py               # CLI: detect / enroll / train / recognize
```

## How it works

**Detection.** Haar cascades slide a window over the image at multiple scales
and check for edge/intensity patterns characteristic of faces — fast but can
miss non-frontal or poorly lit faces. The DNN detector instead runs a small
convolutional network (SSD + ResNet-10 backbone) that directly regresses face
bounding boxes and a confidence score, and is much more robust to real-world
conditions.

**Recognition.** LBPH encodes local texture patterns around each pixel into a
histogram per face region, then compares histograms — cheap to train, works
with very few images per person, but sensitive to pose/lighting changes.
ArcFace instead learns a deep embedding space where a specialized loss
(additive angular margin) pushes embeddings of the same person together and
different people apart, which is why comparing embeddings with cosine
similarity generalizes far better — this is the same family of technique
(deep metric learning) used by production face-ID systems.

## Extending this further

- **Face alignment**: detect eye landmarks and warp faces to a canonical pose
  before recognition — improves both LBPH and ArcFace accuracy meaningfully.
- **Anti-spoofing**: add a liveness check before trusting a recognition result
  for any access-control use case.
- **Batch/video optimization**: run detection every N frames and track boxes
  in between with `cv2.TrackerKCF` or similar for real-time video speedups.
- **Siamese network**: `recognizer.py` is structured so you can drop in a
  custom-trained Siamese/triplet-loss embedding model the same way ArcFace is
  plugged in — same `embed()` → cosine-similarity pattern.
