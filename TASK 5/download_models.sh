#!/usr/bin/env bash
# Downloads optional pre-trained weights not bundled with OpenCV:
#   1) OpenCV DNN SSD face detector (Caffe) -> better than Haar cascades
#   2) An ArcFace-style ONNX embedding model -> for high-accuracy recognition
#
# Run this once, on a machine WITH internet access, before using
# `--method dnn` or `--method arcface`. The app works without this
# script too (it falls back to Haar cascade detection + LBPH recognition,
# both fully offline).

set -e
mkdir -p models

echo "Downloading DNN face detector (prototxt + caffemodel)..."
curl -L -o models/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -L -o models/res10_300x300_ssd_iter_140000.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

echo "Downloading ArcFace ONNX model..."
echo "NOTE: This model is distributed by the InsightFace project. Please review"
echo "its license before commercial use: https://github.com/deepinsight/insightface"
curl -L -o models/arcface.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx

echo "Done. Model files are in ./models/"
