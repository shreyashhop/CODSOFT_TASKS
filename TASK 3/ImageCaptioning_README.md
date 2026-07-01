# Image Captioning: CNN + RNN/Transformer

An image captioning system that combines a pre-trained CNN for visual feature
extraction with a sequence model — either an attention-based LSTM or a
Transformer decoder — to generate natural-language captions for images.

## How it works

**1. Encoder (`encoder.py`) — pre-trained CNN feature extractor**
`EncoderCNN` wraps a pre-trained **ResNet-50** or **VGG16** (ImageNet
weights), strips the final pooling/classification layers, and keeps the
convolutional feature maps. Instead of collapsing the image to one vector, it
outputs a **7×7 grid of feature vectors** (49 "regions"), so the decoder can
attend to different parts of the image at each word — e.g. looking at the
dog when generating "dog" and at the grass when generating "grass". The CNN
is frozen by default (pure feature extractor); pass `--fine_tune_encoder` to
unfreeze its last few layers for end-to-end fine-tuning.

**2. Decoder — turns visual features into a sentence**
Two interchangeable decoders are provided:

- **`decoder.py` — Attention LSTM** ("Show, Attend and Tell" style). At each
  timestep, a Bahdanau-style attention module computes a weighted context
  vector over the 49 image regions based on the current LSTM hidden state.
  That context (gated by a learned scalar) is combined with the previous
  word's embedding and fed into an `LSTMCell` to predict the next word.
  Supports **beam search** at inference time for higher-quality captions.

- **`transformer_decoder.py` — Transformer decoder**. The CNN feature grid is
  projected to `d_model` and used as "memory" for a standard
  `nn.TransformerDecoder` stack (masked self-attention + cross-attention over
  image features + positional encoding), generating the caption
  autoregressively — the same encoder-decoder pattern used in machine
  translation, applied here with a CNN in place of a text encoder.

Both share the same `EncoderCNN`, so you can swap decoders without touching
the vision side.

## Project structure

```
image_captioning/
├── vocabulary.py          # Tokenizer + word<->index mapping
├── dataset.py              # PyTorch Dataset + padding collate_fn
├── encoder.py               # Pre-trained CNN (ResNet50 / VGG16) feature extractor
├── decoder.py                # Attention LSTM decoder + beam search
├── transformer_decoder.py    # Transformer decoder + greedy decoding
├── train.py                   # Training loop
├── inference.py                # Caption generation for a new image
├── requirements.txt
└── data/annotations_example.json  # Expected annotation schema
```

## Setup

```bash
pip install -r requirements.txt
```

## Preparing data

Any dataset with images + human-written captions works (Flickr8k, Flickr30k,
MS-COCO, or your own). Convert the annotations into this JSON schema:

```json
[
  {"image": "img1.jpg", "captions": ["a dog running in a field", "a brown dog on grass"]},
  {"image": "img2.jpg", "captions": ["a plate of pasta with sauce"]}
]
```

See `data/annotations_example.json`. Place the actual image files in one
directory (`--images_dir`).

## Training

```bash
# Attention LSTM decoder with a ResNet50 encoder
python train.py \
  --annotations data/annotations.json \
  --images_dir data/images \
  --decoder_type lstm \
  --backbone resnet50 \
  --epochs 20 --batch_size 32

# Transformer decoder instead
python train.py \
  --annotations data/annotations.json \
  --images_dir data/images \
  --decoder_type transformer \
  --backbone resnet50 \
  --epochs 20
```

Key flags:
- `--backbone {resnet50,vgg16}` — CNN feature extractor
- `--decoder_type {lstm,transformer}` — sequence model
- `--fine_tune_encoder` — unfreeze the last CNN blocks for fine-tuning
- `--embed_dim`, `--hidden_dim` — model capacity

This saves a vocabulary file (`vocab.pkl`) and a checkpoint
(`captioning_model.pt`) after every epoch.

## Generating captions

```bash
python inference.py --image path/to/photo.jpg \
  --checkpoint captioning_model.pt --vocab vocab.pkl --beam_size 3
```

## Design notes / extension ideas

- **Why a spatial feature map instead of one pooled vector?** Attention over
  spatial regions consistently produces more accurate, more detailed
  captions than a single global-average-pooled feature, since the decoder
  can "look" at different objects as it generates each word.
- **Beam search vs. greedy decoding:** the LSTM decoder uses beam search
  (keeps the top-k partial sequences at each step) for better caption
  quality; the Transformer decoder here uses greedy decoding for simplicity
  — swapping in beam search there is a straightforward extension.
- **Evaluation:** for real training, track **BLEU, METEOR, CIDEr, or SPICE**
  scores against multiple reference captions per image rather than just loss.
- **Attention visualization:** the LSTM decoder's `Attention` module returns
  per-region weights (`alpha`) — these can be reshaped to 7×7 and overlaid
  on the image to visualize what the model is "looking at" per word.
- **Scaling up:** for production-quality results, consider larger backbones
  (EfficientNet, ViT) or CLIP image embeddings, and larger decoder capacity
  trained on MS-COCO (~120k images / ~600k captions).
