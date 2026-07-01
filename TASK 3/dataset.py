"""Dataset for image-caption pairs.

Expects an annotations JSON file shaped like:
[
    {"image": "img1.jpg", "captions": ["a dog running", "a brown dog on grass"]},
    {"image": "img2.jpg", "captions": ["a plate of pasta"]},
    ...
]
(This is easy to produce from Flickr8k/Flickr30k or MS-COCO annotation files.)
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageCaptionDataset(Dataset):
    def __init__(self, annotations_path, images_dir, vocabulary, transform=None):
        with open(annotations_path, "r") as f:
            data = json.load(f)

        self.images_dir = images_dir
        self.vocab = vocabulary
        self.transform = transform

        # One training sample per (image, caption) pair.
        self.samples = []
        for item in data:
            for caption in item["captions"]:
                self.samples.append((item["image"], caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_name, caption = self.samples[index]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        numeric_caption = [self.vocab.word2idx[self.vocab.START_TOKEN]]
        numeric_caption += self.vocab.numericalize(caption)
        numeric_caption.append(self.vocab.word2idx[self.vocab.END_TOKEN])

        return image, torch.tensor(numeric_caption, dtype=torch.long)


class CaptionCollate:
    """Pads variable-length captions within a batch to the same length."""

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        captions = [item[1] for item in batch]

        lengths = [len(c) for c in captions]
        max_len = max(lengths)

        padded = torch.full((len(captions), max_len), self.pad_idx, dtype=torch.long)
        for i, c in enumerate(captions):
            padded[i, : len(c)] = c

        return images, padded, torch.tensor(lengths, dtype=torch.long)
