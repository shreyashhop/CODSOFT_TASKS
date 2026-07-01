"""Train the image captioning model.

Example:
    python train.py --annotations data/annotations.json --images_dir data/images \
        --decoder_type lstm --epochs 20
"""

import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from vocabulary import Vocabulary
from dataset import ImageCaptionDataset, CaptionCollate
from encoder import EncoderCNN
from decoder import DecoderRNN
from transformer_decoder import DecoderTransformer


def build_vocab(annotations_path, freq_threshold=5):
    with open(annotations_path) as f:
        data = json.load(f)
    all_captions = [c for item in data for c in item["captions"]]
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocabulary(all_captions)
    return vocab


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    vocab = build_vocab(args.annotations, args.freq_threshold)
    vocab.save(args.vocab_out)
    print(f"Vocabulary size: {len(vocab)} (saved to {args.vocab_out})")

    dataset = ImageCaptionDataset(args.annotations, args.images_dir, vocab, transform)
    pad_idx = vocab.word2idx[vocab.PAD_TOKEN]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=CaptionCollate(pad_idx), num_workers=args.num_workers,
    )

    encoder = EncoderCNN(backbone=args.backbone, fine_tune=args.fine_tune_encoder).to(device)

    if args.decoder_type == "lstm":
        decoder = DecoderRNN(
            vocab_size=len(vocab), embed_dim=args.embed_dim,
            decoder_dim=args.hidden_dim, encoder_dim=encoder.feature_dim,
        ).to(device)
    else:
        decoder = DecoderTransformer(
            vocab_size=len(vocab), encoder_dim=encoder.feature_dim, d_model=args.hidden_dim,
        ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    params = list(decoder.parameters())
    if args.fine_tune_encoder:
        params += list(filter(lambda p: p.requires_grad, encoder.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(args.epochs):
        decoder.train()
        encoder.train(args.fine_tune_encoder)
        total_loss = 0.0

        for images, captions, lengths in loader:
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)

            optimizer.zero_grad()

            if args.decoder_type == "lstm":
                predictions, decode_lengths = decoder(features, captions, lengths.to(device))
                targets = captions[:, 1:][:, : predictions.size(1)]
                loss = criterion(
                    predictions.reshape(-1, predictions.size(-1)), targets.reshape(-1)
                )
            else:
                input_captions = captions[:, :-1]
                target_captions = captions[:, 1:]
                logits = decoder(features, input_captions, pad_idx)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), target_captions.reshape(-1)
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f}")

        torch.save({
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "decoder_type": args.decoder_type,
            "backbone": args.backbone,
        }, args.checkpoint_out)

    print(f"Training complete. Checkpoint saved to {args.checkpoint_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image captioning model")
    parser.add_argument("--annotations", required=True, help="Path to annotations JSON")
    parser.add_argument("--images_dir", required=True, help="Directory containing images")
    parser.add_argument("--vocab_out", default="vocab.pkl")
    parser.add_argument("--checkpoint_out", default="captioning_model.pt")
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "vgg16"])
    parser.add_argument("--decoder_type", default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--freq_threshold", type=int, default=5)
    parser.add_argument("--fine_tune_encoder", action="store_true")
    args = parser.parse_args()

    train(args)
