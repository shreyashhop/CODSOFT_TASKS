"""Generate a caption for a single image using a trained checkpoint.

Example:
    python inference.py --image path/to/photo.jpg \
        --checkpoint captioning_model.pt --vocab vocab.pkl
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms

from vocabulary import Vocabulary
from encoder import EncoderCNN
from decoder import DecoderRNN
from transformer_decoder import DecoderTransformer


def load_model(checkpoint_path, vocab_path, device):
    vocab = Vocabulary.load(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = EncoderCNN(backbone=checkpoint["backbone"], pretrained=False).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    encoder.eval()

    if checkpoint["decoder_type"] == "lstm":
        decoder = DecoderRNN(vocab_size=len(vocab), encoder_dim=encoder.feature_dim).to(device)
    else:
        decoder = DecoderTransformer(vocab_size=len(vocab), encoder_dim=encoder.feature_dim).to(device)
    decoder.load_state_dict(checkpoint["decoder_state"])
    decoder.eval()

    return encoder, decoder, vocab, checkpoint["decoder_type"]


def caption_image(image_path, checkpoint_path, vocab_path, beam_size=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, vocab, decoder_type = load_model(checkpoint_path, vocab_path, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)
        if decoder_type == "lstm":
            caption = decoder.generate_caption(features, vocab, beam_size=beam_size, device=device)
        else:
            caption = decoder.generate_caption(features, vocab, device=device)

    return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a caption for an image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", default="captioning_model.pt")
    parser.add_argument("--vocab", default="vocab.pkl")
    parser.add_argument("--beam_size", type=int, default=3)
    args = parser.parse_args()

    result = caption_image(args.image, args.checkpoint, args.vocab, args.beam_size)
    print(f"Generated caption: {result}")
