"""DecoderTransformer: Transformer-based caption decoder.

Treats the CNN's spatial feature map as "memory" (like the encoder output in
a standard seq2seq Transformer) and autoregressively generates the caption
with a stack of TransformerDecoder layers using masked self-attention plus
cross-attention over the image features.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, encoder_dim=2048, d_model=512,
                 nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1, max_len=50):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.feature_proj = nn.Linear(encoder_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, encoder_out, captions, pad_idx):
        """
        encoder_out: (batch, num_pixels, encoder_dim)
        captions: (batch, seq_len) input tokens, e.g. <start> ... token_{n-1}
        """
        memory = self.feature_proj(encoder_out)  # (batch, num_pixels, d_model)

        tgt = self.embedding(captions) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)

        tgt_mask = self._causal_mask(captions.size(1), captions.device)
        tgt_key_padding_mask = captions == pad_idx

        out = self.transformer_decoder(
            tgt=tgt, memory=memory,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc(out)

    @torch.no_grad()
    def generate_caption(self, encoder_out, vocab, device="cpu"):
        """Greedy autoregressive decoding for a single image (batch=1)."""
        memory = self.feature_proj(encoder_out)
        start_idx = vocab.word2idx[vocab.START_TOKEN]
        end_idx = vocab.word2idx[vocab.END_TOKEN]

        generated = torch.tensor([[start_idx]], device=device)
        for _ in range(self.max_len):
            tgt = self.embedding(generated) * math.sqrt(self.d_model)
            tgt = self.pos_encoding(tgt)
            tgt_mask = self._causal_mask(generated.size(1), device)

            out = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            next_logits = self.fc(out[:, -1, :])
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == end_idx:
                break

        return vocab.denumericalize(generated.squeeze(0).tolist())
