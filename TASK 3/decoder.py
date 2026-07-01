"""DecoderRNN: LSTM decoder with visual (Bahdanau-style) attention.

Architecture follows "Show, Attend and Tell" (Xu et al., 2015): at every
timestep the decoder computes an attention-weighted context vector over the
CNN's spatial feature map, then uses it (with a learned gate) alongside the
previous word embedding to update the LSTM state and predict the next word.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (batch, num_pixels, encoder_dim)
        decoder_hidden: (batch, decoder_dim)
        """
        att1 = self.encoder_att(encoder_out)                      # (batch, num_pixels, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)       # (batch, 1, att_dim)
        att = self.full_att(torch.tanh(att1 + att2)).squeeze(2)    # (batch, num_pixels)
        alpha = F.softmax(att, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)    # (batch, encoder_dim)
        return context, alpha


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, decoder_dim=512,
                 encoder_dim=2048, attention_dim=256, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # gating scalar for the context vector
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, lengths):
        """
        encoder_out: (batch, num_pixels, encoder_dim)
        captions: (batch, max_len) token ids, includes <start> ... <end>
        lengths: (batch,) true caption lengths (including <start>/<end>)

        Teacher forcing: at step t, feed ground-truth token t and predict
        token t+1. Returns predictions for tokens[1:] (i.e. skips <start>).
        """
        batch_size = encoder_out.size(0)
        decode_lengths = (lengths - 1).tolist()  # no need to predict past <end>

        embeddings = self.embedding(captions)  # (batch, max_len, embed_dim)
        h, c = self.init_hidden_state(encoder_out)

        max_decode_len = max(decode_lengths)
        predictions = torch.zeros(
            batch_size, max_decode_len, self.vocab_size, device=encoder_out.device
        )

        for t in range(max_decode_len):
            batch_size_t = sum(l > t for l in decode_lengths)

            context, _ = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = torch.sigmoid(self.f_beta(h[:batch_size_t]))
            context = gate * context

            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], context], dim=1)
            h_t, c_t = self.lstm_cell(lstm_input, (h[:batch_size_t], c[:batch_size_t]))

            if batch_size_t < batch_size:
                h = torch.cat([h_t, h[batch_size_t:]], dim=0)
                c = torch.cat([c_t, c[batch_size_t:]], dim=0)
            else:
                h, c = h_t, c_t

            predictions[:batch_size_t, t, :] = self.fc(self.dropout(h_t))

        return predictions, decode_lengths

    @torch.no_grad()
    def generate_caption(self, encoder_out, vocab, max_len=25, beam_size=3, device="cpu"):
        """Beam search decoding for a single image (encoder_out has batch=1)."""
        k = beam_size
        vocab_size = self.vocab_size
        start_idx = vocab.word2idx[vocab.START_TOKEN]
        end_idx = vocab.word2idx[vocab.END_TOKEN]

        encoder_out = encoder_out.expand(k, -1, -1).contiguous()  # (k, num_pixels, encoder_dim)
        seqs = torch.full((k, 1), start_idx, dtype=torch.long, device=device)
        top_k_scores = torch.zeros(k, 1, device=device)

        h, c = self.init_hidden_state(encoder_out)
        complete_seqs, complete_scores = [], []
        current_k = k
        step = 1

        while True:
            embeddings = self.embedding(seqs[:, -1])
            context, _ = self.attention(encoder_out, h)
            gate = torch.sigmoid(self.f_beta(h))
            context = gate * context

            h, c = self.lstm_cell(torch.cat([embeddings, context], dim=1), (h, c))
            scores = F.log_softmax(self.fc(h), dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(current_k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(current_k, 0, True, True)

            prev_seq_inds = torch.div(top_k_words, vocab_size, rounding_mode="floor")
            next_word_inds = top_k_words % vocab_size

            seqs = torch.cat([seqs[prev_seq_inds], next_word_inds.unsqueeze(1)], dim=1)

            incomplete = [i for i, w in enumerate(next_word_inds.tolist()) if w != end_idx]
            complete = [i for i, w in enumerate(next_word_inds.tolist()) if w == end_idx]

            if complete:
                complete_seqs.extend(seqs[complete].tolist())
                complete_scores.extend(top_k_scores[complete].tolist())
            current_k -= len(complete)

            if current_k == 0 or step >= max_len:
                break

            seqs = seqs[incomplete]
            h = h[prev_seq_inds[incomplete]]
            c = c[prev_seq_inds[incomplete]]
            encoder_out = encoder_out[prev_seq_inds[incomplete]]
            top_k_scores = top_k_scores[incomplete].unsqueeze(1)
            step += 1

        if complete_scores:
            best = complete_scores.index(max(complete_scores))
            best_seq = complete_seqs[best]
        else:
            best_seq = seqs[0].tolist()

        return vocab.denumericalize(best_seq)
