"""Vocabulary: builds a word<->index mapping from a corpus of captions."""

import re
import pickle
from collections import Counter


class Vocabulary:
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self._add_special_tokens()

    def _add_special_tokens(self):
        specials = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        for i, tok in enumerate(specials):
            self.word2idx[tok] = i
            self.idx2word[i] = tok

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9' ]", " ", text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        counter = Counter()
        for sentence in sentence_list:
            counter.update(self.tokenize(sentence))

        idx = len(self.word2idx)
        for word, freq in counter.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]

    def denumericalize(self, indices):
        words = []
        for idx in indices:
            word = self.idx2word.get(int(idx), self.UNK_TOKEN)
            if word == self.END_TOKEN:
                break
            if word not in (self.START_TOKEN, self.PAD_TOKEN):
                words.append(word)
        return " ".join(words)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
