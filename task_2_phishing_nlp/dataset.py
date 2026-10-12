import re
import torch
from torch.utils.data import Dataset, DataLoader

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <URL> ", text)
    text = re.sub(r"[^a-z0-9\s<>]", " ", text)
    return text.split()

class Vocabulary:
    def __init__(self, max_tokens=3000):
        self.max_tokens = max_tokens
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<URL>": 2}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<URL>"}

    def fit(self, texts):
        counts = {}
        for text in texts:
            for token in clean_and_tokenize(text):
                if token not in self.word2idx:
                    counts[token] = counts.get(token, 0) + 1
        sorted_tokens = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        limit = self.max_tokens - len(self.word2idx)
        for token, _ in sorted_tokens[:limit]:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def encode(self, text, max_length=100):
        tokens = clean_and_tokenize(text)
        indices = [self.word2idx.get(tok, 1) for tok in tokens][:max_length]
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)
