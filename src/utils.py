from collections import Counter


class Vocabulary:
    def __init__(self, specials):
        self.token_to_idx = {}
        self.idx_to_token = {}

        for token in specials:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def build_vocab(self, sentences, tokenizer):
        counter = Counter()

        for sentence in sentences:
            counter.update(tokenizer(sentence))

        for token in counter:
            self.add_token(token)

    def numericalize(self, tokens):
        return [self.token_to_idx.get(token, self.token_to_idx['<UNK>']) for token in tokens]

    def decode(self, indices):
        return [self.idx_to_token[idx] for idx in indices]

    def __len__(self):
        return len(self.token_to_idx)
