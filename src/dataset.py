import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        return self.src_sentences[idx], self.tgt_sentences[idx]


def collate_fn(batch, pad_idx):
    src_batch = []
    tgt_batch = []

    for src, tgt in batch:
        src_batch.append(torch.tensor(src))
        tgt_batch.append(torch.tensor(tgt))

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_idx
    )

    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=pad_idx
    )

    return src_batch, tgt_batch
