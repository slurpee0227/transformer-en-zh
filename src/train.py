import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from tokenizer import tokenize_en, tokenize_zh
from utils import Vocabulary
from dataset import TranslationDataset, collate_fn
from TransformerNN import AttentionIsAllYouNeedTransformer


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


train_en = read_file('data/train.en')
train_zh = read_file('data/train.zh')

specials = [
    Config.PAD_TOKEN,
    Config.UNK_TOKEN,
    Config.BOS_TOKEN,
    Config.EOS_TOKEN
]

src_vocab = Vocabulary(specials)
tgt_vocab = Vocabulary(specials)

src_vocab.build_vocab(train_en, tokenize_en)
tgt_vocab.build_vocab(train_zh, tokenize_zh)


def encode_sentence(sentence, tokenizer, vocab):
    tokens = tokenizer(sentence)

    return [
        Config.BOS_IDX,
        *vocab.numericalize(tokens),
        Config.EOS_IDX
    ]


src_data = [
    encode_sentence(sentence, tokenize_en, src_vocab)
    for sentence in train_en
]

tgt_data = [
    encode_sentence(sentence, tokenize_zh, tgt_vocab)
    for sentence in train_zh
]


dataset = TranslationDataset(src_data, tgt_data)

loader = DataLoader(
    dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, Config.PAD_IDX)
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AttentionIsAllYouNeedTransformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=Config.D_MODEL,
    nhead=Config.NHEAD,
    num_encoder_layers=Config.NUM_ENCODER_LAYERS,
    num_decoder_layers=Config.NUM_DECODER_LAYERS,
    dim_feedforward=Config.DIM_FEEDFORWARD,
    dropout=Config.DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=Config.LEARNING_RATE
)


for epoch in range(Config.EPOCHS):
    model.train()

    total_loss = 0

    progress_bar = tqdm(loader)

    for src_batch, tgt_batch in progress_bar:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        src_padding_mask = (src_batch == Config.PAD_IDX)
        tgt_padding_mask = (tgt_input == Config.PAD_IDX)

        logits = model(
            src_batch,
            tgt_input,
            src_padding_mask,
            tgt_padding_mask
        )

        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            tgt_output.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_description(
            f'Epoch {epoch+1} Loss: {loss.item():.4f}'
        )

    avg_loss = total_loss / len(loader)

    print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')


os.makedirs('models', exist_ok=True)

save_data = {
    'model_state_dict': model.state_dict(),
    'src_vocab': src_vocab.token_to_idx,
    'tgt_vocab': tgt_vocab.token_to_idx
}


torch.save(save_data, 'models/transformer_zh_en.pt')

print('Model saved to models/transformer_zh_en.pt')
