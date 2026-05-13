import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from tokenizer import tokenize_en, tokenize_zh
from utils import Vocabulary
from dataset import TranslationDataset, collate_fn
from TransformerNN import AttentionIsAllYouNeedTransformer
from plot_loss import save_loss_curve


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


train_en = read_file('data/train.en')
train_zh = read_file('data/train.zh')

valid_en = read_file('data/valid.en')
valid_zh = read_file('data/valid.zh')

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


train_src = [
    encode_sentence(sentence, tokenize_en, src_vocab)
    for sentence in train_en
]

train_tgt = [
    encode_sentence(sentence, tokenize_zh, tgt_vocab)
    for sentence in train_zh
]

valid_src = [
    encode_sentence(sentence, tokenize_en, src_vocab)
    for sentence in valid_en
]

valid_tgt = [
    encode_sentence(sentence, tokenize_zh, tgt_vocab)
    for sentence in valid_zh
]


train_dataset = TranslationDataset(train_src, train_tgt)
valid_dataset = TranslationDataset(valid_src, valid_tgt)

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, Config.PAD_IDX)
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=2,
    factor=0.5
)

writer = SummaryWriter('runs/transformer_experiment')

best_valid_loss = float('inf')
early_stop_counter = 0

train_losses = []
valid_losses = []

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)

for epoch in range(Config.EPOCHS):
    model.train()

    train_loss = 0

    progress_bar = tqdm(train_loader)

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()

        progress_bar.set_description(
            f'Epoch {epoch+1} Train Loss: {loss.item():.4f}'
        )

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()

    valid_loss = 0

    with torch.no_grad():
        for src_batch, tgt_batch in valid_loader:
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

            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)

    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Valid', avg_valid_loss, epoch)

    scheduler.step(avg_valid_loss)

    print(f'Epoch {epoch+1}')
    print(f'Train Loss: {avg_train_loss:.4f}')
    print(f'Valid Loss: {avg_valid_loss:.4f}')

    checkpoint_path = f'checkpoints/epoch_{epoch+1}.pt'

    torch.save(model.state_dict(), checkpoint_path)

    print(f'Checkpoint saved: {checkpoint_path}')

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        early_stop_counter = 0

        save_data = {
            'model_state_dict': model.state_dict(),
            'src_vocab': src_vocab.token_to_idx,
            'tgt_vocab': tgt_vocab.token_to_idx
        }

        torch.save(save_data, 'models/transformer_zh_en.pt')

        print('Best model updated.')

    else:
        early_stop_counter += 1

    if early_stop_counter >= 3:
        print('Early stopping triggered.')
        break

save_loss_curve(train_losses, valid_losses)

writer.close()

print('Training completed.')
