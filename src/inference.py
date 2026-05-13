import argparse
import torch

from config import Config
from tokenizer import tokenize_en
from TransformerNN import AttentionIsAllYouNeedTransformer


parser = argparse.ArgumentParser()
parser.add_argument('--sentence', type=str, default=None)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('models/transformer_zh_en.pt', map_location=device)

src_vocab = checkpoint['src_vocab']
tgt_vocab = checkpoint['tgt_vocab']

idx_to_token = {v: k for k, v in tgt_vocab.items()}

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

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def encode_sentence(sentence):
    tokens = tokenize_en(sentence)

    token_ids = [
        src_vocab.get(token, Config.UNK_IDX)
        for token in tokens
    ]

    return [Config.BOS_IDX] + token_ids + [Config.EOS_IDX]


def greedy_decode(src_tensor, max_length=50):
    generated = [Config.BOS_IDX]

    for _ in range(max_length):
        tgt_tensor = torch.tensor([generated]).to(device)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)

        next_token = output[:, -1].argmax(-1).item()

        generated.append(next_token)

        if next_token == Config.EOS_IDX:
            break

    return generated


if args.sentence:
    sentence = args.sentence
else:
    sentence = input('Input English: ')

src_ids = encode_sentence(sentence)

src_tensor = torch.tensor([src_ids]).to(device)

output_ids = greedy_decode(src_tensor)

output_tokens = []

for idx in output_ids:
    token = idx_to_token.get(idx, '<UNK>')

    if token in ['<PAD>', '<BOS>', '<EOS>']:
        continue

    output_tokens.append(token)

print('Output Chinese:', ' '.join(output_tokens))
