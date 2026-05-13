# Transformer English-to-Chinese Translation

本專案實作 Attention Is All You Need 論文中的 Transformer Encoder-Decoder 架構，並自行訓練英翻中模型。

本專案未使用 HuggingFace、未使用任何 pretrained model，也不依賴線上 pretrained tokenizer。

## 專案目標

- 實作 Transformer Encoder-Decoder 英翻中模型
- 使用 PyTorch 自行訓練模型
- 支援訓練後輸出 `.pt` 模型檔
- 支援輸入英文句子並輸出中文翻譯

## 專案結構

```text
transformer-en-zh/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── train.en
│   ├── train.zh
│   ├── valid.en
│   └── valid.zh
├── models/
│   └── README.md
├── output/
│   └── README.md
└── src/
    ├── TransformerNN.py
    ├── config.py
    ├── dataset.py
    ├── inference.py
    ├── tokenizer.py
    ├── train.py
    └── utils.py
```

## 環境需求

建議使用：

- Python 3.10 或 3.11
- PyTorch
- jieba
- tqdm

安裝套件：

```bash
pip install -r requirements.txt
```

## 資料格式

請將英中文平行語料放在 `data/` 資料夾中：

```text
data/train.en
data/train.zh
data/valid.en
data/valid.zh
```

檔案格式為一行英文對應一行中文，例如：

`train.en`

```text
I love artificial intelligence.
How are you?
```

`train.zh`

```text
我喜歡人工智慧。
你好嗎？
```

## 訓練模型

```bash
python src/train.py
```

訓練完成後會輸出：

```text
models/transformer_zh_en.pt
```

此 `.pt` 檔即為作業要求的訓練後模型輸出檔。

## 推論翻譯

使用互動模式：

```bash
python src/inference.py
```

也可以直接輸入句子：

```bash
python src/inference.py --sentence "I love artificial intelligence."
```

輸出範例：

```text
Input English: I love artificial intelligence.
Output Chinese: 我 喜歡 人工智慧 。
```

## 模型架構

本專案實作 Transformer Encoder-Decoder 架構，包含：

- Token Embedding
- Sinusoidal Positional Encoding
- Multi-Head Attention
- Transformer Encoder
- Transformer Decoder
- Causal Mask
- Padding Mask
- Feed Forward Network
- Linear Output Projection

## 訓練方式

訓練時採用 teacher forcing：

- Decoder input 使用 `tgt[:, :-1]`
- Label 使用 `tgt[:, 1:]`

Loss function 使用：

```python
nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

可忽略 `<PAD>` token，避免 padding 影響訓練。

## 注意事項

本專案不得加入以下程式碼：

```python
from transformers import ...
```

也不得載入任何 pretrained model。

## Google Colab 使用建議

若在 Google Colab 執行，建議開啟 GPU：

```text
Runtime → Change runtime type → GPU
```

接著執行：

```bash
pip install -r requirements.txt
python src/train.py
python src/inference.py --sentence "I like machine learning."
```

## 繳交建議

請將整個資料夾壓縮成指定格式，例如：

```text
N98123456_王小明.zip
```

zip 內需包含：

- README.md
- requirements.txt
- 所有 `.py` 程式碼
- 訓練完成的 `.pt` 模型檔
