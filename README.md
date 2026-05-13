# Transformer English-to-Chinese Translation

本專案實作 **Attention Is All You Need** 論文中的 Transformer Encoder-Decoder 架構，並自行訓練英翻中模型。

本專案符合以下限制：

- 不使用 HuggingFace
- 不使用任何 pretrained model
- 不使用任何線上 pretrained tokenizer
- 模型權重由本專案資料自行訓練產生

---

## 1. 專案目標

- 實作 Transformer Encoder-Decoder 英翻中模型
- 使用 PyTorch 自行訓練模型
- 支援英翻中推論
- 訓練完成後輸出 `.pt` 模型檔
- 提供 Google Colab 訓練流程
- 提供 BLEU Score、Loss Curve、TensorBoard、Checkpoint、Early Stopping 等加分功能

---

## 2. 專案結構

```text
transformer-en-zh/
├── README.md
├── requirements.txt
├── .gitignore
├── colab/
│   └── Transformer_EN_ZH_Colab.ipynb
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
├── checkpoints/
│   └── epoch_x.pt              # 訓練後自動產生
├── runs/
│   └── transformer_experiment  # TensorBoard 記錄，訓練後自動產生
└── src/
    ├── TransformerNN.py
    ├── attention_visualization.py
    ├── beam_search.py
    ├── bleu_score.py
    ├── config.py
    ├── dataset.py
    ├── inference.py
    ├── plot_loss.py
    ├── tokenizer.py
    ├── train.py
    └── utils.py
```

---

## 3. 環境需求

建議使用：

- Python 3.10 或 3.11
- PyTorch
- Google Colab GPU 或本機 CUDA GPU

安裝套件：

```bash
pip install -r requirements.txt
```

`requirements.txt` 主要包含：

```text
torch
numpy
pandas
jieba
tqdm
matplotlib
sacrebleu
tensorboard
```

---

## 4. 資料集格式

資料集已放在 `data/` 目錄中：

```text
data/train.en
data/train.zh
data/valid.en
data/valid.zh
```

格式為一行英文對應一行中文。

範例：

`train.en`

```text
I love artificial intelligence.
The model is training now.
```

`train.zh`

```text
我喜歡人工智慧。
模型現在正在訓練。
```

---

## 5. 模型架構

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

核心模型檔案：

```text
src/TransformerNN.py
```

---

## 6. 訓練方式

執行：

```bash
python src/train.py
```

訓練流程包含：

- Training Loop
- Validation Loop
- Teacher Forcing
- CrossEntropyLoss
- Padding Mask
- Causal Mask
- Gradient Clipping
- Learning Rate Scheduler
- Early Stopping
- Best Model Save
- Checkpoint Save
- TensorBoard Logging
- Loss Curve Output

訓練時採用 teacher forcing：

```python
tgt_input = tgt_batch[:, :-1]
tgt_output = tgt_batch[:, 1:]
```

Loss function：

```python
nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

可忽略 `<PAD>` token，避免 padding 影響訓練。

---

## 7. 訓練輸出

訓練完成後會產生：

```text
models/transformer_zh_en.pt
```

此檔案為作業要求的訓練後模型輸出檔。

同時會產生：

```text
checkpoints/epoch_1.pt
checkpoints/epoch_2.pt
...
output/loss_curve.png
runs/transformer_experiment/
```

說明：

- `models/transformer_zh_en.pt`：validation loss 最佳模型
- `checkpoints/epoch_x.pt`：每個 epoch 的 checkpoint
- `output/loss_curve.png`：train loss 與 valid loss 曲線
- `runs/transformer_experiment/`：TensorBoard 記錄

---

## 8. 推論翻譯

互動模式：

```bash
python src/inference.py
```

指定句子：

```bash
python src/inference.py --sentence "I love artificial intelligence."
```

輸出範例：

```text
Output Chinese: 我 喜歡 人工智慧 。
```

---

## 9. Google Colab 使用方式

Colab notebook 位於：

```text
colab/Transformer_EN_ZH_Colab.ipynb
```

建議在 Colab 中設定 GPU：

```text
Runtime → Change runtime type → GPU
```

Notebook 已包含：

1. GPU 檢查
2. Clone GitHub Repo
3. 安裝 requirements
4. 檢查資料集
5. 執行訓練
6. 檢查 `.pt` 模型
7. 推論測試
8. 顯示 loss curve
9. 開啟 TensorBoard
10. 下載模型與輸出檔案

---

## 10. 加分功能

### 10.1 BLEU Score

檔案：

```text
src/bleu_score.py
```

用於計算翻譯品質。

### 10.2 Loss Curve

檔案：

```text
src/plot_loss.py
```

訓練後輸出：

```text
output/loss_curve.png
```

### 10.3 Beam Search

檔案：

```text
src/beam_search.py
```

提供比 greedy decoding 更進階的翻譯生成方式。

### 10.4 Attention Visualization

檔案：

```text
src/attention_visualization.py
```

可輸出 attention heatmap。

### 10.5 TensorBoard

執行：

```bash
tensorboard --logdir=runs
```

可查看 train loss 與 valid loss。

### 10.6 Early Stopping 與 Best Model Save

當 validation loss 連續多次沒有改善時會自動停止訓練，並保留 validation loss 最佳模型。

---

## 11. 禁止事項

本專案不得加入以下程式碼：

```python
from transformers import ...
```

也不得載入任何 HuggingFace 或其他網站的 pretrained model。

---

## 12. 繳交方式

請將整個資料夾壓縮成指定格式，例如：

```text
N98123456_王小明.zip
```

zip 內需包含：

- README.md
- requirements.txt
- 所有 `.py` 程式碼
- data 資料集
- Colab notebook
- 訓練完成的 `.pt` 模型檔
- output/loss_curve.png

注意：若 GitHub `.gitignore` 排除了 `.pt`，請務必在壓縮繳交前手動確認 zip 中有包含：

```text
models/transformer_zh_en.pt
```
