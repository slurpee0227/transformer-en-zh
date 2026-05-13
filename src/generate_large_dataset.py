import random
import os

subjects = [
    ('I', '我'),
    ('We', '我們'),
    ('They', '他們'),
    ('The engineer', '工程師'),
    ('The operator', '操作員'),
    ('The system', '系統'),
    ('The model', '模型'),
    ('The machine', '機器'),
    ('The robot', '機器人'),
    ('The server', '伺服器')
]

verbs = [
    ('checks', '檢查'),
    ('builds', '建立'),
    ('stores', '儲存'),
    ('processes', '處理'),
    ('reads', '讀取'),
    ('analyzes', '分析'),
    ('improves', '改善'),
    ('trains', '訓練'),
    ('translates', '翻譯'),
    ('monitors', '監控')
]

objects = [
    ('the data', '資料'),
    ('the model', '模型'),
    ('the production line', '產線'),
    ('the csv file', 'csv 檔案'),
    ('the barcode', '條碼'),
    ('the transformer network', 'Transformer 網路'),
    ('the training result', '訓練結果'),
    ('the translation output', '翻譯輸出'),
    ('the machine status', '機器狀態'),
    ('the attention map', 'attention 圖')
]

extra_phrases = [
    ('every day', '每天'),
    ('carefully', '仔細地'),
    ('very quickly', '非常快速地'),
    ('during training', '在訓練期間'),
    ('in the factory', '在工廠中'),
    ('with high accuracy', '以高準確率'),
    ('using artificial intelligence', '使用人工智慧'),
    ('for machine learning', '用於機器學習')
]


def generate_sentence():
    s_en, s_zh = random.choice(subjects)
    v_en, v_zh = random.choice(verbs)
    o_en, o_zh = random.choice(objects)
    e_en, e_zh = random.choice(extra_phrases)

    en = f'{s_en} {v_en} {o_en} {e_en}.'
    zh = f'{s_zh}{v_zh}{o_zh}{e_zh}。'

    return en, zh


def main():
    os.makedirs('data', exist_ok=True)

    train_en = []
    train_zh = []

    valid_en = []
    valid_zh = []

    for _ in range(8000):
        en, zh = generate_sentence()
        train_en.append(en)
        train_zh.append(zh)

    for _ in range(1000):
        en, zh = generate_sentence()
        valid_en.append(en)
        valid_zh.append(zh)

    with open('data/train.en', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_en))

    with open('data/train.zh', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_zh))

    with open('data/valid.en', 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_en))

    with open('data/valid.zh', 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_zh))

    print('Large synthetic dataset generated successfully.')


if __name__ == '__main__':
    main()
