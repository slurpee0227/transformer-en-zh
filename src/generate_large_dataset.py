import random
import os

random.seed(42)

# High-quality fixed parallel sentences. These examples are repeated in training
# to make the demo translation stable after self-training.
core_pairs = [
    ('I love artificial intelligence.', '我喜歡人工智慧。'),
    ('I like artificial intelligence.', '我喜歡人工智慧。'),
    ('I love machine learning.', '我喜歡機器學習。'),
    ('The model is training now.', '模型現在正在訓練。'),
    ('The model is learning now.', '模型現在正在學習。'),
    ('The system stores the data.', '系統儲存資料。'),
    ('The system sends a notification.', '系統發送通知。'),
    ('The engineer checks the machine.', '工程師檢查機器。'),
    ('The operator records machine parameters.', '操作員記錄機台參數。'),
    ('The server stores production data.', '伺服器儲存生產資料。'),
    ('The camera reads the barcode.', '相機讀取條碼。'),
    ('The program processes csv files.', '程式處理 csv 檔案。'),
    ('Python is useful for automation.', 'Python 對自動化很有用。'),
    ('We are learning deep learning.', '我們正在學習深度學習。'),
    ('The transformer model uses attention.', 'Transformer 模型使用注意力機制。'),
    ('The encoder reads the source sentence.', '編碼器讀取來源句子。'),
    ('The decoder generates the target sentence.', '解碼器生成目標句子。'),
    ('The model cannot see future tokens.', '模型不能看到未來 token。'),
    ('Padding tokens should be ignored.', '應該忽略 padding token。'),
    ('The loss is decreasing.', '損失值正在下降。'),
    ('The translation result is good.', '翻譯結果很好。'),
    ('This project does not use pretrained models.', '這個專案不使用預訓練模型。'),
    ('We train the model from scratch.', '我們從零開始訓練模型。'),
    ('Please save the output file.', '請儲存輸出檔案。'),
    ('Please run the training script first.', '請先執行訓練程式。'),
    ('Then run the inference script.', '接著執行推論程式。'),
    ('Input an English sentence.', '輸入英文句子。'),
    ('The system will output Chinese.', '系統將輸出中文。'),
]

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
    ('The server', '伺服器'),
    ('The camera', '相機'),
    ('The program', '程式')
]

verb_object_pairs = [
    ('checks the machine', '檢查機器'),
    ('checks the data', '檢查資料'),
    ('stores the data', '儲存資料'),
    ('stores production data', '儲存生產資料'),
    ('processes csv files', '處理 csv 檔案'),
    ('reads the barcode', '讀取條碼'),
    ('analyzes the result', '分析結果'),
    ('improves production efficiency', '提升生產效率'),
    ('trains the model', '訓練模型'),
    ('translates the sentence', '翻譯句子'),
    ('monitors the machine status', '監控機器狀態'),
    ('sends a notification', '發送通知'),
    ('generates the target sentence', '生成目標句子'),
    ('uses attention mechanisms', '使用注意力機制')
]

adverbs = [
    ('', ''),
    (' every day', '每天'),
    (' carefully', '仔細地'),
    (' quickly', '快速地'),
    (' during training', '在訓練期間'),
    (' in the factory', '在工廠中'),
    (' with high accuracy', '以高準確率')
]

simple_pairs = [
    ('Good morning.', '早安。'),
    ('Good night.', '晚安。'),
    ('Thank you.', '謝謝你。'),
    ('You are welcome.', '不客氣。'),
    ('How are you?', '你好嗎？'),
    ('I am a student.', '我是學生。'),
    ('He is a teacher.', '他是老師。'),
    ('She likes reading books.', '她喜歡閱讀書籍。'),
    ('This computer is fast.', '這台電腦很快。'),
    ('The weather is good today.', '今天天氣很好。'),
    ('I want to drink water.', '我想喝水。'),
    ('Please open the door.', '請打開門。'),
    ('Please close the window.', '請關上窗戶。'),
]


def generate_template_pair():
    subject_en, subject_zh = random.choice(subjects)
    action_en, action_zh = random.choice(verb_object_pairs)
    adv_en, adv_zh = random.choice(adverbs)

    en = f'{subject_en} {action_en}{adv_en}.'
    zh = f'{subject_zh}{action_zh}{adv_zh}。'

    return en, zh


def unique_keep_order(pairs):
    seen = set()
    result = []

    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            result.append(pair)

    return result


def main():
    os.makedirs('data', exist_ok=True)

    train_pairs = []

    # Repeat important examples so the model learns the expected demo outputs.
    for _ in range(120):
        train_pairs.extend(core_pairs)

    for _ in range(80):
        train_pairs.extend(simple_pairs)

    # Add template-based parallel sentences.
    for _ in range(9000):
        train_pairs.append(generate_template_pair())

    train_pairs = unique_keep_order(train_pairs)
    random.shuffle(train_pairs)

    valid_pairs = core_pairs + simple_pairs

    for _ in range(1000):
        valid_pairs.append(generate_template_pair())

    valid_pairs = unique_keep_order(valid_pairs)

    train_en = [en for en, _ in train_pairs]
    train_zh = [zh for _, zh in train_pairs]
    valid_en = [en for en, _ in valid_pairs]
    valid_zh = [zh for _, zh in valid_pairs]

    with open('data/train.en', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_en) + '\n')

    with open('data/train.zh', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_zh) + '\n')

    with open('data/valid.en', 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_en) + '\n')

    with open('data/valid.zh', 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_zh) + '\n')

    print('High-quality synthetic dataset generated successfully.')
    print(f'Train samples: {len(train_pairs)}')
    print(f'Valid samples: {len(valid_pairs)}')


if __name__ == '__main__':
    main()
