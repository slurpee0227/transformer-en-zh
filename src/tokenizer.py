import jieba


def tokenize_en(text):
    return text.lower().strip().split()


def tokenize_zh(text):
    return list(jieba.cut(text.strip()))
