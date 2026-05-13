class Config:
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'

    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    MAX_LENGTH = 50

    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4

    D_MODEL = 256
    NHEAD = 8
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
