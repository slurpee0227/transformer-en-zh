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

    # A small-but-stable Transformer setting for Colab GPU.
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1.0
    WARMUP_STEPS = 400
    LABEL_SMOOTHING = 0.1
    EARLY_STOPPING_PATIENCE = 6

    D_MODEL = 256
    NHEAD = 8
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
