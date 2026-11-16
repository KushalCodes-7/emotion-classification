# =====================================
# TOKENIZATION CONFIGURATION
# =====================================

MAX_LENGTH = 48

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
UNK_ID = 1


# =====================================
# MODEL CONFIGURATION
# =====================================

VOCAB_SIZE = 15214

EMBEDDING_DIM = 128

HIDDEN_DIM = 128

NUM_LAYERS = 1

NUM_CLASSES = 6

DROPOUT = 0.3


# =====================================
# TRAINING CONFIGURATION
# =====================================

BATCH_SIZE = 32

LEARNING_RATE = 0.001

EPOCHS = 15