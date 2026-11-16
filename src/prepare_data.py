import json
import os

import pandas as pd

from src.tokenization import (
    build_vocabulary,
    encode_text,
    pad_or_truncate
)

from src.config import MAX_LENGTH


# =====================================
# 1. LOAD DATASETS
# =====================================

train_df = pd.read_csv("data/processed/train.csv")
validation_df = pd.read_csv("data/processed/validation.csv")
test_df = pd.read_csv("data/processed/test.csv")


# =====================================
# 2. BUILD VOCABULARY
# =====================================

vocabulary = build_vocabulary(train_df["text"])

print("Vocabulary size:", len(vocabulary))

# Save vocabulary
os.makedirs("artifacts", exist_ok=True)

with open("artifacts/vocabulary.json", "w", encoding="utf-8") as file:
    json.dump(vocabulary, file, ensure_ascii=False, indent=4)

print("Vocabulary saved successfully!")
# =====================================
# 3. CONVERT TEXT INTO SEQUENCES
# =====================================

def prepare_sequences(texts, vocabulary, max_length):

    sequences = []

    for text in texts:

        encoded = encode_text(text, vocabulary)

        padded = pad_or_truncate(
            encoded,
            max_length
        )

        sequences.append(padded)

    return sequences


# =====================================
# 4. PREPARE ALL DATASETS
# =====================================

X_train = prepare_sequences(
    train_df["text"],
    vocabulary,
    MAX_LENGTH
)

X_validation = prepare_sequences(
    validation_df["text"],
    vocabulary,
    MAX_LENGTH
)

X_test = prepare_sequences(
    test_df["text"],
    vocabulary,
    MAX_LENGTH
)


# =====================================
# 5. EXTRACT LABELS
# =====================================

y_train = train_df["label"].tolist()
y_validation = validation_df["label"].tolist()
y_test = test_df["label"].tolist()


# =====================================
# 6. VERIFY DATA
# =====================================

print("\n========== PREPARED DATA SUMMARY ==========")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_validation))
print("Testing samples:", len(X_test))

print("\nSequence length:", len(X_train[0]))

print("\nFirst encoded sequence:")
print(X_train[0])

print("\nFirst corresponding label:")
print(y_train[0])