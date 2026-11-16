
from collections import Counter
import numpy as np
import re


# =====================================
# 1. TOKENIZATION
# =====================================

def tokenize(text):
    """
    Split text into words and punctuation tokens.
    """
    return re.findall(r"\w+|[^\w\s]", text)


# =====================================
# 2. BUILD VOCABULARY
# =====================================

def build_vocabulary(texts, min_frequency=1):

    word_counts = Counter()

    for text in texts:
        tokens = tokenize(text)
        word_counts.update(tokens)

    vocabulary = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, count in word_counts.items():

        if count >= min_frequency:
            vocabulary[word] = len(vocabulary)

    return vocabulary


# =====================================
# 3. ENCODE TEXT
# =====================================

def encode_text(text, vocabulary):
    """
    Convert a sentence into numerical token IDs.
    """

    tokens = tokenize(text)

    token_ids = [
        vocabulary.get(token, vocabulary["<UNK>"])
        for token in tokens
    ]

    return token_ids


# =====================================
# 4. PADDING AND TRUNCATION
# =====================================

def pad_or_truncate(sequence, max_length):

    if len(sequence) < max_length:

        sequence = sequence + [0] * (max_length - len(sequence))

    else:

        sequence = sequence[:max_length]

    return sequence


# =====================================
# 5. SEQUENCE LENGTH ANALYSIS
# =====================================

def analyze_sequence_lengths(texts, vocabulary):

    lengths = []

    for text in texts:

        encoded = encode_text(text, vocabulary)

        lengths.append(len(encoded))

    lengths = np.array(lengths)

    print("\n========== SEQUENCE LENGTH ANALYSIS ==========")

    print("Minimum:", lengths.min())
    print("Maximum:", lengths.max())
    print("Mean:", lengths.mean())
    print("Median:", np.median(lengths))

    print("90th percentile:", np.percentile(lengths, 90))
    print("95th percentile:", np.percentile(lengths, 95))
    print("99th percentile:", np.percentile(lengths, 99))

    return lengths


# =====================================
# 6. TESTING
# =====================================

if __name__ == "__main__":

    import pandas as pd

    print("Tokenization script started!")

    # Load training dataset
    train_df = pd.read_csv("data/processed/train.csv")

    print("Dataset loaded!")

    # Build vocabulary
    vocabulary = build_vocabulary(train_df["text"])

    print("\nVocabulary size:", len(vocabulary))

    print("\nFirst 20 vocabulary entries:")
    print(list(vocabulary.items())[:20])

    # Tokenization test
    sample = "i am feeling happy"

    tokens = tokenize(sample)

    print("\nOriginal sentence:", sample)
    print("Tokens:", tokens)

    token_ids = encode_text(sample, vocabulary)

    print("Token IDs:", token_ids)

    # Encoding test
    print("\n========== ENCODING TEST ==========")

    encoded = encode_text(sample, vocabulary)

    print("Original text:", sample)
    print("Encoded sequence:", encoded)
    print("Sequence length:", len(encoded))

    # Padding test
    print("\n========== PADDING TEST ==========")

    sample_sequence = [2, 31, 9, 661]

    padded_sequence = pad_or_truncate(
        sample_sequence,
        max_length=8
    )

    print("Original sequence:", sample_sequence)
    print("Padded sequence:", padded_sequence)
    print("Padded length:", len(padded_sequence))

    # Truncation test
    print("\n========== TRUNCATION TEST ==========")

    long_sequence = [2, 31, 9, 661, 45, 78, 91, 23, 56]

    truncated_sequence = pad_or_truncate(
        long_sequence,
        max_length=6
    )

    print("Original sequence:", long_sequence)
    print("Truncated sequence:", truncated_sequence)
    print("Truncated length:", len(truncated_sequence))

    # Sequence length analysis
    analyze_sequence_lengths(
        train_df["text"],
        vocabulary
    )