from src.preprocessing import clean_text
import pandas as pd
import matplotlib.pyplot as plt
# =====================================
# 1. LOAD DATASETS
# =====================================

train_df = pd.read_csv("data/raw/training.csv")
validation_df = pd.read_csv("data/raw/validation.csv")
test_df = pd.read_csv("data/raw/test.csv")

# Apply text cleaning to all datasets
train_df["text"] = train_df["text"].apply(clean_text)

validation_df["text"] = validation_df["text"].apply(clean_text)

test_df["text"] = test_df["text"].apply(clean_text)

print("\n========== CLEANED TEXT SAMPLES ==========")

print(train_df["text"].head(5))

from src.preprocessing import clean_text

# Apply text cleaning
train_df["text"] = train_df["text"].apply(clean_text)
validation_df["text"] = validation_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)

print("\n========== CLEANED TEXT SAMPLES ==========")

print(train_df["text"].head(5))

# =====================================
# 2. DISPLAY FIRST FIVE RECORDS
# =====================================

print("\n========== TRAINING DATASET ==========")
print(train_df.head())

print("\n========== VALIDATION DATASET ==========")
print(validation_df.head())

print("\n========== TEST DATASET ==========")
print(test_df.head())


# =====================================
# 3. DATASET SHAPE
# =====================================

print("\n========== DATASET SHAPE ==========")

print("Training:", train_df.shape)
print("Validation:", validation_df.shape)
print("Testing:", test_df.shape)


# =====================================
# 4. COLUMN INFORMATION
# =====================================

print("\n========== COLUMN INFORMATION ==========")

print(train_df.columns.tolist())


# =====================================
# 5. DATA TYPES
# =====================================

print("\n========== DATASET INFORMATION ==========")

train_df.info()
validation_df.info()
test_df.info()


# =====================================
# 6. DATASET SIZE
# =====================================

train_samples = train_df.shape[0]
validation_samples = validation_df.shape[0]
test_samples = test_df.shape[0]

total_samples = train_samples + validation_samples + test_samples

print("\n========== DATASET SUMMARY ==========")

print("Training samples:", train_samples)
print("Validation samples:", validation_samples)
print("Testing samples:", test_samples)
print("Total samples:", total_samples)


# =====================================
# 7. MISSING VALUE ANALYSIS
# =====================================

print("\n========== MISSING VALUE ANALYSIS ==========")

print("\nTraining:")
print(train_df.isnull().sum())

print("\nValidation:")
print(validation_df.isnull().sum())

print("\nTesting:")
print(test_df.isnull().sum())

# =====================================
# 8. DUPLICATE RECORD ANALYSIS
# =====================================

print("\n========== DUPLICATE ANALYSIS ==========")

print("\nTraining Duplicates:")
print(train_df.duplicated().sum())

print("\nValidation Duplicates:")
print(validation_df.duplicated().sum())

print("\nTesting Duplicates:")
print(test_df.duplicated().sum())

# =====================================
# 9. EMOTION DISTRIBUTION ANALYSIS
# =====================================

emotion_labels = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

print("\n========== EMOTION DISTRIBUTION ==========")

for label, emotion in emotion_labels.items():

    train_count = (train_df["label"] == label).sum()
    validation_count = (validation_df["label"] == label).sum()
    test_count = (test_df["label"] == label).sum()

    print(f"\nEmotion: {emotion}")
    print(f"Training: {train_count}")
    print(f"Validation: {validation_count}")
    print(f"Testing: {test_count}")
    print(f"Total: {train_count + validation_count + test_count}")

import matplotlib.pyplot as plt

# =====================================
# 10. EMOTION DISTRIBUTION VISUALIZATION
# =====================================

emotion_counts = train_df["label"].value_counts().sort_index()

emotion_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

plt.figure(figsize=(10, 5))

plt.bar(emotion_names, emotion_counts.values)

plt.title("Emotion Distribution in Training Dataset")
plt.xlabel("Emotion")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.show()

# =====================================
# 11. TEXT LENGTH ANALYSIS
# =====================================

train_df["text_length"] = train_df["text"].apply(
    lambda x: len(x.split())
)

print("\n========== TEXT LENGTH ANALYSIS ==========")

print("Minimum length:", train_df["text_length"].min())

print("Maximum length:", train_df["text_length"].max())

print("Average length:", train_df["text_length"].mean())

print("Median length:", train_df["text_length"].median())

# =====================================
# 12. CROSS-SPLIT DUPLICATE ANALYSIS
# =====================================

train_texts = set(train_df["text"])
validation_texts = set(validation_df["text"])
test_texts = set(test_df["text"])

train_val_overlap = train_texts.intersection(validation_texts)
train_test_overlap = train_texts.intersection(test_texts)
val_test_overlap = validation_texts.intersection(test_texts)

print("\n========== CROSS-SPLIT DUPLICATE ANALYSIS ==========")

print("Train-Validation overlap:", len(train_val_overlap))
print("Train-Test overlap:", len(train_test_overlap))
print("Validation-Test overlap:", len(val_test_overlap))

import re


def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove leading and trailing spaces
    text = text.strip()

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text

import os

# Create processed directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Save cleaned datasets
train_df.to_csv("data/processed/train.csv", index=False)
validation_df.to_csv("data/processed/validation.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("\nCleaned datasets saved successfully!")