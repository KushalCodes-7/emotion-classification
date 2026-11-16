from torch.utils.data import DataLoader

from src.dataset import EmotionDataset
from src.prepare_data import (
    X_train,
    X_validation,
    X_test,
    y_train,
    y_validation,
    y_test
)


# Create Dataset objects

train_dataset = EmotionDataset(X_train, y_train)

validation_dataset = EmotionDataset(
    X_validation,
    y_validation
)

test_dataset = EmotionDataset(X_test, y_test)


# Create DataLoaders

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Verify DataLoader

if __name__ == "__main__":

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Testing batches:", len(test_loader))

    sample_sequences, sample_labels = next(iter(train_loader))

    print("\nBatch sequence shape:", sample_sequences.shape)
    print("Batch label shape:", sample_labels.shape)

    print("\nSample sequence:")
    print(sample_sequences[0])

    print("\nSample label:")
    print(sample_labels[0])