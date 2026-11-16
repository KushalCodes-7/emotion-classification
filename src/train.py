import os
import torch
import torch.nn as nn

from src.model import EmotionBiGRU
from src.dataloader import train_loader, validation_loader

from src.config import LEARNING_RATE, EPOCHS


# =====================================
# 1. DEVICE CONFIGURATION
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================
# 2. INITIALIZE MODEL
# =====================================

model = EmotionBiGRU().to(device)


# =====================================
# 3. LOSS FUNCTION
# =====================================

criterion = nn.CrossEntropyLoss()


# =====================================
# 4. OPTIMIZER
# =====================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

os.makedirs("artifacts", exist_ok=True)

best_val_loss = float("inf")
patience = 3
counter = 0

# =====================================
# 5. TRAINING FUNCTION
# =====================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in loader:

        sequences = sequences.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequences)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()

        # Calculate predictions
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


# =====================================
# 6. VALIDATION FUNCTION
# =====================================

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for sequences, labels in loader:

            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


# =====================================
# 7. TRAINING LOOP
# =====================================

# =====================================
# 7. TRAINING LOOP WITH EARLY STOPPING
# =====================================

import os

if __name__ == "__main__":

    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)

    # Early stopping configuration
    best_val_loss = float("inf")
    patience = 3
    counter = 0

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device
        )

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            counter = 0

            torch.save(
                model.state_dict(),
                "artifacts/best_model.pth"
            )

            print("Best model saved!")

        else:

            counter += 1

            print(f"No improvement: {counter}/{patience}")

        # Early stopping
        if counter >= patience:

            print("\nEarly stopping triggered!")
            print("Training stopped.")

            break

    print("\nTraining completed!")
    print("Best validation loss:", best_val_loss)