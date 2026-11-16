import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.model import EmotionBiGRU
from src.dataloader import test_loader


# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 2. Load the model
model = EmotionBiGRU().to(device)

checkpoint = torch.load(
    "artifacts/best_model.pth",
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()


# 3. Define loss function
criterion = nn.CrossEntropyLoss()

total_loss = 0
all_predictions = []
all_labels = []


# 4. Evaluate model
with torch.no_grad():

    for sequences, labels in test_loader:

        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs = model(sequences)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# 5. Calculate metrics
test_loss = total_loss / len(test_loader)

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)


# 6. Print results
emotion_labels = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

print("\n========== TEST RESULTS ==========")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3, 4, 5],
        target_names=emotion_labels,
        zero_division=0
    )
)

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3, 4, 5]
    )
)