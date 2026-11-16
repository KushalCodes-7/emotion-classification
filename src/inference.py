import json
import torch
import torch.nn.functional as F

from src.model import EmotionBiGRU
from src.config import MAX_LENGTH
from src.tokenization import encode_text, pad_or_truncate


# Emotion mapping
EMOTION_LABELS = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load vocabulary
with open("artifacts/vocabulary.json", "r", encoding="utf-8") as file:
    vocabulary = json.load(file)


# Load trained model
model = EmotionBiGRU().to(device)

checkpoint = torch.load(
    "artifacts/best_model.pth",
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()


def predict_emotion(text):

    # Convert text into numerical sequence
    sequence = encode_text(text, vocabulary)

    # Pad or truncate sequence
    sequence = pad_or_truncate(sequence, MAX_LENGTH)

    # Convert into PyTorch tensor
    input_tensor = torch.tensor(
        [sequence],
        dtype=torch.long
    ).to(device)

    # Predict
    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_emotion = EMOTION_LABELS[predicted_class.item()]

    emotion_probabilities = {
        EMOTION_LABELS[index]: round(probability.item(), 4)
        for index, probability in enumerate(probabilities[0])
    }

    return {
        "text": text,
        "predicted_emotion": predicted_emotion,
        "confidence": round(confidence.item(), 4),
        "probabilities": emotion_probabilities
    }