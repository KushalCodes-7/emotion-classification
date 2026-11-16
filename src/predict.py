import json
import torch
import torch.nn.functional as F

from src.model import EmotionBiGRU
from src.config import MAX_LENGTH
from src.tokenization import encode_text, pad_or_truncate


# 1. Emotion mapping
EMOTION_LABELS = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}


# 2. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 3. Load saved vocabulary
with open("artifacts/vocabulary.json", "r", encoding="utf-8") as file:
    vocabulary = json.load(file)


# 4. Load trained model
model = EmotionBiGRU().to(device)

checkpoint = torch.load(
    "artifacts/best_model.pth",
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()


# 5. Prediction function
def predict_emotion(text):

    sequence = encode_text(text, vocabulary)

    sequence = pad_or_truncate(sequence, MAX_LENGTH)

    input_tensor = torch.tensor(
        [sequence],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    emotion = EMOTION_LABELS[predicted_class.item()]

    print("\n========== PREDICTION ==========")
    print("Input:", text)
    print("Predicted Emotion:", emotion)
    print(f"Confidence: {confidence.item() * 100:.2f}%")

    print("\nEmotion Probabilities:")

    for index, probability in enumerate(probabilities[0]):

        print(
            f"{EMOTION_LABELS[index]}: "
            f"{probability.item() * 100:.2f}%"
        )


if __name__ == "__main__":

    text = input("Enter a sentence: ")

    predict_emotion(text)