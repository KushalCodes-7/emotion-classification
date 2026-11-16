# Emotion Classification Using Deep Learning

A six-class text emotion classification API built using **Python, PyTorch, FastAPI, and Docker**. The project uses a Bidirectional Gated Recurrent Unit (Bi-GRU) neural network to identify emotions expressed in textual input.

**Project Status: Completed and Deployed**

**Live API:** https://emotion-classification-api-83c8.onrender.com/

**API Documentation:** https://emotion-classification-api-83c8.onrender.com/docs

---

## Project Overview

The objective of this project is to develop a deep learning-based Natural Language Processing (NLP) application that identifies emotions expressed in text.

The application accepts textual input and predicts the corresponding emotion using a trained Bi-GRU model. It returns the predicted emotion, confidence score, and probability distribution across all six emotion categories.

### Emotion Categories

| Emotion | Description |
|---|---|
| Sadness | Expressions of sadness or disappointment |
| Joy | Expressions of happiness or excitement |
| Love | Expressions of affection or care |
| Anger | Expressions of frustration or anger |
| Fear | Expressions of worry or fear |
| Surprise | Expressions of unexpected reactions |

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Domain | Natural Language Processing (NLP) |
| Model Architecture | Bidirectional GRU (Bi-GRU) |
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Containerization | Docker |
| Deployment | Render |
| Version Control | Git and GitHub |

---

## Dataset

The project uses the **Emotion Dataset by Parul Pandey**, containing textual samples labelled with six different emotions.

### Dataset Distribution

| Dataset | Number of Samples |
|---|---:|
| Training | 16,000 |
| Validation | 2,000 |
| Testing | 2,000 |
| **Total** | **20,000** |

### Data Preprocessing

The following preprocessing steps were implemented:

- Converted text into lowercase.
- Removed unnecessary leading and trailing whitespace.
- Normalized consecutive whitespace.
- Tokenized text using regular expressions.
- Created a vocabulary from training data.
- Converted tokens into numerical sequences.
- Applied padding and truncation to maintain a fixed sequence length.

### Vocabulary Configuration

| Parameter | Value |
|---|---:|
| Vocabulary Size | 15,214 |
| Maximum Sequence Length | 48 |
| Padding Token | `<PAD>` |
| Unknown Token | `<UNK>` |

---

## Model Architecture

The project uses a **Bidirectional GRU (Bi-GRU)** architecture for text emotion classification.

The model processes text sequences in both forward and backward directions to capture contextual information.

### Model Workflow

```text
Input Text
    |
    v
Text Preprocessing
    |
    v
Tokenization
    |
    v
Vocabulary Mapping
    |
    v
Embedding Layer
    |
    v
Bidirectional GRU
    |
    v
Concatenation of Hidden States
    |
    v
Dropout
    |
    v
Linear Classification Layer
    |
    v
Six Emotion Logits
    |
    v
Softmax Probability Distribution
    |
    v
Predicted Emotion
```

### Model Configuration

| Hyperparameter | Value |
|---|---:|
| Embedding Dimension | 128 |
| Hidden Dimension | 128 |
| GRU Layers | 1 |
| Number of Classes | 6 |
| Dropout | 0.3 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Maximum Epochs | 15 |

The model was trained using Cross Entropy Loss and evaluated using a separate test dataset.

---

## Model Evaluation

The trained model was evaluated using 2,000 unseen test samples.

### Overall Performance

| Evaluation Metric | Score |
|---|---:|
| Test Accuracy | **88.90%** |
| Test Loss | 0.2987 |
| Macro Precision | 0.84 |
| Macro Recall | 0.86 |
| Macro F1-score | 0.85 |
| Weighted F1-score | 0.89 |

### Classification Report

| Emotion | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Sadness | 0.94 | 0.92 | 0.93 | 581 |
| Joy | 0.91 | 0.91 | 0.91 | 695 |
| Love | 0.71 | 0.74 | 0.73 | 159 |
| Anger | 0.90 | 0.89 | 0.89 | 275 |
| Fear | 0.88 | 0.85 | 0.87 | 224 |
| Surprise | 0.68 | 0.85 | 0.76 | 66 |
| **Macro Average** | **0.84** | **0.86** | **0.85** | **2000** |
| **Weighted Average** | **0.89** | **0.89** | **0.89** | **2000** |

### Confusion Matrix

Rows represent actual emotions, while columns represent predicted emotions.

Emotion order: Sadness, Joy, Love, Anger, Fear, Surprise.

```text
[[534  20   4  12  11   0]
 [  9 635  40   3   4   4]
 [  7  29 118   3   2   0]
 [ 11   7   4 244   6   3]
 [  6   0   0   8 191  19]
 [  1   5   0   2   2  56]]
```

The model achieved 88.90% test accuracy. Some confusion remains between semantically similar emotions, particularly Joy and Love, and Fear and Surprise.

---

## FastAPI Application

The trained model is integrated into a FastAPI application that exposes REST API endpoints.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check API availability |
| POST | `/predict` | Predict emotion from text |
| GET | `/docs` | Interactive Swagger documentation |
| GET | `/openapi.json` | OpenAPI specification |

### Example API Request

**Endpoint:** `/predict`

**Method:** POST

Request:

```json
{
    "text": "I am extremely happy today!"
}
```

Example response structure:

```json
{
    "text": "I am extremely happy today!",
    "predicted_emotion": "Joy",
    "confidence": 0.7743,
    "probabilities": {
        "Sadness": 0.01,
        "Joy": 0.7743,
        "Love": 0.10,
        "Anger": 0.02,
        "Fear": 0.05,
        "Surprise": 0.0457
    }
}
```

*The probability values above illustrate the response structure; actual values depend on the input.*

The API also implements input validation and returns an HTTP 400 response when empty text is submitted.

---

## Docker Containerization

The application is containerized using Docker to ensure consistent execution across environments.

### Build Docker Image

```bash
docker build -t emotion-classification .
```

### Run Docker Container

```bash
docker run -d -p 8000:8000 --name emotion-api emotion-classification
```

### Access Local API

```text
http://localhost:8000/
```

### Access Swagger Documentation

```text
http://localhost:8000/docs
```

---

## Live Deployment

The FastAPI application is deployed on Render using Docker.

| Resource | URL |
|---|---|
| Live API | https://emotion-classification-api-83c8.onrender.com/ |
| Swagger Documentation | https://emotion-classification-api-83c8.onrender.com/docs |
| OpenAPI Specification | https://emotion-classification-api-83c8.onrender.com/openapi.json |

The deployed API was tested successfully for emotion predictions and invalid input handling.

---

## Project Structure

```text
emotion-classification/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── src/
│   ├── config.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── model.py
│   ├── predict.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── tokenization.py
│   └── train.py
│
├── artifacts/
│   ├── best_model.pth
│   └── vocabulary.json
│
├── screenshots/
│
├── data/
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── evaluation_results.txt
└── README.md
```

---

## Key Features

- Six-class text emotion classification.
- Deep learning-based Bi-GRU architecture.
- Custom text preprocessing and tokenization pipeline.
- Vocabulary-based numerical text representation.
- Model training with validation and early stopping.
- Comprehensive model evaluation.
- REST API development using FastAPI.
- Prediction confidence and probability distribution.
- Input validation and error handling.
- Docker-based application containerization.
- Cloud deployment using Render.

---

## Future Improvements

- Experiment with Transformer-based architectures.
- Improve classification performance for underrepresented emotion classes.
- Apply techniques to address class imbalance.
- Improve confidence calibration.
- Develop a responsive frontend using React.
- Add prediction history and visualization dashboard.

---

## License

This project is available for educational and portfolio purposes. A formal open-source license can be added separately.
