# Emotion Classification API

A six-class text emotion classification API built using Python, PyTorch, FastAPI, and Docker. The project aims to classify text into six emotion categories: sadness, joy, love, anger, fear, and surprise.

**Project Status:** In Progress 

This project is being developed from scratch, covering the complete machine learning workflow, from text preprocessing and model training to API development and deployment.

## Project Overview

The objective is to develop a deep learning-based Natural Language Processing (NLP) application that identifies the emotion expressed in a given text.

The application will accept text as input and predict the corresponding emotion using a trained neural network.

### Emotion Categories

| Emotion | Description |
|---|---|
| Sadness | Expressions of sadness or disappointment |
| Joy | Expressions of happiness or excitement |
| Love | Expressions of affection or care |
| Anger | Expressions of frustration or anger |
| Fear | Expressions of worry or fear |
| Surprise | Expressions of unexpected reactions |

## Technology Stack

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Domain:** Natural Language Processing (NLP)
- **Model Architecture:** Bidirectional GRU (Bi-GRU)
- **API Framework:** FastAPI
- **ASGI Server:** Uvicorn
- **Containerization:** Docker

## Model Architecture

A Bidirectional GRU (Bi-GRU) is the initial architecture planned for this project.

The model workflow:

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
Vocabulary Lookup
    |
    v
Embedding Layer
    |
    v
Bidirectional GRU
    |
    v
Dropout
    |
    v
Linear Classifier
    |
    v
Six Emotion Probabilities
    |
    v
Predicted Emotion
```

The final architecture and hyperparameters will be documented after implementation and experimentation.

## Project Structure

```text
emotion-classification/
│
├── api/
│   └── main.py
│
├── src/
│   ├── model.py
│   ├── tokenizer.py
│   └── preprocessing.py
│
├── data/
│
├── artifacts/
│
├── tests/
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## Planned API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health and model status |
| POST | `/predict` | Predict emotion from input text |
| GET | `/docs` | Interactive Swagger API documentation |

### Example Request

```json
{
  "text": "I am extremely happy today"
}
```

The API will return the predicted emotion along with confidence and probability scores once implemented.

## Development Roadmap

- [ ] Dataset collection and exploration
- [ ] Text preprocessing and cleaning
- [ ] Tokenization and vocabulary creation
- [ ] PyTorch model implementation
- [ ] Bi-GRU model training
- [ ] Model evaluation and performance analysis
- [ ] Model serialization and loading
- [ ] FastAPI backend development
- [ ] Prediction and health-check endpoints
- [ ] API testing and validation
- [ ] Docker integration
- [ ] Cloud deployment
- [ ] Documentation and final testing

## Model Evaluation

The model will be evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix

Evaluation results will be added after training and testing the model on a held-out dataset.

## Getting Started

Installation and execution instructions will be added as the implementation progresses.

The planned setup includes:

1. Clone the repository.
2. Create a Python virtual environment.
3. Install project dependencies.
4. Prepare the dataset.
5. Train the model.
6. Run the FastAPI application.

## Future Improvements

- Improve classification performance.
- Experiment with different deep learning architectures.
- Handle ambiguous and overlapping emotions.
- Add robust input validation and error handling.
- Optimize inference performance.
- Deploy the API using Docker.

## License

License information will be added after selecting the appropriate licensing terms for the project.
