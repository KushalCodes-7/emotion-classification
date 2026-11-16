
import torch
import torch.nn as nn

from src.config import (
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    NUM_CLASSES,
    DROPOUT,
    PAD_ID
)


class EmotionBiGRU(nn.Module):

    def __init__(self):

        super(EmotionBiGRU, self).__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=PAD_ID
        )

        # 2. Bidirectional GRU
        self.gru = nn.GRU(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0 if NUM_LAYERS == 1 else DROPOUT
        )

        # 3. Dropout
        self.dropout = nn.Dropout(DROPOUT)

        # 4. Fully Connected Layer
        self.fc = nn.Linear(
            HIDDEN_DIM * 2,
            NUM_CLASSES
        )

    def forward(self, input_ids):

        # Convert token IDs into embeddings
        embedded = self.embedding(input_ids)

        # Pass embeddings through Bi-GRU
        gru_output, hidden = self.gru(embedded)

        # Get the final hidden states
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        # Combine forward and backward representations
        combined_hidden = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        # Apply dropout
        dropped = self.dropout(combined_hidden)

        # Classification
        output = self.fc(dropped)

        return output


if __name__ == "__main__":

    model = EmotionBiGRU()

    sample_input = torch.randint(
        0,
        VOCAB_SIZE,
        (32, 48)
    )

    output = model(sample_input)

    print("Model architecture:")
    print(model)

    print("\nInput shape:", sample_input.shape)
    print("Output shape:", output.shape)
