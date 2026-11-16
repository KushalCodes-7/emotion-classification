import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):

    def __init__(self, sequences, labels):

        self.sequences = torch.tensor(
            sequences,
            dtype=torch.long
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, index):

        return self.sequences[index], self.labels[index]