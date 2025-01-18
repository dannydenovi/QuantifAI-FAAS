import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


# Dataset Handling
class GenericDataset(Dataset):
    def __init__(self, data_file: str, batch_size):
        dataset = torch.load(data_file)
        self.data = dataset["data"]
        self.labels = dataset["labels"]

        self.input_format = self.data[0].shape


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        if self.input_format:
            data = data.view(*self.input_format)
        return data, label
