import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomSpikeDataset(Dataset):

    def __init__(self, x_data, y_data):
        """
        Class to create a custom data set that can be loaded with dataloaders.

        arguments
            x_data: torch.Tensor
                input data, shape should be (batch_size x time_steps x input_neurons)
            y_data: torch.Tensor
                target data, shape should be (batch_size)
        """
        self.labels = y_data
        self.data = x_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.data[idx]  # x_data[idx]
        return data, label



