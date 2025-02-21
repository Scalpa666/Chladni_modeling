import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# Displacement data set of particles at different frequencies and positions
class DispDataset(Dataset):
    def __init__(self, data_path):
        # Read the data
        data_df = pd.read_csv(data_path)
        data = data_df.values

        # Fetch frequency, position, and displacement
        self.freq_pos = torch.from_numpy(data[:, :3]).float()
        self.disp = torch.from_numpy(data[:, 3:]).float()

    def __getitem__(self, index):
        x = self.freq_pos[index]
        y = self.disp[index]
        return x, y

    def __len__(self):
        return len(self.freq_pos)