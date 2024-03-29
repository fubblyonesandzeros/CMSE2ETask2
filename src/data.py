import torch
import numpy as np
from torch.utils.data import Dataset


class QuarkGluonClassificationDataset(Dataset):
    def __init__(self, x_file, y_file):
        self.x = np.load(x_file)
        self.y = np.load(y_file)

        self.min = -6.39
        self.max = 16189.07

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        
        x = self.x[index]
        y = self.y[index]

        x = torch.from_numpy(x).permute(2, 1, 0)
        x = (x - self.min)/(self.max - self.min)

        x = x.float()

        return x, y