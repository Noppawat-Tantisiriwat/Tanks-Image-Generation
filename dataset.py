import torch
from torch.utils.data import Dataset

import numpy as np

class TanksDataset(Dataset):

    def __init__(self, transform=None):
        
        path = "alltanks.npy"

        images_data = np.load(path)

        data = np.swapaxes(images_data, 3, 1)

        self.data = data

        self.transform = transform
    
    def __getitem__(self, index):

        if self.transform:

            return self.transform(self.data[index])

    def __len__(self):
        
        return self.data.shape[0]



class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        x = sample
        return torch.from_numpy(x)