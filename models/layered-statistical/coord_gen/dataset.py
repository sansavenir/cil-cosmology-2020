from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import glob


class CSVDataset(Dataset):
    """Face label dataset."""

    def __init__(self, root_dir, max=25):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.files = glob.glob(root_dir+'*.csv')
        self.max = max

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.files[idx]
        file = np.genfromtxt(file_name , delimiter=',', dtype=float)
        res = np.ones((self.max,2))*-100
        res[:file.shape[0]] = file

        sample = {'file': res.flatten()/1000.}

        return sample