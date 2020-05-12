from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from os import listdir



class CSVDataset(Dataset):
    """Face label dataset."""

    def __init__(self, root_dir, csv_file=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.eval = csv_file is None
        if not self.eval:
            self.label_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        if self.eval:
            self.files = listdir(self.root_dir)

    def __len__(self):
        if not self.eval:
            return len(self.label_frame)
        else:
            return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.eval:
            img_name = os.path.join(self.root_dir,
                                    str(self.label_frame.iloc[idx, 0])+'.png')
            image = Image.open(img_name)
            label = self.label_frame.iloc[idx, 1:]
            label = np.array([label])
            label = label.astype('float')
        else:
            img_name = self.root_dir+'/'+self.files[idx]
            image = Image.open(img_name)
            img_name = self.files[idx]

        if self.transform:
            image = self.transform(image)

        if not self.eval:
            sample = {'image': image, 'label': label, 'name': img_name}
        else:
            sample = {'image': image, 'name': img_name}

        return sample