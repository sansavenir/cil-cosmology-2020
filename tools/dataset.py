from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class CSVDataset(Dataset):
    """Face label dataset."""

    def __init__(self, root_dir, scored=False, labeled=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_frame = None
        if scored:
            self.label_frame = pd.read_csv(os.path.join(root_dir, 'scored.csv'))
            self.label_frame['dir'] = ['scored']*self.label_frame.shape[0]
        if labeled:
            lb_frame = pd.read_csv(os.path.join(root_dir, 'labeled.csv'))
            lb_frame['dir'] = ['labeled']*lb_frame.shape[0]
            self.label_frame = self.label_frame.append(lb_frame) if self.label_frame is not None else lb_frame
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir+str(self.label_frame.iloc[idx]['dir']),
                                str(self.label_frame.iloc[idx, 0])+'.png')

        image = Image.open(img_name)
        label = self.label_frame.iloc[idx]['Actual']
        if self.label_frame.iloc[idx]['dir'] == 'labeled' and label == 1.:
            label = 8.
        label = np.array([label])
        label = label.astype('float')

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'name': self.label_frame.iloc[idx, 0]}

        return sample