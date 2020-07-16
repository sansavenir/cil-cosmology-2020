import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import transform
from PIL import Image
import glob
import imageio

class CosmologyDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir

        labeled_path = os.path.join(root_dir, 'labeled.csv')
        labeled = np.genfromtxt(labeled_path, delimiter=',', skip_header=1, dtype=np.float32)
        labeled = labeled[np.where(labeled[:, 1] > 0)]
        labeled = np.hstack([labeled, np.ones([labeled.shape[0], 1])])

        scored_path = os.path.join(root_dir, 'scored.csv')
        scored = np.genfromtxt(scored_path, delimiter=',', skip_header=1, dtype=np.float32)
        scored = scored[np.where(scored[:, 1] > 0.5)]
        scored = np.hstack([scored, np.zeros([scored.shape[0], 1])])
        self.data = np.vstack([labeled, scored])


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, score, is_labeled = self.data[idx]

        dir_name = 'labeled' if is_labeled else 'scored'
        img_path = os.path.join(self.root_dir, dir_name, str(int(img_name)) + '.png')
        img = imageio.imread(img_path, as_gray=True)

        histogram, _ = np.histogram(img, bins=np.arange(256), density=True)

        img = transform.resize(img, (256, 256), preserve_range=True).astype(np.uint8)

        img_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5],[0.5])])
        img = img_transform(img)

        return {'image': img, 
                'pixels': torch.FloatTensor(histogram), 
                'score': torch.FloatTensor([score])}