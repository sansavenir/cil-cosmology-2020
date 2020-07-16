import os
import datetime

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable

from dataset import CosmologyDataset
from model import Scorer

# DATA_DIR = '/kaggle/input/cosmology/cosmology_aux_data_170429/cosmology_aux_data_170429'
DATA_DIR = '../../data'
NUM_EPOCHS = 30
BATCH_SIZE = 32
NUM_WORKERS = 2

np.random.seed(123)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    dataset = CosmologyDataset(DATA_DIR)

    num_samples = len(dataset)
    num_train_samples = int(0.9 * num_samples)
    num_test_samples = num_samples - num_train_samples

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train_samples, num_test_samples])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True, 
                                                   num_workers=NUM_WORKERS)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True, 
                                                  num_workers=NUM_WORKERS)

    net = Scorer().to(device)
    net.apply(weights_init)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = nn.L1Loss() 

    for epoch in (range(NUM_EPOCHS)):
        with tqdm(total=len(train_dataloader), desc='Training') as t:
            for i, data in enumerate(train_dataloader):
                img = Variable(data['image'])
                ps = Variable(data['pixels'])
                score_gt = Variable(data['score'])

                score_pd = net(img, ps)
                loss = loss_func(score_pd, score_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss=loss.item())
                t.update()

        total_loss = 0
        with tqdm(total=len(test_dataloader), desc='Evaluation') as t:
            with torch.no_grad():
                for i, data in enumerate(test_dataloader):
                    img = Variable(data['image'])
                    ps = Variable(data['pixels'])
                    score_gt = Variable(data['score'])

                    score_pd = net(img, ps)
                    loss = loss_func(score_pd, score_gt)
                    total_loss += loss

                    t.set_postfix(loss=loss.item())
                    t.update()

        print('Epoch', epoch+1, 'mean absolute error:', total_loss.item()/len(test_dataloader))


main()