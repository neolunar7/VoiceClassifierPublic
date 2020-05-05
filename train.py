from dataset import VoiceDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os

from config import args
from torch.utils import data
from tqdm import tqdm

class CategoryTrainer:
    def __init__(self, model, optimizer, device, test_per_epoch, num_epochs, weight_path, train_data, test_data, use_wandb):
        self.device = device
        self.use_wandb = use_wandb
        self.weight_path = weight_path
        self.num_epochs = num_epochs
        self.test_per_epoch = test_per_epoch

        self.train_data = train_data
        self.test_data = test_data
        self.test_generator = data.DataLoader(dataset=test_data, shuffle=False, batch_size=args.test_batch, num_workers=args.num_workers)
        self.optimizer = optimizer
        self.model = model
        self.loss = nn.MSELoss(reduce=False)

        self.train_loss = 0.5
        self.test_loss = 0.5
        self.train_acc = 0.5
        self.test_acc = 0.5

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def write_wandb(self):
        import wandb
        wandb.log({
            'Train Loss': self.train_loss,
            'Test Loss': self.test_loss,
        })

    def forward(self, generator, training_mode):
        loss_list = []
        for melspectrogram, categoryVectors, _ in tqdm(generator):
            output = self.model(melspectrogram.unsqueeze(1))
            groundTruth = categoryVectors
            loss = self.loss(output, groundTruth)
            loss_by_batch = loss.sum(-1)
            epoch_loss = loss_by_batch.mean()

            if training_mode == 'train':
                self.update(epoch_loss)
            loss_list.append(epoch_loss.item())

        epoch_mean_loss = np.mean(loss_list)
        print_val = f'loss: {epoch_mean_loss:.4f}'

        if training_mode == 'train':
            self.train_loss = epoch_mean_loss
            print(f'[Train]     {print_val}')
        elif training_mode == 'test':
            self.test_loss = epoch_mean_loss
            print(f'[Test]      {print_val}')

    def train(self):
        for epoch in range(self.num_epochs):
            print(f'\nEpoch: {epoch:03d} re-shuffling...')

            # train
            train_generator = data.DataLoader(dataset=self.train_data, shuffle=True, batch_size=args.train_batch, num_workers=args.num_workers)
            self.model.train()
            self.forward(train_generator, training_mode='train')

            # save_parameters
            cur_weight = self.model.state_dict()
            torch.save(cur_weight, f'{self.weight_path}/{epoch.pt}')

            # test
            if (epoch % self.test_per_epoch) == (self.test_per_epoch-1):
                with torch.no_grad():
                    self.model.eval()
                    self.forward(self.test_generator, training_mode='test')

            # # write_wandb
            if self.use_wandb:
                self.write_wandb()