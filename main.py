from dataset import VoiceDataset
import torch.nn as nn
import numpy as np
import torch
import glob
import os
import pandas as pd
from config import args, print_args
from train import CategoryTrainer
from model.cnn import CategoryCNN, TagCNN

def set_weight(model, args):
    trained_weight = torch.load(f'{args.weight_path}{args.pt}.pt', map_location=args.device)
    model.load_state_dict(trained_weight)
    return model

if __name__ == '__main__':
    if args.use_wandb:
        import wandb
        wandb.init(project=args.project, name=args.name, tags=args.tags, config=args)
    else:
        pass

    print_args(args)

    train_data = VoiceDataset()
    test_data = VoiceDataset()

    if args.model_arc == 'TagCNN':
        model = TagCNN()
    elif args.model_arc == 'CategoryCNN':
        model = CategoryCNN()
    else:
        raise AssertionError

    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.use_wandb:
        wandb.watch(model)

    trainer = CategoryTrainer(model, optimizer, args.device, args.test_per_epoch, args.num_epochs, args.weight_path, train_data, test_data, args.use_wandb)
    trainer.train()