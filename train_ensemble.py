import os

import torch
torch.manual_seed(3407)

import random
random.seed(3407)

import argparse
import pandas as pd
import numpy as np
np.random.seed(3407)

from PIL import Image
from torch import nn
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor

from torch.optim.lr_scheduler import StepLR as StepLR
from torch.utils.data import random_split, Subset

from model import EarlyEnsemble_model


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn, phase='val'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"{phase}: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

# Define a function to split the dataset while maintaining class distribution for CoLeaf datasets
def split_dataset(indices, classes, split_ratio=0.2):
    class_counts = {cls: 0 for cls in classes}
    for idx in indices:
        _, label = dataset.samples[idx]
        class_counts[dataset.classes[label]] += 1

    train_indices = []
    val_indices = []
    for cls in classes:
        cls_indices = [idx for idx in indices if dataset.classes[dataset.samples[idx][1]] == cls]
        np.random.shuffle(cls_indices)
        split = int(np.floor(split_ratio * len(cls_indices)))
        val_indices.extend(cls_indices[:split])
        train_indices.extend(cls_indices[split:])
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set

if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='Train Early Ensemble Model')
    parser.add_argument('--data', type=str, default='BRICOL', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train (default: 60)')
    parser.add_argument('--model_path', type=str, default='./Ckpt/', help='path to save the best model')
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Define data augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomApply([
            transforms.RandomRotation(10)],p=0.3),
        transforms.RandomApply([
            transforms.RandomAffine(10)],p=0.3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1)],p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    if args.data == 'BRICOL':
        train_set  = datasets.ImageFolder(f'./data/{args.data}/symptom/train', transform_train)
        val_set  = datasets.ImageFolder(f'./data/{args.data}/symptom/val', transform_test)
        test_set  = datasets.ImageFolder(f'./data/{args.data}/symptom/test', transform_test)

    elif args.data == 'CoLeaf':
        train_set  = datasets.ImageFolder(f'./data/{args.data}/train', transform_train)
        val_set  = datasets.ImageFolder(f'./data/{args.data}/val', transform_test)
        test_set  = datasets.ImageFolder(f'./data/{args.data}/test', transform_test)
    else:
        raise ValueError('Dataset not supported')

    classes = train_set.classes

    # Create data loaders.
    train_dataloader = DataLoader(train_set, batch_size = args.batch_size, drop_last = True)
    valid_dataloader = DataLoader(val_set, batch_size = args.batch_size, drop_last = True)
    test_dataloader =  DataLoader(test_set, batch_size = args.batch_size)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    ensemble_model = EarlyEnsemble_model(len(classes))
    ensemble_model = ensemble_model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ensemble_model.parameters(), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    acc_val = 0
    acc_test =0
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, ensemble_model, loss_fn, optimizer)
        current_acc = test(valid_dataloader, ensemble_model, loss_fn, 'Validation')
        if current_acc > acc_val:
            acc_val = current_acc
            torch.save(ensemble_model.state_dict(), args.model_path+ 'best_acc.pth')
        test(test_dataloader,ensemble_model,loss_fn, 'Test')
        scheduler.step()
    print("Done!")
