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

from torch.optim.lr_scheduler import CosineAnnealingLR as CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR as StepLR

from model import EarlyEnsembleModel, EfficientNet, MobileNet, ViT, ResNet50

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


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='Train Early Ensemble Model')
    parser.add_argument('--data', type=str, default='BRACOL', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--saved', type=str, default='./Ckpt/', help='path to save the best model')
    parser.add_argument('--use_efficient', type=bool, default=False, help='Use EfficientNet in EarlyEnsemble_model')
    parser.add_argument('--use_mobile', type=bool, default=False, help='Use MobileNet in EarlyEnsemble_model')
    parser.add_argument('--use_vit', type=bool, default=False, help='Use Vision Transformer in EarlyEnsemble_model')
    args = parser.parse_args()

    # Load config
    dataset_name = args.data
    epochs = args.epochs
    model_name = args.model_name
    use_efficient = args.use_efficient
    use_mobile = args.use_mobile
    use_vit = args.use_vit
    saved = args.saved

    save_model_path= os.path.join(saved, model_name)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path, exist_ok=True)

    saved_name =f'{use_efficient}_{use_mobile}_{use_vit}_bestacc.pth'
    model_dict = {
        "EarlyEnsemble": EarlyEnsembleModel,
        "EfficientNet": EfficientNet,
        "MobileNet": MobileNet,
        "ViT": ViT,
        "ResNet50": ResNet50,
    }

    # Define data augmentation
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomApply([
            transforms.RandomRotation(10)],p=0.3),
        transforms.RandomApply([
            transforms.RandomAffine(10)],p=0.3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1)],p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    if args.data == 'BRACOL':
        train_set  = datasets.ImageFolder(f'../lara2018/classification/dataset/symptom/train', transform_train)
        val_set  = datasets.ImageFolder(f'../lara2018/classification/dataset/symptom/val', transform_test)
        test_set  = datasets.ImageFolder(f'../lara2018/classification/dataset/symptom/test', transform_test)

    elif args.data == 'CoLeaf':
        train_set  = datasets.ImageFolder(f'./data/{args.data}/train', transform_train)
        val_set  = datasets.ImageFolder(f'./data/{args.data}/val', transform_test)
        test_set  = datasets.ImageFolder(f'./data/{args.data}/test', transform_test)
    else:
        raise ValueError('Dataset not supported')

    classes = train_set.classes
    # Create data loaders.
    train_dataloader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True,drop_last = True)
    valid_dataloader = DataLoader(val_set, batch_size = args.batch_size, shuffle=False,drop_last = True)
    test_dataloader =  DataLoader(test_set, batch_size = args.batch_size)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    if model_name in model_dict:
        if model_name == "EarlyEnsemble":
            model = model_dict[model_name](len(classes), use_efficient=use_efficient, use_mobile=use_mobile, use_vit=use_vit)  # Adjust as needed
        else:
            model = model_dict[model_name](len(classes))
    else:
        raise ValueError(f"Model {model_name} not recognized")

    model = EfficientNet(len(classes))
    
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    acc_val = 0
    print(epochs)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        current_acc = test(valid_dataloader, model, loss_fn, 'Validation')
        if current_acc > acc_val:
            acc_val = current_acc
            torch.save(model.state_dict(), os.path.join(save_model_path,saved_name))
        test(test_dataloader,model,loss_fn, 'Test')
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr}")
        scheduler.step()
    print("Done!")
