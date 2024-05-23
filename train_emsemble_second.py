import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import numpy as np
import torch
from torch import nn
import random
from torch.optim.lr_scheduler import StepLR as StepLR
from PIL import Image
from torch.utils.data import random_split, Subset

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.RandomApply([
        transforms.RandomRotation(10)],p=0.3),
    transforms.RandomApply([
        transforms.RandomAffine(10)],p=0.3),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

dataset = datasets.ImageFolder('CoLeaf')
# Load the dataset with the same ImageFolder for both train and validation

# Define a function to split the dataset while maintaining class distribution
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
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset

# Get the indices and classes from the original dataset
indices = list(range(len(dataset)))
classes = dataset.classes

# Split the dataset into train and test datasets
train_dataset, test_dataset = split_dataset(indices, classes, split_ratio=0.2)

# Further split the train dataset into train and validation datasets
train_dataset, val_dataset = split_dataset(list(train_dataset.indices), classes, split_ratio=0.1)

train_dataset.dataset.transform= transform_train
val_dataset.dataset.transform= transform_test
test_dataset.dataset.transform = transform_test

# Create DataLoaders for training and validation
batch_size=64
# Create data loaders.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
test_dataloader =  DataLoader(test_dataset, batch_size=batch_size)


class EarlyEnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EarlyEnsembleModel, self).__init__()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)

        self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)


        
        # Replace the classification layer of EfficientNet
        num_features = self.efficientnet.classifier.fc.in_features
        self.efficientnet.classifier.fc = nn.Linear(num_features, 128)
        # Replace the classification layer of MobileNet
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, 128)

        self.fc = nn.Linear(128,num_classes)
    def forward(self, x):
        out_efficientnet = self.efficientnet(x)
        out_mobilenet = self.mobilenet(x)

        out = self.fc(out_efficientnet + out_mobilenet)  # Combine the predictions
        return out
    def freeze(self):
        for param in self.densenet.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
ensemblemodel = EarlyEnsembleModel(len(dataset.classes))
ensemblemodel = ensemblemodel.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ensemblemodel.parameters(), lr=1e-2)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

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
            
def test(dataloader, model, loss_fn):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct
if __name__ =='__main__':
    epochs = 100
    acc_val = 0
    acc_test =0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, ensemblemodel, loss_fn, optimizer)
        current_acc = test(valid_dataloader, ensemblemodel, loss_fn)
        if current_acc > acc_val:
            acc_val = current_acc
            torch.save(ensemblemodel.state_dict(), './Ckpt_second/best_acc_efficiennet_mobilenet.pth')
        test(test_dataloader,ensemblemodel,loss_fn)
        scheduler.step()
    print("Done!")