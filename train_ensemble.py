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


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class CoffeeLeafDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]))
        #image = utils.prepare_input_from_uri(img_path+'.jpg')
        #image = torch.squeeze(image)
        image = Image.open(img_path+'.jpg')
        label = 0 if np.sum(self.img_labels.iloc[idx, 1:4]) == 0 else np.argmax(self.img_labels.iloc[idx, 1:4])+1 
        if self.transform:
            image = self.transform(image)
        return image, label


class EarlyEnsemble_model(nn.Module):
    def __init__(self, num_classes):
        super(EarlyEnsemble_model, self).__init__()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.vit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        
        # Replace the classification layer of EfficientNet
        num_features = self.efficientnet.classifier.fc.in_features
        self.efficientnet.classifier.fc = nn.Linear(num_features, 128)
        # Replace the classification layer of MobileNet
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, 128)
        # Replace the classification layer of  Vision Transformer
        num_features = self.vit.head.in_features
        self.vit.head = nn.Linear(num_features, 128)

        self.fc = nn.Linear(128,num_classes)
    def forward(self, x):
        out_efficientnet = self.efficientnet(x)
        out_mobilenet = self.mobilenet(x)
        out_vit = self.vit(x)

        out = self.fc(out_efficientnet + out_mobilenet + out_vit)  # Combine the predictions
        return out
    def freeze(self):
        for param in self.densenet.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False

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
    # parser.add_argument('--num_labels', type=int, default=5, help='number of labels in the dataset')
    parser.add_argument('--data', type=str, default='coffee-leaf-diseases', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train (default: 60)')
    parser.add_argument('--model_path', type=str, default='./Ckpt/', help='path to save the best model')
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)


    if args.data == 'coffee-leaf-diseases':
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.CenterCrop(224),
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
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        train_set  = dataset.ImageFolder('./data/symptom/train', transform_train)
        val_set  = dataset.ImageFolder('./data/symptom/val', transform_test)
        test_set  = dataset.ImageFolder('./data/symptom/test', transform_test)
        classes = train_set.classes
        # dataset = CoffeeLeafDataset('./data/coffee-leaf-diseases/train_classes.csv','./data/coffee-leaf-diseases/train/images/', transform)
        # test_set = CoffeeLeafDataset('./data/coffee-leaf-diseases/test_classes.csv','./data/coffee-leaf-diseases/test/images/', transform_test)
        # train_set, val_set = torch.utils.data.random_split(dataset, [1000, 264],generator=torch.Generator().manual_seed(42))
    elif args.data == 'co-leaf':
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
        dataset = datasets.ImageFolder('./data/CoLeaf')
        # Get the indices and classes from the original dataset
        indices = list(range(len(dataset)))
        classes = dataset.classes

        # Split the dataset into train and test datasets
        train_set, test_set = split_dataset(indices, classes, split_ratio=0.2)

        # Further split the train dataset into train and validation datasets
        train_set, val_set = split_dataset(list(train_set.indices), classes, split_ratio=0.1)

        train_set.dataset.transform = transform_train
        val_set.dataset.transform = transform_test
        test_set.dataset.transform = transform_test
        # Define the path where you want to save the indices
        output_file_path = './test_set_indices.txt'

        # Write the indices to the text file
        with open(output_file_path, 'w') as file:
            for index in test_indices:
            file.write(f"{index}\n")
            else:
            raise ValueError('Dataset not supported')
    else:
        raise ValueError('Dataset not supported')

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
    
    # assert 1==2
    ensemble_model = EarlyEnsemble_model(classes)
    ensemble_model = ensemble_model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ensemble_model.parameters(), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # Define early stopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # Create a new data loader for validation set

    # acc_val = 0
    # acc_test =0
    # for t in range(args.epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train(train_dataloader, ensemble_model, loss_fn, optimizer)
    #     current_acc = test(valid_dataloader, ensemble_model, loss_fn)
    #     if current_acc > acc_val:
    #         acc_val = current_acc
    #         torch.save(ensemble_model.state_dict(), args.model_path+ 'best_acc.pth')
    #     test(test_dataloader,ensemble_model,loss_fn)
    #     scheduler.step()
    # print("Done!")

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, ensemble_model, loss_fn, optimizer)
        
        # Evaluate on validation set after each epoch
        with torch.no_grad():
            ensemble_model.eval()
            val_loss = 0
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                pred = ensemble_model(X)
                val_loss += loss_fn(pred, y).item()
            val_loss /= len(val_dataloader)
            print(f"Validation Loss: {val_loss}")
            
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.stopped:
            print("Early stopping triggered!")
            break

        test(test_dataloader, ensemble_model, loss_fn)
        scheduler.step()

    print("Done!")
