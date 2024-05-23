import os
import argparse
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR as StepLR
from PIL import Image

torch.manual_seed(3407)
import random
random.seed(3407)
import numpy as np
np.random.seed(3407)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

transform = transforms.Compose([
    transforms.Resize(224),
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
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
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
dataset = CoffeeLeafDataset('./data/coffee-leaf-diseases/train_classes.csv','./data/coffee-leaf-diseases/train/images/',transform)
test_dataset = CoffeeLeafDataset('./data/coffee-leaf-diseases/test_classes.csv','./data/coffee-leaf-diseases/test/images/',transform_test)
train_set, val_set = torch.utils.data.random_split(dataset, [1000, 264],generator=torch.Generator().manual_seed(42))
batch_size=64
# Create data loaders.
train_dataloader = DataLoader(train_set, batch_size=batch_size,drop_last=True)
valid_dataloader = DataLoader(val_set, batch_size=batch_size,drop_last=True)
test_dataloader =  DataLoader(test_dataset, batch_size=batch_size)


class EarlyEnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EarlyEnsembleModel, self).__init__()
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
# assert 1==2
ensemblemodel = EarlyEnsembleModel(5)
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
    import argparse
    parser = argparse.ArgumentParser(description='Train Early Ensemble Model')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train (default: 60)')
    parser.add_argument('--model_path', type=str, default='./Ckpt/', help='path to save the best model')
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    epochs = args.epochs
    acc_val = 0
    acc_test =0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, ensemblemodel, loss_fn, optimizer)
        current_acc = test(valid_dataloader, ensemblemodel, loss_fn)
        if current_acc > acc_val:
            acc_val = current_acc
            torch.save(ensemblemodel.state_dict(), args.model_path+ 'best_acc.pth')
        test(test_dataloader,ensemblemodel,loss_fn)
        scheduler.step()
    print("Done!")
