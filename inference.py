from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
from torchvision import datasets,transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from model import EarlyEnsemble_model
import glob
import argparse

def inference(path_model, dataset, rootdir='./data/BRICOL/symptom/test/'):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    if dataset =='bricol':
        model = EarlyEnsemble_model(5)
        model.load_state_dict(torch.load(path_model))
        model = model.to(device)
        model.eval()
        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        test_set = datasets.ImageFolder(rootdir)
        mapping_label = test_set.class_to_idx

        list_img = glob.glob(rootdir+'*/*')
        list_true = []
        list_predict =[]
        for img_path in list_img: 
            image = Image.open(img_path)
            label = int(mapping_label[img_path.split('/')[-2]]) 
            list_true.append(label)
            img = transform_test(image)
            img = torch.unsqueeze(img,dim=0)
            img = img.to(device)
            output = model(img)
            prediction_label = torch.argmax(output).item()
            list_predict.append(prediction_label)
        accuracy = accuracy_score(list_predict, list_true)
        f1 = f1_score(list_true, list_predict,average='micro')
        classification_rp = classification_report(list_true, list_predict, digits=4)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Classification report:\n", classification_rp)
        print(np.unique(list_true))
    else:
        raise ValueError('Dataset not supported')
        
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Infer Early Ensemble Model')
    parser.add_argument('--model_path', type=str, default='./efficient_mobile', help='path to the model')
    parser.add_argument('--dataset', type=str, default='bricol', help='dataset to test')
    parser.add_argument('--rootdir', type=str, default='./data/BRICOL/symptom/test/', help='path to the test dataset')
    args = parser.parse_args()
    inference(args.model_path + '/best_acc.pth','1')
