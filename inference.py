from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
from torchvision import datasets,transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from model import EarlyEnsembleModel, EfficientNet, MobileNet, ViT, ResNet50
import glob
import argparse

def inference(path_model, dataset, rootdir='./data/BRACOL/symptom/test/'):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    config = path_model.split('_')
    model_name = config[-4].split('/')[2]
    use_efficient, use_mobile, use_vit = config[-4].split('/')[-1], config[-3], config[-2]
    print(f"Using EfficientNet: {use_efficient}, MobileNet: {use_mobile}, ViT: {use_vit}")
    param_dict = {
        'True': True,
        'False': False
    }
    print(param_dict[use_efficient], param_dict[use_mobile], param_dict[use_vit])   
    model_dict = {
        "EarlyEnsemble": EarlyEnsembleModel,
        "EfficientNet": EfficientNet,
        "MobileNet": MobileNet,
        "ViT": ViT,
        "ResNet50": ResNet50,
    }
    if dataset =='bracol':
        if model_name == 'EarlyEnsemble':
            model = model_dict[model_name](5, param_dict[use_efficient], param_dict[use_mobile], param_dict[use_vit])
        else:
            model = model_dict[model_name](5)
        model.load_state_dict(torch.load(path_model))
        model = model.to(device)
        model.eval()
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
    parser.add_argument('--model_path', type=str, default='./efficient_mobile/False_False_False_bestacc.pth', help='path to the model')
    parser.add_argument('--dataset', type=str, default='bracol', help='dataset to test')
    parser.add_argument('--rootdir', type=str, default='./data/BRICOL/symptom/test/', help='path to the test dataset')
    args = parser.parse_args()
    inference(args.model_path, args.dataset, args.rootdir)
