from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
from torchvision import datasets,transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from model import EarlyEnsembleModel

def inference(path_model, dataset, rootdir='./data/coffee-leaf-diseases/'):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    if dataset =='1':
        model = EarlyEnsembleModel(5)
        model.load_state_dict(torch.load(path_model))
        model = model.to(device)
        model.eval()
        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        list_img = pd.read_csv(os.path.join(rootdir,'test_classes.csv'))
        test_img_dir = os.path.join(rootdir,'test/images/')
        list_true = []
        list_predict =[]
        for i in range(0,len(list_img)): 
            img_path = os.path.join(test_img_dir, str(list_img.iloc[i, 0]))
            image = Image.open(img_path+'.jpg')
            label = 0 if np.sum(list_img.iloc[i, 1:4]) == 0 else np.argmax(list_img.iloc[i, 1:4])+1 
            list_true.append(label)
            img = transform_test(image)
            img = torch.unsqueeze(img,dim=0)
            img = img.to(device)
            output = model(img)
            prediction_label = torch.argmax(output).item()
            list_predict.append(prediction_label)
        accuracy = accuracy_score(list_predict, list_true)
        f1 = f1_score(list_true, list_predict,average='micro')
        classification_rp = classification_report(list_predict, list_true)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Classification report:\n", classification_rp)
        print(np.unique(list_true))
    else:
        # TODO for the second dataset
        pass
if __name__ =='__main__':
    inference('efficient_mobile_vit/best_acc.pth','1')