import torch.nn as nn
import torch



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
        # out = self.fc(out_vit) 
        # out = self.fc(out_efficientnet + out_mobilenet) 
        return out
    def freeze(self):
        for param in self.densenet.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False
