import torch.nn as nn
import torch
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