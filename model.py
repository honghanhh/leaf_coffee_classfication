import torch.nn as nn
import torch



class EarlyEnsembleModel(nn.Module):
    def __init__(self, num_classes, use_efficient=False, use_mobile=False, use_vit=False):
        super(EarlyEnsembleModel, self).__init__()

        self.use_efficient = use_efficient
        self.use_mobile = use_mobile
        self.use_vit = use_vit
        
        if use_efficient:
            self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
            num_features = self.efficientnet.classifier.fc.in_features
            self.efficientnet.classifier.fc = nn.Linear(num_features, 128)
        
        if use_mobile:
            self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            num_features = self.mobilenet.classifier[1].in_features
            self.mobilenet.classifier[1] = nn.Linear(num_features, 128)
        
        if use_vit:
            self.vit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        # Replace the classification layer of  Vision Transformer
            num_features = self.vit.head.in_features
            self.vit.head = nn.Linear(num_features, 128)

        self.fc = nn.Linear(128,num_classes)

    def forward(self, x):
        out = None
        if self.use_efficient:
            out_efficientnet = self.efficientnet(x)
            if out is None:
                out = out_efficientnet
            else:
                out += out_efficientnet
        if self.use_mobile:
            out_mobilenet = self.mobilenet(x)
            if out is None:
                out = out_mobilenet
            else:
                out += out_mobilenet
        if self.use_vit:
            out_vit = self.vit(x)
            if out is None:
                out = out_vit
            else:
                out += out_vit

        out = self.fc(out)  # Combine the predictions

        return out

class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        num_features = self.efficientnet.classifier.fc.in_features
        self.efficientnet.classifier.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)


class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()
        mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        num_features = mobilenet.classifier[1].in_features
        mobilenet.classifier[1] = nn.Linear(num_features, num_classes)
        self.model = mobilenet

    def forward(self, x):
        return self.model(x)


class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        vit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        num_features = vit.head.in_features
        vit.head = nn.Linear(num_features, num_classes)
        self.model = vit

    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        num_features = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_features, num_classes)
        self.model = resnet50

    def forward(self, x):
        return self.model(x)