import torch
import torch.nn as nn


class ResNet:
    def __init__(self, pretrained=False, num_classes=10):
        self.pretrained = pretrained
        self.num_classes = num_classes

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18',
                               pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, self.num_classes)
        self.model = model

    def __call__(self):
        return self.model
