import torch
import torch.nn as nn


class SqueezeNet:
    def __init__(self, pretrained=False):
        self.pretrained = pretrained

        model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1',
                               pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
        self.model = model

    def __call__(self):
        return self.model
