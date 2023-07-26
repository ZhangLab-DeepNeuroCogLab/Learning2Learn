import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, num_classes):
        super(Head, self).__init__()
        self.num_classes = num_classes

        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.final_conv = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.dropout(x)
        x = self.final_conv(x)
        x = self.relu(x)
        x = self.pool(x)
        out = torch.flatten(x, 1)
        return out


class MTL(nn.Module):
    def __init__(self, base_model, num_classes):
        super(MTL, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        self.classifier_1, self.classifier_2 = Head(self.num_classes), Head(self.num_classes)

    def forward(self, x):
        try:
            x = self.base_model.features(x)
        except AttributeError:
            x = self.base_model.module.features(x)
        out1, out2 = self.classifier_1(x), self.classifier_2(x)
        return out1, out2
