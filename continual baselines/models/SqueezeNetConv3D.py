import torch
import torch.nn as nn
from torch.autograd import Variable

class SqueezeNetConv3D(nn.Module):
    def __init__(
            self, pretrained_snet,
            output_size=2, dropout=0.25
    ):
        super(SqueezeNetConv3D, self).__init__()
        try:
            self.pretrained_snet = pretrained_snet.module
        except Exception:
            self.pretrained_snet = pretrained_snet
        self.output_size = output_size
        self.dropout = dropout
        self.adjust_base_model()

        self._features = {"features": torch.empty(0)}
        layer = dict([*self.pretrained_snet.named_modules()])['features']
        layer.register_forward_hook(self.save_outputs_hook('features'))

        self.conv_1 = SqueezeNetConv3D._custom_3d(512, 32)
        self.conv_2 = SqueezeNetConv3D._custom_3d(32, 64)
        self.fc_1 = nn.Linear(192, 128)
        self.fc_2 = nn.Linear(128, self.output_size)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=self.dropout)

    def adjust_base_model(self):
        self.pretrained_snet.classifier = nn.Identity()
        for parameter in self.pretrained_snet.features[12].parameters():
            parameter.requires_grad = True

    @staticmethod
    def _custom_3d(in_c, out_c):
        conv = nn.Sequential(
            nn.Conv3d(
                in_c, out_c, kernel_size=(3, 3, 3), padding=0
            ),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        return conv

    def save_outputs_hook(self, layer):
        def fn(_, __, output):
            self._features[layer] = output
        return fn

    def forward(self, input):
        feature = []
        for inp in input:
            _ = self.pretrained_snet(inp)
            feature.append(self._features['features'].permute(1, 0, 2, 3))
        feature = torch.stack(feature, 0)
        print(feature.shape)
        out = self.conv_1(feature)
        out = self.conv_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)

        return self.fc_2(out)


