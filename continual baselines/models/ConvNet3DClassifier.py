import torch
import torch.nn as nn


class ConvNet3DClassifier(nn.Module):
    def __init__(self, output_size, dropout=0.25):
        super(ConvNet3DClassifier, self).__init__()
        self.output_size = output_size
        self.dropout = dropout

        self.conv_1 = ConvNet3DClassifier._custom_3d(3, 32)
        self.conv_2 = ConvNet3DClassifier._custom_3d(32, 64)
        self.fc_1 = nn.Linear(559872, 128)
        self.fc_2 = nn.Linear(128, self.output_size)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=self.dropout)

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

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)

        return self.fc_2(out)
