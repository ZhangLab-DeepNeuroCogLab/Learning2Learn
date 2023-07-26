import torch
import torch.nn as nn
from torch.autograd import Variable

class SqueezeNetLSTM(nn.Module):
    def __init__(
            self, pretrained_snet,
            hidden_size, embedding_length,
            dropout=0.25, num_layers=1,
            output_size=2
    ):
        super(SqueezeNetLSTM, self).__init__()
        self.output_size = output_size
        try:
            self.pretrained_snet = pretrained_snet.module
        except Exception:
            self.pretrained_snet = pretrained_snet
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length
        self.dropout = dropout
        self.num_layers = num_layers
        if self.num_layers < 2:
            self.dropout = 0

        try:
            self.pretrained_snet.module.classifier[1] = nn.Conv2d(
                512, self.embedding_length, kernel_size=(1, 1), stride=(1, 1)
            )
        except Exception:
            self.pretrained_snet.classifier[1] = nn.Conv2d(
                512, self.embedding_length, kernel_size=(1, 1), stride=(1, 1)
            )
        self.adjust_base_model()
        self.lstm = nn.LSTM(
            input_size=self.embedding_length,
            hidden_size=self.hidden_size,
            bidirectional=True,
            dropout=self.dropout,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(2*self.hidden_size, self.output_size)

    def adjust_base_model(self):
        self.pretrained_snet.train()
        for parameter in self.pretrained_snet.features.parameters():
            parameter.requires_grad = False

    def forward(self, input):
        feature = []
        for inp in input:
            feature.append(self.pretrained_snet(inp))
        feature = torch.stack(feature, dim=0)
        output, (final_hidden_state, final_cell_state) = self.lstm(feature)
        hidden = torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:]), dim=1)
        return self.linear(hidden)
