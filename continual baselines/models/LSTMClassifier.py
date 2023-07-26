import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_length, batch_size, dropout=0.25, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_layers = num_layers
        if self.num_layers < 2:
            self.dropout = 0

        self.lstm = nn.LSTM(
            input_size=self.embedding_length,
            hidden_size=self.hidden_size,
            bidirectional=True,
            dropout=self.dropout,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(2*self.hidden_size, self.output_size)

    def forward(self, input):
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        hidden = torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:]), dim=1)
        return self.linear(hidden)

