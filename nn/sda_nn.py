import torch
import torch.nn as nn
import numpy as np


class SdaNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SdaNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, input):
        output, _ = self.lstm(input)
        ln_o = self.fc(output)
        res = torch.sigmoid(ln_o)
        return res
