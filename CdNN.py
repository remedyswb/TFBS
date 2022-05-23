import torch
import torch.nn as nn
from torch.nn import functional as F


class CdNN(nn.Module):
    def __init__(self, dropout):
        super(CdNN, self).__init__()
        self.conv_1 = nn.Sequential(nn.ConstantPad1d(23, 0.25),
                                    nn.Conv1d(in_channels=4, out_channels=16, kernel_size=24, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=21, padding=1))
        self.conv_2 = nn.Sequential(nn.Conv1d(in_channels=6, out_channels=1, kernel_size=1, bias=False),
                                    nn.ReLU())
        self.dense = nn.Sequential(nn.Dropout(dropout), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, X):
        X = self.conv_1(X)
        X = X.permute(0, 2, 1)
        X = self.conv_2(X).reshape(-1, 16)
        output = self.dense(X)
        return output