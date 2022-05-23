import torch
import torch.nn as nn
from torch.nn import functional as F


class CfNN(nn.Module):
    def __init__(self, dropout):
        super(CfNN, self).__init__()
        self.conv = nn.Sequential(nn.ConstantPad1d(23, 0.25),
                                  nn.Conv1d(in_channels=4, out_channels=16, kernel_size=24, stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=21, padding=1))
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(nn.Dropout(dropout), nn.Linear(96, 1), nn.Sigmoid())

    def forward(self, X):
        X = self.conv(X)
        X = self.flatten(X)
        output = self.dense(X)
        return output






