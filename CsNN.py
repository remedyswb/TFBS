import torch
import torch.nn as nn
from torch.nn import functional as F


class CsNN(nn.Module):
    def __init__(self, dropout):
        super(CsNN, self).__init__()
        self.conv = nn.Sequential(nn.ConstantPad1d(23, 0.25),
                                  nn.Conv1d(in_channels=4, out_channels=16, kernel_size=24, stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=21, padding=1))
        self.weighted_sum = nn.Parameter(torch.randn(16, 6))
        self.dense = nn.Sequential(nn.Dropout(dropout), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, X):
        X = self.conv(X)
        X = self.seperable_conv(X)
        output = self.dense(X)
        return output

    def seperable_conv(self, X): #size of X:(batch_size, num_kernels, hidden_size)
        Y = X * self.weighted_sum
        output = torch.sum(Y, dim=-1)
        return output





