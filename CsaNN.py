import torch
from torch import nn
from math import sqrt
import torch.nn.functional as F


class Self_Attention(nn.Module):
    def __init__(self, i, d, v):
        super(Self_Attention, self).__init__()
        self.Wq = nn.Linear(i, d)
        self.Wk = nn.Linear(i, d)
        self.Wv = nn.Linear(i, v)
        self.norm = 1 / sqrt(d)

    def forward(self, X): #size of input:(batch_size, L, i)
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        A = F.softmax(torch.bmm(Q, K.permute(0, 2, 1)) * self.norm, dim=-1)
        output = torch.bmm(A, V)
        return output


class CsaNN(nn.Module):
    def __init__(self, dropout):
        super(CsaNN, self).__init__()
        self.convolution = nn.Sequential(nn.ConstantPad1d(23, 0.25),
                                         nn.Conv1d(in_channels=4, out_channels=16, kernel_size=24, stride=1),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=3, padding=1))
        self.attention = Self_Attention(16, 4, 4)
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(42, 1), nn.Sigmoid())

    def forward(self, X):
        X = self.convolution(X).permute(0, 2, 1)
        X = self.attention(X)
        X = self.flatten(X)
        output = self.dense(X)
        return output