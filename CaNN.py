import torch
import torch.nn as nn
import torch.nn.functional as F


#模型
class CaNN(nn.Module):
    def __init__(self, dropout):
        super(CaNN, self).__init__()
        self.convolution = nn.Sequential(nn.ConstantPad1d(23, 0.25),
                                         nn.Conv1d(in_channels=4, out_channels=16, kernel_size=24, stride=1),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=21, padding=1))

        self.key = nn.Linear(16, 1)
        self.dense = nn.Sequential(nn.Dropout(dropout), nn.Linear(16, 1), nn.Sigmoid())

    def attention(self, X):  #size of input is (batch_size, num_kernels, hidden_size)
        E = self.key(X.permute(0, 2, 1))
        A = F.softmax(E, dim=1)
        output = torch.bmm(X, A)
        return output.squeeze(-1)

    def forward(self, X):
        X = self.convolution(X)
        X = self.attention(X)
        output = self.dense(X)
        return output




