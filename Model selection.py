import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn import metrics
import random
import numpy as np
import math
from math import sqrt

def logsampler(a, b):
    x = np.random.uniform(low=0, high=1)
    y = 10**((math.log10(b)-math.log10(a))*x+math.log10(a))
    return y


def sqrtsampler(a, b):
    x = np.random.uniform(low=0, high=1)
    y = (b - a) * math.sqrt(x) + a
    return y


def Onehot(text):
    alphabet = 'AGCT'
    x = 4
    y = len(text[0])
    z = len(text)
    features = torch.zeros(z, x, y)
    for k in range(z):
        for i in range(x):
            for j in range(y):
                if text[k][j] == alphabet[i]:
                    features[k, i, j] = 1
    return features


def StoT(y):
    for i in range(len(y)):
        if y[i] == '1':
            y[i] = 1.
        else:
            y[i] = 0.
    return torch.tensor(y).unsqueeze(1)


#数据预处理
class ChIP_data(Dataset):
    def __init__(self, seqs, labels):
        self.features = seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


with open(r'C:\Users\DELL\Desktop\毕设\TFBS 数据集\CTCF\train.txt') as f:
    data = f.read().split()
    seqs = [data[i] for i in range(1, len(data), 3)]
    labels = [data[i] for i in range(2, len(data), 3)]

size = 10000
eval_seq1, eval_label1 = Onehot(seqs[:size]), StoT(labels[:size])
eval_seq2, eval_label2 = Onehot(seqs[size:2*size]), StoT(labels[size:2*size])
eval_seq3, eval_label3 = Onehot(seqs[2*size:3*size]), StoT(labels[2*size:3*size])
train_seq1, train_label1 = eval_seq2+eval_seq3, eval_label2+eval_label3
train_seq2, train_label2 = eval_seq1+eval_seq3, eval_label1+eval_label3
train_seq3, train_label3 = eval_seq1+eval_seq2, eval_label1+eval_label2

train_data1 = ChIP_data(train_seq1, train_label1)
train_data2 = ChIP_data(train_seq2, train_label2)
train_data3 = ChIP_data(train_seq3, train_label3)
eval_data1 = ChIP_data(eval_seq1, eval_label1)
eval_data2 = ChIP_data(eval_seq2, eval_label2)
eval_data3 = ChIP_data(eval_seq3, eval_label3)

train_dataloader1 = DataLoader(train_data1, batch_size=100, shuffle=True)
train_dataloader2 = DataLoader(train_data2, batch_size=100, shuffle=True)
train_dataloader3 = DataLoader(train_data3, batch_size=100, shuffle=True)
eval_dataloader1 = DataLoader(eval_data1, batch_size=100, shuffle=False)
eval_dataloader2 = DataLoader(eval_data2, batch_size=100, shuffle=False)
eval_dataloader3 = DataLoader(eval_data3, batch_size=100, shuffle=False)

train_dataloader = [train_dataloader1, train_dataloader2, train_dataloader3]
eval_dataloader = [eval_dataloader1, eval_dataloader2, eval_dataloader3]


#模型
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


class DeepBind(nn.Module):
    def __init__(self, dropout):
        super(DeepBind, self).__init__()
        self.convolution = nn.Sequential(nn.ConstantPad1d(23, 0.25),
                                         nn.Conv1d(in_channels=4, out_channels=16, kernel_size=24, stride=1),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=124))
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, X):
        X = self.convolution(X)
        X = self.flatten(X)
        output = self.dense(X)
        return output


#模型评估与超参数选择
learning_steps_list = [1000, 2000, 3000, 4000, 5000]
best_auc = 0
calibration_auc = []
for calibration in range(10):
    dropout_list = [0, 0.25, 0.5]
    dropout = random.choice(dropout_list)
    weight_decay = logsampler(1e-10, 1e-3)
    learning_rate = logsampler(7e-5, 7e-3)

    model = DeepBind(dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()
    eval_auc = [[], [], []]

    for k in range(3):
        train_loader = train_dataloader[k]
        eval_loader = eval_dataloader[k]
        learning_steps = 0
        while learning_steps < 5000:
            model.train()
            for X, y in train_loader:
                pred = model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                learning_steps += 1

                if learning_steps % 1000 == 0:
                    pred_all = []
                    label_all = []
                    with torch.no_grad():
                        model.eval()
                        for X, y in eval_loader:
                            pred = model(X)
                            label_all.extend(y.detach().numpy())
                            pred_all.extend(pred.detach().numpy())

                        eval_auc[k].append(metrics.roc_auc_score(label_all, pred_all))

    for n in range(5):
        auc = (eval_auc[0][n]+eval_auc[1][n]+eval_auc[2][n])/3
        if auc > best_auc:
            best_auc = auc
            best_learning_steps = learning_steps_list[n]
            best_learning_rate = learning_rate
            best_dropout = dropout
            best_weight_decay = weight_decay
    calibration_auc.append(best_auc)

print('best AUC =', best_auc)
print('best_learning_steps =', best_learning_steps)
print('best_learning_rate =', best_learning_rate)
print('best_dropout =', best_dropout)
print('best_weight_decay =', best_weight_decay)

plt.plot(calibration_auc)
plt.title('Best AUC of each calibration')
plt.show()













