import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn import functional as F
from math import sqrt


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

train_seq, train_label = Onehot(seqs[:30000]), StoT(labels[:30000])
train_data = ChIP_data(train_seq, train_label)
train_dataloader = DataLoader(train_data, batch_size=100, shuffle=True)


#模型
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


#终极训练
learning_steps = 5000
learning_rate = 0.000667568714897817
dropout = 0
weight_decay = 5.34898042551208E-06

best_auc = 0
for n in range(5):
    model = DeepBind(dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()
    train_loss = []
    steps = 0
    model.train()
    while steps <= learning_steps:
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

            if steps % 100 == 0:
                train_loss.append(loss.item())

    pred_all = []
    label_all = []
    with torch.no_grad():
        model.eval()
        for X, y in train_dataloader:
            pred = model(X)
            label_all.extend(y.detach().numpy())
            pred_all.extend(pred.detach().numpy())
        auc = metrics.roc_auc_score(label_all, pred_all)
        print('AUC for model', n+1, ':', auc)
        if auc > best_auc:
            best_auc = auc
            best_loss = train_loss
            torch.save(model, r'C:\Users\DELL\PycharmProjects\pythonProject1\model.pth')

plt.plot(best_loss, 'r')
plt.title('Training loss')
plt.show()

#测试
with open(r'C:\Users\DELL\Desktop\毕设\TFBS 数据集\CTCF\test.txt') as f:
    data = f.read().split()
    seqs = [data[i] for i in range(1, len(data), 3)]
    labels = [data[i] for i in range(2, len(data), 3)]

test_seq, test_label = Onehot(seqs[:10000]), StoT(labels[:10000])
test_data = ChIP_data(test_seq, test_label)
test_dataloader = DataLoader(test_data, batch_size=100, shuffle=False)

model = torch.load(r'C:\Users\DELL\PycharmProjects\pythonProject1\model.pth')
model.eval()
pred_all = []
label_all = []
for X, y in test_dataloader:
    pred = model(X)
    label_all.extend(y.detach().numpy())
    pred_all.extend(pred.detach().numpy())
auc = metrics.roc_auc_score(label_all, pred_all)
print('Test AUC:', auc)


