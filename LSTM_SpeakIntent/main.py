# Reference code link:
# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
# https://youngforever.tech/posts/2020-03-07-lstm%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF/
# https://blog.csdn.net/l8947943/article/details/103733473

import numpy as np
import matplotlib
import matplotlib as plt
import torch.nn as nn
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from LSTM import simpleLSTM
import data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#download and load data
if not os.path.exists('x_data.npy'):
    x_data, y_data = data.load_data()
    np.save('x_data', x_data)
    np.save('y_data', y_data)
x_data = np.load('x_data.npy')
y_data = np.load('y_data.npy')
print(np.shape(x_data))
print(np.shape(y_data))
x_train = x_data[:int(len(x_data)/4*3)]
y_train = y_data[:int(len(y_data)/4*3)]
x_test = x_data[(int(len(x_data)/4*3)+1):]
y_test = y_data[(int(len(y_data)/4*3)+1):]
print(np.shape(x_train))
print(np.shape(y_train))
# Input Data
training_data_count = len(x_train)  # number of training series
testing_data_count = len(x_test)
num_steps = len(x_train[0])  # timesteps per series
num_input = 3 * 64  # input parameters per timestep

# use GetLoader to load the data and return Dataset object, which contains data and labels
torch_data = data.GetLoader(x_train, y_train)
train_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)
torch_data = data.GetLoader(x_test, y_test)
test_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)


# Hyper Parameters
epochs = 20
hidden_size = 64
num_layers = 5
num_classes = 2
lr = 0.01           # learning rate

model = simpleLSTM(num_input, hidden_size, num_layers, num_classes).to(device)
# loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(epochs):
    C = np.zeros(2)
    if epoch % 5 == 0:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for i, (data, labels) in enumerate(train_data):
        # data = data.reshape(-1, num_steps, num_input).to(device)
        data = data.to(device)
        label = []
        if labels[0].size() != num_classes:
            for j in range(len(labels)):
                label.append(np.eye(num_classes, dtype=float)[int(labels[j])])
                if labels[j] == 0:
                    C[0] += 1
                else:
                    C[1] += 1
        else:
            label = labels
        label = torch.tensor(np.array(label)).to(device)

        optimizer.zero_grad()
        # forward pass
        outputs = model(data.float())
        loss = criterion(outputs, label)

        # backward and optimize
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, epochs, i+1, training_data_count / 128, loss.item()))
    print(C)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    C = np.zeros((2, 2))
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for data, label in test_data:
        data = data.reshape(-1, num_steps, num_input).to(device)
        label = label.to(device)
        label = label.squeeze()
        outputs = model(data.float())
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        for i in range(len(predicted)):
            if predicted[i] == 0:
                C[0][0] += 1
            else:
                C[0][1] += 1
            if label[i] == 0:
                C[1][0] += 1
            else:
                C[1][1] += 1

        # TP    predict 和 label 同时为1
        TP += ((predicted == 1) & (label == 1)).cpu().sum()
        # TN    predict 和 label 同时为0
        TN += ((predicted == 0) & (label == 0)).cpu().sum()
        # FN    predict 0 label 1
        FN += ((predicted == 0) & (label == 1)).cpu().sum()
        # FP    predict 1 label 0
        FP += ((predicted == 1) & (label == 0)).cpu().sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    print('Test F1 score of the model on the {} test mmWave data: {}'.format(testing_data_count, F1))
    print('Test C of the model on the {} test mmWave data: {}'.format(testing_data_count, C))
    print('Test Accuracy of the model on the {} test mmWave data: {} %'.format(testing_data_count, 100 * correct / total))