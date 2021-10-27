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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#download and load data
data.download_data()
x_train, x_test, y_train, y_test = data.load_data()

# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
torch_data = data.GetLoader(x_train, y_train)
train_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)
torch_data = data.GetLoader(x_test, y_test)
test_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)

# Input Data
training_data_count = len(x_train)  # 7352 training series (with 50% overlap between each serie)
testing_data_count = len(x_test)
num_steps = len(x_train[0])  # 128 timesteps per series
num_input = len(x_train[0][0])  # 9 input parameters per timestep

# Hyper Parameters
epochs = 50           # 训练整批数据多少次
hidden_size = 64
num_layers = 1
num_classes = 6
lr = 0.01           # learning rate

model = simpleLSTM(num_input, hidden_size, num_layers, num_classes)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(epochs):
    for i, (data, labels) in enumerate(train_data):
        data = data.reshape(-1, num_steps, num_input).to(device)
        labels = labels.to(device)
        label = []
        if labels[0].size() != num_classes:
            for j in range(len(labels)):
                y_ = np.zeros(num_classes, dtype=np.int32)
                label.append(np.eye(num_classes, dtype=float)[labels[j][0]])
        else:
            label = labels
        label = torch.tensor(np.array(label))

        # forward pass
        outputs = model(data)
        loss = criterion(outputs, label)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, epochs, i+1, training_data_count, loss.item()))

# Test the model
# https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval
# torch.max()用法.https://blog.csdn.net/weixin_43255962/article/details/84402586
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, label in test_data:
        data = data.reshape(-1, num_steps, num_input).to(device)
        label = label.to(device)
        label = label.squeeze()
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print('Test Accuracy of the model on the {} test images: {} %'.format(testing_data_count, 100 * correct / total))