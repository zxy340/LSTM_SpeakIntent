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

Concepts = [
    'Blink',  # AU45
    'Brow_Lowerer',  # AU4
    'Cheek_Raiser',  # AU6
    'Inner_Brow_Raiser',  # AU1
    'Jaw_Drop',  # AU26
    'Lid_Tightener',  # AU7
    'Lip_Corner_Depressor',  # AU15
    'Lip_Corner_Puller',  # AU 12
    'Lip_stretcher',  # AU20
    'Lip_Suck',  # AU28
    'Lip_Tightener',  # AU23
    'Nose_Wrinkler',  # AU9
    'Outer_Brow_Raiser',  # AU2
    'Upper_Lid_Raiser',  # AU5
    'Upper_Lip_Raiser',  # AU10
    'Chin_Raiser',  # AU17
    'Dimpler',  # AU14
    'Lips_part'  # AU25
]
# Concepts = [
#     'Inner_Brow_Raiser',     # AU01
#     'Outer_Brow_Raiser',     # AU02
#     'Brow_Lowerer',          # AU04
#     'Upper_Lid_Raiser',      # AU05
#     'Cheek_Raiser',          # AU06
#     'Lid_Tightener',         # AU07
#     'Nose_Wrinkler',         # AU09
#     'Upper_Lip_Raiser',      # AU10
#     'Lip_Corner_Puller',     # AU12
#     'Dimpler',               # AU14
#     'Lip_Corner_Depressor',  # AU15
#     'Chin_Raiser',           # AU17
#     'Lip_stretcher',         # AU20
#     'Lip_Tightener',         # AU23
#     'Lips_part'              # AU25
#     'Jaw_Drop',              # AU26
#     'Lip_Suck',              # AU28
#     'Blink',                 # AU45
# ]
PATH = "LSTM_model"  # the stored model parameter
path = '/home/xiaoyu/blink_mmwave/'  # the stored mmWave data and labels
label_index = 12  # indicate which concept to train the model

# ........................read and process data...............................
# find if data has been processed and saved in local
# if not, read data from local files and process the data
# after processing, save the data in local
os.chdir('data/' + Concepts[label_index] + '/')
if not os.path.exists('x_data.npy'):
    os.chdir('..')
    os.chdir('..')
    x_data, y_data = data.load_data(Concepts[label_index], path)
    os.chdir('data/' + Concepts[label_index] + '/')
    np.save('x_data', x_data)
    np.save('y_data', y_data)
    print('Dataset is now located at: ' + 'data/' + Concepts[label_index] + '/')
# ...............................................................................

# ...........................load data...........................................
# load data from saved files, the variable "x_data" stores mmWave data, the variable "y_data" stores labels
# we split 3/4 data as training data, and 1/4 data as testing data
# the variable "x_train" stores mmWave data for training set, the variable "y_train" stores labels for training set
# the variable "x_test" stores mmWave data for testing set, the variable "y_test" stores labels for testing set
x_data = np.load('x_data.npy')
y_data = np.load('y_data.npy')
print(np.shape(x_data))
print(np.shape(y_data))
os.chdir('..')
os.chdir('..')
x_train = x_data[:int(len(x_data)/4*3)]
y_train = y_data[:int(len(y_data)/4*3)]
x_test = x_data[(int(len(x_data)/4*3)+1):]
y_test = y_data[(int(len(y_data)/4*3)+1):]
print(np.shape(x_train))
print(np.shape(y_train))
# .................................................................................

# .............basic information of training and testing set.......................
training_data_count = len(x_train)  # number of training series
testing_data_count = len(x_test)
num_steps = len(x_train[0])  # timesteps per series
num_input = len(x_train[0][0])  # input parameters per timestep
# ..................................................................................

# ..................................................................................
# use GetLoader to load the data and return Dataset object, which contains data and labels
torch_data = data.GetLoader(x_train, y_train)
train_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)
torch_data = data.GetLoader(x_test, y_test)
test_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)
# ....................................................................................

# .............Hyper Parameters and initial model parameters..........................
epochs = 20
hidden_size = 64
num_layers = 5
num_classes = 2
lr = 0.01           # learning rate
# initial model
model = simpleLSTM(num_input, hidden_size, num_layers, num_classes).to(device)
# loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)
# .....................................................................................

# ...........................train and store the model.................................
# train the model
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
# store the model
torch.save(model.state_dict(), PATH)
# ........................................................................................

# ............................load and test the trained model.............................
# load the model
if os.path.exists("LSTM_model"):
    model.load_state_dict(torch.load(PATH))
    print("the model has been successfully loaded!")
else:
    print("the local doesn't store the model for testing, please train the model!")
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
# ..................................................................................................