# Reference code link:
# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
# https://youngforever.tech/posts/2020-03-07-lstm%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF/
# https://blog.csdn.net/l8947943/article/details/103733473

import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from LSTM import simpleLSTM
import data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Concepts = [
    'Inner_Brow_Raiser',     # AU01
    'Outer_Brow_Raiser',     # AU02
    'Brow_Lowerer',          # AU04
    'Upper_Lid_Raiser',      # AU05
    'Cheek_Raiser',          # AU06
    'Lid_Tightener',         # AU07
    'Nose_Wrinkler',         # AU09
    'Upper_Lip_Raiser',      # AU10
    'Lip_Corner_Puller',     # AU12
    'Dimpler',               # AU14
    'Lip_Corner_Depressor',  # AU15
    'Chin_Raiser',           # AU17
    'Lip_stretcher',         # AU20
    'Lip_Tightener',         # AU23
    'Lips_part',             # AU25
    'Jaw_Drop',              # AU26
    'Lip_Suck',              # AU28
    'Blink'                  # AU45
]
users = [
    'adityarathore',      # 00
    'Caitlin_Chan',       # 01
    'Amy_Zhang',          # 02
    'Anarghya',           # 03
    'aniruddh',           # 04
    'anthony',            # 05
    'baron_huang',        # 06
    'bhuiyan',            # 07
    'chandler',           # 08
    'chenyi_zou',         # 09
    'deepak_joseph',      # 10
    'dunjiong_lin',       # 11
    'Eric_Kim',           # 12
    'FrankYang',          # 13
    'giorgi_datashvili',  # 14
    'Huining_Li',         # 15
    'jonathan',           # 16
    'Kunjie_Lin',         # 17
    'lauren',             # 18
    'moohliton',          # 19
    'phoung',             # 20
    'Tracy_chen'          # 21
]
label_index = 0  # indicate which concept to train the model
data_path = '/mnt/stuff/xiaoyu/data/'  # the path where 'x_data.npy' and 'y_data.npy' are located
model_type = 'LSTM/'

# ...........................load data...........................................
# load data from saved files, the variable "x_data" stores mmWave data, the variable "y_data" stores labels
# we split 3/4 data as training data, and 1/4 data as testing data
# the variable "x_train" stores mmWave data for training set, the variable "y_train" stores labels for training set
# the variable "x_test" stores mmWave data for testing set, the variable "y_test" stores labels for testing set
x_train = np.zeros((1, 128, 192))
y_train = np.zeros((1,))
x_test = np.zeros((1, 128, 192))
y_test = np.zeros((1,))
for i in range(int(len(users)/16*9)):
    user = users[i]
    x_train = np.concatenate((x_train, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    y_train = np.concatenate((y_train, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
for i in range(int(len(users)/16*9), int(len(users)/4*3)):
    user = users[i]
    x_test = np.concatenate((x_test, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    y_test = np.concatenate((y_test, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
x_train = x_train[1:]
y_train = y_train[1:]
x_test = x_test[1:]
y_test = y_test[1:]
print("the length of x_train is {}".format(np.shape(x_train)))
print("the length of y_train is {}".format(np.shape(y_train)))
print("the length of x_test is {}".format(np.shape(x_test)))
print("the length of y_test is {}".format(np.shape(y_test)))
seq = np.arange(0, len(x_train), 1)
np.random.shuffle(seq)
x_train = x_train[seq[:]]
y_train = y_train[seq[:]]
seq = np.arange(0, len(x_test), 1)
np.random.shuffle(seq)
x_test = x_test[seq[:]]
y_test = y_test[seq[:]]
print("the length of x_train is {}".format(np.shape(x_train)))
print("the length of y_train is {}".format(np.shape(y_train)))
print("the length of x_test is {}".format(np.shape(x_test)))
print("the length of y_test is {}".format(np.shape(y_test)))
# .................................................................................

# .............basic information of training and testing set.......................
training_data_count = len(x_train)  # number of training series
testing_data_count = len(x_test)
num_steps = len(x_test[0])  # timesteps per series
num_input = len(x_test[0][0])  # input parameters per timestep
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
    print('Train C of the model on the {} train mmWave data: {}'.format(training_data_count, C))
# store the model
if not os.path.exists('model/' + model_type + Concepts[label_index]):
    os.makedirs('model/' + model_type + Concepts[label_index])
torch.save(model.state_dict(), 'model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model')
# ........................................................................................

# ............................load and test the trained model.............................
# load the model
model.load_state_dict(torch.load('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'))
print("the model has been successfully loaded!")
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
        data = data.to(device)
        label = label.to(device)
        label = label.squeeze()
        outputs = model(data.float())
        _, predicted = torch.max(outputs, 1)
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

    print('TP = {}, TN = {}, FN = {}, FP = {}'.format(TP, TN, FN, FP))
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    print('Test F1 score of the model on the {} test mmWave data: {}'.format(testing_data_count, F1))
    print('Test C of the model on the {} test mmWave data: {}'.format(testing_data_count, C))
    print('Test Accuracy of the model on the {} test mmWave data: {} %'.format(testing_data_count, 100 * correct / total))
# ..................................................................................................