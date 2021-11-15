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
    'adityarathore',
    'Alex',
    'Amy_Zhang',
    'Anarghya',
    'aniruddh',
    'anthony',
    'baron_huang',
    'bhuiyan',
    'Eric',
    'chandler',
    'chenyi_zou',
    'deepak_joseph',
    'dunjiong_lin',
    # 'Eric_Kim',
    # 'FrankYang',
    # 'giorgi_datashvili',
    # 'Huining_Li',
    # 'jonathan',
    # 'Kunjie_Lin',
    # 'lauren',
    # 'moohliton',
    # 'phoung',
    # 'Tracy_chen'
]
label_index = 0  # indicate which concept to train the model

# ...........................load data...........................................
# load data from saved files, the variable "x_data" stores mmWave data, the variable "y_data" stores labels
# we split 3/4 data as training data, and 1/4 data as testing data
# the variable "x_train" stores mmWave data for training set, the variable "y_train" stores labels for training set
# the variable "x_test" stores mmWave data for testing set, the variable "y_test" stores labels for testing set
x_data = np.zeros((1, 128, 192))
y_data = np.zeros((1,))
for i in range(int(len(users)/4*3), len(users)):
    user = users[i]
    x_data = np.concatenate((x_data, np.load('data/' + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    y_data = np.concatenate((y_data, np.load('data/' + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
x_data = x_data[1:]
y_data = y_data[1:]
print("the length of x_data is {}".format(np.shape(x_data)))
print("the length of y_data is {}".format(np.shape(y_data)))
seq = np.arange(0, len(x_data), 1)
np.random.shuffle(seq)
x_data = x_data[seq[:]]
y_data = y_data[seq[:]]
x_train = x_data[:int(len(x_data)/4*3)]
y_train = y_data[:int(len(y_data)/4*3)]
x_test = x_data[(int(len(x_data)/4*3)+1):]
y_test = y_data[(int(len(y_data)/4*3)+1):]
print("the length of x_train is {}".format(np.shape(x_train)))
print("the length of y_train is {}".format(np.shape(y_train)))
print("the length of x_test is {}".format(np.shape(x_test)))
print("the length of y_test is {}".format(np.shape(y_test)))
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

# Hyper Parameters and initial model parameters
hidden_size = 64
num_layers = 5
num_classes = 2
lr = 0.01           # learning rate
# initial model
model = simpleLSTM(num_input, hidden_size, num_layers, num_classes).to(device)

# load the model
model.load_state_dict(torch.load('LSTM_model'))
print("the model has been successfully loaded!")

fc_dic = {}

for layer in range(num_layers):
    name = "fc" + str(layer)
    # create each fc
    fc = nn.Linear(hidden_size, num_classes).to(device)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(fc.parameters(), lr)
    for i, (data, labels) in enumerate(train_data):
        data = data.to(device)
        outputs, h_n, c_n = model(data.float())

        label = []
        if labels[0].size() != num_classes:
            for j in range(len(labels)):
                label.append(np.eye(num_classes, dtype=float)[int(labels[j])])
        else:
            label = labels
        label = torch.tensor(np.array(label)).to(device)

        optimizer.zero_grad()
        outputs = fc(h_n[layer].float())
        loss = criterion(outputs, label)

        # backward and optimize
        loss.backward()
        optimizer.step()

        if i % 30 == 0:
            print('Step [{}/{}], Loss: {:.4f}'
                   .format(i+1, training_data_count / 128, loss.item()))
    fc_dic[name] = fc

for layer in range(num_layers):
    name = "fc" + str(layer)
    fc = fc_dic[name]
    for parameters in fc.parameters():
        print("parameters of the {} layer is {}".format(layer, parameters))
    # Test the model
    fc.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, label in test_data:
            data = data.to(device)
            outputs, h_n, c_n = model(data.float())
            label = label.to(device)
            label = label.squeeze()
            outputs = fc(h_n[layer].float())
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('Test Accuracy of the model on the {} test mmWave data: {} %'.format(testing_data_count,
                                                                                   100 * correct / total))