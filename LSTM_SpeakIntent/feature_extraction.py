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
    'Eric_Kim',
    'FrankYang',
    'giorgi_datashvili',
    'Huining_Li',
    'jonathan',
    'Kunjie_Lin',
    'lauren',
    'moohliton',
    'phoung',
    'Tracy_chen'
]
label_index = 0  # indicate which concept to train the model
model_type = 'LSTM/'

# ...........................load data...........................................
# load data from saved files, the variable "x_data" stores mmWave data, the variable "y_data" stores labels
# since former 3/4 users' data have been used to train and test sub_model, we use the left 1/4 users' data to extract features
# based on the left 1/4 users' data, we split 3/4 users' data as training data, and 1/4 users' data as testing data
# the variable "x_train" stores mmWave data for training set, the variable "y_train" stores labels for training set
# the variable "x_test" stores mmWave data for testing set, the variable "y_test" stores labels for testing set
x_train = np.zeros((1, 128, 192))
y_train = np.zeros((1,))
x_test = np.zeros((1, 128, 192))
y_test = np.zeros((1,))
for i in range(int(len(users)/4*3), int(len(users)-len(users)/16)):
    user = users[i]
    x_train = np.concatenate((x_train, np.load('data/' + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    y_train = np.concatenate((y_train, np.load('data/' + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
for i in range(int(len(users)-len(users)/16), len(users)):
    user = users[i]
    x_test = np.concatenate((x_test, np.load('data/' + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    y_test = np.concatenate((y_test, np.load('data/' + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
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
hidden_size = 64
num_layers = 5
num_classes = 2
lr = 0.01           # learning rate
# initial model
model = simpleLSTM(num_input, hidden_size, num_layers, num_classes).to(device)
# .....................................................................................

# ............................load the trained model...................................
# load the model
model.load_state_dict(torch.load('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'))
print("the model has been successfully loaded!")
# .....................................................................................

# ............................full-connected layer training............................
# we first get the number of layers in the loaded model, and train fc for each layer of the model
# then we store each layer of the fc model into the dictionary named "fc_dic"
# for example, as for 2 layers LSTM model, the structure of "fc_dic" is as follows:
# layer1: array(64*2)
# layer2: array(64*2)
# where 64 is the number of hidden parameters in each LSTM layer, 2 is the number of classes of labels
fc_dic = {}  # record the parameters of each layer fc
for layer in range(num_layers):
    name = "fc" + str(layer)
    # create each fc
    fc = nn.Linear(hidden_size, num_classes).to(device)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(fc.parameters(), lr)
    for i, (data, labels) in enumerate(train_data):
        data = data.to(device)
        output_unused, h_n, c_n = model(data.float())

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
# ...................................................................................

# ............................full-connected layer testing...........................
# we test the accuracy of each fc after the corresponding output from the loaded layer
# todo: then we find the target layer with the maximum accuracy and further extract the features
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
            output_unused, h_n, c_n = model(data.float())
            label = label.to(device)
            label = label.squeeze()
            outputs = fc(h_n[layer].float())
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('Test Accuracy of the model on the {} test mmWave data: {} %'.format(testing_data_count,
                                                                                   100 * correct / total))
# ...................................................................................