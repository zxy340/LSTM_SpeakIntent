import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from TDNN import simpleTDNN
import data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Concepts = [
    'Inner_Brow_Raiser',     # AU01   # 01
    'Outer_Brow_Raiser',     # AU02   # 02
    'Brow_Lowerer',          # AU04   # 03
    'Upper_Lid_Raiser',      # AU05   # 04
    'Cheek_Raiser',          # AU06   # 05
    'Lid_Tightener',         # AU07   # 06
    'Nose_Wrinkler',         # AU09   # 07
    'Upper_Lip_Raiser',      # AU10   # 08
    'Lip_Corner_Puller',     # AU12   # 09
    'Dimpler',               # AU14   # 10
    'Lip_Corner_Depressor',  # AU15   # 11
    'Chin_Raiser',           # AU17   # 12
    'Lip_stretcher',         # AU20   # 13
    'Lip_Tightener',         # AU23   # 14
    'Lips_part',             # AU25   # 15
    'Jaw_Drop',              # AU26   # 16
    'Lip_Suck',              # AU28   # 17
    'Blink'                  # AU45   # 18
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
label_index = 7  # indicate which concept to train the model
data_path = '/mnt/stuff/xiaoyu/data/'  # the path where 'x_data.npy' and 'y_data.npy' are located
model_type = 'TDNN/'

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
    print('Current added user is {}'.format(user))
    if not os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy'):
        continue
    x_train = np.concatenate((x_train, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    y_train = np.concatenate((y_train, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
for i in range(int(len(users)/16*9), int(len(users)/4*3)):
    user = users[i]
    print('Current added user is {}'.format(user))
    if not os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy'):
        continue
    x_test = np.concatenate((x_test, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    y_test = np.concatenate((y_test, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
x_train = x_train[1:]
y_train = y_train[1:]
x_test = x_test[1:]
y_test = y_test[1:]
seq = np.arange(0, len(x_train), 1)
np.random.shuffle(seq)
x_train = x_train[seq[:30000]]
y_train = y_train[seq[:30000]]
seq = np.arange(0, len(x_test), 1)
np.random.shuffle(seq)
x_test = x_test[seq[:10000]]
y_test = y_test[seq[:10000]]
print("the length of x_train is {}".format(np.shape(x_train)))
print("the length of y_train is {}".format(np.shape(y_train)))
print("the length of x_test is {}".format(np.shape(x_test)))
print("the length of y_test is {}".format(np.shape(y_test)))
# .................................................................................

