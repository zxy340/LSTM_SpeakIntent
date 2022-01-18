import numpy as np
import torch
from data_loader import data_loading_concept, data_loading_speak
from main_function import feature_extrac, speak_detection
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Concepts = [
    'Inner_Brow_Raiser',     # AU01   # 00
    'Outer_Brow_Raiser',     # AU02   # 01
    'Brow_Lowerer',          # AU04   # 02
    'Upper_Lid_Raiser',      # AU05   # 03
    'Cheek_Raiser',          # AU06   # 04
    'Lid_Tightener',         # AU07   # 05
    'Nose_Wrinkler',         # AU09   # 06
    'Upper_Lip_Raiser',      # AU10   # 07
    'Lip_Corner_Puller',     # AU12   # 08
    'Dimpler',               # AU14   # 09
    'Lip_Corner_Depressor',  # AU15   # 10
    'Chin_Raiser',           # AU17   # 11
    'Lip_stretcher',         # AU20   # 12
    'Lip_Tightener',         # AU23   # 13
    'Lips_part',             # AU25   # 14
    'Jaw_Drop',              # AU26   # 15
    'Lip_Suck',              # AU28   # 16
    # 'Blink'                  # AU45   # 17
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
    # 'Tracy_chen'          # 21
]

if __name__ == '__main__':
    data_path = '/mnt/stuff/xiaoyu/data/'  # the path where 'x_data.npy' and 'y_data.npy' are located
    model_type = 'LSTM/'  # the type of the sub_model

    features = []  # the index of feature in target layer for all concepts
    layers = []  # the index of target layer for all concepts
    # features = np.zeros(len(Concepts))  # the index of feature in target layer for all concepts
    # features = features.astype(int)
    # layers = np.zeros(len(Concepts))  # the index of target layer for all concepts
    # layers = layers.astype(int)
    # train all the sub_models for all the concept respectively
    for label_index in range(len(Concepts)):
        x_train, y_train, x_test, y_test = data_loading_concept(label_index, data_path)
        feature_index, max_layer = feature_extrac(x_train, y_train, x_test, y_test, label_index, model_type)
        features.append(feature_index)
        layers.append(max_layer)

    # since the time cost of the function data_load_speak is too high, we save the outcome into the directory "data"
    if not os.path.exists('data/x_train.npy'):
        x_train, y_train, label_train, level_train, x_test, y_test, label_test, level_test = data_loading_speak(data_path)
        np.save('data/x_train.npy', x_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/label_train.npy', label_train)
        np.save('data/level_train.npy', level_train)
        np.save('data/x_test.npy', x_test)
        np.save('data/y_test.npy', y_test)
        np.save('data/label_test.npy', label_test)
        np.save('data/level_test.npy', level_test)

    # load the data for speak intent model training and testing
    x_train = np.load('data/x_train.npy')
    y_train = np.load('data/y_train.npy')
    label_train = np.load('data/label_train.npy')
    level_train = np.load('data/level_train.npy')
    x_test = np.load('data/x_test.npy')
    y_test = np.load('data/y_test.npy')
    label_test = np.load('data/label_test.npy')
    level_test = np.load('data/level_test.npy')

    # get data with effective labels
    yes_concept = np.argwhere(label_train == 1)  # frames that all its chirps are this concept
    no_concept = np.argwhere(label_train == 0)  # frames that all its chirps are not this concept
    print('the shape of the data with speak intent label for training is {}'.format(np.shape(yes_concept)))
    print('the shape of the data without speak intent label for training is {}'.format(np.shape(no_concept)))
    # get the min frame number of yes_concept and no_concept and choose the same number of the other
    n_sample = min(np.shape(yes_concept)[0], np.shape(no_concept)[0])
    if n_sample == 0:  # if the availabel data is 0, then return 0 to indicate no data
        print("We don't have data for speak intent training and testing!")
    if n_sample > 35000:  # since the memory is not big enough, we should restrict the number of one label data within 35000 samples
        n_sample = 35000
    seq_train = np.concatenate([yes_concept[:n_sample], no_concept[:n_sample]], axis=0)  # concatenate yes_concept and no_concept sequences
    np.random.shuffle(seq_train)  # randomly shuffle the sequences
    # x_train = x_train[seq_train[:int(len(seq_train)/9*8)].squeeze()]
    # y_train = y_train[seq_train[:int(len(seq_train)/9*8)].squeeze()]
    # label_train = label_train[seq_train[:int(len(seq_train)/9*8)].squeeze()]
    # level_train = level_train[seq_train[:int(len(seq_train)/9*8)].squeeze()]

    # get data with effective labels
    yes_concept = np.argwhere(label_test == 1)  # frames that all its chirps are this concept
    no_concept = np.argwhere(label_test == 0)  # frames that all its chirps are not this concept
    print('the shape of the data with speak intent label for testing is {}'.format(np.shape(yes_concept)))
    print('the shape of the data without speak intent label for testing is {}'.format(np.shape(no_concept)))
    # get the min frame number of yes_concept and no_concept and choose the same number of the other
    n_sample = min(np.shape(yes_concept)[0], np.shape(no_concept)[0])
    if n_sample == 0:  # if the availabel data is 0, then return 0 to indicate no data
        print("We don't have data for speak intent training and testing!")
    if n_sample > 35000:  # since the memory is not big enough, we should restrict the number of one label data within 35000 samples
        n_sample = 35000
    seq_test = np.concatenate([yes_concept[:n_sample], no_concept[:n_sample]], axis=0)  # concatenate yes_concept and no_concept sequences
    np.random.shuffle(seq_test)  # randomly shuffle the sequences
    x_test = np.concatenate([x_train[seq_train[int(len(seq_train)/9*8):].squeeze()], x_test[seq_test[:].squeeze()]], axis=0)
    y_test = np.concatenate([y_train[seq_train[int(len(seq_train)/9*8):].squeeze()], y_test[seq_test[:].squeeze()]], axis=0)
    label_test = np.concatenate([label_train[seq_train[int(len(seq_train)/9*8):].squeeze()], label_test[seq_test[:].squeeze()]], axis=0)
    level_test = np.concatenate([level_train[seq_train[int(len(seq_train)/9*8):].squeeze()], level_test[seq_test[:].squeeze()]], axis=0)
    x_train = x_train[seq_train[:int(len(seq_train)/9*8)].squeeze()]
    y_train = y_train[seq_train[:int(len(seq_train)/9*8)].squeeze()]
    label_train = label_train[seq_train[:int(len(seq_train)/9*8)].squeeze()]
    level_train = level_train[seq_train[:int(len(seq_train)/9*8)].squeeze()]

    if len(x_train > 0) & len(x_test) > 0:
        # speak intent model training and testing
        speak_detection(x_train, y_train, label_train, level_train, x_test, y_test, label_test, level_test, features, layers, model_type)