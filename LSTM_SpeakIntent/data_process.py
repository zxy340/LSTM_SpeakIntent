import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from LSTM import simpleLSTM
from data import load_data
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
    'Blink'                  # AU45   # 17
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
current_user_data = users[0]  # the folder "data" has several users
path = '/mnt/stuff/data/' + current_user_data + '/output/'
data_save_path = '/mnt/stuff/xiaoyu/data/'  # the path where 'x_data.npy' and 'y_data.npy' are located

# ........................read and process data...............................
# find if data has been processed and saved in local
# if not, read data from local files and process the data
# after processing, save the data in local
for concept in Concepts:
    print('Current processing concept is {}'.format(concept))
    if not (os.path.exists(data_save_path + current_user_data + '/' + concept + '/x_data.npy') and
        os.path.exists(data_save_path + current_user_data + '/' + concept + '/y_data.npy') and
        os.path.exists(data_save_path + current_user_data + '/' + concept + '/labels.npy') and
        os.path.exists(data_save_path + current_user_data + '/' + concept + '/levels.npy')):
        x_data, y_data, labels, levels = load_data(concept, path)
        if np.all(x_data) == 0:
            print(current_user_data + ' has no data of ' + concept)
        else:
            if not os.path.exists(data_save_path + current_user_data + '/' + concept):
                os.makedirs(data_save_path + current_user_data + '/' + concept)
            np.save(data_save_path + current_user_data + '/' + concept + '/x_data', x_data)
            np.save(data_save_path + current_user_data + '/' + concept + '/y_data', y_data)
            np.save(data_save_path + current_user_data + '/' + concept + '/labels', labels)
            np.save(data_save_path + current_user_data + '/' + concept + '/levels', levels)
            print('Dataset is now located at: ' + data_save_path + current_user_data + '/' + concept + '/')
    else:
        print('{} dataset had already been processed before!'.format(concept))
# ...............................................................................