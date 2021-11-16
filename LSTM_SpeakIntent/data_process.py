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
    'adityarathore',      #00
    'Alex',               #01
    'Amy_Zhang',          #02
    'Anarghya',           #03
    'aniruddh',           #04
    'anthony',            #05
    'baron_huang',        #06
    'bhuiyan',            #07
    'Eric',               #08
    'chandler',           #09
    'chenyi_zou',         #10
    'deepak_joseph',      #11
    'dunjiong_lin',       #12
    'Eric_Kim',           #13
    'FrankYang',          #14
    'giorgi_datashvili',  #15
    'Huining_Li',         #16
    'jonathan',           #17
    'Kunjie_Lin',         #18
    'lauren',             #19
    'moohliton',          #20
    'phoung',             #21
    'Tracy_chen'          #22
]
current_user_data = users[0]  # the folder "data" has several users
path = '/mnt/stuff/data/' + current_user_data + '/output/'

# ........................read and process data...............................
# find if data has been processed and saved in local
# if not, read data from local files and process the data
# after processing, save the data in local
for concept in Concepts:
    if not os.path.exists('data/' + current_user_data + '/' + concept + '/x_data.npy'):
        x_data, y_data = data.load_data(concept, path)
        if x_data == 0:
            print(current_user_data + ' has no data of ' + concept)
        else:
            if not os.path.exists('data/' + current_user_data + '/' + concept):
                os.makedirs('data/' + current_user_data + '/' + concept)
            np.save('data/' + current_user_data + '/' + concept + '/x_data', x_data)
            np.save('data/' + current_user_data + '/' + concept + '/y_data', y_data)
            print('Dataset is now located at: ' + 'data/' + current_user_data + '/' + concept + '/')
# ...............................................................................