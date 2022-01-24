# this file is just used to test the correctness of the code
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from LSTM import simpleLSTM
import data
import os
current_user_labeldata = np.array(['True', 'False', 'True', 'False'])
print(current_user_labeldata)
yes_concept = np.argwhere(current_user_labeldata == 'True')
print(yes_concept)