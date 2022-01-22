# this file is just used to test the correctness of the code
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from LSTM import simpleLSTM
import data
import os

y_pred = np.array([[1, 0], [2, 3], [4, 5], [6, 7], [8, 9]])
y_pred = [list(x).index(max(x)) for x in y_pred]
print(y_pred)