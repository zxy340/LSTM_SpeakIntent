# this file is just used to test the correctness of the code
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from LSTM import simpleLSTM
import data
import os

a = np.tile(np.arange(1, 129, 1, dtype=int), (1000, 1))
print(type(a))