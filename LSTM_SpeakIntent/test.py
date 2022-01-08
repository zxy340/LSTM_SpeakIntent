# this file is just used to test the correctness of the code
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from LSTM import simpleLSTM
import data
import os

a = [1, 2, 3, 4, 5, 6]
print(min(sum(a[0:3]), sum(a[3:])))