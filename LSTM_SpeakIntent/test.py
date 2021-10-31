import numpy as np
import torch.nn as nn
import torch

a = [[1, 2], [3, 4], [5, 6], [7, 8]]
print(np.shape(a))
b = []
for i in range(10):
    b.append(a)
b = np.transpose(b, (2, 1, 0))
print(np.shape(b))