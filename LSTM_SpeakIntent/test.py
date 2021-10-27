import numpy as np
import torch.nn as nn
import torch

a = np.arange(10, dtype=np.int32)
b = np.eye(6, dtype=np.int32)[a[1]]
print(b)