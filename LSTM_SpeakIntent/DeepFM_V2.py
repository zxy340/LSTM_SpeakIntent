import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten
from tensorflow.keras.layers import Concatenate, dot, Activation, Reshape
from tensorflow.keras.layers import BatchNormalization, concatenate, Dropput, Add
from tensorflow.keras.layers import RepeatVector, Subtract, Lamda, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2 as l2_reg
from tensorflow.keras.regularizers import l1_l2 as l1_l2
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

from tensorflow.keras import backend as K
from tensorflow.keras.layers import layer

import random as rn
import update_sample
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)

tf.keras.backend.set_session(sess)

np.random.seed(1993)
rn.seed(1993)
tf.set_random_seed(1993)

