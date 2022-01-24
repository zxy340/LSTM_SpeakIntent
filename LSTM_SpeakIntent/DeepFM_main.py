# Reference code link:
# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
# https://youngforever.tech/posts/2020-03-07-lstm%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF/
# https://blog.csdn.net/l8947943/article/details/103733473

import torch
import numpy as np
from data_loader import data_loading_LSTM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from DeepFM import DeepFM
from sklearn.metrics import roc_auc_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Concepts = [
    'Inner_Brow_Raiser',     # AU01   # 00
    # 'Outer_Brow_Raiser',     # AU02   # 01
    # 'Brow_Lowerer',          # AU04   # 02
    # 'Upper_Lid_Raiser',      # AU05   # 03
    # 'Cheek_Raiser',          # AU06   # 04
    # 'Lid_Tightener',         # AU07   # 05
    # 'Nose_Wrinkler',         # AU09   # 06
    # 'Upper_Lip_Raiser',      # AU10   # 07
    # 'Lip_Corner_Puller',     # AU12   # 08
    # 'Dimpler',               # AU14   # 09
    # 'Lip_Corner_Depressor',  # AU15   # 10
    # 'Chin_Raiser',           # AU17   # 11
    # 'Lip_stretcher',         # AU20   # 12
    # 'Lip_Tightener',         # AU23   # 13
    # 'Lips_part',             # AU25   # 14
    # 'Jaw_Drop',              # AU26   # 15
    # 'Lip_Suck',              # AU28   # 16
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
    'Tracy_chen'          # 21
]
data_path = '/mnt/stuff/xiaoyu/data/'  # the path where 'x_data.npy' and 'y_data.npy' are located
model_type = 'DeepFM/'

# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": 2017
}

# train the LSTM model for each concept
for label_index in range(len(Concepts)):
    # load data
    x_train, y_train, x_test, y_test = data_loading_LSTM(label_index, data_path)
    # for frame in range(len(x_train)):
    #     if y_train[frame] == 0:
    #         continue
    #     fig = plt.figure(figsize=(16,16))
    #     for i in range(16):
    #         ax = fig.add_subplot(4, 4, i+1)
    #         plt.plot(x_train[frame, i, :])
    #         ax.set_ylabel("value")
    #         ax.set_xlabel("range")
    #         plt.title("chirp=" + str(i) + " and frame_index=" + str(frame) + " and label=" + str(y_train[frame]))
    #         plt.tight_layout()
    #     plt.show()
    print('the shape of x_train, y_train, x_test, y_test are {}, {}, {}, {} respectively'.format(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)))
    # if the current concept doesn't have data, then jump to the next concept
    if len(x_train) == 0:
        continue

    # prepare training and validation data in the required format
    Xi_train = np.tile(np.arange(1, 129, 1, dtype=int), (len(x_train), 1))
    Xi_valid = np.tile(np.arange(1, 129, 1, dtype=int), (len(x_test), 1))
    Xv_train = np.reshape(x_train, (len(x_train), -1))
    Xv_valid = np.reshape(x_test, (len(x_test), -1))
    y_train = y_train.ravel()
    y_valid = y_test.ravel()

    # init a DeepFM model
    dfm = DeepFM(**dfm_params)

    # fit a DeepFM model
    dfm.fit(Xi_train, Xv_train, y_train)

    # make prediction
    dfm.predict(Xi_valid, Xv_valid)

    # evaluate a trained model
    dfm.evaluate(Xi_valid, Xv_valid, y_valid)