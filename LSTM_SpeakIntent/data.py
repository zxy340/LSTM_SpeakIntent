import numpy as np
import torch
from urllib.request import urlretrieve
import zipfile
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

DATASET_PATH = "UCI HAR Dataset/"
TRAIN = "train/"
TEST = "test/"

def un_zip(file_name):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name+'.zip')
    if os.path.isdir(file_name):
        pass
    else:
        os.mkdir(file_name)
    for names in zip_file.namelist():
        zip_file.extract(names,file_name)
    zip_file.close()

def download_data():
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    filename = 'UCI HAR Dataset'
    print("Downloading...")
    if not os.path.exists("UCI HAR Dataset.zip"):
        urlretrieve(URL, filename + '.zip')
        un_zip(filename)
        print("Downloading done.\n")
    else:
        print("Dataset already downloaded. Did not download twice.\n")

def load_X(X_signals_paths):
    X_signals = []
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                line.replace('  ', ' ').strip().split(' ') for line in file
             ]]
        )
        file.close()
    return np.transpose(np.array(X_signals), (1, 2, 0))

def load_Y(y_path):
    file = open(y_path, 'r')
    y = np.array(
        [elem for elem in [
            line.replace('  ', ' ').strip().split() for line in file
        ]],
        dtype=np.int32
    )
    file.close()
    return y - 1

def load_data():
    x_train_signals_paths = [DATASET_PATH + "UCI HAR Dataset/" + TRAIN + "Inertial Signals/"
                             + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
    x_test_signals_paths = [DATASET_PATH + "UCI HAR Dataset/" + TEST + "Inertial Signals/"
                             + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]
    x_train = load_X(x_train_signals_paths)
    x_test = load_X(x_test_signals_paths)

    y_train_path = DATASET_PATH + "UCI HAR Dataset/" + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + "UCI HAR Dataset/" + TEST + "y_test.txt"
    y_train = load_Y(y_train_path)
    y_test = load_Y(y_test_path)

    return x_train, x_test, y_train, y_test

class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)