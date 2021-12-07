import numpy as np
import torch
from os.path import getsize
import matplotlib.pyplot as plt
import os

def awr1642_lazy(dfile, num_frame, num_chirp, num_channel=4, num_sample=256):
    """ Lazy load adc data

    Parameters
    ----------
    dfile : str
        an reorganized adc_data file
    num_frame : int
    num_chirp : int
    num_sample : int

    Returns
    -------
    np.memmap
        the I, Q data of each sample are not reordered
    """
    return np.memmap(dfile, mode='r', dtype=np.int16, shape=(num_frame, num_chirp, num_channel, num_sample * 2))


def reorder_IQ(d):
    """ reorder the IIQQ format of data to I+1j*Q, the returned data is in memory
    Parameters
    ----------
    d : np.memmap
        the data loaded by awr1642_lazy

    Returns
    -------
    ndarray
        (num_frame, num_chirp, num_channel, num_sample)
    """
    if len(d.shape) == 3:
        num_chirp, num_channel, num_IQsample = d.shape
        d = d.reshape((num_chirp, num_channel, -1, 4))
        d = d[..., :2] + 1j * d[..., 2:]
        d = d.reshape((num_chirp, num_channel, -1))
    elif len(d.shape) == 4:
        num_frame, num_chirp, num_channel, num_IQsample = d.shape
        d = d.reshape((num_frame, num_chirp, num_channel, -1, 4))
        d = d[..., :2] + 1j * d[..., 2:]
        d = d.reshape((num_frame, num_chirp, num_channel, -1))
    return d


def range_profile(d):
    """ Calculate range profiles for d

    Parameters
    ----------
    d : ndarray
        the IF signals that calculation happens. It assumes last axis is the adc data, i.e., (..., adc_data)

    Returns
    -------
    ndarray
        the calculated range profiles, i.e., (..., range_profiles)
    """
    w = np.hanning(d.shape[-1])
    corr = 20 * np.log10(2 ** 15) + 20 * np.log10(w.sum()) - 20 * np.log10(np.sqrt(2))
    range_fft = np.apply_along_axis(lambda x: np.fft.fft(w * x), -1, d)
    fft_amp = np.abs(range_fft)
    fft_amp = 20 * np.log10(fft_amp) - corr  # dBFs, plus 10 if dBm
    return fft_amp, range_fft


def steering_vector(angle, num_channel):
    """ Get steering vector for each Rx. It takes the counter-clock wise as positive angle. The distance between two Rx is assumed to be wavelength. Rx0 is the most left Rx

    Parameters
    ----------
    angle : int
        the angle of steering, in degree
    num_channel : int
        the number of Rx

    Returns
    -------
    ndarray
        sterring vector
    """
    angle = angle * np.pi / 180
    omega = 2 * np.pi * np.sin(angle)
    sv = np.arange(num_channel) * omega * -1j
    sv = np.exp(sv)
    return sv

def process(signal):
    """
    mmWave Signal processing

    Parameters
    ----------
    signal : ndarray(shape=(C, 4, 256))
        the mmWave signal to be processed, where C is the number of chirps (fast time)

    Returns
    -------
    ndarray(shape=(C, 3, 64))
        processed signal
    """
    sv_0 = steering_vector(45, 4)
    sv_1 = steering_vector(-45, 4)
    sv_2 = steering_vector(0, 4)
    r0, f0 = range_profile(np.einsum('j, hjk', sv_0, signal))
    r1, f1 = range_profile(np.einsum('j, hjk', sv_1, signal))
    r2, f2 = range_profile(np.einsum('j, hjk', sv_2, signal))
    return np.concatenate((r0[:, np.newaxis, :64], r1[:, np.newaxis, :64], r2[:, np.newaxis, :64]), axis=1)


def process_ex(signal):
    """
    mmWave Signal processing extra version

    Parameters
    ----------
    signal : ndarray(shape=(C, 4, 256))
        the mmWave signal to be processed, where C is the number of chirps (fast time)

    Returns
    -------
    ndarray(shape=(C, 3, 64))
        processed signal
    """
    sv_0 = steering_vector(45, 4)
    sv_1 = steering_vector(-45, 4)
    sv_2 = steering_vector(0, 4)
    r0, f0 = range_profile(np.einsum('j, hjk', sv_0, signal))
    r1, f1 = range_profile(np.einsum('j, hjk', sv_1, signal))
    r2, f2 = range_profile(np.einsum('j, hjk', sv_2, signal))
    f0 = np.unwrap(np.angle(f0))
    f1 = np.unwrap(np.angle(f1))
    f2 = np.unwrap(np.angle(f2))
    return np.concatenate((f0[:, np.newaxis, :64], f1[:, np.newaxis, :64], f2[:, np.newaxis, :64]), axis=1)


def load_data(concept, Path):
    """
    load mmWave data from the path

    Parameters
    ----------
    concept : string
        the name of the label to be classified
    Path : string
        the path of mmWave data and concept label
    Returns
    -------
    X : (n_sample * 2, 128, 3 * 64)
        array of the mmWave data, n_sample is the frame number of label 1, which indicates the data has the concept action,
        n_sample * 2 is because we control the label 0 and label 1 has the same number for training and testing,
        128 is the number of chirps of one frame, 3 * 64 is the data size of one chirp.
    Y : (n_sample * 2, )
        array of the labels
    """
    n_channle = 4
    n_adc = 256
    n_chirp = 128
    n_frame = int(getsize(Path + 'mmwave.bin') / 2 / n_chirp / n_adc / n_channle / 2)  # get the number of frames of all the mmWave data
    data = awr1642_lazy(Path + 'mmwave.bin', n_frame, 128, n_channle, n_adc)  # get all the mmWave data
    if not os.path.exists(Path + concept + '.npz'):  # if the concept of the user is not existed, return 0 to indicate no data
        return 0, 0, 0, 0
    label = np.load(Path + concept + '.npz')['label'][:, :128]  # get all the concept labels
    mask = np.load(Path + concept + '.npz')['mask'][:, :128]  # get all the masks, if mask is "False", this mmWave data is not available
    intent_type = np.load(Path + 'intent.npz')['intent_type1'][:, :128]  # get all the speak intent labels
    intent_level = np.load(Path + 'intent.npz')['intent_level'][:, :128]  # get all the speak intent labels

    # delete mmWave data which is all 0
    delete = []
    for i in range(len(data)):
        for j in range(128):
            if np.all(data[i][j] == 0):
                delete.append(i)
    data = np.delete(data, delete, axis=0)
    label = np.delete(label, delete, axis=0)
    mask = np.delete(mask, delete, axis=0)
    intent_type = np.delete(intent_type, delete, axis=0)
    intent_level = np.delete(intent_level, delete, axis=0)

    # get data with effective labels
    yes_concept = np.argwhere(((label == 1).sum(axis=1)) == 128)  # frames that all its chirps are this concept
    no_concept = np.argwhere(((label == 0).sum(axis=1)) == 128)  # frames that all its chirps are not this concept
    print(np.shape(yes_concept))
    print(np.shape(no_concept))

    # get the min frame number of yes_concept and no_concept and choose the same number of the other
    n_sample = min(np.shape(yes_concept)[0], np.shape(no_concept)[0])
    if n_sample == 0:  # if the availabel data is 0, then return 0 to indicate no data
        return 0, 0, 0, 0
    if n_sample > 35000:  # since the memory is not big enough, we should restrict the number of one label data within 35000 samples
        n_sample = 35000
    seq = np.concatenate([yes_concept[:n_sample], no_concept[:n_sample]], axis=0)  # concatenate yes_concept and no_concept sequences
    print(np.shape(seq))
    np.random.shuffle(seq)  # randomly shuffle the sequences
    print(np.shape(seq))

    # process mmWave data: reorder_IQ->process_ex, and then concatenate them together
    samples = reorder_IQ(data[seq[:]].squeeze())
    print(np.shape(samples))
    temp = process_ex(samples[0].reshape(-1, 4, 256)).reshape(1, 128, -1)
    X = np.zeros((np.shape(samples)[0], np.shape(samples)[1], np.shape(temp)[2]), dtype=np.float)
    for i in range(len(samples)):
        X[i] = process_ex(samples[i].reshape(-1, 4, 256)).reshape(1, 128, -1)
        if i % 1000 == 0:
            print(i)
    print('the size of the processed mmWave data is {}'.format(X.shape))
    print(np.shape(label))
    # get the corresponding labels of mmWave data
    Y = label[seq[:]][:, 0, 0]
    print('the size of the concept label is {}'.format(Y.shape))
    Z1 = intent_type[seq[:]][:, 0, 0]
    print('the size of the intent_type label is {}'.format(Z1.shape))
    Z2 = intent_level[seq[:]][:, 0, 0]
    print('the size of the intent_type label is {}'.format(Z2.shape))

    return X, Y, Z1, Z2

class GetLoader(torch.utils.data.Dataset):
    # initial function, get the data and label
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index is divided based on the batchsize, finally return the data and corresponding labels
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        if labels == "True":
            labels = 1
        elif labels == "False":
            labels = 0
        return data, labels
    # for DataLoader better dividing the data, we use this function to return the length of the data
    def __len__(self):
        return len(self.data)