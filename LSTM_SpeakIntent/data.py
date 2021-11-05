import numpy as np
import torch
from os.path import getsize


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


def load_data():
    n_channle = 4
    n_adc = 256
    n_chirp = 128
    n_frame = int(getsize("/home/xiaoyu/blink_mmwave/mmwave.bin") / 2 / n_chirp / n_adc / n_channle / 2)
    data = awr1642_lazy("/home/xiaoyu/blink_mmwave/mmwave.bin", n_frame, 128, n_channle, n_adc)
    label = np.load("/home/xiaoyu/blink_mmwave/blink.npz")['label'][:, :128]
    mask = np.load("/home/xiaoyu/blink_mmwave/blink.npz")['mask'][:, :128]
    delete = []
    for i in range(len(data)):
        for j in range(128):
            if np.all(data[i][j] == 0):
                delete.append(i)
    data = np.delete(data, delete, axis=0)
    label = np.delete(label, delete, axis=0)
    mask = np.delete(mask, delete, axis=0)

    # frames that all its chirps are blink
    blink = np.argwhere(((label == 1).sum(axis=1)) == 128)
    # frames that all its chirps are non-blink
    noblink = np.argwhere(((label == 0).sum(axis=1)) == 128)
    print(np.shape(blink))
    n_sample = 8790
    seq = np.concatenate([blink[:n_sample], noblink[:n_sample]], axis=0)
    print(np.shape(seq))
    np.random.shuffle(seq)
    print(np.shape(seq))
    samples = reorder_IQ(data[seq[:]].squeeze())
    print(np.shape(samples))
    X = process_ex(samples[0].reshape(-1, 4, 256)).reshape(1, 128, -1)
    for i in range(1, len(samples)):
        print(np.shape(X))
        X = np.concatenate([X, process_ex(samples[i].reshape(-1, 4, 256)).reshape(1, 128, -1)], axis=0)
    print(X.shape)

    Y = label[seq[:]][:, 0, 0]
    print(Y.shape)

    return X, Y

class GetLoader(torch.utils.data.Dataset):
    # initial function, get the data and label
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index is divided based on the batchsize, finally return the data and corresponding labels
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # for DataLoader better dividing the data, we use this function to return the length of the data
    def __len__(self):
        return len(self.data)