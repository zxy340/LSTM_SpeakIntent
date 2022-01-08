import numpy as np
import torch
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Concepts = [
    'Inner_Brow_Raiser',     # AU01   # 00
    'Outer_Brow_Raiser',     # AU02   # 01
    'Brow_Lowerer',          # AU04   # 02
    'Upper_Lid_Raiser',      # AU05   # 03
    'Cheek_Raiser',          # AU06   # 04
    'Lid_Tightener',         # AU07   # 05
    'Nose_Wrinkler',         # AU09   # 06
    'Upper_Lip_Raiser',      # AU10   # 07
    'Lip_Corner_Puller',     # AU12   # 08
    'Dimpler',               # AU14   # 09
    'Lip_Corner_Depressor',  # AU15   # 10
    'Chin_Raiser',           # AU17   # 11
    'Lip_stretcher',         # AU20   # 12
    'Lip_Tightener',         # AU23   # 13
    'Lips_part',             # AU25   # 14
    'Jaw_Drop',              # AU26   # 15
    'Lip_Suck',              # AU28   # 16
    'Blink'                  # AU45   # 17
]
users = [
    'adityarathore',      # 00
    'Caitlin_Chan',       # 01
    # 'Amy_Zhang',          # 02
    # 'Anarghya',           # 03
    # 'aniruddh',           # 04
    # 'anthony',            # 05
    # 'baron_huang',        # 06
    # 'bhuiyan',            # 07
    # 'chandler',           # 08
    # 'chenyi_zou',         # 09
    # 'deepak_joseph',      # 10
    # 'dunjiong_lin',       # 11
    # 'Eric_Kim',           # 12
    # 'FrankYang',          # 13
    # 'giorgi_datashvili',  # 14
    # 'Huining_Li',         # 15
    # 'jonathan',           # 16
    # 'Kunjie_Lin',         # 17
    # 'lauren',             # 18
    # 'moohliton',          # 19
    # 'phoung',             # 20
    # 'Tracy_chen'          # 21
]

def CFAR_CA(data, T, G, offset):
    data = data.squeeze()
    # print('the size of the temporary processed data should be (128, 192), actually it is {}'.format(np.shape(data)))
    # fig = plt.figure(1)
    # ax1 = plt.subplot(3, 1, 1)
    # # 在画纸1上绘图
    # plt.title('the figure of signal before cfar')
    # plt.plot(data[0])

    data = data.reshape(128, 3, 64)
    # print('the size of the temporary processed data should be (128, 3, 64), actually it is {}'.format(np.shape(data)))

    # Vector to hold threshold values
    threshold_cfar = []

    # Vector to hold final signal after thresholding
    signal_cfar = []

    for chirp in range(128):
        threshold_chirp = []
        signal_chirp = []
        for channel in range(len(data[chirp])):
            # Slide window across the signal length
            for i in range((len(data[chirp][channel]) - (2*G+2*T))):

                # Determine the noise threshold by measuring it within the training cells
                noise_level = sum(data[chirp][channel][i:i+T]) + sum(data[chirp][channel][i+T+2*G+1:i+2*T+2*G+2])

                # Measuring the signal within the CUT
                threshold = (noise_level / (2*T)) * offset
                threshold_chirp.append(threshold)

                signal = data[chirp][channel][i+T+G]

                # Filter the signal above the threshold
                if signal < threshold:
                    signal = 0
                signal_chirp.append(signal)

        threshold_cfar.append(threshold_chirp)
        signal_cfar.append(signal_chirp)
    # # 选择画纸2
    # ax2 = plt.subplot(3, 1, 2)
    # # 在画纸2上绘图
    # plt.title('the figure of signal after cfar')
    # plt.plot(signal_cfar[0])
    # # 选择画纸3
    # ax3 = plt.subplot(3, 1, 3)
    # # 在画纸3上绘图
    # plt.title('the figure of threshold')
    # plt.plot(threshold_cfar[0])
    # # 显示图像
    # plt.show()
    return signal_cfar

def CFAR_MIN(data, T, G, offset):
    data = data.squeeze()
    # print('the size of the temporary processed data should be (128, 192), actually it is {}'.format(np.shape(data)))
    # fig = plt.figure(1)
    # ax1 = plt.subplot(3, 1, 1)
    # # 在画纸1上绘图
    # plt.title('the figure of signal before cfar')
    # plt.plot(data[0])

    data = data.reshape(128, 3, 64)
    # print('the size of the temporary processed data should be (128, 3, 64), actually it is {}'.format(np.shape(data)))

    # Vector to hold threshold values
    threshold_cfar = []

    # Vector to hold final signal after thresholding
    signal_cfar = []

    for chirp in range(128):
        threshold_chirp = []
        signal_chirp = []
        for channel in range(len(data[chirp])):
            # Slide window across the signal length
            for i in range((len(data[chirp][channel]) - (2*G+2*T))):

                # Determine the noise threshold by measuring it within the training cells
                noise_level = min(sum(data[chirp][channel][i:i+T]), sum(data[chirp][channel][i+T+2*G+1:i+2*T+2*G+1]))

                # Measuring the signal within the CUT
                threshold = (noise_level / T) * offset
                threshold_chirp.append(threshold)

                signal = data[chirp][channel][i+T+G]

                # Filter the signal above the threshold
                if signal < threshold:
                    signal = 0
                signal_chirp.append(signal)

        threshold_cfar.append(threshold_chirp)
        signal_cfar.append(signal_chirp)
    # # 选择画纸2
    # ax2 = plt.subplot(3, 1, 2)
    # # 在画纸2上绘图
    # plt.title('the figure of signal after cfar')
    # plt.plot(signal_cfar[0])
    # # 选择画纸3
    # ax3 = plt.subplot(3, 1, 3)
    # # 在画纸3上绘图
    # plt.title('the figure of threshold')
    # plt.plot(threshold_cfar[0])
    # # 显示图像
    # plt.show()
    return signal_cfar

def CFAR_MAX(data, T, G, offset):
    data = data.squeeze()
    # print('the size of the temporary processed data should be (128, 192), actually it is {}'.format(np.shape(data)))
    # fig = plt.figure(1)
    # ax1 = plt.subplot(3, 1, 1)
    # # 在画纸1上绘图
    # plt.title('the figure of signal before cfar')
    # plt.plot(data[0])

    data = data.reshape(128, 3, 64)
    # print('the size of the temporary processed data should be (128, 3, 64), actually it is {}'.format(np.shape(data)))

    # Vector to hold threshold values
    threshold_cfar = []

    # Vector to hold final signal after thresholding
    signal_cfar = []

    for chirp in range(128):
        threshold_chirp = []
        signal_chirp = []
        for channel in range(len(data[chirp])):
            # Slide window across the signal length
            for i in range((len(data[chirp][channel]) - (2*G+2*T))):

                # Determine the noise threshold by measuring it within the training cells
                noise_level = max(sum(data[chirp][channel][i:i+T]), sum(data[chirp][channel][i+T+2*G+1:i+2*T+2*G+1]))

                # Measuring the signal within the CUT
                threshold = (noise_level / T) * offset
                threshold_chirp.append(threshold)

                signal = data[chirp][channel][i+T+G]

                # Filter the signal above the threshold
                if signal < threshold:
                    signal = 0
                signal_chirp.append(signal)

        threshold_cfar.append(threshold_chirp)
        signal_cfar.append(signal_chirp)
    # # 选择画纸2
    # ax2 = plt.subplot(3, 1, 2)
    # # 在画纸2上绘图
    # plt.title('the figure of signal after cfar')
    # plt.plot(signal_cfar[0])
    # # 选择画纸3
    # ax3 = plt.subplot(3, 1, 3)
    # # 在画纸3上绘图
    # plt.title('the figure of threshold')
    # plt.plot(threshold_cfar[0])
    # # 显示图像
    # plt.show()
    return signal_cfar

def CFAR_OS(data, T, G, offset, k):
    data = data.squeeze()
    # print('the size of the temporary processed data should be (128, 192), actually it is {}'.format(np.shape(data)))
    # fig = plt.figure(1)
    # ax1 = plt.subplot(3, 1, 1)
    # # 在画纸1上绘图
    # plt.title('the figure of signal before cfar')
    # plt.plot(data[0])

    data = data.reshape(128, 3, 64)
    # print('the size of the temporary processed data should be (128, 3, 64), actually it is {}'.format(np.shape(data)))

    # Vector to hold threshold values
    threshold_cfar = []

    # Vector to hold final signal after thresholding
    signal_cfar = []

    for chirp in range(128):
        threshold_chirp = []
        signal_chirp = []
        for channel in range(len(data[chirp])):
            # Slide window across the signal length
            for i in range((len(data[chirp][channel]) - (2*G+2*T))):
                temp = []

                # Determine the noise threshold by measuring it within the training cells
                for j in range(i, i+T):
                    temp.append(data[chirp, channel, j])
                for j in range(i+T+2*G+1, i+2*T+2*G+1):
                    temp.append(data[chirp, channel, j])
                temp.sort()
                noise_level = temp[k-1]

                # Measuring the signal within the CUT
                threshold = noise_level * offset
                threshold_chirp.append(threshold)

                signal = data[chirp, channel, i+T+G]

                # Filter the signal above the threshold
                if signal < threshold:
                    signal = 0
                signal_chirp.append(signal)

        threshold_cfar.append(threshold_chirp)
        signal_cfar.append(signal_chirp)
    # # 选择画纸2
    # ax2 = plt.subplot(3, 1, 2)
    # # 在画纸2上绘图
    # plt.title('the figure of signal after cfar')
    # plt.plot(signal_cfar[0])
    # # 选择画纸3
    # ax3 = plt.subplot(3, 1, 3)
    # # 在画纸3上绘图
    # plt.title('the figure of threshold')
    # plt.plot(threshold_cfar[0])
    # # 显示图像
    # plt.show()
    return signal_cfar

def data_loading_LSTM(label_index, data_path):
    """
    load data from saved files, the variable "x_data" stores mmWave data, the variable "y_data" stores labels
    we split 3/4 data as training data, and 1/4 data as testing data
    the variable "x_train" stores mmWave data for training set, the variable "y_train" stores labels for training set
    the variable "x_test" stores mmWave data for testing set, the variable "y_test" stores labels for testing set

    Parameters
    ----------
    label_index : int
        indicate the index of the concept
    data_path : string
        the path of mmWave data and concept label
    Returns
    -------
    x_train : np.array(num_frames, 128, 3 * 64)
        array of the mmWave data for training, num_frames is the number of frames from several users,
        128 is the number of chirps of one frame, 3 * 64 is the data size of one chirp.
    y_train : np.array(num_frames, )
        array of the labels
    x_test : np.array(num_frames, 128, 3 * 64)
        array of the mmWave data for testing, num_frames is the number of frames from several different users from x_train,
        128 is the number of chirps of one frame, 3 * 64 is the data size of one chirp.
    y_train : np.array(num_frames, )
        array of the labels
    """
    x_train = np.zeros((1, 128, 192))
    y_train = np.zeros((1,))
    x_test = np.zeros((1, 128, 192))
    y_test = np.zeros((1,))
    # for i in range(int(len(users)/8*3)):
    for i in range(1):
        user = users[i]
        print('Current added user is {}'.format(user))
        if not os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy'):
            continue
        x_train = np.concatenate((x_train, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')),
                                 axis=0)
        y_train = np.concatenate((y_train, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')),
                                 axis=0)
    # for i in range(int(len(users)/8*3), int(len(users)/2)):
    for i in range(1):
        user = users[i+1]
        print('Current added user is {}'.format(user))
        if not os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy'):
            continue
        x_test = np.concatenate((x_test, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')),
                                axis=0)
        y_test = np.concatenate((y_test, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')),
                                axis=0)
    x_train = x_train[1:]
    y_train = y_train[1:]
    x_test = x_test[1:]
    y_test = y_test[1:]
    # x_train = x_train[:int(len(x_train) / 8 * 3)]
    # y_train = y_train[:int(len(y_train) / 8 * 3)]
    # x_test = x_test[int(len(x_test) / 8 * 3):int(len(x_test) / 2)]
    # y_test = y_test[int(len(y_test) / 8 * 3):int(len(y_test) / 2)]
    # print("the length of x_train is {}".format(np.shape(x_train)))
    # print("the length of y_train is {}".format(np.shape(y_train)))
    # print("the length of x_test is {}".format(np.shape(x_test)))
    # print("the length of y_test is {}".format(np.shape(y_test)))
    seq = np.arange(0, len(x_train), 1)
    np.random.shuffle(seq)
    x_train = x_train[seq[:]]
    y_train = y_train[seq[:]]
    seq = np.arange(0, len(x_test), 1)
    np.random.shuffle(seq)
    x_test = x_test[seq[:]]
    y_test = y_test[seq[:]]
    # print("the length of x_train is {}".format(np.shape(x_train)))
    # print("the length of y_train is {}".format(np.shape(y_train)))
    # print("the length of x_test is {}".format(np.shape(x_test)))
    # print("the length of y_test is {}".format(np.shape(y_test)))

    # fig = plt.figure(1, figsize=(16, 16))
    # count = 0
    # for i in range(len(x_train)):
    #     if y_train[i] == 1:
    #         count += 1
    #         if count > 16:
    #             break
    #         ax = fig.add_subplot(4, 4, count)
    #         plt.plot(x_train[i][127])
    #         # plt.title('the {}th frame'.format(i))
    # fig = plt.figure(2, figsize=(16, 16))
    # count = 0
    # for i in range(len(x_test)):
    #     if y_test[i] == 1:
    #         count += 1
    #         if count > 16:
    #             break
    #         ax = fig.add_subplot(4, 4, count)
    #         plt.plot(x_test[i][127])
    #         # plt.title('the {}th frame'.format(i))
    # plt.show()

    x_train_cfar = []
    for i in range(len(x_train)):
        # x_train_cfar.append(CFAR_CA(x_train[i], T=5, G=15, offset=1))
        # x_train_cfar.append(CFAR_MIN(x_train[i], T=5, G=15, offset=1))
        # x_train_cfar.append(CFAR_MAX(x_train[i], T=5, G=15, offset=1))
        x_train_cfar.append(CFAR_OS(x_train[i], T=5, G=15, offset=1, k=3))
    x_test_cfar = []
    for i in range(len(x_test)):
        # x_test_cfar.append(CFAR_CA(x_test[i], T=5, G=15, offset=1))
        # x_test_cfar.append(CFAR_MIN(x_test[i], T=5, G=15, offset=1))
        # x_test_cfar.append(CFAR_MAX(x_test[i], T=5, G=15, offset=1))
        x_test_cfar.append(CFAR_OS(x_test[i], T=5, G=15, offset=1, k=3))
    x_train_cfar = np.array(x_train_cfar)
    x_test_cfar = np.array(x_test_cfar)

    return x_train_cfar, y_train, x_test_cfar, y_test

def data_loading_concept(label_index, data_path):
    """
    load data from saved files, the variable "x_data" stores mmWave data, the variable "y_data" stores labels
    since former 3/4 users' data have been used to train and test sub_model, we use the left 1/4 users' data to extract features
    based on the left 1/4 users' data, we split 3/4 users' data as training data, and 1/4 users' data as testing data
    the variable "x_train" stores mmWave data for training set, the variable "y_train" stores labels for training set
    the variable "x_test" stores mmWave data for testing set, the variable "y_test" stores labels for testing set

    Parameters
    ----------
    label_index : int
        indicate the index of the concept
    data_path : string
        the path of mmWave data and concept label
    Returns
    -------
    x_train : np.array(num_frames, 128, 3 * 64)
        array of the mmWave data for training, num_frames is the number of frames from several users,
        128 is the number of chirps of one frame, 3 * 64 is the data size of one chirp.
    y_train : np.array(num_frames, )
        array of the labels
    x_test : np.array(num_frames, 128, 3 * 64)
        array of the mmWave data for testing, num_frames is the number of frames from several different users from x_train,
        128 is the number of chirps of one frame, 3 * 64 is the data size of one chirp.
    y_train : np.array(num_frames, )
        array of the labels
    """
    # create arrays
    x_train = np.zeros((1, 128, 192))
    y_train = np.zeros((1,))
    x_test = np.zeros((1, 128, 192))
    y_test = np.zeros((1,))

    # concatenate several users' data
    # for i in range(int(len(users)/2), int(len(users)/16*11)):
    for i in range(1):
        user = users[i]
        print('Current added user is {}'.format(user))
        if not (os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy') and
            os.path.exists(data_path + user + '/' + Concepts[label_index] + '/y_data.npy') and
            os.path.exists(data_path + user + '/' + Concepts[label_index] + '/labels.npy') and
            os.path.exists(data_path + user + '/' + Concepts[label_index] + '/levels.npy')):
            continue
        x_train = np.concatenate((x_train, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
        y_train = np.concatenate((y_train, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
    # for i in range(int(len(users)/16*11), int(len(users)/4*3)):
    for i in range(1):
        user = users[i]
        print('Current added user is {}'.format(user))
        if not (os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy') and
            os.path.exists(data_path + user + '/' + Concepts[label_index] + '/y_data.npy') and
            os.path.exists(data_path + user + '/' + Concepts[label_index] + '/labels.npy') and
            os.path.exists(data_path + user + '/' + Concepts[label_index] + '/levels.npy')):
            continue
        x_test = np.concatenate((x_test, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
        y_test = np.concatenate((y_test, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)

    # remove the initial part of data that is not belong to users
    x_train = x_train[1:]
    y_train = y_train[1:]
    x_test = x_test[1:]
    y_test = y_test[1:]
    x_train = x_train[int(len(x_train)/2):int(len(x_train)/16*11)]
    y_train = y_train[int(len(y_train)/2):int(len(y_train)/16*11)]
    x_test = x_test[int(len(x_test)/16*11):int(len(x_test)/4*3)]
    y_test = y_test[int(len(y_test)/16*11):int(len(y_test)/4*3)]
    print("the length of x_train is {}".format(np.shape(x_train)))
    print("the length of y_train is {}".format(np.shape(y_train)))
    print("the length of x_test is {}".format(np.shape(x_test)))
    print("the length of y_test is {}".format(np.shape(y_test)))

    # randomly sort the data for further using
    seq = np.arange(0, len(x_train), 1)
    np.random.shuffle(seq)
    x_train = x_train[seq[:]]
    y_train = y_train[seq[:]]
    seq = np.arange(0, len(x_test), 1)
    np.random.shuffle(seq)
    x_test = x_test[seq[:]]
    y_test = y_test[seq[:]]
    print("the length of x_train is {}".format(np.shape(x_train)))
    print("the length of y_train is {}".format(np.shape(y_train)))
    print("the length of x_test is {}".format(np.shape(x_test)))
    print("the length of y_test is {}".format(np.shape(y_test)))

    return x_train, y_train, x_test, y_test

def data_loading_speak(data_path):
    """
    load data from saved files, the variable "x_data" stores mmWave data, the variable "y_data" stores labels
    since former 3/4 users' data have been used to train and test sub_model, we use the left 1/4 users' data to extract features
    based on the left 1/4 users' data, we split 3/4 users' data as training data, and 1/4 users' data as testing data
    the variable "x_train" stores mmWave data for training set, the variable "y_train" stores labels for training set
    the variable "x_test" stores mmWave data for testing set, the variable "y_test" stores labels for testing set
    there is one thing needs attention: the labels of "label_train" and "level_train" are speak intent labels,
    while the labels of "y_train" are concept action labels, concept action labels are used to train sub_model,
    and speak intent labels are used to train final speak intent model.
    Parameters
    ----------
    data_path : string
        the path of mmWave data and concept label
    Returns
    -------
    x_train : np.array(num_frames, 128, 3 * 64)
        array of the mmWave data for training, num_frames is the number of frames from several users,
        128 is the number of chirps of one frame, 3 * 64 is the data size of one chirp.
    y_train : np.array(num_frames, )
        array of the labels for the x_train corresponding concept
    label_train : np.array(num_frames, )
        array of the labels for speak intent, that is whether has the concept action
    level_train : np.array(num_frames, )
        array of the labels for the level of speak intent, the value can be 0, 1, 2, 3, while high value means high intent level
    x_test : np.array(num_frames, 128, 3 * 64)
        array of the mmWave data for testing, num_frames is the number of frames from several different users from x_train,
        128 is the number of chirps of one frame, 3 * 64 is the data size of one chirp.
    y_train : np.array(num_frames, )
        array of the labels
    label_test : np.array(num_frames, )
        array of the labels for testing
    level_test : np.array(num_frames, )
        array of the labels for testing
    """
    x_train = np.zeros((1, 128, 192))
    y_train = np.zeros((1,))
    label_train = np.zeros((1,))
    level_train = np.zeros((1,))
    x_test = np.zeros((1, 128, 192))
    y_test = np.zeros((1,))
    label_test = np.zeros((1,))
    level_test = np.zeros((1,))
    # for i in range(int(len(users)/4*3), int(len(users)/16*15)):
    for i in range(1):
        user = users[i]
        print('Current added user is {}'.format(user))
        for label_index in range(len(Concepts)):
            if not (os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy') and
                    os.path.exists(data_path + user + '/' + Concepts[label_index] + '/y_data.npy') and
                    os.path.exists(data_path + user + '/' + Concepts[label_index] + '/labels.npy') and
                    os.path.exists(data_path + user + '/' + Concepts[label_index] + '/levels.npy')):
                continue
            x_train = np.concatenate((x_train, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
            y_train = np.concatenate((y_train, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
            label_train = np.concatenate((label_train, np.load(data_path + user + '/' + Concepts[label_index] + '/labels.npy')), axis=0)
            level_train = np.concatenate((level_train, np.load(data_path + user + '/' + Concepts[label_index] + '/levels.npy')), axis=0)
    # for i in range(int(len(users)/16*15), int(len(users))):
    for i in range(1):
        user = users[i]
        print('Current added user is {}'.format(user))
        for label_index in range(len(Concepts)):
            if not (os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy') and
                    os.path.exists(data_path + user + '/' + Concepts[label_index] + '/y_data.npy') and
                    os.path.exists(data_path + user + '/' + Concepts[label_index] + '/labels.npy') and
                    os.path.exists(data_path + user + '/' + Concepts[label_index] + '/levels.npy')):
                continue
            x_test = np.concatenate((x_test, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
            y_test = np.concatenate((y_test, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
            label_test = np.concatenate((label_test, np.load(data_path + user + '/' + Concepts[label_index] + '/labels.npy')), axis=0)
            level_test = np.concatenate((level_test, np.load(data_path + user + '/' + Concepts[label_index] + '/levels.npy')), axis=0)
    x_train = x_train[1:]
    y_train = y_train[1:]
    label_train = label_train[1:]
    level_train = level_train[1:]
    x_test = x_test[1:]
    y_test = y_test[1:]
    label_test = label_test[1:]
    level_test = level_test[1:]
    x_train = x_train[int(len(x_train)/4*3):int(len(x_train)/16*15)]
    y_train = y_train[int(len(y_train)/4*3):int(len(y_train)/16*15)]
    label_train = label_train[int(len(label_train)/4*3):int(len(label_train)/16*15)]
    level_train = level_train[int(len(level_train)/4*3):int(len(level_train)/16*15)]
    x_test = x_test[int(len(x_test)/16*15):int(len(x_test))]
    y_test = y_test[int(len(y_test)/16*15):int(len(y_test))]
    label_test = label_test[int(len(label_test)/16*15):int(len(label_test))]
    level_test = level_test[int(len(level_test)/16*15):int(len(level_test))]
    print("the length of x_train is {}".format(np.shape(x_train)))
    print("the length of y_train is {}".format(np.shape(y_train)))
    print("the length of x_test is {}".format(np.shape(x_test)))
    print("the length of y_test is {}".format(np.shape(y_test)))
    seq = np.arange(0, len(x_train), 1)
    np.random.shuffle(seq)
    x_train = x_train[seq[:]]
    y_train = y_train[seq[:]]
    label_train = label_train[seq[:]]
    level_train = level_train[seq[:]]
    seq = np.arange(0, len(x_test), 1)
    np.random.shuffle(seq)
    x_test = x_test[seq[:]]
    y_test = y_test[seq[:]]
    label_test = label_test[seq[:]]
    level_test = level_test[seq[:]]
    print("the length of x_train is {}".format(np.shape(x_train)))
    print("the length of y_train is {}".format(np.shape(y_train)))
    print("the length of label_train is {}".format(np.shape(label_train)))
    print("the length of level_train is {}".format(np.shape(level_train)))
    print("the length of x_test is {}".format(np.shape(x_test)))
    print("the length of y_test is {}".format(np.shape(y_test)))
    print("the length of label_test is {}".format(np.shape(label_test)))
    print("the length of level_test is {}".format(np.shape(level_test)))

    return x_train, y_train, label_train, level_train, x_test, y_test, label_test, level_test
    # .................................................................................