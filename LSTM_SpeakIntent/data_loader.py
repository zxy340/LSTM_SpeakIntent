import numpy as np
import torch
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
    # 'Tracy_chen'          # 21
]

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
    # x_test = np.zeros((1, 128, 192))
    # y_test = np.zeros((1,))
    # for i in range(int(len(users)/8*3)):
    for i in range(int(len(users))):
    # for i in range(1):
    # for i in range(int(len(users)/4*3)):
        user = users[i]
        print('Current added user for training is {}'.format(user))
        if not os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy'):
            continue
        current_user_xdata = np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')
        x_train = np.concatenate((x_train, current_user_xdata[:10000]),
                                 axis=0)
        current_user_ydata = np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')
        y_train = np.concatenate((y_train, current_user_ydata[:10000]),
                                 axis=0)
    # for i in range(int(len(users)/8*3), int(len(users)/2)):
    # for i in range(1):
    # for i in range(int(len(users)/4*3), int(len(users))):
    #     user = users[i]
    #     print('Current added user for testing is {}'.format(user))
    #     if not os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy'):
    #         continue
    #     x_test = np.concatenate((x_test, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')),
    #                             axis=0)
    #     y_test = np.concatenate((y_test, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')),
    #                             axis=0)
    x_train = x_train[1:]
    y_train = y_train[1:]
    # x_test = x_test[1:]
    # y_test = y_test[1:]
    seq = np.arange(0, len(x_train), 1)
    np.random.shuffle(seq)
    x_test = x_train[seq[int(len(x_train)/8*3):int(len(x_train)/2)]]
    y_test = y_train[seq[int(len(y_train)/8*3):int(len(y_train)/2)]]
    x_train = x_train[seq[:int(len(x_train)/8*3)]]
    y_train = y_train[seq[:int(len(y_train)/8*3)]]
    # seq = np.arange(0, len(x_test), 1)
    # np.random.shuffle(seq)
    print("the length of x_train is {}".format(np.shape(x_train)))
    print("the length of y_train is {}".format(np.shape(y_train)))
    print("the length of x_test is {}".format(np.shape(x_test)))
    print("the length of y_test is {}".format(np.shape(y_test)))

    return x_train, y_train, x_test, y_test

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
    # x_test = np.zeros((1, 128, 192))
    # y_test = np.zeros((1,))

    # concatenate several users' data
    # for i in range(int(len(users)/2), int(len(users)/16*11)):
    # for i in range(1):
    for i in range(int(len(users))):
        user = users[i]
        print('Current added user is {}'.format(user))
        if not (os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy') and
            os.path.exists(data_path + user + '/' + Concepts[label_index] + '/y_data.npy') and
            os.path.exists(data_path + user + '/' + Concepts[label_index] + '/labels.npy') and
            os.path.exists(data_path + user + '/' + Concepts[label_index] + '/levels.npy')):
            continue
        current_user_xdata = np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')
        x_train = np.concatenate((x_train, current_user_xdata[:10000]),
                                 axis=0)
        current_user_ydata = np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')
        y_train = np.concatenate((y_train, current_user_ydata[:10000]),
                                 axis=0)
    # for i in range(int(len(users)/16*11), int(len(users)/4*3)):
    # for i in range(1):
    #     user = users[i]
    #     print('Current added user is {}'.format(user))
    #     if not (os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy') and
    #         os.path.exists(data_path + user + '/' + Concepts[label_index] + '/y_data.npy') and
    #         os.path.exists(data_path + user + '/' + Concepts[label_index] + '/labels.npy') and
    #         os.path.exists(data_path + user + '/' + Concepts[label_index] + '/levels.npy')):
    #         continue
    #     x_test = np.concatenate((x_test, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    #     y_test = np.concatenate((y_test, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)

    # remove the initial part of data that is not belong to users
    x_train = x_train[1:]
    y_train = y_train[1:]
    # x_test = x_test[1:]
    # y_test = y_test[1:]
    seq = np.arange(0, len(x_train), 1)
    np.random.shuffle(seq)
    x_test = x_train[seq[int(len(x_train)/16*11):int(len(x_train)/4*3)]]
    y_test = y_train[seq[int(len(y_train)/16*11):int(len(y_train)/4*3)]]
    x_train = x_train[seq[int(len(x_train)/2):int(len(x_train)/16*11)]]
    y_train = y_train[seq[int(len(y_train)/2):int(len(y_train)/16*11)]]
    # seq = np.arange(0, len(x_test), 1)
    # np.random.shuffle(seq)
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
    # x_test = np.zeros((1, 128, 192))
    # y_test = np.zeros((1,))
    # label_test = np.zeros((1,))
    # level_test = np.zeros((1,))
    # for i in range(int(len(users)/4*3), int(len(users)/16*15)):
    # for i in range(1):
    for i in range(int(len(users))):
        user = users[i]
        print('Current added user is {}'.format(user))
        for label_index in range(len(Concepts)):
            if not (os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy') and
                    os.path.exists(data_path + user + '/' + Concepts[label_index] + '/y_data.npy') and
                    os.path.exists(data_path + user + '/' + Concepts[label_index] + '/labels.npy') and
                    os.path.exists(data_path + user + '/' + Concepts[label_index] + '/levels.npy')):
                continue
            current_user_xdata = np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')
            x_train = np.concatenate((x_train, current_user_xdata[:10000]),
                                     axis=0)
            current_user_ydata = np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')
            y_train = np.concatenate((y_train, current_user_ydata[:10000]),
                                     axis=0)
            current_user_labeldata = np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')
            y_train = np.concatenate((y_train, current_user_labeldata[:10000]),
                                     axis=0)
    # for i in range(int(len(users)/16*15), int(len(users))):
    # for i in range(1):
    #     user = users[i]
    #     print('Current added user is {}'.format(user))
    #     for label_index in range(len(Concepts)):
    #         if not (os.path.exists(data_path + user + '/' + Concepts[label_index] + '/x_data.npy') and
    #                 os.path.exists(data_path + user + '/' + Concepts[label_index] + '/y_data.npy') and
    #                 os.path.exists(data_path + user + '/' + Concepts[label_index] + '/labels.npy') and
    #                 os.path.exists(data_path + user + '/' + Concepts[label_index] + '/levels.npy')):
    #             continue
    #         x_test = np.concatenate((x_test, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    #         y_test = np.concatenate((y_test, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
    #         label_test = np.concatenate((label_test, np.load(data_path + user + '/' + Concepts[label_index] + '/labels.npy')), axis=0)
    #         level_test = np.concatenate((level_test, np.load(data_path + user + '/' + Concepts[label_index] + '/levels.npy')), axis=0)
    x_train = x_train[1:]
    y_train = y_train[1:]
    label_train = label_train[1:]
    level_train = level_train[1:]
    # x_test = x_test[1:]
    # y_test = y_test[1:]
    # label_test = label_test[1:]
    # level_test = level_test[1:]
    seq = np.arange(0, len(x_train), 1)
    np.random.shuffle(seq)
    x_test = x_train[int(len(x_train)/16*15):int(len(x_train))]
    y_test = y_train[int(len(y_train)/16*15):int(len(y_train))]
    label_test = label_train[int(len(label_train)/16*15):int(len(label_train))]
    x_train = x_train[int(len(x_train)/4*3):int(len(x_train)/16*15)]
    y_train = y_train[int(len(y_train)/4*3):int(len(y_train)/16*15)]
    label_train = label_train[int(len(label_train)/4*3):int(len(label_train)/16*15)]
    # seq = np.arange(0, len(x_train), 1)
    # np.random.shuffle(seq)
    # x_train = x_train[seq[:]]
    # y_train = y_train[seq[:]]
    # label_train = label_train[seq[:]]
    # level_train = level_train[seq[:]]
    # seq = np.arange(0, len(x_test), 1)
    # np.random.shuffle(seq)
    # x_test = x_test[seq[:]]
    # y_test = y_test[seq[:]]
    # label_test = label_test[seq[:]]
    # level_test = level_test[seq[:]]
    print("the length of x_train is {}".format(np.shape(x_train)))
    print("the length of y_train is {}".format(np.shape(y_train)))
    print("the length of label_train is {}".format(np.shape(label_train)))
    print("the length of x_test is {}".format(np.shape(x_test)))
    print("the length of y_test is {}".format(np.shape(y_test)))
    print("the length of label_test is {}".format(np.shape(label_test)))

    return x_train, y_train, label_train, x_test, y_test, label_test
    # .................................................................................