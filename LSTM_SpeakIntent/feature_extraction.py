import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from LSTM import simpleLSTM
from data import GetLoader
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

def data_loading(label_index, data_path):
    # ...........................load data...........................................
    # load data from saved files, the variable "x_data" stores mmWave data, the variable "y_data" stores labels
    # since former 3/4 users' data have been used to train and test sub_model, we use the left 1/4 users' data to extract features
    # based on the left 1/4 users' data, we split 3/4 users' data as training data, and 1/4 users' data as testing data
    # the variable "x_train" stores mmWave data for training set, the variable "y_train" stores labels for training set
    # the variable "x_test" stores mmWave data for testing set, the variable "y_test" stores labels for testing set
    x_train = np.zeros((1, 128, 192))
    y_train = np.zeros((1,))
    x_test = np.zeros((1, 128, 192))
    y_test = np.zeros((1,))
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
    # .................................................................................

def feature_extrac(x_train, y_train, x_test, y_test, label_index):
    print('current extracted feature is from {}'.format(Concepts[label_index]))
    # .............basic information of training and testing set.......................
    training_data_count = len(x_train)  # number of training series
    if training_data_count == 0:
        return 0, 0
    testing_data_count = len(x_test)
    num_steps = len(x_train[0])  # timesteps per series
    num_input = len(x_train[0][0])  # input parameters per timestep
    # ..................................................................................

    # ..................................................................................
    # use GetLoader to load the data and return Dataset object, which contains data and labels
    torch_data = GetLoader(x_train, y_train)
    train_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)
    torch_data = GetLoader(x_test, y_test)
    test_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)
    # ....................................................................................

    # .............Hyper Parameters and initial model parameters..........................
    hidden_size = 64
    num_layers = 5
    num_classes = 2
    lr = 0.01           # learning rate
    # initial model
    model = simpleLSTM(num_input, hidden_size, num_layers, num_classes).to(device)
    # .....................................................................................

    # ............................load the trained model...................................
    # load the model
    if not os.path.exists('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'):
        return 0, 0
    model.load_state_dict(torch.load('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'))
    print("the model has been successfully loaded!")
    # .....................................................................................

    # ............................full-connected layer training............................
    # we first get the number of layers in the loaded model, and train fc for each layer of the model
    # then we store each layer of the fc model into the dictionary named "fc_dic"
    # for example, as for 2 layers LSTM model, the structure of "fc_dic" is as follows:
    # layer1: array(64*2)
    # layer2: array(64*2)
    # where 64 is the number of hidden parameters in each LSTM layer, 2 is the number of classes of labels
    fc_dic = {}  # record the parameters of each layer fc
    for layer in range(num_layers):
        name = "fc" + str(layer)
        # create each fc
        fc = nn.Linear(hidden_size, num_classes, bias=False).to(device)
        # loss and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(fc.parameters(), lr)
        for i, (data, labels) in enumerate(train_data):
            data = data.to(device)
            output_unused, h_n, c_n = model(data.float())

            label = []
            if labels[0].size() != num_classes:
                for j in range(len(labels)):
                    label.append(np.eye(num_classes, dtype=float)[int(labels[j])])
            else:
                label = labels
            label = torch.tensor(np.array(label)).to(device)

            optimizer.zero_grad()
            outputs = fc(h_n[layer].float())
            loss = criterion(outputs, label)

            # backward and optimize
            loss.backward()
            optimizer.step()

            if i % 30 == 0:
                print('Step [{}/{}], Loss: {:.4f}'
                       .format(i+1, training_data_count / 128, loss.item()))
        fc_dic[name] = fc
    # ...................................................................................

    # ............full-connected layer testing and target layer finding..................
    # we test the accuracy of each fc after the corresponding output from the loaded layer
    # we further get the target layer with the max accuracy.
    # variable "max_Acc" stores the maximum accuracy of all the layers
    # variable "max_layer" stores the name of the target layer
    # variable "max_fc" stores the parameters of the fc layer for the target layer
    max_Acc = 0
    for layer in range(num_layers):
        name = "fc" + str(layer)
        fc = fc_dic[name]
        # Test the model
        fc.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, label in test_data:
                data = data.to(device)
                output_unused, h_n, c_n = model(data.float())
                label = label.to(device)
                label = label.squeeze()
                outputs = fc(h_n[layer].float())
                _, predicted = torch.max(outputs, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            print('Test Accuracy of the model on the {} test mmWave data: {} %'.format(testing_data_count,
                                                                                       100 * correct / total))
            if max_Acc < 100 * correct / total:
                max_Acc = 100 * correct / total
                max_layer = layer
                max_fc = fc.state_dict()['weight']
    print('the target layer is {}th layer with the maximum accuracy {}'.format(max_layer, max_Acc))
    # ...................................................................................

    # .........................feature extraction........................................
    feature_index = 0
    sec_feature = 0
    max_feature = max_fc[0][1]
    for i in range(max_fc.size()[0]):
        tem_feature = max_fc[i][1]
        if tem_feature > max_feature:
            feature_index = i
            sec_feature = max_feature
            max_feature = tem_feature
    print('the target feature is the {}th neuron with {} difference value for label 1 to label 0'.format(feature_index, max_feature))
    print('the second largest feature has the {} difference value for label 1 to label 0'.format(sec_feature))
    # ...................................................................................

    return feature_index, max_layer

def data_loading_speak(data_path):
    # ...........................load data...........................................
    # load data from saved files, the variable "x_data" stores mmWave data, the variable "y_data" stores labels
    # since former 3/4 users' data have been used to train and test sub_model, we use the left 1/4 users' data to extract features
    # based on the left 1/4 users' data, we split 3/4 users' data as training data, and 1/4 users' data as testing data
    # the variable "x_train" stores mmWave data for training set, the variable "y_train" stores labels for training set
    # the variable "x_test" stores mmWave data for testing set, the variable "y_test" stores labels for testing set
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

def speak_detection(x_train, y_train, label_train, level_train, x_test, y_test, label_test, level_test, features, layers):
    # .............basic information of training and testing set.......................
    training_data_count = len(x_train)  # number of training series
    testing_data_count = len(x_test)
    num_steps = len(x_train[0])  # timesteps per series
    num_input = len(x_train[0][0])  # input parameters per timestep
    # ..................................................................................

    # .............Hyper Parameters and initial model parameters..........................
    epochs = 20
    hidden_size = 64
    num_layers = 5
    num_classes = 2
    batch_size = 128
    lr = 0.01           # learning rate
    # initial model
    model = simpleLSTM(num_input, hidden_size, num_layers, num_classes).to(device)
    # .....................................................................................

    # ..................................................................................
    # use GetLoader to load the data and return Dataset object, which contains data and labels
    torch_data = GetLoader(x_train, label_train)
    train_data = DataLoader(torch_data, batch_size, shuffle=True, drop_last=False)
    torch_data = GetLoader(x_test, label_test)
    test_data = DataLoader(torch_data, batch_size, shuffle=True, drop_last=False)
    # ....................................................................................

    # .............load the target layer and train the speak intent model..................
    feature_layer = torch.zeros(batch_size, hidden_size)
    fc = nn.Linear(hidden_size, num_classes, bias=False).to(device)
    # loss and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(fc.parameters(), lr)

    # train the model
    for epoch in range(epochs):
        C = np.zeros(2)
        if epoch % 5 == 0:
            lr = lr / 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for i, (data, labels) in enumerate(train_data):
            data = data.to(device)
            for label_index in range(len(Concepts)):
                # load the model
                if not os.path.exists('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'):
                    feature_layer[:, label_index] = 0
                else:
                    model.load_state_dict(torch.load('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'))
                    print('the model of {} has been successfully loaded!'.format(Concepts[label_index]))
                    # .....................................................................................
                    output_unused, h_n, c_n = model(data.float())
                    for batch in range(batch_size):
                        feature_layer[batch, label_index] = h_n[layers[label_index], batch, features[label_index]]

            feature_layer = feature_layer.to(device)
            print(feature_layer)
            label = []
            if labels[0].size() != num_classes:
                for j in range(len(labels)):
                    label.append(np.eye(num_classes, dtype=float)[int(labels[j])])
                    if labels[j] == 0:
                        C[0] += 1
                    else:
                        C[1] += 1
            else:
                label = labels
            label = torch.tensor(np.array(label)).to(device)


            outputs = fc(feature_layer.float())
            optimizer.zero_grad()
            loss = criterion(outputs.float(), label.float())
            # backward and optimize
            loss.backward()
            optimizer.step()

            if i % 30 == 0:
                print('Step [{}/{}], Loss: {:.4f}'
                       .format(i+1, training_data_count / 128, loss.item()))
        print('Train C of the model on the {} train mmWave data: {}'.format(training_data_count, C))
    # .....................................................................................

    # ......................test the trained speak intent model............................
    # Test the model
    fc.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        C = np.zeros((2, 2))
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for data, label in test_data:
            data = data.to(device)
            for label_index in range(len(Concepts)):
                # load the model
                if not os.path.exists('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'):
                    feature_layer[:, label_index] = 0
                else:
                    model.load_state_dict(torch.load('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'))
                    print('the model of {} has been successfully loaded!'.format(Concepts[label_index]))
                    # .....................................................................................
                    output_unused, h_n, c_n = model(data.float())
                    for batch in range(batch_size):
                        feature_layer[batch, label_index] = h_n[layers[label_index], batch, features[label_index]]
            feature_layer = feature_layer.to(device)
            label = label.to(device)
            label = label.squeeze()
            outputs = fc(feature_layer.float())
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            for i in range(len(predicted)):
                if predicted[i] == 0:
                    C[0][0] += 1
                else:
                    C[0][1] += 1
                if label[i] == 0:
                    C[1][0] += 1
                else:
                    C[1][1] += 1

            # TP    predict 和 label 同时为1
            TP += ((predicted == 1) & (label == 1)).cpu().sum()
            # TN    predict 和 label 同时为0
            TN += ((predicted == 0) & (label == 0)).cpu().sum()
            # FN    predict 0 label 1
            FN += ((predicted == 0) & (label == 1)).cpu().sum()
            # FP    predict 1 label 0
            FP += ((predicted == 1) & (label == 0)).cpu().sum()

        print('TP = {}, TN = {}, FN = {}, FP = {}'.format(TP, TN, FN, FP))
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        print('Test F1 score of the model on the {} test mmWave data: {}'.format(testing_data_count, F1))
        print('Test C of the model on the {} test mmWave data: {}'.format(testing_data_count, C))
        print('Test Accuracy of the model on the {} test mmWave data: {} %'.format(testing_data_count,
                                                                                   100 * correct / total))
    # .....................................................................................

if __name__ == '__main__':
    data_path = '/mnt/stuff/xiaoyu/data/'  # the path where 'x_data.npy' and 'y_data.npy' are located
    model_type = 'LSTM/'  # the type of the sub_model

    # features = []  # the index of feature in target layer for all concepts
    # layers = []  # the index of target layer for all concepts
    features = np.zeros(len(Concepts))  # the index of feature in target layer for all concepts
    features = features.astype(int)
    layers = np.zeros(len(Concepts))  # the index of target layer for all concepts
    layers = layers.astype(int)
    # indicate which concept to train the model
    # for label_index in range(len(Concepts)):
    #     x_train, y_train, x_test, y_test = data_loading(label_index, data_path)
    #     feature_index, max_layer = feature_extrac(x_train, y_train, x_test, y_test, label_index)
    #     features.append(feature_index)
    #     layers.append(max_layer)

    if not os.path.exists('data/x_train.npy'):
        x_train, y_train, label_train, level_train, x_test, y_test, label_test, level_test = data_loading_speak(data_path)
        np.save('data/x_train.npy', x_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/label_train.npy', label_train)
        np.save('data/level_train.npy', level_train)
        np.save('data/x_test.npy', x_test)
        np.save('data/y_test.npy', y_test)
        np.save('data/label_test.npy', label_test)
        np.save('data/level_test.npy', level_test)
    x_train = np.load('data/x_train.npy')
    y_train = np.load('data/y_train.npy')
    label_train = np.load('data/label_train.npy')
    level_train = np.load('data/level_train.npy')
    x_test = np.load('data/x_test.npy')
    y_test = np.load('data/y_test.npy')
    label_test = np.load('data/label_test.npy')
    level_test = np.load('data/level_test.npy')
    speak_detection(x_train, y_train, label_train, level_train, x_test, y_test, label_test, level_test, features, layers)