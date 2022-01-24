import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from LSTM import simpleLSTM
from TDNN import simpleTDNN
from data import GetLoader, GetLoaderID
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
    # 'Tracy_chen'          # 21 :  since current he has no speak intent label data
]
def LSTM_train(x_train, y_train, id_train, x_test, y_test, id_test, label_index, model_type):
    """
    train the LSTM model for specific concept and test the model

    Parameters
    ----------
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
    label_index : int
        indicate the index of the concept
    model_type : string
        indicate the type of the LSTM model the concept
    Returns
    -------
    """
    # .............basic information of training and testing set.......................
    training_data_count = len(x_train)  # number of training series
    testing_data_count = len(x_test)
    num_steps = len(x_test[0])  # timesteps per series
    num_input = len(x_test[0][0])  # input parameters per timestep
    # ..................................................................................

    # ..................................................................................
    # use GetLoader to load the data and return Dataset object, which contains data and labels
    torch_data = GetLoader(x_train, y_train, id_train)
    train_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)
    torch_data = GetLoader(x_test, y_test, id_test)
    test_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)
    # ....................................................................................

    # .............Hyper Parameters and initial model parameters..........................
    epochs = 1
    hidden_size = 16
    num_layers = 5
    num_classes = 2
    lr = 0.01           # learning rate
    # initial model
    model = simpleLSTM(num_input, hidden_size, num_layers, num_classes).to(device)
    # for name, param in model.named_parameters():
    #     if name.startswith("lstm.1.weight"):
    #         nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
    #     else:
    #         nn.init.constant_(param, 0.3)
    # loss and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr)
    # .....................................................................................

    # ...........................train and store the model.................................
    # train the model
    for epoch in range(epochs):
        C = np.zeros(2)
        if epoch % 10 == 0:
            lr = lr / 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for i, (data, labels, id) in enumerate(train_data):
            data = data.to(device)
            id = id.to(device)
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

            optimizer.zero_grad()
            # forward pass
            outputs, _, _ = model(data.float(), id.float())
            loss = criterion(outputs.float(), label.float())

            # backward and optimize
            loss.backward()
            optimizer.step()

            if i % 30 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, epochs, i+1, training_data_count / 128, loss.item()))
        print('Train C of the {} model for the {} concept on the {} train mmWave data: {}'.format(model_type, Concepts[label_index], training_data_count, C))
    # store the model
    if not os.path.exists('model/' + model_type + Concepts[label_index]):
        os.makedirs('model/' + model_type + Concepts[label_index])
    torch.save(model.state_dict(), 'model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model')
    # ........................................................................................

    # ............................load and test the trained model.............................
    # load the model
    model.load_state_dict(torch.load('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'))
    print('the {} model for the {} concept has been successfully loaded!'.format(model_type, Concepts[label_index]))
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        C = np.zeros((2, 2))
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for data, label, id in test_data:
            data = data.to(device)
            label = label.to(device)
            id = id.to(device)
            # label = label.squeeze()
            outputs, _, _ = model(data.float(), id.float())
            _, predicted = torch.max(outputs.float(), 1)
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
        print('Test F1 score of the {} model for the concept {} on the {} test mmWave data: {}'.format(model_type, Concepts[label_index], testing_data_count, F1))
        print('Test C of the {} model for the concept {} on the {} test mmWave data: {}'.format(model_type, Concepts[label_index], testing_data_count, C))
        print('Test Accuracy of the {} model for the concept {} on the {} test mmWave data: {} %'.format(model_type, Concepts[label_index], testing_data_count, 100 * correct / total))
    # ..................................................................................................

def feature_extrac(x_train, y_train, x_test, y_test, label_index, model_type):
    """
    extract features from the trained sub_model

    Parameters
    ----------
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
    label_index : int
        indicate the index of the concept
    model_type : string
        indicate the type of the model for feature extraction
    Returns
    -------
    feature_index : int
        the index of the feature in the target layer
    max_layer : int
        the index of the layer with maximum accuracy
    """
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
    hidden_size = 16
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

def speak_detection(x_train, y_train, label_train, level_train, x_test, y_test, label_test, level_test, features, layers, model_type):
    """
    extract features from the trained sub_model

    Parameters
    ----------
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
    features : list
        the length of features equals the number of concepts, while each element of the list indicates the feature index of the target layer
        for the corresponding concept
    layers: list
        the length of layers equals the number of concepts, while each element of the list indicates the layer index of the corresponding sub_model
    model_type : string
        indicate the type of the model for feature extraction
    Returns
    -------
    """
    # .............basic information of training and testing set.......................
    training_data_count = len(x_train)  # number of training series
    testing_data_count = len(x_test)
    num_steps = len(x_train[0])  # timesteps per series
    num_input = len(x_train[0][0])  # input parameters per timestep
    # ..................................................................................

    # .............Hyper Parameters and initial model parameters..........................
    epochs = 50
    hidden_size = 16
    num_layers = 5
    num_classes = 2
    batch_size = 8
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
    feature_layer = torch.zeros(batch_size, len(Concepts))
    fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(len(Concepts), num_classes, bias=False),
    ).to(device)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(fc.parameters(), lr)

    # train the model
    # model.eval()
    for i in model.parameters():
        i.requires_grad = False
    for epoch in range(epochs):
        C = np.zeros(2)
        # if epoch % 10 == 0:
        #     lr = lr / 2
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        for i, (data, labels) in enumerate(train_data):
            if len(data) < batch_size:
                print(len(data))
                continue
            data = data.to(device)
            for label_index in range(len(Concepts)):
                # load the model
                if not os.path.exists('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'):
                    feature_layer[:, label_index] = 0
                else:
                    model.load_state_dict(torch.load('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'))
                    # print('the model of {} has been successfully loaded!'.format(Concepts[label_index]))
                    # .....................................................................................
                    output_unused, h_n, c_n = model(data.float())
                    feature_layer[:, label_index] = h_n[layers[label_index], :, features[label_index]]
            feature_layer = feature_layer.to(device)
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

            # if i % 30 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, epochs, i+1, training_data_count / batch_size, loss.item()))
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
            if len(data) < batch_size:
                print(len(data))
                continue
            data = data.to(device)
            for label_index in range(len(Concepts)):
                # load the model
                if not os.path.exists('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'):
                    feature_layer[:, label_index] = 0
                else:
                    model.load_state_dict(torch.load('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'))
                    # print('the model of {} has been successfully loaded!'.format(Concepts[label_index]))
                    # .....................................................................................
                    output_unused, h_n, c_n = model(data.float())
                    feature_layer[:, label_index] = h_n[layers[label_index], :, features[label_index]]
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

def TDNN_train(x_train, y_train, id_train, x_test, y_test, id_test, label_index, model_type):

    # .............basic information of training and testing set.......................
    training_data_count = len(x_train)  # number of training series
    testing_data_count = len(x_test)
    num_steps = len(x_test[0])  # timesteps per series
    num_input = len(x_test[0][0])  # input parameters per timestep
    # ..................................................................................

    # ..................................................................................
    # use GetLoader to load the data and return Dataset object, which contains data and labels
    torch_data = GetLoader(x_train, y_train, id_train)
    train_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)
    torch_data = GetLoader(x_test, y_test, id_test)
    test_data = DataLoader(torch_data, batch_size=128, shuffle=True, drop_last=False)
    # ....................................................................................

    # .............Hyper Parameters and initial model parameters..........................
    epochs = 30
    # hidden_size = 64
    # num_layers = 5
    num_classes = 2
    lr = 0.01  # learning rate
    # initial model
    model = simpleTDNN(num_classes).to(device)
    for name, param in model.named_parameters():
        nn.init.constant_(param, 0.3)
    # loss and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # .....................................................................................

    # ...........................train and store the model.................................
    # train the model
    for epoch in range(epochs):
        C = np.zeros(2)
        if epoch % 10 == 0:
            lr = lr / 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for i, (data, labels, id) in enumerate(train_data):
            data = data.to(device)
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

            optimizer.zero_grad()
            # forward pass
            outputs = model(data.float())
            loss = criterion(outputs.float(), label.float())

            # backward and optimize
            loss.backward()
            optimizer.step()

            if i % 30 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, training_data_count / 128, loss.item()))
        print('Train C of the model on the {} train mmWave data: {}'.format(training_data_count, C))
    # store the model
    if not os.path.exists('model/' + model_type + Concepts[label_index]):
        os.makedirs('model/' + model_type + Concepts[label_index])
    torch.save(model.state_dict(), 'model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model')
    # ........................................................................................

    # ............................load and test the trained model.............................
    # load the model
    model.load_state_dict(torch.load('model/' + model_type + Concepts[label_index] + '/' + 'LSTM_model'))
    print("the model has been successfully loaded!")
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        C = np.zeros((2, 2))
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for data, label, id in test_data:
            data = data.to(device)
            label = label.to(device)
            outputs = model(data.float())
            _, predicted = torch.max(outputs.float(), 1)
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
        print('Test Accuracy of the model on the {} test mmWave data of concept {}: {} %'.format(testing_data_count,
                                                                                                 Concepts[label_index],
                                                                                                 100 * correct / total))
    # ..................................................................................................