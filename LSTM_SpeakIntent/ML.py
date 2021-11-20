import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

Concepts = [
    'Inner_Brow_Raiser',     # AU01
    'Outer_Brow_Raiser',     # AU02
    'Brow_Lowerer',          # AU04
    'Upper_Lid_Raiser',      # AU05
    'Cheek_Raiser',          # AU06
    'Lid_Tightener',         # AU07
    'Nose_Wrinkler',         # AU09
    'Upper_Lip_Raiser',      # AU10
    'Lip_Corner_Puller',     # AU12
    'Dimpler',               # AU14
    'Lip_Corner_Depressor',  # AU15
    'Chin_Raiser',           # AU17
    'Lip_stretcher',         # AU20
    'Lip_Tightener',         # AU23
    'Lips_part',             # AU25
    'Jaw_Drop',              # AU26
    'Lip_Suck',              # AU28
    'Blink'                  # AU45
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
label_index = 0  # indicate which concept to train the model
data_path = '/mnt/stuff/xiaoyu/data/'  # the path where 'x_data.npy' and 'y_data.npy' are located
model_type = 'LSTM/'

# ...........................load data...........................................
# load data from saved files, the variable "x_data" stores mmWave data, the variable "y_data" stores labels
# we split 3/4 data as training data, and 1/4 data as testing data
# the variable "x_train" stores mmWave data for training set, the variable "y_train" stores labels for training set
# the variable "x_test" stores mmWave data for testing set, the variable "y_test" stores labels for testing set
x_train = np.zeros((1, 128, 192))
y_train = np.zeros((1,))
x_test = np.zeros((1, 128, 192))
y_test = np.zeros((1,))
for i in range(int(len(users)/16*9)):
    user = users[i]
    x_train = np.concatenate((x_train, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    y_train = np.concatenate((y_train, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
for i in range(int(len(users)/16*9), int(len(users)/4*3)):
    user = users[i]
    x_test = np.concatenate((x_test, np.load(data_path + user + '/' + Concepts[label_index] + '/x_data.npy')), axis=0)
    y_test = np.concatenate((y_test, np.load(data_path + user + '/' + Concepts[label_index] + '/y_data.npy')), axis=0)
x_train = x_train[1:]
y_train = y_train[1:]
x_test = x_test[1:]
y_test = y_test[1:]
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
# .................................................................................

# ......................train and test random forest model.........................
forest = RandomForestClassifier()
forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)
print('the final confusion_matrix outcome of RandomForest model is {}'.format(confusion_matrix(y_test, y_pred)))
print('the classification report of the concept {} is {}'.format(Concepts[label_index], classification_report(y_test, y_pred)))
print('the accuracy of the concept {} is {}'.format(Concepts[label_index], accuracy_score(y_test, y_pred)))
# .................................................................................

# ......................train and test SVM model...................................
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(x_test)
print('the final confusion_matrix outcome of SVM model is {}'.format(confusion_matrix(y_test, y_pred)))
print('the classification report of the concept {} is {}'.format(Concepts[label_index], classification_report(y_test, y_pred)))
print('the accuracy of the concept {} is {}'.format(Concepts[label_index], accuracy_score(y_test, y_pred)))
# .................................................................................

# ......................train and test Naive Bayes model...........................
Bayes = GaussianNB()
Bayes.fit(x_train, y_train)
y_pred = Bayes.predict(x_test)
print('the final confusion_matrix outcome of NaiveBayes model is {}'.format(confusion_matrix(y_test, y_pred)))
print('the classification report of the concept {} is {}'.format(Concepts[label_index], classification_report(y_test, y_pred)))
print('the accuracy of the concept {} is {}'.format(Concepts[label_index], accuracy_score(y_test, y_pred)))
# .................................................................................