# Reference code link:
# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
# https://youngforever.tech/posts/2020-03-07-lstm%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF/
# https://blog.csdn.net/l8947943/article/details/103733473

import torch
import numpy as np
from data_loader import data_loading_LSTM, data_loading_speak
from main_function import LSTM_train, TDNN_train
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

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
data_path = '/mnt/stuff/xiaoyu/data/'  # the path where 'x_data.npy' and 'y_data.npy' are located
model_type = 'LSTM/'

# # train the LSTM model for each concept
# for label_index in range(len(Concepts)):
#     # load data
#     x_train, y_train, id_train, x_test, y_test, id_test = data_loading_LSTM(label_index, data_path)
#     # for frame in range(len(x_train)):
#     #     if y_train[frame] == 0:
#     #         continue
#     #     fig = plt.figure(figsize=(16,16))
#     #     for i in range(16):
#     #         ax = fig.add_subplot(4, 4, i+1)
#     #         plt.plot(x_train[frame, i, :])
#     #         ax.set_ylabel("value")
#     #         ax.set_xlabel("range")
#     #         plt.title("chirp=" + str(i) + " and frame_index=" + str(frame) + " and label=" + str(y_train[frame]))
#     #         plt.tight_layout()
#     #     plt.show()
#
#     print('the shape of x_train, y_train, id_train, x_test, y_test, id_test are {}, {}, {}, {}, {}, {} respectively'.format(np.shape(x_train), np.shape(y_train), np.shape(id_train),
#                                                                                                                np.shape(x_test), np.shape(y_test), np.shape(id_test)))
#     # if the current concept doesn't have data, then jump to the next concept
#     # if len(x_train) == 0:
#     if len(x_train) <= 100:
#         continue
#     LSTM_train(x_train, y_train, id_train, x_test, y_test, id_test, label_index, model_type)
#     # TDNN_train(x_train, y_train, id_train, x_test, y_test, id_test, label_index, model_type)
#
#
#     # x_train = np.reshape(x_train, (len(x_train), -1))
#     # y_train = np.reshape(y_train, (len(y_train), -1))
#     # x_test = np.reshape(x_test, (len(x_test), -1))
#     # y_test = np.reshape(y_test, (len(y_test), -1))
#     # ......................train and test random forest model.........................
#     # forest = RandomForestClassifier()
#     # forest.fit(x_train, y_train.ravel())
#     # y_pred = forest.predict(x_test)
#     # print('the final result of RandomForest model for concept {} is:'.format(Concepts[label_index]))
#     # print(confusion_matrix(y_test.ravel(), y_pred))
#     # print(classification_report(y_test.ravel(), y_pred))
#     # print('the RandomForest accuracy of the concept {} is {}'.format(Concepts[label_index], accuracy_score(y_test.ravel(), y_pred)))
#
#     # # ......................train and test Naive Bayes model...........................
#     # Bayes = GaussianNB()
#     # Bayes.fit(x_train, y_train.ravel())
#     # y_pred = Bayes.predict(x_test)
#     # print('the final result of NaiveBayes model for concept {} is:'.format(Concepts[label_index]))
#     # print(confusion_matrix(y_test.ravel(), y_pred))
#     # print(classification_report(y_test.ravel(), y_pred))
#     # print('the Bayes accuracy of the concept {} is {}'.format(Concepts[label_index], accuracy_score(y_test.ravel(), y_pred)))
#     #
#     # ......................train and test LGB model...........................
#     # # 转换为Dataset数据格式
#     # train_data = lgb.Dataset(x_train, label=y_train.ravel())
#     # validation_data = lgb.Dataset(x_test, label=y_test.ravel())
#     # # 参数
#     # # params = {
#     # #     'learning_rate': 0.1,
#     # #     'lambda_l1': 100,
#     # #     'lambda_l2': 100,
#     # #     'max_depth': 6,
#     # #     'num_leaves': 31,
#     # #     'objective': 'multiclass',  # 目标函数
#     # #     'num_class': 2,
#     # #     'n_estimators': 100,
#     # #     'early_stopping_round': 20,
#     # # }
#     # params = {
#     #     'seed': 1993,
#     #     'n_estimators': 500,
#     #     'max_depth': 7,
#     #     'nthread': 10,
#     #     'reg_alpha': 100,
#     #     'reg_lambda': 100,
#     #     'gamma': 1,
#     #     'learning_rate': 0.1,
#     #     'tree_method': "gpu_hist",
#     #     'gpu_id': 0
#     # }
#     # # 模型训练
#     # gbm = lgb.train(params, train_data, valid_sets=[validation_data])
#     # # 模型预测
#     # y_pred = gbm.predict(x_test)
#     # for index in range(len(y_pred)):
#     #     if y_pred[index] < 0.5:
#     #         y_pred[index] = 0
#     #     else:
#     #         y_pred[index] = 1
#     # # 模型评估
#     # print("The confusion_matrix of LGB model is {}".format(confusion_matrix(y_test.ravel(), y_pred)))
#     # print("The accuarcy of LGB is : %.2f%%" % (accuracy_score(y_test.ravel(), y_pred) * 100.0))
#     #
#     # # ......................train and test XGBoost model...........................
#     # params = {
#     #     'booster': 'gbtree',
#     #     'objective': 'multi:softmax',
#     #     'num_class': 2,
#     #     'gamma': 0.1,
#     #     'max_depth': 8,
#     #     # 'silent': 1,
#     #     'eta': 0.1,
#     #     'eval_metric': 'mlogloss',
#     # }
#     # plst = params.items()
#     # dtrain = xgb.DMatrix(x_train, y_train.ravel())  # 生成数据集格式
#     # model = xgb.train(params,
#     #                   dtrain,  # 训练的数据
#     #                   )  # xgboost模型训练
#     # # 对测试集进行预测
#     # dtest = xgb.DMatrix(x_test)
#     # y_pred = model.predict(dtest)
#     # # 计算准确率
#     # accuracy = accuracy_score(y_test.ravel(), y_pred)
#     # print("The confusion_matrix of XGBoost model is {}".format(confusion_matrix(y_test.ravel(), y_pred)))
#     # print("The accuarcy of XGBoost is : %.2f%%" % (accuracy * 100.0))
#     #
#     # # ......................train and test SVM model...................................
#     # svclassifier = svm.SVC(kernel='rbf', C=0.5, gamma=1, max_iter=1000)
#     # svclassifier.fit(x_train, y_train.ravel())
#     # y_pred = svclassifier.predict(x_test)
#     # print('the final result of SVM model for concept {} is:'.format(Concepts[label_index]))
#     # print(confusion_matrix(y_test.ravel(), y_pred))
#     # print(classification_report(y_test.ravel(), y_pred))
#     # print('the SVM accuracy of the concept {} is {}'.format(Concepts[label_index], accuracy_score(y_test.ravel(), y_pred)))

x_train, y_train, x_test, y_test = data_loading_speak(data_path)
print('the shape of x_train, y_train, x_test, y_test are {}, {}, {}, {} respectively'.format(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)))
x_train = np.reshape(x_train, (len(x_train), -1))
y_train = np.reshape(y_train, (len(y_train), -1))
x_test = np.reshape(x_test, (len(x_test), -1))
y_test = np.reshape(y_test, (len(y_test), -1))
np.save('./data/speakintent_data/x_train', x_train)
np.save('./data/speakintent_data/y_train', y_train)
np.save('./data/speakintent_data/x_test', x_test)
np.save('./data/speakintent_data/y_test', y_test)
x_train = np.load('./data/speakintent_data/x_train.npy')
y_train = np.load('./data/speakintent_data/y_train.npy')
x_test = np.load('./data/speakintent_data/x_test.npy')
y_test = np.load('./data/speakintent_data/y_test.npy')
# # ......................train and test random forest model.........................
forest = RandomForestClassifier()
forest.fit(x_train, y_train.ravel())
y_pred = forest.predict(x_test)
print('the final result of RandomForest model for speak intent is:')
print(confusion_matrix(y_test.ravel(), y_pred))
print(classification_report(y_test.ravel(), y_pred))
print('the RandomForest accuracy of the speak intent is {}'.format(accuracy_score(y_test.ravel(), y_pred)))

# ......................train and test Naive Bayes model...........................
Bayes = GaussianNB()
Bayes.fit(x_train, y_train.ravel())
y_pred = Bayes.predict(x_test)
print('the final result of NaiveBayes model for speak intent is:')
print(confusion_matrix(y_test.ravel(), y_pred))
print(classification_report(y_test.ravel(), y_pred))
print('the Bayes accuracy of the speak intent is {}'.format(accuracy_score(y_test.ravel(), y_pred)))

# ......................train and test LGB model...........................
# 转换为Dataset数据格式
train_data = lgb.Dataset(x_train, label=y_train.ravel())
validation_data = lgb.Dataset(x_test, label=y_test.ravel())
# 参数
# params = {
#     'learning_rate': 0.1,
#     'lambda_l1': 100,
#     'lambda_l2': 100,
#     'max_depth': 6,
#     'num_leaves': 31,
#     'objective': 'multiclass',  # 目标函数
#     'num_class': 2,
#     'n_estimators': 100,
#     'early_stopping_round': 20,
# }
params = {
    'seed': 1993,
    'n_estimators': 500,
    'max_depth': 7,
    'nthread': 10,
    'reg_alpha': 100,
    'reg_lambda': 100,
    'gamma': 1,
    'learning_rate': 0.1,
    'tree_method': "gpu_hist",
    'gpu_id': 0
}
# 模型训练
gbm = lgb.train(params, train_data, valid_sets=[validation_data])
# 模型预测
y_pred = gbm.predict(x_test)
for index in range(len(y_pred)):
    if y_pred[index] < 0.5:
        y_pred[index] = 0
    else:
        y_pred[index] = 1
# 模型评估
print("The confusion_matrix of LGB model for speak intent is {}".format(confusion_matrix(y_test.ravel(), y_pred)))
print("The accuarcy of LGB for speak intent is : %.2f%%" % (accuracy_score(y_test.ravel(), y_pred) * 100.0))