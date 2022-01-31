import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import dropout

class deepfm(nn.Module):
    def __init__(self):
        super(deepfm, self).__init__()
        self.deep_layers = [64, 32, 16]

        self.dense_1 = torch.nn.Sequential(
			torch.nn.Linear(128*192,64), # adjust by yourself
			torch.nn.Dropout(p=0.2),
			torch.nn.Tanh(),
			nn.BatchNorm1d(64)
		)

        self.dense_2 = torch.nn.Sequential(
			torch.nn.Linear(64,32), # adjust by yourself
			torch.nn.Dropout(p=0.2),
			torch.nn.Tanh(),
			nn.BatchNorm1d(32)
		)

        self.dense_3 = torch.nn.Sequential(
			torch.nn.Linear(32,16), # adjust by yourself
			torch.nn.Dropout(p=0.2),
			torch.nn.Tanh(),
			nn.BatchNorm1d(16)
		)

        self.dense_4 = torch.nn.Sequential(
			torch.nn.Linear(24688, 512),  # adjust by yourself
			torch.nn.Tanh(),
			nn.BatchNorm1d(512)
		)

        self.dense_5 = torch.nn.Sequential(
		    torch.nn.Linear(512, 64),  # adjust by yourself
		    torch.nn.Tanh(),
		    nn.BatchNorm1d(64)
	    )

        self.deep_layers_detail = [self.dense_1, self.dense_2, self.dense_3]
        # self.deep_layers_detail = [self.dense_1, self.dense_2]

        self.dense_o = torch.nn.Sequential(
				torch.nn.Linear(64,2), # adjust by yourself
			)

    def forward(self, input):
        # glorot = 0.02
        feat_value = torch.tensor(input)
        # label_value= torch.tensor(label)

        feat_value = dropout(feat_value, p=0.2)

        y_first_order = feat_value
        self.y_deep = dropout(y_first_order, p=0.2)

        self.y_deep_ = []
        for i in range(0, len(self.deep_layers)):
            self.y_deep = self.deep_layers_detail[i](self.y_deep)
            self.y_deep_.append(self.y_deep)

        self.y_first_order = dropout(y_first_order)
        self.concat_y = torch.cat(self.y_deep_,1)
        self.concat_y = torch.cat([self.y_first_order, self.concat_y],1)
        # print(np.shape(self.concat_y))
        self.output = self.dense_4(self.concat_y)
        self.output = self.dense_5(self.output)

        self.output = self.dense_o(self.output)

        return self.output









