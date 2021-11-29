# LSTM
import torch.nn as nn
import torch
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# __init__ is basically a function which will "initialize"/"activate" the properties of the class for a specific object
# self represents that object which will inherit those properties
class simpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(simpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=False, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # c_n shape (n_layers, batch, hidden_size)

        # forward propagate lstm
        out, (h_n, c_n) = self.lstm(x)

        # select the output of the last moment.
        out = self.fc(out[:, -1, :])
        # return out
        return out, h_n, c_n
