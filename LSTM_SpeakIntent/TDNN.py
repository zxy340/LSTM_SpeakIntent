import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):

    def __init__(
            self,
            input_dim=192,
            output_dim=512,
            context_size=5,
            stride=1,
            dilation=1,
            batch_norm=True,
            dropout_p=0.0
    ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)

        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x

class simpleTDNN(TDNN):
    def __init__(self, num_classes):
        super(simpleTDNN, self).__init__()
        self.frame1 = TDNN(input_dim=192, output_dim=512, context_size=5, dilation=1)
        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.fc1 = nn.Sequential(
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.Sigmoid())
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Sigmoid())
        self.fc3 = nn.Sequential(
            nn.Linear(500, 128),
            nn.ReLU(),
            nn.Sigmoid())
        self.fc4 = nn.Linear(128, num_classes)
    def forward(self, x):
        # Input to frame1 is of shape (batch_size, T, 24)
        # Output of frame5 will be (batch_size, T-14, 1500)
        out1 = self.frame1(x)
        out2 = self.frame2(out1)
        out3 = self.frame3(out2)
        out4 = self.frame4(out3)
        out5 = self.frame5(out4)
        out = self.fc1(out5)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out