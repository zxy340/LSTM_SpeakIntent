import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),  # 对这16个结果进行规范处理，
            nn.ReLU(),	 # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),	 # 14+2*2-5+1=14  该次卷积后output_size = 14*14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))	 # 14/2=7 池化后为7*7

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),	 # 14+2*2-5+1=14  该次卷积后output_size = 14*14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))	 # 14/2=7 池化后为7*7

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),	 # 14+2*2-5+1=14  该次卷积后output_size = 14*14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))	 # 14/2=7 池化后为7*7

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),	 # 14+2*2-5+1=14  该次卷积后output_size = 14*14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))	 # 14/2=7 池化后为7*7

        self.fc = nn.Linear(2*2*64, num_classes)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out = out5.reshape(out5.size(0), -1)
        out = self.fc(out)
        # return out
        return out, out1, out2, out3, out4, out5