##################################################
# DeepConvLSTM model as proposed by Ordonez and Roggen (2015) in "Deep Convolutional and LSTM Recurrent Neural Networks
# for Multimodal Wearable Activity Recognition (https://doi.org/10.3390/s16010115)
##################################################
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
##################################################

import torch.nn as nn
from utils import makedir


class DeepConvLSTM(nn.Module):

    def __init__(self, n_channels, n_classes, dataset, experiment='default', conv_kernels=64,
                 kernel_size=5, lstm_units=128, lstm_layers=2, model='DeepConvLSTM'):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, lstm_units, num_layers=lstm_layers)

        self.classifier = nn.Linear(lstm_units, n_classes)

        self.activation = nn.ReLU()

        self.model = model
        self.dataset = dataset
        self.experiment = experiment

        makedir(self.path_checkpoints)
        makedir(self.path_logs)
        makedir(self.path_visuals)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)

        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        x, h = self.lstm(x)
        x = x[-1, :, :]

        out = self.classifier(x)

        return None, out

    @property
    def path_checkpoints(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/checkpoints/"

    @property
    def path_logs(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/logs/"

    @property
    def path_visuals(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/visuals/"