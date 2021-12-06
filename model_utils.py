##################################################
# Helper functions for model training and evaluation.
##################################################
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
##################################################

import torch
import numpy as np
from torch import nn


def init_weights(model, method):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network)

    :param network: network of which weights are to be initialised
    :return: network with initialised weights
    """
    for m in model.modules():
        if isinstance(m, nn.Linear) or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            if method == 'normal':
                torch.nn.init.normal_(m.weight)
            elif method == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight)
            elif method == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight)
            elif method == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.weight)
            elif method == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight)
            elif method == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            # LSTM initialisation
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    if method == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif method == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif method == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif method == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif method == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif method == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    if method == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif method == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif method == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif method == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif method == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif method == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)


def init_loss(loss, smoothing, weights):
    """
    Initialises an loss object for a given network.

    :return: loss object
    """
    if loss == 'CrossEntropy' or loss == 'cross-entropy' or loss == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
    return criterion


def init_optimizer(network, optimizer, lr, weight_decay):
    """
    Initialises an optimizer object for a given network.

    :param network: network for which optimizer and loss are to be initialised
    :return: optimizer object
    """
    # define optimizer and loss
    if optimizer == 'adadelta' or optimizer == 'AdaDelta':
        opt = torch.optim.Adadelta(network.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'adam' or optimizer == 'Adam':
        opt = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'rmsprop' or optimizer == 'RMSProp':
        opt = torch.optim.RMSprop(network.parameters(), lr=lr, weight_decay=weight_decay)
    return opt


def init_scheduler(optimizer, lr_schedule, lr_step, lr_decay):
    if lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)
    elif lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_step, factor=lr_decay)
    return scheduler
