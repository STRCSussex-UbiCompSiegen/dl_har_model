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


def init_weights(network):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network)

    :param network: network of which weights are to be initialised
    :return: network with initialised weights
    """
    for m in network.modules():
        # linear layers and conv layers
        if isinstance(m, nn.Linear) or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            if network.weights_init == 'normal':
                torch.nn.init.normal_(m.weight)
            elif network.weights_init == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight)
            elif network.weights_init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight)
            elif network.weights_init == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.weight)
            elif network.weights_init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight)
            elif network.weights_init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # LSTM initialisation
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    if network.weights_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif network.weights_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif network.weights_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif network.weights_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif network.weights_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif network.weights_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    if network.weights_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif network.weights_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif network.weights_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif network.weights_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif network.weights_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif network.weights_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)
    return network


def init_loss(config):
    """
    Initialises an loss object for a given network.

    :return: loss object
    """
    if config['loss'] == 'cross-entropy':
        criterion = nn.CrossEntropyLoss(label_smoothing=config['smoothing'])
    return criterion


def init_optimizer(network, config):
    """
    Initialises an optimizer object for a given network.

    :param network: network for which optimizer and loss are to be initialised
    :return: optimizer object
    """
    # define optimizer and loss
    if config['optimizer'] == 'adadelta':
        opt = torch.optim.Adadelta(network.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        opt = torch.optim.Adam(network.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'rmsprop':
        opt = torch.optim.RMSprop(network.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    return opt


def init_scheduler(optimizer, config):
    if config['lr_schedule'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_step'], config['lr_decay'])
    elif config['lr_schedule'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['lr_step'], factor=config['lr_decay'])
    return scheduler


def sliding_window_samples(data, samples_per_window, overlap_ratio):
    """
    Return a sliding window measured in number of samples over a data array.

    :param data: input array, can be numpy or pandas dataframe
    :param samples_per_window: window length as number of samples per window
    :param overlap_ratio: overlap is meant as percentage and should be an integer value
    :return: tuple of windows and indices
    """
    windows = []
    indices = []
    curr = 0
    win_len = int(samples_per_window)
    if overlap_ratio is not None:
        overlapping_elements = int((overlap_ratio / 100) * (win_len))
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    else:
        overlapping_elements = 0
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        curr = curr + win_len - overlapping_elements
    try:
        result_windows = np.array(windows)
        result_indices = np.array(indices)
    except:
        result_windows = np.empty(shape=(len(windows), win_len, data.shape[1]), dtype=object)
        result_indices = np.array(indices)
        for i in range(0, len(windows)):
            result_windows[i] = windows[i]
            result_indices[i] = indices[i]
    return result_windows, result_indices


def apply_sliding_window(data_x, data_y, sliding_window_size, sliding_window_overlap):
    """
    Function which transforms a dataset into windows of a specific size and overlap.

    :param data_x: numpy float array
        Array containing the features (can be 2D)
    :param data_y: numpy float array
        Array containing the corresponding labels to the dataset (is 1D)
    :param sliding_window_size: integer or float
        Size of each window (either in seconds or units)
    :param sliding_window_overlap: integer
        Amount of overlap between the sliding windows (measured in percentage, e.g. 20 is 20%)
    :return:
    """
    full_data = np.concatenate((data_x, data_y[:, None]), axis=1)
    output_x = None
    output_y = None

    for i, subject in enumerate(np.unique(full_data[:, 0])):
        subject_data = full_data[full_data[:, 0] == subject]
        subject_x, subject_y = subject_data[:, :-1], subject_data[:, -1]
        tmp_x, _ = sliding_window_samples(subject_x, sliding_window_size, sliding_window_overlap)
        tmp_y, _ = sliding_window_samples(subject_y, sliding_window_size, sliding_window_overlap)
        if output_x is None:
            output_x = tmp_x
            output_y = tmp_y
        else:
            output_x = np.concatenate((output_x, tmp_x), axis=0)
            output_y = np.concatenate((output_y, tmp_y), axis=0)
    output_y = [[i[-1]] for i in output_y]
    return output_x, np.array(output_y).flatten()
