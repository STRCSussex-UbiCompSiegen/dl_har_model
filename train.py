##################################################
# All functions related to training a deep learning architecture using sensor-based activity data.
##################################################
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
##################################################

import os
import random
import time
from copy import copy
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight
from torch.utils.data import DataLoader

from dl_har_dataloader.datasets import SensorDataset
from dl_har_model.eval import eval_one_epoch, eval_model
from dl_har_model.model.DeepConvLSTM import DeepConvLSTM
from utils import paint, AverageMeter
from dl_har_model.model_utils import init_weights, apply_sliding_window, init_loss, init_optimizer, init_scheduler, \
    seed_torch

train_on_gpu = torch.cuda.is_available()  # Check for cuda


def cross_validate(valid_type, args, verbose=False):
    """
    Train model for a number of epochs.

    :param dataset: A SensorDataset containing the complete data to be used for training and validation.
    :param valid_type: Type of validation which is employed. Options: 'loso'
    :param args: A dict containing config options for the training.
    Required keys:
                    'model': str, model architecture to be used (default deepconvlstm).
                    'batch_size_train': int, number of windows to process in each training batch (default 256)
                    'batch_size_test': int, number of windows to process in each testing batch (default 256)
                    'optimizer': str, optimizer function to use. Options: 'Adam' or 'RMSProp'. Default 'Adam'.
                    'use_weights': bool, whether to use weighted or non-weighted CE-loss. Default 'True'.
                    'lr': float, maximum initial learning rate. Default 0.001.
                    'lr_schedule': str, type of learning rate schedule to use. Default 'step'
                    'lr_step': int, interval at which to decrease the learning rate. Default 10.
                    'lr_decay': float, factor by which to  decay the learning rate. Default 0.9.
                    'init_weights': str, How to initialize weights. Options 'orthogonal' or None. Default 'orthogonal'.
                    'epochs': int, Total number of epochs to train the model for. Default 300.

    :param verbose: A boolean indicating whether or not to print results.
    :return: training and validation losses, accuracies, f1 weighted and macro across epochs
    """
    if verbose:
        print(paint("Running HAR training loop ..."))
    start_time = time.time()

    config_dataset = {
        "dataset": args['dataset'],
        "window": args['window'],
        "stride": args['stride'],
        "stride_test": args['stride_test'],
        "path_processed": f"data/{args['dataset']}",
    }

    results_array = pd.DataFrame(columns=['v_type', 'seed', 'sbj', 't_loss', 't_acc', 't_fm', 't_fw', 'v_loss', 'v_acc',
                                          'v_fm', 'v_fw'])
    test_results_array = pd.DataFrame(columns=['v_type', 'seed', 'test_loss', 'test_acc', 'test_fm', 'test_fw'])

    preds_array = pd.DataFrame(columns=['v_type', 'seed', 'sbj', 'val_preds', 'test_preds'])

    for seed in args['seeds']:
        if verbose:
            print(paint("Running with random seed set to {0}...".format(str(seed))))
        seed_torch(seed)
        if valid_type == 'loso':
            if verbose:
                print(paint("Applying Leave-One-Subject-Out Cross-Validation ..."))

            dataset = SensorDataset(**config_dataset)

            args['n_classes'] = dataset.n_classes
            args['n_channels'] = dataset.n_channels

            for sbj in range(dataset.num_sbj):
                t_data, v_data = dataset.loso_split(sbj)
                if verbose:
                    print(paint("Training Subject {0}/{1} ...".format(str(sbj + 1), dataset.num_sbj)))
                    print(paint("Train Dataset Size: {0}".format(len(t_data))))
                    print(paint("Valid Dataset Size: {0}".format(len(v_data))))

                t_data = copy(dataset).alter_data(t_data[:, :-1], t_data[:, -1], 'train')
                v_data = copy(dataset).alter_data(v_data[:, :-1], v_data[:, -1], 'val')

                # Normalize data wrt. statistics of the training set
                mean = np.mean(t_data.data[:, 1:], axis=0)
                std = np.std(t_data.data[:, 1:], axis=0)

                t_data.normalize(mean, std)
                v_data.normalize(mean, std)

                train_x, train_y = apply_sliding_window(t_data.data, t_data.target, t_data.window, t_data.stride)
                valid_x, valid_y = apply_sliding_window(v_data.data, v_data.target, v_data.window, v_data.stride)

                t_data = t_data.alter_data(train_x[:, :, 1:], train_y, 'train')
                v_data = v_data.alter_data(valid_x[:, :, 1:], valid_y, 'val')

                t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, model, criterion = train_model(t_data, v_data, args, verbose)

                results_row = {'v_type': 'loso',
                               'seed': seed,
                               'sbj': sbj,
                               't_loss': t_loss,
                               't_acc': t_acc,
                               't_fm': t_fm,
                               't_fw': t_fw,
                               'v_loss': v_loss,
                               'v_acc': v_acc,
                               'v_fm': v_fm,
                               'v_fw': v_fw
                               }
                results_array = results_array.append(results_row, ignore_index=True)

                _, _, _, _, _, val_preds = eval_model(model, criterion, v_data, args)

                preds_row = {'v_type': 'loso',
                             'seed': seed,
                             'sbj': sbj,
                             'val_preds': val_preds.tolist(),
                             'test_preds': None,
                             }

                preds_array = preds_array.append(preds_row, ignore_index=True)

                if verbose:
                    print("SUBJECT: {}/{}".format(sbj + 1, dataset.num_sbj),
                          "\nAvg. Train Loss: {:.4f}".format(np.mean(t_loss)),
                          "Train Acc: {:.4f}".format(t_acc[-1]),
                          "Train F1 (M): {:.4f}".format(t_fm[-1]),
                          "Train F1 (W): {:.4f}".format(t_fw[-1]),
                          "\nValid Loss: {:.4f}".format(np.mean(v_loss)),
                          "Valid Acc: {:.4f}".format(v_acc[-1]),
                          "Valid F1 (M): {:.4f}".format(v_fm[-1]),
                          "Valid F1 (W): {:.4f}".format(v_fw[-1]))

        elif valid_type == 'split':
            if verbose:
                print(paint("Applying Train-Valid Split ..."))
            config_dataset['prefix'] = 'train'
            t_data = SensorDataset(**config_dataset)
            config_dataset['prefix'] = 'val'
            v_data = SensorDataset(**config_dataset)
            config_dataset['prefix'] = 'test'
            test_data = SensorDataset(**config_dataset)

            # Normalize data wrt. statistics of the training set
            mean = np.mean(t_data.data[:, 1:], axis=0)
            std = np.std(t_data.data[:, 1:], axis=0)

            t_data.normalize(mean, std)
            v_data.normalize(mean, std)
            test_data.normalize(mean, std)

            args['n_classes'] = t_data.n_classes
            args['n_channels'] = t_data.n_channels

            train_x, train_y = apply_sliding_window(t_data.data, t_data.target, t_data.window, t_data.stride)
            valid_x, valid_y = apply_sliding_window(v_data.data, v_data.target, v_data.window, v_data.stride)
            test_x, test_y = apply_sliding_window(v_data.data, v_data.target, v_data.window, v_data.stride_test)

            t_data = t_data.alter_data(train_x[:, :, 1:], train_y, 'train')
            v_data = v_data.alter_data(valid_x[:, :, 1:], valid_y, 'val')
            test_data.alter_data(test_x[:, :, 1:], test_y, 'test')

            t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, model, criterion = train_model(t_data, v_data, args, verbose)

            if verbose:
                print("Avg. Train Loss: {:.4f}".format(np.mean(t_loss)),
                      "Train Acc: {:.4f}".format(t_acc[-1]),
                      "Train F1 (M): {:.4f}".format(t_fm[-1]),
                      "Train F1 (W): {:.4f}".format(t_fw[-1]),
                      "\nValid Loss: {:.4f}".format(np.mean(v_loss)),
                      "Valid Acc: {:.4f}".format(v_acc[-1]),
                      "Valid F1 (M): {:.4f}".format(v_fm[-1]),
                      "Valid F1 (W): {:.4f}".format(v_fw[-1]))

            results_row = {'v_type': 'split',
                           'seed': seed,
                           'sbj': -1,
                           't_loss': t_loss,
                           't_acc': t_acc,
                           't_fm': t_fm,
                           't_fw': t_fw,
                           'v_loss': v_loss,
                           'v_acc': v_acc,
                           'v_fm': v_fm,
                           'v_fw': v_fw
                           }

            results_array = results_array.append(results_row, ignore_index=True)

            _, _, _, _, _, val_preds = eval_model(model, criterion, v_data, args)
            loss_test, acc_test, fm_test, fw_test, elapsed, test_preds = eval_model(model, criterion, test_data, args)

            tests_results_row = {'v_type': 'split',
                                 'seed': seed,
                                 'test_loss': loss_test,
                                 'test_acc': acc_test,
                                 'test_fm': fm_test,
                                 'test_fw': fw_test,
                                 }

            preds_row = {'v_type': 'split',
                         'seed': seed,
                         'sbj': -1,
                         'val_preds': val_preds.tolist(),
                         'test_preds': test_preds.tolist(),
                        }

            test_results_array = test_results_array.append(tests_results_row, ignore_index=True)
            preds_array = preds_array.append(preds_row, ignore_index=True)

        elapsed = round(time.time() - start_time)
        elapsed = str(timedelta(seconds=elapsed))
        if verbose:
            print(paint(f"Finished HAR training loop (h:m:s): {elapsed}"))
            print(paint("--" * 50, "blue"))

    if args['valid_type'] == 'split':
        return results_array, test_results_array, preds_array
    elif args['valid_type'] == 'loso':
        return results_array, None, preds_array


def train_model(train_data, val_data, args, verbose=False):
    """
    Train model for a number of epochs.

    :param train_data: A SensorDataset containing the data to be used for training the model.
    :param val_data: A SensorDataset containing the data to be used for validating the model.
    :param args: A dict containing config options for the training.
    Required keys:
                    'model': str, model architecture to be used (default deepconvlstm).
                    'batch_size_train': int, number of windows to process in each training batch (default 256)
                    'batch_size_test': int, number of windows to process in each testing batch (default 256)
                    'optimizer': str, optimizer function to use. Options: 'Adam' or 'RMSProp'. Default 'Adam'.
                    'use_weights': bool, whether to use weighted or non-weighted CE-loss. Default 'True'.
                    'lr': float, maximum initial learning rate. Default 0.001.
                    'lr_schedule': str, type of learning rate schedule to use. Default 'step'
                    'lr_step': int, interval at which to decrease the learning rate. Default 10.
                    'lr_decay': float, factor by which to  decay the learning rate. Default 0.9.
                    'init_weights': str, How to initialize weights. Options 'orthogonal' or None. Default 'orthogonal'.
                    'epochs': int, Total number of epochs to train the model for. Default 300.

    :param verbose: A boolean indicating whether or not to print results.
    :return: training and validation losses, accuracies, f1 weighted and macro across epochs
    """
    loader = DataLoader(train_data, args['batch_size_train'], True)
    loader_val = DataLoader(val_data, args['batch_size_test'], False)

    if args['model'] == 'deepconvlstm':
        model = DeepConvLSTM(n_channels=train_data.n_channels, n_classes=args['n_classes'], dataset=args['dataset'],
                             weights_init=args['weights_init'])
    elif args['model'] == 'shallow_deepconvlstm':
        model = DeepConvLSTM(n_channels=train_data.n_channels, n_classes=args['n_classes'], dataset=args['dataset'],
                             weights_init=args['weights_init'], lstm_layers=1)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_data.target + 1),
                                                      y=train_data.target + 1)
    criterion = init_loss(args)
    if args['use_weights']:
        criterion.weight = torch.FloatTensor(class_weights)

    if train_on_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = init_optimizer(model, args)

    if args['lr_step'] > 0:
        scheduler = init_scheduler(optimizer, args)

    if verbose:
        print(paint("[-] Initializing weights (" + args['weights_init'] + ")..."))
    init_weights(model)

    metric_best = 0.0

    t_loss, t_acc, t_fm, t_fw = [], [], [], []
    v_loss, v_acc, v_fm, v_fw = [], [], [], []

    for epoch in range(args['epochs']):
        if verbose:
            print("--" * 50)
            print("[-] Learning rate: ", optimizer.param_groups[0]["lr"])
        train_one_epoch(model, loader, criterion, optimizer, verbose=verbose)
        loss, acc, fm, fw = eval_one_epoch(model, loader, criterion)
        loss_val, acc_val, fm_val, fw_val = eval_one_epoch(model, loader_val, criterion)

        t_loss.append(loss)
        t_acc.append(acc)
        t_fm.append(fm)
        t_fw.append(fw)
        v_loss.append(loss_val)
        v_acc.append(acc_val)
        v_fm.append(fm_val)
        v_fw.append(fw_val)

        if verbose:
            print(
                paint(
                    f"[-] Epoch {epoch + 1}/{args['epochs']}"
                    f"\tTrain loss: {loss:.2f} \tacc: {100 * acc:.2f}(%)\tfm: {100 * fm:.2f}(%)\tfw: {100 * fw:.2f}"
                    f"(%)\t"
                )
            )

            print(
                paint(
                    f"[-] Epoch {epoch + 1}/{args['epochs']}"
                    f"\tVal loss: {loss_val:.2f} \tacc: {100 * acc_val:.2f}(%)\tfm: {100 * fm_val:.2f}(%)"
                    f"\tfw: {100 * fw_val:.2f}(%)"
                )
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }

        metric = fm_val
        if metric >= metric_best:
            if verbose:
                print(paint(f"[*] Saving checkpoint... ({metric_best}->{metric})", "blue"))
            metric_best = metric
            torch.save(
                checkpoint, os.path.join(model.path_checkpoints, "checkpoint_best.pth")
            )

        if epoch % 5 == 0:
            torch.save(
                checkpoint,
                os.path.join(model.path_checkpoints, f"checkpoint_{epoch}.pth"),
            )

        if args['lr_step'] > 0:
            scheduler.step()

    return t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, model, criterion


def train_one_epoch(model, loader, criterion, optimizer, print_freq=100, verbose=False):
    """
    Train model for a one of epoch.

    :param model: Model which is to be trained.
    :param loader: A DataLoader object containing the data to be used for training the model.
    :param criterion: The loss object.
    :param optimizer: The optimizer object.
    :param print_freq: int, How often to print loss during each epoch if verbose=True. Default 100.

    :param verbose: A boolean indicating whether or not to print results.
    :return: training and validation losses, accuracies, f1 weighted and macro across epochs
    """

    losses = AverageMeter("Loss")
    model.train()

    for batch_idx, (data, target, idx) in enumerate(loader):
        if train_on_gpu:
            data = data.cuda()
            target = target.view(-1).cuda()
        else:
            target = target.view(-1)
        z, logits = model(data)
        loss = criterion(logits, target)
        losses.update(loss.item(), data.shape[0])

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if verbose:
            if batch_idx % print_freq == 0:
                print(f"[-] Batch {batch_idx + 1}/{len(loader)}\t Loss: {str(losses)}")
