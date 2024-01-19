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
from datetime import timedelta
from glob import glob

import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight
from torch.utils.data import DataLoader

from dl_har_model.eval import eval_one_epoch, eval_model
from utils import paint, AverageMeter
from dl_har_model.train_utils import compute_center_loss, get_center_delta, mixup_data, MixUpLoss, init_weights, \
    init_loss, init_optimizer, init_scheduler, seed_torch
from dl_har_dataloader.datasets import SensorDataset

train_on_gpu = torch.cuda.is_available()  # Check for cuda


def split_validate(model, train_args, dataset_args, seeds=None, verbose=False):
    """
    Train model for a number of epochs using split validation.

    :param model: A pytorch model for training. Must implement forward function and allow backprop.
    :param dict train_args: A dict containing args for training. For allowed keys see train_model arguments.
    :param dict dataset_args: A dict containing args for SensorDataset class excluding the prefix. For allowed keys see
    SensorDataset.__init__ arguments.
    :param verbose: A boolean indicating whether to print results.
    :param list seeds: A dict containing all random seeds used for training.

    :return: training and validation losses, accuracies, f1 weighted and macro across epochs and raw predictions
    """

    train_data = SensorDataset(prefix='train', **dataset_args)
    val_data = SensorDataset(prefix='val', **dataset_args)
    test_data = SensorDataset(prefix='test', **dataset_args)
    if seeds is None:
        seeds = [1]
    if verbose:
        print(paint("Running HAR training loop ..."))
    start_time = time.time()

    if verbose:
        print(paint("Applying Split-Validation..."))

    # Initialize lists of dictionaries to store results
    results_list = []
    test_results_list = []
    preds_list = []

    base_path_checkpoints = model.path_checkpoints

    for seed in seeds:
        if verbose:
            print(paint("Running with random seed set to {0}...".format(str(seed))))
        model.path_checkpoints = base_path_checkpoints + f"/seed_{seed}"
        seed_torch(seed)
        t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, criterion = \
            train_model(model, train_data, val_data, seed=seed, verbose=True, **train_args)
        _, _, _, _, _, val_preds = eval_model(model, val_data, criterion, seed=seed)
        loss_test, acc_test, fm_test, fw_test, elapsed, test_preds = eval_model(model, test_data, criterion, seed=seed)

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

        results_list = results_list.append(results_row)
        test_results_list = test_results_list.append(tests_results_row)
        preds_list = preds_list(preds_row)

    # After the loop, convert lists of dictionaries to DataFrames
    results_array = pd.DataFrame(results_list)
    test_results_array = pd.DataFrame(test_results_list)
    preds_array = pd.DataFrame(preds_list)

    elapsed = round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    if verbose:
        print(paint(f"Finished HAR training loop (h:m:s): {elapsed}"))
        print(paint("--" * 50, "blue"))

    return results_array, test_results_array, preds_array


def loso_cross_validate(model, train_args, dataset_args, seeds, verbose=False):
    """
    Train model for a number of epochs.

    :param model: A pytorch model for training. Must implement forward function and allow backprop.
    :param dict train_args: A dict containing args for training. For allowed keys see train_model arguments.
    :param dict dataset_args: A dict containing args for SensorDataset class excluding the prefix. For allowed keys see
    SensorDataset.__init__ arguments.
    :param seeds: A list of random seeds which are used during training runs.
    :param verbose: A boolean indicating whether or not to print results.

    :return: training and validation losses, accuracies, f1 weighted and macro across epochs and raw predictions
    """
    if verbose:
        print(paint("Running HAR training loop ..."))
    start_time = time.time()

    if verbose:
        print(paint("Applying Leave-One-Subject-Out Cross-Validation ..."))

    results_array = pd.DataFrame(columns=['v_type', 'seed', 'sbj', 't_loss', 't_acc', 't_fm', 't_fw', 'v_loss', 'v_acc',
                                          'v_fm', 'v_fw'])

    preds_array = pd.DataFrame(columns=['v_type', 'seed', 'sbj', 'val_preds', 'test_preds'])

    num_users = len(glob(os.path.join(dataset_args['path_processed'], 'User_*.npz')))
    users = [os.path.splitext(os.path.basename(x))[0] for x in glob(os.path.join(dataset_args['path_processed'],
                                                                                 'User_*.npz'))]

    base_path_checkpoints = model.path_checkpoints

    for seed in seeds:
        if verbose:
            print(paint("Running with random seed set to {0}...".format(str(seed))))
        seed_torch(seed)
        for i, val_user in enumerate(users):

            model.path_checkpoints = base_path_checkpoints + f"/seed_{seed}/" + val_user

            train_users = users.copy()
            train_users.remove(val_user)

            train_dataset = SensorDataset(prefix=train_users, **dataset_args)
            val_dataset = SensorDataset(prefix=val_user, **dataset_args)

            t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, criterion = \
                train_model(model, train_dataset, val_dataset, seed=seed, verbose=True, **train_args)

            results_row = {'v_type': 'loso',
                           'seed': seed,
                           'sbj': val_user,
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

            _, _, _, _, _, val_preds = eval_model(model, val_dataset, criterion, seed=seed)

            preds_row = {'v_type': 'loso',
                         'seed': seed,
                         'sbj': val_user,
                         'val_preds': val_preds.tolist(),
                         'test_preds': None,
                         }

            preds_array = preds_array.append(preds_row, ignore_index=True)

            if verbose:
                print("SUBJECT: {}/{}".format(i + 1, num_users),
                      "\nAvg. Train Loss: {:.4f}".format(np.mean(t_loss)),
                      "Train Acc: {:.4f}".format(t_acc[-1]),
                      "Train F1 (M): {:.4f}".format(t_fm[-1]),
                      "Train F1 (W): {:.4f}".format(t_fw[-1]),
                      "\nValid Loss: {:.4f}".format(np.mean(v_loss)),
                      "Valid Acc: {:.4f}".format(v_acc[-1]),
                      "Valid F1 (M): {:.4f}".format(v_fm[-1]),
                      "Valid F1 (W): {:.4f}".format(v_fw[-1]))

    elapsed = round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    if verbose:
        print(paint(f"Finished HAR training loop (h:m:s): {elapsed}"))
        print(paint("--" * 50, "blue"))

    return results_array, None, preds_array


def train_model(model, train_data, val_data, batch_size_train=256, batch_size_test=256, optimizer='Adam',
                use_weights=True, lr=0.001, lr_schedule='step', lr_step=10, lr_decay=0.9, weights_init='orthogonal',
                epochs=300, print_freq=100, loss='CrossEntropy', smoothing=0.0, weight_decay=0.0, seed=1,
                centerloss=False, lr_cent=1e-4, beta=0.5, mixup=False, alpha=0.5, verbose=False, save_checkpoints=False):
    """
    Train model for a number of epochs.

    :param model: A network object used for training and prediction.
    :param train_data: A SensorDataset containing the data to be used for training the model.
    :param val_data: A SensorDataset containing the data to be used for validating the model.
    :param int batch_size_train: Number of windows to process in each training batch (default 256)
    :param int batch_size_test: Number of windows to process in each testing batch (default 256)
    :param str optimizer: Optimizer function to use. Options: 'Adam' or 'RMSProp'. Default 'Adam'.
    :param bool use_weights: Whether to use weighted or non-weighted CE-loss. Default True.
    :param float lr: Maximum initial learning rate. Default 0.001.
    :param str lr_schedule: Type of learning rate schedule to use. Default 'step'
    :param int lr_step: Interval at which to decrease the learning rate. Default 10.
    :param float lr_decay: Factor by which to  decay the learning rate. Default 0.9.
    :param str weights_init: How to initialize weights. Options 'orthogonal' or None. Default 'orthogonal'.
    :param int epochs: Total number of epochs to train the model for. Default 300.
    :param int print_freq: How often to print loss during each epoch if verbose=True. Default 100.
    :param str loss: Loss function to use. Default 'Adam'.
    :param float smoothing: Amount of label smoothing to apply. Default 0.0 (no smoothing).
    :param float weight_decay: Amount of weight decay applied per batch. Default 0.0 (no decay).
    :param bool centerloss: Whether to augment the loss function with centerloss. In this case the model must implement
    centers. Default False.
    :param float lr_cent: Learning rate for centerloss.
    :param float beta: Weighting for centerloss.
    :param bool mixup: Whether to implement data augmentation with mixup.
    :param float alpha: Controls the distribution of labels for mixup.

    :param bool save_checkpoints: A boolean indicating whether to save checkpoints during model training.
    :param bool verbose: A boolean indicating whether to print results.
    :param int seed: Random seed which is to be used.
    :return: training and validation losses, accuracies, f1 weighted and macro across epochs
    """

    loader = DataLoader(train_data, batch_size_train, True, worker_init_fn=np.random.seed(int(seed)))
    loader_val = DataLoader(val_data, batch_size_test, False, worker_init_fn=np.random.seed(int(seed)))

    if use_weights:
        class_weights = torch.from_numpy(class_weight.compute_class_weight('balanced',
                                                                           classes=np.unique(train_data.target + 1),
                                                                           y=train_data.target + 1)).float()
    else:
        class_weights = None

    if train_on_gpu:
        criterion = init_loss(loss, smoothing, class_weights, train_on_gpu).cuda()
    else:
        criterion = init_loss(loss, smoothing, class_weights, train_on_gpu)

    optimizer = init_optimizer(model, optimizer, lr, weight_decay)

    if lr_step != 0:
        scheduler = init_scheduler(optimizer, lr_schedule, lr_step, lr_decay)

    if verbose:
        print(paint(f"Initializing weights ({weights_init})..."))

    init_weights(model, weights_init)

    metric_best = 0.0

    t_loss, t_acc, t_fm, t_fw = [], [], [], []
    v_loss, v_acc, v_fm, v_fw = [], [], [], []

    path_checkpoints = getattr(model, 'path_checkpoints', './models/custom_model/checkpoints')

    for epoch in range(epochs):
        if verbose:
            print("--" * 50)
            print("[-] Learning rate: ", optimizer.param_groups[0]["lr"])
        train_one_epoch(model, loader, criterion, optimizer, print_freq, centerloss, lr_cent, beta,
                    mixup, alpha, verbose)
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
                    f"[-] Epoch {epoch + 1}/{epochs}"
                    f"\tTrain loss: {loss:.2f} \tacc: {100 * acc:.2f}(%)\tfm: {100 * fm:.2f}(%)\tfw: {100 * fw:.2f}"
                    f"(%)\t"
                )
            )

            print(
                paint(
                    f"[-] Epoch {epoch + 1}/{epochs}"
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
                checkpoint, os.path.join(path_checkpoints, "checkpoint_best.pth")
            )

        if save_checkpoints:
            if epoch % 5 == 0:
                torch.save(
                    checkpoint,
                    os.path.join(path_checkpoints, f"checkpoint_{epoch}.pth")
                )

        if lr_step > 0:
            scheduler.step()

    return t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, criterion


def train_one_epoch(model, loader, criterion, optimizer, print_freq, centerloss, lr_cent, beta,
                    mixup, alpha, verbose):
    """
    Train model for a one of epoch.

    :param model: Model which is to be trained.
    :param loader: A DataLoader object containing the data to be used for training the model.
    :param criterion: The loss object.
    :param optimizer: The optimizer object.
    :param int print_freq: How often to print loss during each epoch if verbose=True. Default 100.
    :param bool centerloss: Enable loss function augmentation with centerloss.
    :param float lr_cent: Learning rate to for the centerloss.
    :param float beta: Weighting for the centerloss.
    :param bool mixup: Enable data augmentation with MixUp.
    :param float alpha: Mixup scaling factor.

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

        if centerloss:
            assert hasattr(model, 'centers'), "Model must implement class center tracking to enable centerloss."
            centers = model.centers

        if mixup:
            data, y_a_y_b_lam = mixup_data(data, target, alpha)

        z, logits = model(data)

        if mixup:
            criterion = MixUpLoss(criterion)
            loss = criterion(logits, y_a_y_b_lam)
        else:
            loss = criterion(logits, target)

        if centerloss:
            center_loss = compute_center_loss(z, centers, target)
            loss = loss + beta * center_loss

        losses.update(loss.item(), data.shape[0])

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if centerloss:
            center_deltas = get_center_delta(z.data.float(), centers.float(), target, lr_cent, train_on_gpu)
            model.centers = centers - center_deltas

        if verbose:
            if batch_idx % print_freq == 0:
                print(f"[-] Batch {batch_idx + 1}/{len(loader)}\t Loss: {str(losses)}")

        if mixup:
            criterion = criterion.get_old()
