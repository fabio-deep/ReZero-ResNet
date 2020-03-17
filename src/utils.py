# -*- coding: utf-8 -*-
"""
    A few useful utilities.
"""
import os
import logging
import numpy as np

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """ Creates a custom pytorch dataset, mainly
        used for creating validation set splits.
    """

    def __init__(self, data, labels, transform):
        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])
        if isinstance(data, torch.Tensor):
            data = data.numpy() # to work with `ToPILImage'
        self.data = data[idx]
        self.labels = labels[idx]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.labels[idx]


def random_split(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.
    """
    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'train': torch.stack(train_x), 'valid': torch.stack(valid_x)}, \
            {'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'train': torch.from_numpy(np.stack(train_x)),
            'valid': torch.from_numpy(np.stack(valid_x))}, \
        {'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}


def sample_weights(labels):
    """ Calculates per sample weights. """
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]


class Standardize(object):
    """ Standardizes a 'PIL Image' such that each channel
        gets zero mean and unit variance. """

    def __call__(self, img):
        return (img - img.mean(dim=(1, 2), keepdim=True)) \
            / torch.clamp(img.std(dim=(1, 2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def experiment_config(args):
    """ Handles experiment configuration and creates new dirs for model.
    """
    # check number of models already saved in 'experiments' dir, add 1 to get new model number
    experiments_dir = os.path.join(os.path.split(os.getcwd())[0], 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    model_num = len(os.listdir(experiments_dir)) + 1

    # create all save dirs
    model_dir = os.path.join(os.path.split(os.getcwd())[0],
                             'experiments', 'Model_'+str(model_num))
    args.summaries_dir = os.path.join(model_dir, 'summaries')
    args.checkpoint_dir = os.path.join(model_dir, 'checkpoint.pt')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(args.summaries_dir, exist_ok=True)

    # save hyperparameters in .txt file
    with open(os.path.join(model_dir, 'hyperparams.txt'), 'w') as logs:
        for key, value in vars(args).items():
            logs.write('--{0}={1} '.format(str(key), str(value)))

    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(os.path.join(model_dir, 'trainlogs.txt')),
                                  logging.StreamHandler()])
    return args


def print_network(model, args):
    """ Utility for printing out a model's architecture.
    """
    logging.info('-'*70)  # print some info on architecture
    logging.info('{:>25} {:>27} {:>15}'.format('Layer.Parameter', 'Shape', 'Param#'))
    logging.info('-'*70)

    for param in model.state_dict():
        p_name = param.split('.')[-2]+'.'+param.split('.')[-1]
        # don't print batch norm layers for prettyness
        if p_name[:2] != 'BN' and p_name[:2] != 'bn':
            logging.info(
                '{:>25} {:>27} {:>15}'.format(
                    p_name,
                    str(list(model.state_dict()[param].squeeze().size())),
                    '{0:,}'.format(np.product(list(model.state_dict()[param].size())))
                )
            )
    logging.info('-'*70)

    logging.info('\nTotal params: {:,}\n\nSummaries dir: {}\n'.format(
        sum(p.numel() for p in model.parameters()),
        args.summaries_dir))

    for key, value in vars(args).items():
        if str(key) != 'print_progress':
            logging.info('--{0}: {1}'.format(str(key), str(value)))
