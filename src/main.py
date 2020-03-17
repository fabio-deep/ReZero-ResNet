#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Distributed training using Pytorch boilerplate.
"""
import os
import logging
import random
import argparse
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import resnet
from train import train
from evaluate import evaluate
from datasets import get_dataloaders
from utils import experiment_config, print_network

warnings.filterwarnings("ignore")

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--dataset', default='cifar10',
                    help='e.g. cifar10, svhn, fashionmnist, mnist')
PARSER.add_argument('--n_epochs', type=int, default=1000,
                    help='number of epochs to train for.')
PARSER.add_argument('--batch_size', type=int, default=128,
                    help='number of images used to approx. gradient.')
PARSER.add_argument('--learning_rate', type=float, default=.1,
                    help='step size.')
PARSER.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay regularisation factor.')
PARSER.add_argument('--decay_rate', type=float, default=0.1,
                    help='factor to multiply with learning rate.')
PARSER.add_argument('--decay_steps', type=int, default=0,
                    help='decay learning rate every n steps.')
PARSER.add_argument('--optimiser', default='sgd',
                    help='e.g. sgd, adam')
PARSER.add_argument('--decay_milestones', nargs='+', type=int, default=[0],
                    help='epochs at which to multiply learning rate with decay rate.')
PARSER.add_argument('--padding', type=int, default=4,
                    help='padding augmentation factor.')
PARSER.add_argument('--brightness', type=float, default=0,
                    help='brightness augmentation factor.')
PARSER.add_argument('--contrast', type=float, default=0,
                    help='contrast augmentation factor.')
PARSER.add_argument('--patience', default=60,
                    help='number of epochs to wait for improvement.')
PARSER.add_argument('--crop_dim', type=int, default=32,
                    help='height and width of input cropping.')
PARSER.add_argument('--load_checkpoint_dir', default=None,
                    help='directory to load a checkpoint from.')
PARSER.add_argument('--no_distributed', dest='distributed', action='store_false',
                    help='choose whether or not to use distributed training.')
PARSER.set_defaults(distributed=True)
PARSER.add_argument('--inference', dest='inference', action='store_true',
                    help='infer from checkpoint rather than training.')
PARSER.set_defaults(inference=False)
PARSER.add_argument('--half_precision', dest='half_precision', action='store_true',
                    help='train using fp16.')
PARSER.set_defaults(half_precision=False)


def setup(distributed):
    """ Sets up for optional distributed training.

    For distributed training run as:
        python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py
    To kill zombie processes use:
        kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')

    For data parallel training on GPUs or CPU training run as:
        python main.py --no_distributed
    """
    if distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device = torch.device(f'cuda:{local_rank}')  # unique on individual node

        print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
            os.environ.get('WORLD_SIZE'),
            os.environ.get('RANK'),
            os.environ.get('LOCAL_RANK'),
            os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
    else:
        local_rank = None
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 8  # 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True

    return device, local_rank


def main():
    """ Main method. """
    args = PARSER.parse_known_args()[0]

    # sets up the backend for distributed training (optional)
    device, local_rank = setup(distributed=args.distributed)

    # retrieve the dataloaders for the chosen dataset
    dataloaders, args = get_dataloaders(args)

    # make dirs for current experiment logs, summaries etc
    args = experiment_config(args)

    # initialise the model
    model = resnet.resnet20(args)

    # place model onto GPU(s)
    if args.distributed:
        torch.cuda.set_device(device)
        torch.set_num_threads(1)  # n cpu threads / n processes per node
        model = DistributedDataParallel(model.cuda(),
                                        device_ids=[local_rank], output_device=local_rank)
        # only print stuff from process (rank) 0
        args.print_progress = True if int(os.environ.get('RANK')) == 0 else False
    else:
        if args.half_precision:
            model.half()  # convert to half precision
            for layer in model.modules():
                # keep batchnorm in 32 for convergence reasons
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        print('\nUsing', torch.cuda.device_count(), 'GPU(s).\n')
        model.to(device)
        args.print_progress = True

    if args.print_progress:
        print_network(model, args) # prints out the network architecture etc
        logging.info('\ntrain: {} - valid: {} - test: {}'.format(
            len(dataloaders['train'].dataset), len(dataloaders['valid'].dataset),
            len(dataloaders['test'].dataset)))

    # launch model training or inference
    if not args.inference:
        train(model, dataloaders, args)

        if args.distributed:  # cleanup
            torch.distributed.destroy_process_group()
    else:
        model.load_state_dict(torch.load(args.load_checkpoint_dir))
        test_loss, test_acc = evaluate(model, args, dataloaders['test'])
        print('[Test] loss {:.4f} - acc {:.4f} - acc_topk {:.4f}'.format(
            test_loss, test_acc[0], test_acc[1]))


if __name__ == '__main__':
    main()
