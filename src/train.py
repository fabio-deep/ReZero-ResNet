# -*- coding: utf-8 -*-
"""
    Training loop.
"""
import gc
import time
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from evaluate import evaluate


def train(model, dataloaders, args):
    """ Trains a given model and dataset.
    """
    # optimisers
    if args.optimiser == 'adam':
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    elif args.optimiser == 'sgd':
        optimiser = optim.SGD(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise NotImplementedError('{} not setup.'.format(args.optimiser))

    # lr schedulers
    if args.decay_steps > 0:
        lr_decay = lr_scheduler.ExponentialLR(optimiser, gamma=args.decay_rate)
    elif args.decay_milestones[0] > 0:
        lr_decay = lr_scheduler.MultiStepLR(optimiser, milestones=args.decay_milestones,
                                            gamma=args.decay_rate)
    else:
        lr_decay = lr_scheduler.ReduceLROnPlateau(optimiser, factor=args.decay_rate, mode='max',
                                                  patience=30, cooldown=20, min_lr=1e-6, verbose=True)

    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    best_valid_acc = 0
    patience_counter = 0

    since = time.time()
    for epoch in range(args.n_epochs):

        model.train()
        sample_count = 0
        running_loss = 0
        running_acc = 0

        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['train'])
        else:
            train_dataloader = dataloaders['train']

        for i, (inputs, labels) in enumerate(train_dataloader):
            args.step = (epoch * len(dataloaders['train'])) + i + 1  # calc current step

            if args.half_precision:
                inputs = inputs.type(torch.HalfTensor).cuda(non_blocking=True)
            else:
                inputs = inputs.type(torch.FloatTensor).cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimiser.zero_grad()
            yhat = model(inputs)
            loss = F.nll_loss(F.log_softmax(yhat), labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()

            sample_count += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)  # smaller batches count less
            running_acc += (yhat.argmax(-1) == labels).sum().item()  # num corrects

            if args.print_progress:
                # inspect gradient L2 norm
                total_norm = torch.zeros(1).cuda()
                for name, param in model.named_parameters():
                    try:
                        total_norm += param.grad.data.norm(2)**2
                    except:
                        pass
                total_norm = total_norm**(1/2)
                args.writer.add_scalar('grad_L2_norm', total_norm, args.step)

        epoch_train_loss = running_loss / sample_count
        epoch_train_acc = running_acc / sample_count

        # reduce lr
        if args.decay_steps > 0 or args.decay_milestones[0] > 0:
            lr_decay.step()
        else:  # reduce on plateau, evaluate to keep track of acc in each process
            epoch_valid_loss, epoch_valid_acc = evaluate(model, dataloaders['valid'], args)
            lr_decay.step(epoch_valid_acc[0])

        if args.print_progress:  # only validate using process 0
            if epoch_valid_loss is None:  # check if process 0 already validated
                epoch_valid_loss, epoch_valid_acc = evaluate(model, dataloaders['valid'], args)

            logging.info('\n[Train] loss: {:.4f} - acc: {:.4f} | [Valid] loss: {:.4f} - acc: {:.4f} - acc_topk: {:.4f}'.format(
                epoch_train_loss, epoch_train_acc,
                epoch_valid_loss, epoch_valid_acc[0], epoch_valid_acc[1]))

            epoch_valid_acc = epoch_valid_acc[0]  # discard top k acc
            args.writer.add_scalars('epoch_loss', {'train': epoch_train_loss,
                                                   'valid': epoch_valid_loss}, epoch+1)
            args.writer.add_scalars('epoch_acc', {'train': epoch_train_acc,
                                                  'valid': epoch_valid_acc}, epoch+1)
            args.writer.add_scalars('epoch_error', {'train': 1-epoch_train_acc,
                                                    'valid': 1-epoch_valid_acc}, epoch+1)

            # # inspect weights and gradients
            # for name, p in model.named_parameters():
            #     try:
            #         name = '.'.join(name.split('.')[1:]) \
            #             if name.split('.')[0] == 'module' else name
            #         args.writer.add_histogram(name, p.data.clone().cpu().numpy(), args.step)
            #         args.writer.add_histogram(name, p.data.grad.clone().cpu().numpy(), args.step)
            #     except:
            #         pass

            # save model and early stopping
            if epoch_valid_acc >= best_valid_acc:
                patience_counter = 0
                best_epoch = epoch + 1
                best_valid_acc = epoch_valid_acc
                best_valid_loss = epoch_valid_loss
                # saving using process (rank) 0 only as all processes are in sync
                torch.save(model.state_dict(), args.checkpoint_dir)
            else:
                patience_counter += 1
                if patience_counter == (args.patience-10):
                    logging.info('\nPatience counter {}/{}.'.format(
                        patience_counter, args.patience))
                elif patience_counter == args.patience:
                    logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                        args.patience))
                    break
            epoch_valid_loss = None  # reset loss

        gc.collect()  # release unreferenced memory

    if args.print_progress:
        time_elapsed = time.time() - since
        logging.info('\nTraining time: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        model.load_state_dict(torch.load(args.checkpoint_dir))  # load best model

        test_loss, test_acc = evaluate(model, dataloaders['test'], args)

        logging.info('\nBest [Valid] | epoch: {} - loss: {:.4f} - acc: {:.4f}'.format(
            best_epoch, best_valid_loss, best_valid_acc))
        logging.info('[Test] loss {:.4f} - acc: {:.4f} - acc_topk: {:.4f}'.format(
            test_loss, test_acc[0], test_acc[1]))
