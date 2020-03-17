# -*- coding: utf-8 -*-
"""
    Evaluation loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate(model, dataloader, args):
    """ Evaluates a given model and dataset.
    """
    model.eval()
    sample_count = 0
    running_loss = 0
    running_acc = 0
    running_acc_topk = 0
    k = 2

    with torch.no_grad():

        for inputs, labels in dataloader:

            if args.half_precision:
                inputs = inputs.type(torch.HalfTensor).cuda(non_blocking=True)
            else:
                inputs = inputs.type(torch.FloatTensor).cuda(non_blocking=True)
            labels = labels.type(torch.LongTensor).cuda(non_blocking=True)

            yhat = model(inputs)
            loss = F.nll_loss(F.log_softmax(yhat), labels)

            sample_count += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)  # smaller batches count less
            running_acc += (yhat.argmax(-1) == labels).sum().item()  # num corrects
            _, yhat = yhat.topk(k, 1, True, True)
            running_acc_topk += (yhat == labels.view(-1, 1).expand_as(yhat)
                                 ).sum().item()  # num corrects

        loss = running_loss / sample_count
        acc = running_acc / sample_count
        top_k_acc = running_acc_topk / sample_count

    return loss, (acc, top_k_acc)
