# -*- coding: utf-8 -*-

import datetime
import os
import torch
from torch import nn
from torch.autograd import Variable


def test(args, model, dataloader, type):
    assert type in ['val', 'test']

    result = {}

    loss_records = []

    batch_size = dataloader.batch_size
    epoch_step = len(dataloader)
    data_num = len(dataloader.dataset)

    val_aux_correct = 0
    val_prim_correct = 0

    model.train(False)

    with torch.no_grad():

        cur_step = 0

        for batch_no, data in enumerate(dataloader):

            cur_step += 1

            aux_views, prim_views, labels = data

            if args.cuda:
                aux_views = Variable(aux_views.cuda())
                prim_views = Variable(prim_views.cuda())
                labels = Variable(labels.cuda()).long()
            else:
                aux_views = Variable(aux_views)
                prim_views = Variable(prim_views)
                labels = Variable(labels).long()

            aux_out, prim_out = model(aux_views, prim_views)

            loss = 0

            aux_loss = nn.CrossEntropyLoss()(aux_out, labels)

            loss += aux_loss * (1.0 - args.lambda_)

            prim_loss = nn.CrossEntropyLoss()(prim_out, labels)

            loss += prim_loss * args.lambda_

            loss_records.append(loss.detach().item())

            print(type + '_step: {:-8d} / {:d}, loss: {:6.4f}, aux_loss: {:6.4f}, prim_loss: {:6.4f}.'
                  .format(cur_step, epoch_step, loss.detach().item(), aux_loss.detach().item(),
                          prim_loss.detach().item()), flush=True)

            _, aux_pred = torch.topk(aux_out, 1)
            val_aux_correct += torch.sum((aux_pred[:, 0] == labels)).data.item()

            _, prim_pred = torch.topk(prim_out, 1)
            val_prim_correct += torch.sum((prim_pred[:, 0] == labels)).data.item()

    val_aux_acc = val_aux_correct / data_num
    val_prim_acc = val_prim_correct / data_num

    result['loss_records'] = loss_records
    result[type + '_aux_acc'] = val_aux_acc
    result[type + 'val_prim_acc'] = val_prim_acc

    return result
