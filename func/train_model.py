# -*- coding: utf-8 -*-

import datetime
import os
import torch
from torch import nn
from torch.autograd import Variable

from func.test_model import *


def train(args, model, optimizer, learning_rate_scheduler, dataloaders):
    loss_records = []

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    train_batch_size = dataloaders['train'].batch_size
    train_epoch_step = len(dataloaders['train'])

    savepoint = args.savepoint
    checkpoint = args.checkpoint
    if savepoint > train_epoch_step:
        save_point = train_epoch_step
        checkpoint = save_point

    last_time, cur_time = None, datetime.datetime.now()

    for epoch in range(args.start_epoch, args.epoch_num):

        if learning_rate_scheduler:
            learning_rate_scheduler.step(epoch)

        cur_step = 0

        for batch_no, data in enumerate(dataloaders['train']):

            last_time = cur_time

            cur_step += 1

            model.train(True)

            aux_views, prim_views, labels = data

            if args.cuda:
                aux_views = Variable(aux_views.cuda())
                prim_views = Variable(prim_views.cuda())
                labels = Variable(labels.cuda()).long()
            else:
                aux_views = Variable(aux_views)
                prim_views = Variable(prim_views)
                labels = Variable(labels).long()

            optimizer.zero_grad()

            aux_out, prim_out = model(aux_views, prim_views)

            loss = 0

            aux_loss = nn.CrossEntropyLoss()(aux_out, labels)

            loss += aux_loss * (1.0 - args.lambda_)

            prim_loss = nn.CrossEntropyLoss()(prim_out, labels)

            loss += prim_loss * args.lambda_

            loss.backward()

            if args.cuda:
                torch.cuda.synchronize()

            optimizer.step()

            if args.cuda:
                torch.cuda.synchronize()

            cur_time = datetime.datetime.now()

            loss_records.append(loss.detach().item())

            print('train_step: {:-8d} / {:d}, loss: {:6.4f}, aux_loss: {:6.4f}, prim_loss: {:6.4f}.'
                  .format(cur_step, train_epoch_step, loss.detach().item(), aux_loss.detach().item(),
                          prim_loss.detach().item()), flush=True)

            print(cur_time - last_time)

        print('epoch: {:-4d}, start_epoch: {:-4d}, epoch_num: {:-4d}.'
              .format(epoch, args.start_epoch, args.epoch_num))

        if args.type == 'val':
            val_result = test(args, model=model, dataloader=dataloaders['val'], type='val')
            val_aux_acc = val_result['val_aux_acc']
            val_prim_acc = val_result['val_prim_acc']
            print('val_aux_acc: {:6.4f}, val_prim_acc: {:6.4f}.'
                  .format(val_aux_acc, val_prim_acc))

        save_path = os.path.join(args.save_dir, 'weight_epoch_' + str(epoch) + '.pth')

        if args.cuda:
            torch.cuda.synchronize()

        torch.save(model.state_dict(), save_path)

        if args.cuda:
            torch.cuda.empty_cache()

