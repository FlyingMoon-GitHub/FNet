# -*- coding: utf-8 -*-

import argparse
import torch
from torch.utils.data import DataLoader

from data.dataset import *
from func.train_model import *
from model.ufnet import *
from model.bfnet import *
from util.weight_init import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', dest='root_path',
                        default=os.path.join('.', 'data', 'cervical'), type=str)
    parser.add_argument('--img_dir', dest='image_dir',
                        default='image', type=str)
    parser.add_argument('--anno_dir', dest='anno_dir',
                        default='anno', type=str)
    parser.add_argument('--img_views', dest='image_views',
                        default='VIA3,VILI', type=str)
    parser.add_argument('--prim_view', dest='prim_view',
                        default=0, type=int)
    parser.add_argument('--img_format', dest='image_format',
                        default='jpg', type=str)
    parser.add_argument('--target_size', dest='target_size',
                        default=512, type=int)

    parser.add_argument('--model', dest='model',
                        default='ufnet', type=str)
    parser.add_argument('--type', dest='type',
                        default='val', type=str)
    parser.add_argument('--class_num', dest='class_num',
                        default=4, type=int)
    parser.add_argument('--lambda', dest='lambda_',
                        default=0.75, type=float)

    parser.add_argument('--epoch_num', dest='epoch_num',
                        default=360, type=int)
    parser.add_argument('--train_batch', dest='train_batch',
                        default=8, type=int)
    parser.add_argument('--val_batch', dest='val_batch',
                        default=8, type=int)
    parser.add_argument('--savepoint', dest='savepoint',
                        default=5000, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        default=5000, type=int)

    parser.add_argument('--lr', dest='learning_rate',
                        default=0.001 * 10, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        default=0.9, type=float)
    parser.add_argument('--decay_step', dest='decay_step',
                        default=20, type=int)
    parser.add_argument('--decay_gamma', dest='decay_gamma',
                        default=0.3, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0, type=int)
    parser.add_argument('--train_num_workers', dest='train_num_workers',
                        default=16, type=int)
    parser.add_argument('--val_num_workers', dest='val_num_workers',
                        default=32, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        default=torch.cuda.is_available(), type=bool)

    parser.add_argument('--savepoint_file', dest='savepoint_file',
                        default=None, type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        default=os.path.join('.', 'checkpoint'), type=str)

    # LACK OF TRANSFORMATION-RELATED ARGUMENTS

    args = parser.parse_args()

    # CUSTOM SETTINGS
    args.save_dir = os.path.join('.', 'checkpoint', 'cervical')
    # CUSTOM SETTINGS END

    args.cuda = args.cuda and torch.cuda.is_available()

    assert args.model in ['ufnet', 'bfnet']
    assert args.type in ['train', 'val']

    dataloaders = {}

    train_data_config = getDatasetConfig(args, 'train')
    train_dataset = MyDataset(train_data_config)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch,
                                  shuffle=True,
                                  num_workers=args.train_num_workers,
                                  drop_last=False,
                                  pin_memory=True)
    dataloaders['train'] = train_dataloader

    if args.type == 'val':
        val_data_config = getDatasetConfig(args, 'val')
        val_dataset = MyDataset(val_data_config)
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=args.val_batch,
                                    shuffle=True,
                                    num_workers=args.val_num_workers,
                                    drop_last=False,
                                    pin_memory=True)
        dataloaders['val'] = val_dataloader

    model_config = getModelConfig(args, args.type)

    model = None
    if args.model == 'ufnet':
        model = UFNet(model_config)
    elif args.model == 'bfnet':
        model = BFNet(model_config)

    if args.savepoint_file:
        model_dict = model.state_dict()
        model_dict.update(torch.load(args.savepoint_file))
        model.load_state_dict(model_dict)
    else:
        model.apply(weightInit)

    if args.cuda:
        model = model.cuda()

    model.summary()

    if args.cuda:
        model = nn.DataParallel(model)

    optimizer = None
    if args.cuda:
        optimizer = optim.SGD(model.module.parameters(), lr=args.learning_rate, momentum=args.momentum)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    learning_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)

    train(args=args,
          model=model,
          optimizer=optimizer,
          learning_rate_scheduler=learning_rate_scheduler,
          dataloaders=dataloaders)
