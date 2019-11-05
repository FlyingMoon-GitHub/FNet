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
    parser.add_argument('--class_num', dest='class_num',
                        default=4, type=int)
    parser.add_argument('--lambda', dest='lambda_',
                        default=0.75, type=float)

    parser.add_argument('--epoch_num', dest='epoch_num',
                        default=360, type=int)
    parser.add_argument('--test_batch', dest='test_batch',
                        default=16, type=int)

    parser.add_argument('--test_num_workers', dest='test_num_workers',
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

    test_data_config = getDatasetConfig(args, 'test')
    test_dataset = MyDataset(test_data_config)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.test_batch,
                                 shuffle=True,
                                 num_workers=args.test_num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    model_config = getModelConfig(args, 'test')

    model = None
    if args.model == 'ufnet':
        model = UFNet(model_config)
    elif args.model == 'bfnet':
        model = BFNet(model_config)

    if args.savepoint_file:
        model_dict = model.state_dict()
        model_dict.update(torch.load(args.savepoint_file))
        model.load_state_dict({(k if args.cuda else k.replace('module.','')):v for k,v in model_dict})
    else:
        model.apply(weightInit)

    if args.cuda:
        model = model.cuda()

    model.summary()

    if args.cuda:
        model = nn.DataParallel(model)

    test_result = test(args, model=model, dataloader=test_dataloader, type='test')

    test_aux_acc = test_result['test_aux_acc']
    test_prim_acc = test_result['test_prim_acc']
    test_aux_cfs_mat = test_result['test_aux_cfs_mat']
    test_prim_cfs_mat = test_result['test_prim_cfs_mat']
    print('test_aux_acc: {:6.4f}, test_prim_acc: {:6.4f}.'
          .format(test_aux_acc, test_prim_acc))
    print('test_aux_cfs_mat')
    print(test_aux_cfs_mat)
    print('test_prim_cfs_mat')
    print(test_prim_cfs_mat)
