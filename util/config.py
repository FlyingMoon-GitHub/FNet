# -*- coding: utf-8 -*-

import argparse
import torch.optim as optim
from torch.optim import lr_scheduler

from util.transformation import *

def getDatasetConfig(args, type):
    assert type in ['train', 'val', 'test']

    config = {}
    config['root_path'] = args.root_path
    config['image_dir'] = args.image_dir
    config['anno_dir'] = args.anno_dir
    config['image_views'] = args.image_views.replace(' ','').strip().split(',')
    config['prim_view'] = args.prim_view
    config['image_format'] = args.image_format.lower()
    assert len(config['image_views']) == 2
    config['anno_file'] = type + '.txt'

    config['class_num'] = args.class_num

    config['img_aug'] = getTransformation(args, type)

    return config

def getModelConfig(args, type):
    assert type in ['train', 'val', 'test']
    config = {}

    config['model'] = args.model
    config['class_num'] = args.class_num
    config['lambda'] = args.lambda_
    config['target_size'] = args.target_size
    config['use_cuda'] = args.use_cuda

    return config