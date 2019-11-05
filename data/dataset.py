# -*- coding: utf-8 -*-

import argparse

import cv2
import os
from PIL import Image, ImageFile
from torch.utils.data import *

from util.config import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyDataset(Dataset):
    def __init__(self, config):
        super(MyDataset, self).__init__()

        self.root_path = config['root_path']
        self.image_dir = config['image_dir']
        self.image_views = config['image_views']
        self.prim_view = config['prim_view']
        self.image_format = config['image_format']
        self.anno_dir = config['anno_dir']
        self.class_num = config['class_num']
        self.anno_file = config['anno_file']

        self.anno = []
        with open(os.path.join(self.root_path, self.anno_dir, self.anno_file), 'r') as f:
            line = f.readline().replace('\n', '')
            while line:
                line_items = line.split(' ')
                self.anno.append((line_items[0], int(line_items[1])))
                line = f.readline().replace('\n', '')

        assert self.anno, 'No available data?'

        self.img_aug = config['img_aug']

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        images = []

        for image_view in self.image_views:
            image = Image.open(
                os.path.join(self.root_path, self.image_dir, self.anno[index][0], image_view + '.' + self.image_format))
            images.append(image.convert('RGB'))

        images = self.img_aug(images)

        return images[1 - self.prim_view], images[self.prim_view], self.anno[index][1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', dest='root_path',
                        default=r'C:\Users\FlyingMoon\PycharmProjects\FNet\data\cervical', type=str)
    parser.add_argument('--img_dir', dest='image_dir',
                        default=r'image', type=str)
    parser.add_argument('--anno_dir', dest='anno_dir',
                        default=r'anno', type=str)
    parser.add_argument('--img_views', dest='image_views',
                        default=r'VIA3,VILI', type=str)
    parser.add_argument('--prim_view', dest='prim_view',
                        default=0, type=int)
    parser.add_argument('--img_format', dest='image_format',
                        default=r'jpg', type=str)

    parser.add_argument('--class_num', dest='class_num',
                        default=4, type=int)
    parser.add_argument('--target_size', dest='target_size',
                        default=512, type=int)
    parser.add_argument('--lr', dest='learning_rate',
                        default=0.9, type=float)
    parser.add_argument('--lambda', dest='lambda_',
                        default=0.75, type=float)

    '''
    TRANSFORM-RELATED ARGUMENTS
    '''

    args = parser.parse_args()
    train_data_config = getDatasetConfig(args, 'train')
    val_data_config = getDatasetConfig(args, 'val')

    train_dataset = MyDataset(train_data_config)
    val_dataset = MyDataset(val_data_config)

    item = train_dataset[0]

    print(item)

    import matplotlib.pyplot as plt

    for ind in range(2):
        plt.imshow(item[ind].permute(1, 2, 0))
        plt.show()
