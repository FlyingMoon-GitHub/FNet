# -*- coding: utf-8 -*-

from torchvision import transforms

def getTransformation(args, type):
    assert type in ['train', 'val', 'test']

    transformation_list = []

    if type == 'train':
        transformation_list.append(transforms.RandomRotation(degrees=20))
        transformation_list.append(transforms.RandomResizedCrop(size=args.target_size, scale=(0.9, 1.0)))
        transformation_list.append(transforms.ColorJitter(brightness=0.1))
        transformation_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transformation_list.append(transforms.RandomVerticalFlip(p=0.5))

    transformation_list.append(transforms.Resize((args.target_size, args.target_size)))
    transformation_list.append(transforms.ToTensor()),
    transformation_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transformation = transforms.Compose(transformation_list)

    return transformation
