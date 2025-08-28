# Author: Luciano Filho

import os
import PIL
import time
import math
import torch
import random
import numpy as np
import subprocess
import torchvision
from logging import getLogger
from multiprocessing import Value


from timm.data import create_transform
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms


_GLOBAL_SEED = 0
logger = getLogger()


def build_transform(is_train):

    # Testing imagenet stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    input_size=224
    # train transform
    if is_train:
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.0,
            #auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            #re_prob=0.25,
            #re_mode='pixel',
            #re_count=1, 
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_dataset(is_train, image_folder, ssl_transform=None):
    default_transform = build_transform(is_train)
    root = os.path.join(image_folder, 'train' if is_train else 'val')
    dataset = PairedDataset(root, transform=default_transform, ssl_transform=ssl_transform)
    return dataset

'''
    Paired Batch Dataset:
    Samples two batches of data belonging to the same labels. 
    
    Instead of inputting two different parts of the same input image
    into the context and target encoders, we will input different images
    that belongs to the same class and train the predictor to reconstruct
    the target encoder representation conditioned on the context 
    representation and the target encoder's mask positions. 

    It is expected that in addition to learn features, these models
    learns to cluster the representations according to semantical labels. 
    We are assuming that by reconstructing portions of different images  
    from a same class the model learns find sub-classes of greater
    cohesion in terms of visual similarity. And this is the hypothesis
    that we want to test. 

    In that sense, what does this dataset function has to do?

    It has to drawn two batches of data that belongs to the same set of classes.
    Considering that we have different types of transforms to be applied context-
    wise, because I-JEPA SSL doesn't require augmentations, whereas fine-tuning
    can benefit from using techniques such as rand-augment. 

    Therefore this function should yield data in the following format:

    "for itr, (sb_data, udata_1, udata_2, masks_enc, masks_pred) in enumerate(paired_batch_loader):"
    
    Where: 
        - sb_data: Corresponds to the supervised batch data, i.e., the data 
        that goes into the larger (pretrained) model.
        - udata_1: Fed into the context encoder. 
        - udata_2: Fed into the target encoder. 
        - masks_enc, masks_pred: Applied in the standard way.

'''

class PairedDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, ssl_transform=None):
        super(PairedDataset, self).__init__(root, transform=None)  # Do not apply default transform
        self.transform = transform
        self.ssl_transform = ssl_transform

    def __getitem__(self, index):
        # Get the first image and its label
        path, label1 = self.samples[index]
        image1 = self.loader(path)
        
        # Apply the standard transform to image1
        if self.transform is not None:
            image1 = self.transform(image1)

        # Find all indices of the same class as label1
        indices_same_class = [i for i in range(len(self)) if self.targets[i] == label1]

        # Remove the current index to avoid picking the same image
        indices_same_class.remove(index)

        try:
            # Randomly select another image of the same class
            index2 = random.choice(indices_same_class)
        except IndexError: 
            index2 = index
        path2, label2 = self.samples[index2]
        image2 = self.loader(path2)

        # Apply the ssl_transform to image2
        if self.ssl_transform is not None:
            image2 = self.ssl_transform(image2)
        else:
            image2 = self.transform(image2)

        return (image1, label1), (image2, label2)

def make_paired_batch_dataset(
    ssl_transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True, 
    feature_extraction=False,
    subset_file=None
):
    index_targets = False    

    dataset = build_dataset(is_train=training, image_folder=image_folder, ssl_transform=ssl_transform) 

    logger.info('PairedDataset created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)        
    return dataset, data_loader, dist_sampler
