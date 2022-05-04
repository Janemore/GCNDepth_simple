#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

from functools import partial

from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader
from .sampler import GroupSampler

# https://github.com/pytorch/pytorch/issues/973
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

"""
This method is nothing but building a dataloader using customized sampler; 

input: dataset 
output: dataloader 

one explanation about sampler and dataloader: 
https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
"""

def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     **kwargs):
    shuffle = kwargs.get('shuffle', True)

    sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
    batch_size = num_gpus * imgs_per_gpu
    num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
                             pin_memory=False,
                             **kwargs,
                             drop_last=True
                             )

    return data_loader
