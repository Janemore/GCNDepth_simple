
import os
from .utils import readlines


def get_dataset(cfg, training=True):
    dataset_name = cfg['name']
    if dataset_name == 'kitti':
        from .kitti_dataset import KITTIRAWDataset as dataset
    elif dataset_name == 'kitti_odom':
        from .kitti_dataset import KITTIOdomDataset as dataset

    fpath = os.path.join(os.path.dirname(__file__), "splits", cfg.split, "{}_files.txt")
    filenames = readlines(fpath.format("train")) if training else readlines(fpath.format('val'))
    img_ext = '.png' if cfg.png == True else '.jpg'

    # dataset is defined before

    dataset = dataset(cfg.in_path,
                      filenames,
                      cfg.height,
                      cfg.width,
                      cfg.frame_ids if training else [0],
                      is_train=training,
                      img_ext=img_ext,
                      gt_depth_path=cfg.gt_depth_path)
    # dataset[0]
    return dataset