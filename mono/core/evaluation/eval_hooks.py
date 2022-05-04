import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import Hook
from mmcv.parallel import scatter, collate
from torch.utils.data import Dataset
from .pixel_error import *

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

def change_input_variable(data):
    for k, v in data.items():
        data[k] = torch.as_tensor(v).float()
    return data

def unsqueeze_input_variable(data):
    for k, v in data.items():
        data[k] = torch.unsqueeze(v, dim=0)
    return data


class NonDistEvalHook(Hook):
    def __init__(self, dataset, cfg):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.interval = cfg.get('interval', 1)
        self.out_path = cfg.get('work_dir', './')
        self.cfg = cfg

    def after_train_epoch(self, runner):
        print('evaluation..............................................')

        abs_rel = AverageMeter()
        sq_rel = AverageMeter()
        rmse = AverageMeter()
        rmse_log = AverageMeter()
        a1 = AverageMeter()
        a2 = AverageMeter()
        a3 = AverageMeter()

        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()

        for idx in range(self.dataset.__len__()):
            data = self.dataset[idx]
            data = change_input_variable(data)
            data = unsqueeze_input_variable(data)
            with torch.no_grad():
                result = runner.model(data)

            disp = result[("disp", 0, 0)]
            pred_disp, _ = disp_to_depth(disp)
            pred_disp = pred_disp.cpu()[0, 0].numpy()

            gt_depth = data['gt_depth'].cpu()[0].numpy()
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_, a3_ = compute_errors(gt_depth, pred_depth)

            abs_rel.update(abs_rel_)
            sq_rel.update(sq_rel_)
            rmse.update(rmse_)
            rmse_log.update(rmse_log_)
            a1.update(a1_)
            a2.update(a2_)
            a3.update(a3_)
            print('a1_ is ', a1_)

        print('a1 is ', a1.avg)


