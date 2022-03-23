from config import feat_roots
import os, torch, random, scipy, cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import subprocess
import torch.distributed as dist
from decord import VideoReader

def load(model, ckpt_path, device='cuda'):
    checkpoint = torch.load(ckpt_path, map_location=device)

    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v

    model.load_state_dict(new_s)

    optimizer_state = checkpoint["optimizer"]

    epoch = checkpoint['global_epoch']

    if 'steps' in checkpoint:
        steps = checkpoint['steps']
        return model, optimizer_state, epoch, steps

    return model, optimizer_state, epoch

def batch_IOU(actual, predicted):
    I = (actual * predicted).long().sum(dim=1).float()
    U = (actual + predicted).bool().float().sum(dim=1)
    iou = I / torch.clamp(U, 1e-5, None)

    return iou

def accuracy(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = float(len(act_set & pred_set) > 0)
    return result

def recall(actual, predicted, k, denom=None):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    denom = len(act_set) if denom is None else denom
    result = len(act_set & pred_set) / float(denom)
    return result

def precision(labels, Iscores, denom):
    labels_sorted = labels[Iscores]
    average_precision_array = []
    counter = 0
    for idx,val in enumerate(labels_sorted):
        if val ==1:
            counter +=1
            average_precision_array.append(counter/float(idx+1))
    mean_average_precision = np.sum(np.asarray(average_precision_array))/float(denom)
    return mean_average_precision
