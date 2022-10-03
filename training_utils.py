import numbers
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
#import tensorflow as tf
from PIL import ImageStat
import numpy as np
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def save_model_checkpoint(model_state_dict, exp_stage_dir, is_best, model_filename, epoch_num):
    #Modify this function to get model name
    model_file_save_path = os.path.join(exp_stage_dir, model_filename)
    torch.save(model_state_dict, model_file_save_path)
    if epoch_num+1 in [2,4,7,11]:
        torch.save(model_state_dict, os.path.join(exp_stage_dir,"epoch:{}".format(epoch_num)))
    if is_best or epoch_num==0:
        shutil.copyfile(model_file_save_path, os.path.join(exp_stage_dir,"best_model.pth"))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, initial_stage_lr, stage_num, epoch, args):
    #Adjust these for each stage as needed
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    if(True):#stage_num in [0,1,2,3,4,5,6]):
        lr = initial_stage_lr * (args.lr_decay_r**(epoch//args.lr_decay_every))
        if(args.num_stages==1 or (args.num_stages>1 and stage_num==0)):
            if(epoch<14):
                lr = 0.1
            elif(14<=epoch<24):
                lr = 0.01
            elif(24<=epoch<34):
                lr = 0.001
            else:
                lr = 0.0001
                #lr = initial_stage_lr * (args.lr_decay_r**(epoch//args.lr_decay_every))
        else:
            if(stage_num>0):
                prev_epochs = 0#sum(args.stages_epochs[:stage_num]) if stage_num>0 else 0
                lr = initial_stage_lr*(args.lr_decay_r**((epoch+prev_epochs)//args.lr_decay_every))
        #lr = args.lr * (initial_stage_lr** (epoch // 30))
        #print("LR: {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        