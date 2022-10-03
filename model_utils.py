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
from torch.nn import Linear


    
     
def load_model(prev_exp_stage_dir, stage_num, args, model_arch_name = 'resnet18', load_best_model=True, use_pytmodel = True, mgrowth = False, mgrowth_fn = None, input_refine= False, inp_img_size= 224, output_classes = 1000, gpu_=None):
   
    if(load_best_model):
        model_path = os.path.join(prev_exp_stage_dir, "best_model.pth")
    else:
        model_path = os.path.join(prev_exp_stage_dir, "last_state.pth")
        
    if gpu_ is None:
        checkpoint = torch.load(model_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu_)
        checkpoint = torch.load(model_path, map_location=loc)
    #args.start_epoch = checkpoint['epoch']
    #best_acc1 = checkpoint['best_acc1']
    if gpu_ is not None:
        # best_acc1 may be from a checkpoint from a different GPU
        best_acc1 = best_acc1.to(gpu_)
    model = checkpoint['model'].module
    #optimizer = checkpoint['optimizer']
#    print(model.state_dict()['conv1.weight']==checkpoint['state_dict']['module.conv1.weight'])
 #   print(model.state_dict()['conv1.weight'])#[0])
  #  print(checkpoint['state_dict']['module.conv1.weight'])
    #model.load_state_dict(checkpoint['state_dict'])
    
    
    if(mgrowth):
        if(use_pytmodel):
            print("Model growth is only available for custom models. Set args.use_pytmodel as 'f'")
            sys.exit(0)
        else:
            if(args.arch.startswith('custom_vgg')):
                arch_name = args.arch
                config_map = {'11':'A','13':'B','16':'C','19':'D'}
                bn = arch_name.split('_')[-1]
                if(bn not in ('bn','batchnorm','bnorm')):
                    bn=False
                    config = config_map[arch_name.split('_')[-1]]
                else:
                    bn = True
                    config = config_map[arch_name.split('_')[-2]]
            
                start_block = stage_num
                end_block = min(5,stage_num+1)
                prev_classifier = model.classifier
                prev_block = model.post_stage_block
                model = make_custom_vgg(cfg = config,
                                   batch_norm= bn, 
                                   start_block=start_block, end_block=end_block, 
                                   prev_block= prev_block, 
                                   prev_model_classifier=prev_classifier, 
                                   num_output_classes=output_classes)
                
            else:
                print("Model not found. Currently only vgg custom supported (custom_vgg_11_bn, custom_vgg_13 etc)")
                sys.exit(0)
            #model = _custom_vgg('A', )
        
        
        
    
    
    #model_inp_res = inp_img_size 
    #if(input_refine):
     
    model = check_and_update_model_with_new_output_shape(model, model_arch_name, output_classes)
    return model

def check_and_update_model_with_new_output_shape(model, model_arch_name, new_num_classes):
    #if(model_)
    if(model_arch_name.startswith('vgg')):
        if(new_num_classes != model.classifier[-1].out_features):
            model.classifier[-1] = Linear(in_features=4096, out_features=args.stages_output_classes[stage_num], bias=True)
    
    elif(model_arch_name.startswith("resnet")):
            #101, 152, 50-> 2048
            #18,34 -> 512
        if(new_num_classes != model.fc.out_features):
            if(model_arch_name in ['resnet101','resnet152','resnet50']):
                model.fc = Linear(in_features=2048, out_features=args.stages_output_classes[stage_num], bias=True)
            elif(model_arch_name in ['resnet18','resnet34']):
                model.fc = Linear(in_features=512, out_features=args.stages_output_classes[stage_num], bias=True)
    elif(model_arch_name.startswith("resnext") or model_arch_name.startswith("wide")):
            #101, 152, 50-> 2048
            #18,34 -> 512
        if(new_num_classes != model.fc.out_features):
            #if(model_arch_name in ['resnet101','resnet152','resnet50']):
            model.fc.out_features = args.stages_output_classes[stage_num]
    
    #elif(model_arch_name.starts_)
    elif(model_arch_name.startswith('custom_vgg')):
        if(new_num_classes != model.classifier[-1].out_features):
            model.classifier[-1] = Linear(in_features=4096, out_features=args.stages_output_classes[stage_num], bias=True)
    
    elif(model_arch_name.startswith('densenet')):
        #if(model_arch_name == 'densenet121'):
            #in_features = 2
        if(new_num_classes != model.classifier.out_features):
            model.classifier.out_features = args.stages_output_classes[stage_num]
         #(classifier): Linear(in_features=2208, out_features=1000, bias=True)
    elif(model_arch_name.startswith('alexnet')):
        if(new_num_classes != model.classifier[-1].out_features):
            model.classifier[-1].out_features = args.stages_output_classes[stage_num]
            
    elif(model_arch_name.startswith('inception')):
        if(new_num_classes != model.fc.out_features):
            model.fc.out_features = args.stages_output_classes[stage_num]
        #classifier[-1]
    elif(model_arch_name.startswith('googlenet')):
        if(new_num_classes != model.fc.out_features):
            model.fc.out_features = args.stages_output_classes[stage_num]
        #classifier[-1]
    
    else:
        print("Couldn't find code to change final classes for this architecture...exiting")
        sys.exit(0)
        
    return model






def create_custom_model(arch_name, output_classes=1000, mgrowth=True, num_stages=5, pretrained=False):
    arch_name = arch_name.strip()
    if(arch_name.startswith('custom_vgg')):
        config_map = {'11':'A','13':'B','16':'C','19':'D'}
        bn = arch_name.split('_')[-1]
        if(bn not in ('bn','batchnorm','bnorm')):
            bn=False
            config = config_map[arch_name.split('_')[-1]]
        else:
            bn = True
            config = config_map[arch_name.split('_')[-2]]
        if(mgrowth):
            start_block = 0
            end_block = max(1,5-num_stages+1)
        else:
            start_block = 0
            end_block = 5
            
        custom_model = make_custom_vgg(cfg = config,
                                   batch_norm= bn, 
                                   start_block=start_block, end_block=end_block, 
                                   prev_block= None, 
                                   prev_model_classifier=None, 
                                   num_output_classes=output_classes)
        return custom_model
    else:
        print("Model not found. Currently only vgg custom supported (custom_vgg_11_bn, custom_vgg_13 etc).Exiting..")
        sys.exit(0)
        
def create_model(arch_name, num_classes = 1000, pretrained=False):
    """
    Given arch name, loads and returns the model
    """
    None
    
def update_model_output_shape(new_output_shape = 1000):
    """
    For custom models, changes the classifier to new output shape
    """
    None