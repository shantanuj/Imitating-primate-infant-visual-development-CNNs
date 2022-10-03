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

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class ColorVary(object):
    """
    Change the brightness, contrast, saturation and hue of images
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness #self._check_input(brightness, 'brightness')
        self.contrast = contrast#self._check_input(contrast, 'contrast')
        self.saturation = saturation#self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = brightness #brightnrandom.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if saturation is not None:
            saturation_factor = saturation#random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if contrast is not None:
            contrast_factor = contrast #random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        
        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        #random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
    
    
def get_stage_specific_dataloaders(exp_stage_dir, tr_img_dir, val_img_dir, use_data_aug = False, sat_val=1.0, con_val=1.0, res_val = 224, use_normalization =0, interpolation_type= 2):
    ##0. Make stage directory
    #exp_stage_dir = os.path.join(exp_dir,"stage_{}".format(stage)) 
    #os.makedirs(exp_stage_dir)
    if(use_normalization in [0,'0']):
    #Calculate for a sample
        ##1. Calculate normalization values (and output sample values)
        norm_m, norm_std = get_normalization_values_and_output_sample_images(exp_stage_dir=exp_stage_dir,
                                                                             img_train_dir = tr_img_dir,
                                                                             sat_val=sat_val,con_val=con_val,res_val=res_val 
                                                                             )
        normalize = transforms.Normalize(mean = norm_m, std =norm_std)
    elif(use_normalization in [1,'1']):
    #Use default imagenet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif(use_normalization in [2,'2']):
    #dont use
        normalize = None
    elif(use_normalization in [3,'3']):
        print("Normalizing each input batch not done yet...exiting")
        sys.exit(0)
    else:
        print("Wrong normalization terminal input...")
    
    
    increase_r = 32/256 # based on 224 -> 256 (for other res, use a similar number)
    ##3. Get training (or) val dataloader 
    if(use_data_aug):
        #Data aug = random crop + random horizontal flip
        #No data aug = center crop 
        if(normalize is not None):
            train_dataset = datasets.ImageFolder(
                tr_img_dir,
            #transforms.Resiz([
                transforms.Compose([
                transforms.Resize(int(res_val*(1+increase_r)), interpolation =interpolation_type),
                ColorVary(1,con_val,sat_val,0),
                transforms.RandomCrop(res_val),
                transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
                normalize
            ]))
          
            
        else:
            train_dataset = datasets.ImageFolder(
                tr_img_dir,
            #transforms.Resiz([
                transforms.Compose([
                transforms.Resize(int(res_val*(1+increase_r)), interpolation =interpolation_type),
                ColorVary(1,con_val,sat_val,0),
                transforms.RandomCrop(res_val),
                transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
            ]))
        
        
    else:
        ## No data aug:
            #1)No random crop (only center crop)
            #2)No flips
        if(normalize is not None):
            train_dataset = datasets.ImageFolder(
                tr_img_dir,
                transforms.Compose([
                transforms.Resize(int(res_val*(1+increase_r)), interpolation =interpolation_type),
                ColorVary(1,con_val,sat_val,0),
                transforms.CenterCrop(res_val),
            #transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
                normalize
            ]))
        #if(res_val>=224):
            #training_data_loader = 
        else:
            train_dataset = datasets.ImageFolder(
                tr_img_dir,
                transforms.Compose([
                transforms.Resize(int(res_val*(1+increase_r)), interpolation =interpolation_type),
                ColorVary(1,con_val,sat_val,0),
                transforms.CenterCrop(res_val),
            #transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
                
            ]))
    
    val_dataset = datasets.ImageFolder(val_img_dir, transforms.Compose([
            transforms.Resize(int(res_val*(1+increase_r)), interpolation =interpolation_type),
            ColorVary(1,con_val,sat_val,0),
            transforms.CenterCrop(res_val),
            transforms.ToTensor(),
            normalize,
        ]))
    
    return train_dataset, val_dataset


def get_normalization_values_and_output_sample_images(exp_stage_dir, img_train_dir, sat_val, con_val, res_val, sample_batch_size=1024, max_n_sample_batches =50, interpolation_type=2):
    """Calculates normalization values for a sample of entire dataset"""
    print("Loading dataset train for sample normalization values")
    norm_vals_dir = os.path.join(img_train_dir, 'norm_vals_res{}_sat{}_con_{}/'.format(float(res_val), float(sat_val), float(con_val)))
    norm_vals_pkl = os.path.join(norm_vals_dir,'values.pkl')
    increase_r = 32/224
        #print(int(res*(1+increase_r)))
    train_dataset = datasets.ImageFolder(
            img_train_dir,
            transforms.Compose([
                transforms.Resize(int(res_val*(1+increase_r)), interpolation=interpolation_type),
                ColorVary(1,con_val,sat_val,0),
                transforms.CenterCrop(res_val),
                transforms.ToTensor(),
            ]))
    mid_index= len(train_dataset)//2
    sample_img1 = train_dataset[0][0].permute([1,2,0])
    sample_img2 = train_dataset[mid_index][0].permute([1,2,0])
    sample_img3 = train_dataset[-1][0].permute([1,2,0])
    #print("Norm vals pkl path: {}".format(norm_vals_pkl))
    if(os.path.exists(norm_vals_pkl)):
        #Check if already exists and load
        #load 
        print("--------Loading existing sample normalization values----------------")
        with open(norm_vals_pkl,'rb') as f: 
            mean, std = pickle.load(f)
            print("Existing values: {},{}".format(mean, std))
    else:
        if(not os.path.exists(norm_vals_dir)):
            #shutil.rmtree(norm_vals_dir)
            os.makedirs(norm_vals_dir)
        
        
        #plt.imsave(os.path.join(norm_vals_dir, 'tr_img0.png'), sample_img1, format = 'png')
        
        #plt.imsave(os.path.join(norm_vals_dir, 'tr_imgmiddle.png'), sample_img2, format = 'png')
        #plt.imsave(os.path.join(norm_vals_dir, 'tr_imglast.png'), sample_img3, format = 'png')
       
    
        print("---------Calculating sample normalization values----------------")
        print("res: {}, sat: {}, con:{}".format(res_val, sat_val, con_val))
        #print("Calculating normalization values")
        batch_size= sample_batch_size#len(train_dataset)//1000 #256
        num_batches = len(train_dataset)//batch_size +1
        num_workers = 0
    #max_n_sample_batches = 10
        data_loader = DataLoader(train_dataset,batch_size=batch_size, num_workers=num_workers, shuffle=True)#len(or_train_dataset))
        mean = 0.
        std = 0.
        nb_samples = 0
        time_start = time.clock()
        for i, (images, _) in enumerate(data_loader):
            if(i%5==0):
                print("at sample batch {} out of {}".format(i, max_n_sample_batches))
            
            if(i>max_n_sample_batches):
                break
    
            batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            nb_samples+= batch_samples
            
        print("Total samples: {} out of {} ({}% of population)".format(nb_samples,len(train_dataset), np.round(100*nb_samples/len(train_dataset), decimals=2)))
        mean /= nb_samples
        std /= nb_samples
        time_end = time.clock()
        print("Calculated normalization values from samples", time_end-time_start)
        print(mean, std) 
        
        print("Saving pkl values to: {}".format(norm_vals_pkl))
        
        with open(norm_vals_pkl,'wb') as f: 
            pickle.dump((mean,std),f)
        
        
 
    #####Output images to exp_stage_dir
    plt.imsave(os.path.join(exp_stage_dir, 'tr_img0.png'), sample_img1, format = 'png')
    mid_index= len(train_dataset)//2
    plt.imsave(os.path.join(exp_stage_dir, 'tr_imgmiddle.png'), sample_img2, format = 'png')
    plt.imsave(os.path.join(exp_stage_dir, 'tr_imglast.png'), sample_img3, format = 'png')
    
    return mean, std
