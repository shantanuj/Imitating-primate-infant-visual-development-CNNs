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

###################################  MAIN CODE  ###########################################     
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
from custom_VGG import *
from filter_growth_resnet import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names+= ['custom_vgg_11_bn','custom_vgg_11','custom_vgg_13_bn','custom_vgg_13','custom_vgg_16_bn','custom_vgg_16','custom_vgg_19_bn','custom_vgg_19','fg_resnet18','fg_resnet50','fg_wide_resnet50_2','resnext50_32x4d']
#+ []
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

#####Stage specific training params######
parser.add_argument('--num_stages', default =1, type = int)
parser.add_argument('--mg','--mgrowth',default="f", type= str)
parser.add_argument('--irefine','--ir',default="f", type=str)
parser.add_argument('--stages_con','--sc',default=[1], type=str, help = "Contrast value at each stage")
parser.add_argument('--stages_res','--sr',default = [224],type=str, help = "Resolution value at each stage")
parser.add_argument('--stages_sat','--ss', default=[1],type=str, help="Saturation value at each stage")
parser.add_argument('--stages_epochs','--se',default=[10,10,15,20,50],type=str, help ="Epochs per stage")
parser.add_argument('--stages_lr','--slr', default=[0.1,0.1,0.1,0.1,0.1], type=str, help ="Initial LR for each stage")
parser.add_argument('--stages_bs','--sbs', default=[256,256,256,256,256], type=str, help ="Initial batch size for each stage")

parser.add_argument('--stages_output_classes','--soc', default=[200,200,200,200,200], type=str, help ="Output classes for model at each stage")
#irefine, mg, use_pytmodel, slism
parser.add_argument('--num_classes', default=1000, type=int,
                    help='Number of output classes.')

#Specific to model
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('--filter_growth', '--fg', default="t", type=str, help="Grow filters over stages (only supported for resnet models for now")
parser.add_argument('--change_at_inp_layer', '--cail', default='t', type=str, help="Apply filter growth to input layer as well (only supported for resnet models for now)")

parser.add_argument('--use_pytmodel', '--use_pytm', default="t", type=str, help="Use pytorch preprovided models")
parser.add_argument('--model_growth_fn',default='gradual_layer_add', type=str)
parser.add_argument('--alpha_cutoff_r', default=0.4, type=float, 
                    help='Ratio at which alpha is set to 1 (for gradual growth models)')

parser.add_argument('--cutoff_eps', default=[1,1,2,3,10], type=str, 
                    help='Ratio at which alpha is set to 1 (for gradual growth models)')

parser.add_argument('--use_hierarchy',action='store_true', help="Whether to train with hierarchy levels or not (NOT DONE IN CODE YET))")
parser.add_argument('--hierarchy_level_stages', default=[2,2,2,2,2], type =list,
                   help = "The hierarchy level for supervision at each stage (0 corresponds to superordinates, 1 basic level and 2 subordinates(the original classes))")
parser.add_argument('--use_data_aug',action='store_true', help="Whether to use data augmentation such as horizontal flip or randomresizecrops")
parser.add_argument('--only_last_data_aug','--olda',action='store_true', help="Whether to use data augmentation such as horizontal flip or randomresizecrops")
parser.add_argument('--input_norm',type=int,default=0, help="0:Calculate for a sample (default)\n 1: Use imagenet default \n 2: No normalization \n 3: Calculate batch wise (not done yet)")
parser.add_argument('--slism','--save_last_iter_stage_model', default='f',type=str,help="Continue next stage training by taking last epoch model state of previous stage (instead of using the best model at previous stage)")
parser.add_argument('--exp_aux_name', '--exp_auxiliary_name', type =str, default='', help="An auxiliary name to describe the experiment (e.g. 'custom_lr')")
parser.add_argument('--model_save_analysis', '--msa', action='store_true', help="Whether to store models at specific epochs")
##########################################

parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_decay_every','--lree', default =30, type=int, help ="LR decays to <decay ratio> of previous every <input> epochs")
parser.add_argument('--lr_decay_r','--lrdr', default =0.1, type=float, help ="LR decays to <input> of previous every <decay num> epochs")
parser.add_argument('--lr_buffer', default=0, type=int, metavar='N',
                    help='number of epochs to add to decay (in case of stage-based training)')
parser.add_argument('--lr_first_decay','--lrfd', default=30, type=int, metavar='N',
                    help='number of epochs to do first decay')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_stage_num', default=-1, type=int, 
                    help='The stage number at which to start training (0 corresponds to 1st stage)')


parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--model_save_dir', default ="/data/shantanu/Vision_models/0_BMVC_ivl_exps/", 
                    type=str, help = "Directory where models will be saved/checkpointed and stage wise info be stored as well")
parser.add_argument('--save_every_n_epoch','--save_nepoch', type =int, default =1, help = "Save the model every <input> number epochs" )

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


best_acc1 = 0



def main():
    
    #global save_best_model_from_prv_stage
    args = parser.parse_args()
    args.use_data_aug = True
    #args.only_last_data_aug = True
    if(type(args.stages_con)!=list):
        args.stages_con = [float(x) for x in args.stages_con.split(',')]
        #print(args.stages_con)
    if(type(args.stages_res)!=list):
        args.stages_res = [float(x) for x in args.stages_res.split(',')]
    if(type(args.stages_sat)!=list):
        args.stages_sat = [float(x) for x in args.stages_sat.split(',')]
    if(type(args.stages_epochs)!=list):
        args.stages_epochs = [int(x) for x in args.stages_epochs.split(',')]
    if(type(args.stages_lr)!=list):
        args.stages_lr = [float(x) for x in args.stages_lr.split(',')]
    if(type(args.stages_bs)!=list):
        args.stages_bs = [int(x) for x in args.stages_bs.split(',')]
    if(type(args.stages_output_classes)!=list):
        args.stages_output_classes = [int(x) for x in args.stages_output_classes.split(',')]
        
    if(type(args.cutoff_eps)!=list):
        args.cutoff_eps = [float(x) for x in args.cutoff_eps.split(',')]
    #irefine, mg, use_pytmodel, slism
    args.irefine = str2bool(args.irefine)
    args.mg = str2bool(args.mg)
    args.use_pytmodel = str2bool(args.use_pytmodel)
    args.slism = str2bool(args.slism)
    args.filter_growth = str2bool(args.filter_growth)
    args.change_at_inp_layer = str2bool(args.change_at_inp_layer)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    
    save_best_model_from_prv_stage = True
    #print(args.slism)
    if(args.slism):
        save_best_model_from_prv_stage = False
    #1. Create relevant experiment directories (or check if exists) 
    exp_dir = process_stage_info(model_name = args.arch,
                                 exp_aux_name = args.exp_aux_name,
                                 save_best_model = save_best_model_from_prv_stage,
                                 dataset_dir = args.data,
                                model_save_dir = args.model_save_dir,
                                mgrowth = args.mg, irefine = args.irefine, use_hierarchy = args.use_hierarchy,
                                use_data_aug = args.use_data_aug,
                                num_stages = args.num_stages,
                                stages_con = args.stages_con, stages_sat = args.stages_sat, stages_res= args.stages_res,
                                 stages_epochs = args.stages_epochs, stages_lr = args.stages_lr, stages_bs = args.stages_bs,
                                 num_classes = args.num_classes,
                                 hierarchy_level_stages = args.hierarchy_level_stages,
                                 model_growth_fn = args.model_growth_fn, 
                                 evaluate_mode = args.evaluate,
                                 lr_decay_every = args.lr_decay_every,
                                 pretrained = args.pretrained,
                                 lr_decay_ratio = args.lr_decay_r,
                                 only_last_data_aug = args.only_last_data_aug,
                                 lr_buffer = args.lr_buffer,
                                 lr_first_decay = args.lr_first_decay,
                                 cutoff_eps= args.cutoff_eps,
                                 filter_growth = args.filter_growth,
                                 change_at_inp_layer = args.change_at_inp_layer,
                                 args = args
                                )
    
    stages_list = range(args.num_stages)
    
    #1b. Check if continuing training for a stage
    #if(args.resume and args.resume_stage_num>=0):
     #   stages_list = stages_list[args.resume_stage_num:]
    
    #2. Iterate for each stage
    prev_exp_stage_dir= ""
    for stage_num in stages_list:
        best_acc1 = 0
        print("At stage:{} out of {}".format(stage_num+1, len(stages_list)))
        exp_stage_dir = os.path.join(exp_dir,"stage_{}".format(stage_num)) 
        if(not args.evaluate):
            os.makedirs(exp_stage_dir)
        print("Experiment directory: {}".format(exp_stage_dir))
        if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
            args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, stage_num, exp_stage_dir, prev_exp_stage_dir))
        else:
        # Simply call main_worker function
            main_worker(args.gpu, ngpus_per_node, args, stage_num, exp_stage_dir, prev_exp_stage_dir)
        prev_exp_stage_dir = exp_stage_dir
        print("Experiment directory: {}".format(exp_stage_dir))


        
def main_worker(gpu, ngpus_per_node, args, stage_num, exp_stage_dir, prev_exp_stage_dir):
    global best_acc1
    args.gpu = gpu
    #save_best_model_from_prv_stage = True
    #if(args.save_last_iter_stage_model):
     #   save_best_model_from_prv_stage = False
    save_best_model_from_prv_stage = True
    if(args.slism):
        save_best_model_from_prv_stage = False
        
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if(stage_num==0): 
        #First stage- create or initialize pretrained model
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
#-------> Modify for stages
            if(args.use_pytmodel):
                #print("ERROR: Not supported for filter growth"
                model = models.__dict__[args.arch](pretrained=True)
            else:
            ###TODO: CUstom models function
                print("=> Creating custom model '{}'".format(args.arch))
                model = create_custom_model(args.arch, output_classes=args.stages_output_classes[stage_num], 
                                            mgrowth=args.mg, num_stages=args.num_stages, pretrained=True)
                #create_model(args.arch, args.stages_output_classes[stage_num], pretrained=True) 
        else:
            print("=> creating model '{}'".format(args.arch))
            if(args.use_pytmodel):
                model = models.__dict__[args.arch]()
                model = check_and_update_model_with_new_output_shape(model, args.arch, args.stages_output_classes[stage_num])
            else:
                print("=> Creating custom model '{}'".format(args.arch))
                model = create_custom_model(args.arch, output_classes=args.stages_output_classes[stage_num], 
                                            mgrowth=args.mg, num_stages=args.num_stages, pretrained=False)
       
    
    else:
        # Stage num>0 -> load previous stage best model
#----->>>  TODO1: Make code for load and save stagewise model
 #   --> See whether checkpoints vs creating a new model
   #load_model(prev_exp_stage_dir, stage_num, model_arch_name = 'resnet18', load_best_model=True, use_pytmodel = True, mgrowth = False, mgrowth_fn = None, input_refine= False, inp_img_size= 224, output_classes = 1000, gpu_=None):
   
        model = load_model(prev_exp_stage_dir, 
                           stage_num = stage_num, 
                           args = args,
                           model_arch_name= args.arch,
                           load_best_model = save_best_model_from_prv_stage,
                           use_pytmodel = args.use_pytmodel,
                           mgrowth = args.mg, 
                           mgrowth_fn = args.model_growth_fn, 
                           inp_img_size= args.stages_res[stage_num],
                           output_classes = args.stages_output_classes[stage_num],
                           gpu_ = args.gpu
                          )
        
        
    ###############For DISTRIBUTED TRAINING and GPU SETTING####################
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.stages_bs[stage_num] = int(args.stages_bs[stage_num] / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    ###############For DISTRIBUTED TRAINING and GPU SETTING####################
    
    ##############LOSS FUNCTION AND OPTIMIZER#############################
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.stages_lr[stage_num],
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #####################################################################
    
    
    #########################LOAD MODEL FROM CHECKPOINT########################
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    ########################### CUDNN benchmark##########################################
    
    cudnn.benchmark = True

    #####################################################################
    
    ############################DATALOADERS##################################
    
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val_ptorch_format')
    
    if(not args.use_data_aug):
        use_data_aug = False
    else:
        if(args.only_last_data_aug and stage_num<args.num_stages-1):
            use_data_aug=False
        elif(args.only_last_data_aug and stage_num==args.num_stages-1):
            use_data_aug = True
        else:
            use_data_aug=True
         
    train_dataset, val_dataset = get_stage_specific_dataloaders(exp_stage_dir, 
                                                                tr_img_dir = traindir, 
                                                                val_img_dir = valdir,
                                                                #irefine=args.irefine, 
                                                                use_data_aug = use_data_aug,
                                                                sat_val=args.stages_sat[stage_num],
                                                                con_val=args.stages_con[stage_num],
                                                                res_val = args.stages_res[stage_num],
                                                               use_normalization = args.input_norm)
    
    
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
     #                                std=[0.229, 0.224, 0.225])

    """
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(224+32),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    """
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.stages_bs[stage_num], shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    """
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    """
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.stages_bs[stage_num], shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ###########################################################
    
    ####################EVALUATION CODE##############################
    if args.evaluate:
        validate(val_loader, model, criterion, args, exp_stage_dir)
        return
    ##############################################################
    
    ####################TRAINING CODE############################
    if(args.mg):
        total_epochs = args.stages_epochs[stage_num]
        alpha_cutoff_r = args.alpha_cutoff_r
        cutoff_eps = args.cutoff_eps
        cutoff_epoch = cutoff_eps[stage_num] 
        #cutoff_epoch = int(total_epochs*alpha_cutoff_r)
        alpha_increment = 0 if cutoff_epoch ==0 else 1/cutoff_epoch 
        
    print(model)   
    #print(args.start_epoch, args.stages_epochs[stage_num])
    for epoch in range(args.start_epoch, args.stages_epochs[stage_num]):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        #################lr scheduler########################
#------> TODO: Adjust learning rate
    
        adjust_learning_rate(optimizer, args.stages_lr[stage_num], stage_num, epoch, args)
        for param_group in optimizer.param_groups:
            print("LR: {}".format(param_group['lr']))
            
        print("Batch size: {}".format(args.stages_bs[stage_num]))
        
        if(args.mg):
            print("Pre-training validation accuracy")
            v1pre_, v5pre_ = validate(val_loader, model, criterion, args, exp_stage_dir)
            if(cutoff_epoch==0): 
                model.alpha = 1.0
                
        if(args.mg and stage_num>0):
            print("Alpha: {}".format(model.alpha))
            # evaluate on validation set
            
#optimizer, initial_stage_lr, stage_num, epoch, args
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, args.stages_epochs[stage_num], exp_stage_dir)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args, exp_stage_dir)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if(is_best):
            with(open(os.path.join(exp_stage_dir,"Best_val.txt"), 'a')) as f:
                f.write(str((epoch,acc1,acc5))+"\n")
        
        
#-------> TO DO: Model saving code 
    
        #if(args.save_last_iter_stage_model): #Save the last training iteration as checkpoint
           
        #else:   
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if(epoch%args.save_every_n_epoch ==0 or is_best):
                model_state_dict = {'epoch': epoch + 1,
                        'arch': args.arch,
                        'model':model, 
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }
                model_file_name = "latest_epoch_state.pth"#.format(epoch+1)
                save_model_checkpoint(model_state_dict, exp_stage_dir, is_best, model_file_name, epoch, args.model_save_analysis)
                #def save_model_checkpoint(model_state_dict, exp_stage_dir, is_best, model_filename):
                
            if(epoch==args.stages_epochs[stage_num]-1 and not(save_best_model_from_prv_stage)):
                
                model_state_dict = {'epoch': epoch + 1,
                        'arch': args.arch,
                        'model':model,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }
                model_file_name = "last_state.pth"
                save_model_checkpoint(model_state_dict, exp_stage_dir, False, model_file_name, epoch, args.model_save_analysis)
        ###INCREMENT ALPHA
        if(args.mg and epoch<=cutoff_epoch):
            model.alpha += alpha_increment
            model.alpha = min(1,model.alpha)
            
    print("Best acc: {}".format(best_acc1))
                
                
def train(train_loader, model, criterion, optimizer, epoch, args, total_epochs, exp_stage_dir):
    #print(model)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}/{}]".format(epoch, total_epochs))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            out = progress.display(i)
            with(open(os.path.join(exp_stage_dir,"train_progress.txt"), 'a')) as f:
                f.write(out+"\n")
                
        """
        
        """
        
        #COMMENT THE BELOW OUT JUST TO CHECK CODE FUNCTIONALITY
        #if(i%512==0):
         #   break
def validate(val_loader, model, criterion, args, exp_stage_dir):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                out = progress.display(i)
                
                with(open(os.path.join(exp_stage_dir,"val_progress.txt"), 'a')) as f:
                    f.write(out+"\n")

        # TODO: this should also be done with the ProgressMeter
        f_out = ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
        print(f_out)
    #top1_avg = top1.avg
    #top5_avg = top5.avg
    with(open(os.path.join(exp_stage_dir,"val_progress.txt"), 'a')) as f:
        f.write(f_out+"\n")
    return top1.avg, top5.avg

#save_model_checkpoint(model_state_dict, is_best)
def save_model_checkpoint(model_state_dict, exp_stage_dir, is_best, model_filename, epoch_num, model_save_analysis=False):
    #Modify this function to get model name
    model_file_save_path = os.path.join(exp_stage_dir, model_filename)
    torch.save(model_state_dict, model_file_save_path)
    #if epoch_num in range(0,21):
     #   torch.save(model_state_dict, os.path.join(exp_stage_dir,"epoch:{}".format(epoch_num)))
    if epoch_num+1 in [2,4,7,11] and model_save_analysis:
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
    
    if(True):#stage_num in [0,1,2,3,4,5,6]):
        lr = initial_stage_lr * (args.lr_decay_r**(epoch//args.lr_decay_every))
        decay_wait = args.lr_first_decay #1+2+2+3
        #buffer= args.lr_buffer
        #decay_wait+= buffer
        if(args.num_stages==1): #and not args.resume):
            if(epoch<decay_wait-1):
                lr = initial_stage_lr
             
            #if(epoch==decay_wait-1):
                
            elif(epoch==decay_wait-1 or decay_wait-1<epoch<decay_wait-1+args.lr_decay_every):
                lr = initial_stage_lr * (args.lr_decay_r)
            else:
                lr = initial_stage_lr * (args.lr_decay_r**(1+(epoch-(decay_wait-1))//args.lr_decay_every))

                
            #else:
             #   lr = initial_stage_lr * (args.lr_decay_
            #if(epoch<decay_wait+10):#epoch<14+6):
             #   lr = 0.1
            #elif(decay_wait+10<=epoch<decay_wait+20):
             #   lr = 0.01
            #elif(decay_wait+20<=epoch<decay_wait+30):
             #   lr = 0.001
            #elif(decay_wait+30<=epoch<decay_wait+40):
             #   lr = 0.0001
            #else:
             #   lr = 0.00001
        #elif(args.resume): # so num_stages>1 and args.resume
            #lr = initial_stage_lr * (args.lr_decay_r**(epoch//args.lr_decay_every))
                #lr = initial_stage_lr * (args.lr_decay_r**(epoch//args.lr_decay_every))
        else:#num_stages>1 and not args.resume
            #buffer= args.lr_buffer
            #if(stage_num>=0):
            prev_epochs = sum(args.stages_epochs[:stage_num]) if stage_num>0 else 0
            decay_wait = max(0,decay_wait - prev_epochs)
            if(epoch<decay_wait-1):
                lr = initial_stage_lr
             
            #if(epoch==decay_wait-1):
                
            elif(epoch==decay_wait-1 or decay_wait-1<epoch<decay_wait-1+args.lr_decay_every):
                lr = initial_stage_lr * (args.lr_decay_r)
            else:
                lr = initial_stage_lr * (args.lr_decay_r**(1+(epoch-(decay_wait-1))//args.lr_decay_every))
                
                #if(epoch<decay_wait+10):#epoch<14+6):
                 #   lr = 0.1
                #elif(decay_wait+10<=epoch<decay_wait+20):
                 #   lr = 0.01
                #elif(decay_wait+20<=epoch<decay_wait+30):
                 #   lr = 0.001
                #elif(decay_wait+30<=epoch<decay_wait+40):
                 #   lr = 0.0001
                #else:
                 #   lr = 0.00001
                #lr = initial_stage_lr*(args.lr_decay_r**((epoch+prev_epochs)//args.lr_decay_every))
                #lr = initial_stage_lr*(args.lr_decay_r**((epoch)//args.lr_decay_every))
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
        
    #print(type(sample_img1))    
    #print(sample_img1.shape)
    
    #####Output images to exp_stage_dir
    plt.imsave(os.path.join(exp_stage_dir, 'tr_img0.png'), sample_img1.numpy(), format = 'png')
    mid_index= len(train_dataset)//2
    plt.imsave(os.path.join(exp_stage_dir, 'tr_imgmiddle.png'), sample_img2.numpy(), format = 'png')
    plt.imsave(os.path.join(exp_stage_dir, 'tr_imglast.png'), sample_img3.numpy(), format = 'png')
    
    return mean, std

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def process_stage_info(model_name="vgg16_bn", exp_aux_name = "", save_best_model =True, model_save_dir = "/data/shantanu/Vision_models/0_Final_pytorch_imnet_exps/",  dataset_dir = "/data/shantanu/Image_datasets/200_Imagenet_hierarchy_traversed/",
                       mgrowth=False, irefine = False, use_hierarchy= False,
                       use_data_aug = False,
                       num_stages=5, stages_con=[1,1,1,1,1], stages_sat=[1,1,1,1,1], stages_res=[224,224,224,224,224],
                       stages_epochs=[10,10,15,20,50], stages_lr= [0.1,0.1,0.05,0.01,0.001],
                       stages_bs = [256,256,256,256,256],
                       num_classes = 1000,
                       hierarchy_level_stages = [1,1,2,2,2],
                       model_growth_fn = 'gradual_layer_add',
                       evaluate_mode = False,
                       lr_decay_every = 30,
                      pretrained=False,
                      lr_decay_ratio = 0.1,
                      only_last_data_aug=False,
                      lr_buffer = 0,
                      lr_first_decay = 30, 
                      filter_growth = False,
                      change_at_inp_layer = False, 
                      args = None,
                      cutoff_eps = [1,1,2,3,10]):
    
    original_model_name = model_name
    #print(irefine)
    #There are 8 possible experiment types numbered as follow
        #1: Grow model, refine input, use hierarchy
        #8: No growth, no refinement, no hierarchy
    exp_name = "exptype:{}_"
    mapped_numbers = 2**4
    
   
         
            
    bitwise = "0"
    if(filter_growth):
        if(change_at_inp_layer):
            exp_name += "A_FiltergrowthInp"
        else:
            exp_name += "B_FiltergrowthNotInp"
            bitwise +="1"
    else:
        bitwise +="0"
        
    if(mgrowth):
        exp_name += "_layergrowth"
        bitwise += "1"
    else:
        bitwise += "0"
    
    if(irefine):
        exp_name += "_inpRefine"
        bitwise += "1"
    else:
        bitwise += "0"
    
    if(use_hierarchy):
        exp_name += "_hierarchical"
        bitwise += "1"
        print("Code for hierarchy not done yet. Exiting..")
        sys.exit(0)
    else:
        bitwise += "0"
   
    exp_num = int(bitwise, 2)
    exp_name = exp_name.format(exp_num)
    if(exp_num==0):
        exp_name += "BASELINE_FF_notgrow_notrefine_nothierarchical"
        exp_name += "_same_lr_schedule"

    
        
    if(use_hierarchy):
        print("Code for hierarchy not done yet. Exiting..")
        sys.exit(0)
        
    exp_name += exp_aux_name
    if(dataset_dir.split('/')[-1]==''):
        dataset_name = dataset_dir.split('/')[-2]#'imnet_{}'.format(num_classes)
    else:
        dataset_name = dataset_dir.split('/')[-1]
        
        
    exp_dir = os.path.join(model_save_dir, dataset_name)
    exp_dir = os.path.join(exp_dir, original_model_name)
    exp_dir = os.path.join(exp_dir, exp_name)
    if(save_best_model):
        exp_dir = os.path.join(exp_dir, "use_best_from_prv_stage")
    else:
        exp_dir = os.path.join(exp_dir, "use_last_from_prv_stage")
    
     
    stages_exp_str = "numstages:{}".format(num_stages)
    stages_exp_str += "_epochs:{}".format(str(stages_epochs))
    stages_exp_str += "_lrs:{}".format(str(stages_lr))
    stages_exp_str += "_bs:{}".format(str(stages_bs))
    stages_exp_str += "_lrfd:{}".format(str(lr_first_decay))
    stages_exp_str += "_lrde:{}".format(str(lr_decay_every))
    stages_exp_str += "_lrdr:{}".format(str(lr_decay_ratio))
    
    #if(lr_buffer > 0):
     #   stages_exp_str += "_decbuffer:{}".format(lr_buffer)
    
    if(pretrained):
        stages_exp_str += "FINETUNED"#.format(str(lr_decay_every))
        
    if(irefine):##Implies stages being used
        if(len(stages_sat)!= num_stages or len(stages_res)!=num_stages or len(stages_con)!=num_stages):
            print(stages_sat, num_stages, stages_res, stages_con)
            print("Num stages different from provided stage sat, res and con values. Exiting...")
            sys.exit(0)
            
        stages_exp_str+="_sat:{}_con:{}_res:{}".format(stages_sat, stages_con, stages_res)
    
    if(mgrowth):
        stages_exp_str+="_cutoff:{}_".format(cutoff_eps)
        #stages_exp_str+="_mgrowthfn:{}".format(model_growth_fn)
        
   
        
    if(use_data_aug):
        stages_exp_str+="_dataaug_"
        if(only_last_data_aug):
            stages_exp_str+="laststgonly"
    
    exp_dir = os.path.join(exp_dir,stages_exp_str)
    if(os.path.exists(exp_dir) and not evaluate_mode):
        print("Path already exists: {}".format(exp_dir))
        inp = int(input("0: Exit \n 1: Exit \n 2: Delete and fresh exp \n 3: New trial (adds a trial).\n Enter choice: "))
        if(inp in [2]):
            shutil.rmtree(exp_dir)
        elif(inp in [3]):
            trial_n_inp = input("Enter trial number: ")
            exp_dir = exp_dir+ "trial:{}_".format(trial_n_inp)
        else:
            sys.exit(0)
            
    return exp_dir


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
            model.classifier[-1] = Linear(in_features=4096, out_features=new_num_classes, bias=True)
    
    elif(model_arch_name.startswith("resnet")):
            #101, 152, 50-> 2048
            #18,34 -> 512
        if(new_num_classes != model.fc.out_features):
            if(model_arch_name in ['resnet101','resnet152','resnet50']):
                model.fc = Linear(in_features=2048, out_features=new_num_classes, bias=True)
            elif(model_arch_name in ['resnet18','resnet34']):
                model.fc = Linear(in_features=512, out_features=new_num_classes, bias=True)
                
    elif(model_arch_name.startswith("resnext") or model_arch_name.startswith("wide")):
            #101, 152, 50-> 2048
            #18,34 -> 512
        if(new_num_classes != model.fc.out_features):
            #if(model_arch_name in ['resnet101','resnet152','resnet50']):
            model.fc.out_features = new_num_classes
    
    #elif(model_arch_name.starts_)
    elif(model_arch_name.startswith('custom_vgg')):
        if(new_num_classes != model.classifier[-1].out_features):
            model.classifier[-1] = Linear(in_features=4096, out_features=new_num_classes, bias=True)
    
    elif(model_arch_name.startswith('densenet')):
        #if(model_arch_name == 'densenet121'):
            #in_features = 2
        if(new_num_classes != model.classifier.out_features):
            model.classifier.out_features = new_num_classes
         #(classifier): Linear(in_features=2208, out_features=1000, bias=True)
    elif(model_arch_name.startswith('alexnet')):
        if(new_num_classes != model.classifier[-1].out_features):
            model.classifier[-1].out_features = new_num_classes
            
    elif(model_arch_name.startswith('inception')):
        if(new_num_classes != model.fc.out_features):
            model.fc.out_features = new_num_classes
        #classifier[-1]
    elif(model_arch_name.startswith('googlenet')):
        if(new_num_classes != model.fc.out_features):
            model.fc.out_features = new_num_classes
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
    
    elif(arch_name.startswith('fg_resnet')):
        from filter_growth_resnet import load_resnet_model
        arch_name_rnet = arch_name[3:]
        custom_model = load_resnet_model(arch_name_rnet)
            
        return custom_model
    
    else:
        print("Model not found. Currently only vgg custom supported (custom_vgg_11_bn, custom_vgg_13 etc).Exiting..")
        sys.exit(0)


if __name__ == '__main__':
    main()
