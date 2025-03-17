import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.mask_cls import MaskCls
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

def main(rank, world_size):

    config = load_config(mode=1)

    # initialize distributed training
    dist.init_process_group(
        backend='nccl',
        # init_method='tcp://210.77.19.93:22',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    # cleanup()
    
    torch.cuda.set_device(rank)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda", rank)
    else:
        config.DEVICE = torch.device("cpu")


    torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = MaskCls(config, world_size, rank)

    # if rank == 0:
        # model.load()


    iteration = model.ClsModel.iteration
    if len(config.GPU) > 1 and rank == 0:
        print('GPU:{}'.format(config.GPU))
    
    model.ClsModel.net = nn.parallel.DistributedDataParallel(model.ClsModel.net, device_ids=[rank])

    model.ClsModel.iteration = iteration

    # model training
    if rank == 0:
        print('\nstart training...\n')
    model.train()
    
    cleanup()

def cleanup():
    dist.destroy_process_group()

def load_config(mode=None):


    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./ckpts_cls', help='model checkpoints path')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    config = Config(config_path)
    if len(config.GPU) > 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '22323'
    # train mode
    if mode == 1:
        config.MODE = 1

    # test mode
    elif mode == 2:
        config.MODE = 2


    return config


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    world_size = 1
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)