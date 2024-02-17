import glob
from torch.utils.tensorboard import SummaryWriter
import os
import losses
import utils
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models import VxmDenseBPCAMax_2
import random
import time


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():

    lr = 0.0001
    epoch_start = 0
    max_epoch = 500
    img_size = (160, 192, 224)
    cont_training = False

    '''
    Initialize model
    '''
    model = VxmDenseBPCAMax_2(img_size)
    model.cuda()

    from torchsummary import summary
    summary(model, [(2, 160, 192, 224)])


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    # torch.manual_seed(0)

    # reproducibility
    seed = 42

    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_iden)

    main()
