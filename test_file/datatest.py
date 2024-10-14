import numpy as np
import os
import argparse
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import sys
sys.path.append("..")
from model.cifar.alexnet import *
from data import *
def main():
    #set parameters
    batch_size = args.batch_size

    #set data and dataloader
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if (args.dataset == 'cifar10'):
        train_data = torchvision.datasets.CIFAR10(root='/home/lijiacheng/torhvision_dataset/',train=True,
                                       transform=transform_train,target_transform=None,
                                       download=True)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=False,num_workers=0)


    target_dataset = dataset(dataset_name=args.dataset, gpu=1, membership_attack_number=1,
                             cutout=0,n_holes=1,length=1)
    self_transform_train = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    self_train_set = part_pytorch_dataset(target_dataset.train_data,target_dataset.train_label,train=True,transform=self_transform_train)

    print ("total self numpy array shape:",target_dataset.train_data.shape)

    for i,(images,labels) in enumerate(train_loader):

        print("torch tensor images shape:",images.size())

        self_images = target_dataset.train_data[i*batch_size:(i+1)*batch_size]

        print ("self numpy array shape:",self_images.shape)

        self_images = self_transform_train(self_images[0])

        print ("self tensor images shape:",self_images.shape)

        residual = images - self_images

        print ("residual:",residual)

        if (i>100):
            print ("if all residuals are 0, then data is the same")
            exit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='cifar10')
    parser.add_argument('--batch_size',type=int,default=1)
    args = parser.parse_args()
    main()
    print (vars(args))