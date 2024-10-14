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
from model.cifar.densenet import densenet

from model import *
from utils import *
def main():
    #set parameters
    epochs = args.target_epochs
    learning_rate = args.target_learning_rate
    decay = args.target_l2_ratio
    batch_size = args.target_batch_size

    #set data and dataloader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if (args.dataset == 'cifar10'):
        train_data = torchvision.datasets.CIFAR10(root='/home/lijiacheng/torhvision_dataset/',train=True,
                                       transform=transform_train,target_transform=None,
                                       download=True)
        test_data = torchvision.datasets.CIFAR10(root='/home/lijiacheng/torchvision_dataset/',train=False,
                                        transform=transform_test,target_transform=None,
                                        download=True)
        target_model = models.densenet121(pretrained=True)
        #exit()
        target_model.classifier = nn.Linear(1024,10,bias=True)
        print (target_model)
        target_model = densenet(depth=100,num_classes=10)

    if (args.dataset == 'cifar100'):
        train_data = torchvision.datasets.CIFAR100(root='/home/lijiacheng/torhvision_dataset/',train=True,
                                       transform=transform_train,target_transform=None,
                                       download=True)
        test_data = torchvision.datasets.CIFAR100(root='/home/lijiacheng/torchvision_dataset/',train=False,
                                        transform=transform_test,target_transform=None,
                                        download=True)
        target_model = models.densenet121(pretrained=True)
        #exit()
        target_model.classifier = nn.Linear(1024,100,bias=True)
        print (target_model)
        target_model = densenet(depth=100,num_classes=100)

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=4)

    ## training
    dtype = torch.cuda.FloatTensor
    label_type = torch.cuda.LongTensor
    criterion = nn.CrossEntropyLoss().cuda()
    target_model = torch.nn.DataParallel(target_model).cuda()

    for epoch in range(epochs):
        # adjust learning rate
        optimizer = torch.optim.SGD(target_model.parameters(), lr=learning_rate,momentum=0.9, weight_decay=decay)
        if (epoch in args.schedule):
            learning_rate = learning_rate / 10
            print ("new learning rate = %f" % (learning_rate))
            optimizer = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)

        target_model.train()

        for i, (inputs, targets) in enumerate(train_loader):

            if (args.mixup):
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha)
                optimizer.zero_grad()
                inputs, targets_a, targets_b = Variable(inputs).type(dtype), Variable(targets_a).type(label_type), Variable(targets_b).type(label_type)
                outputs = target_model(inputs)

                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(criterion, outputs)
                loss.backward()
                optimizer.step()
                #total_loss+=loss*inputs.size(0)
            else:
                images = Variable(inputs).type(dtype)
                labels = Variable(targets).type(label_type)
                optimizer.zero_grad()
                outputs = target_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        ### testing accuracy
        correct = 0
        total = 0
        target_model.eval()
        # print ("train finished")
        for images, labels in train_loader:
            images = Variable(images).type(dtype)
            # print (images.size())
            outputs = target_model(images)
            labels = labels.type(label_type)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            # print (outputs.data)
            # print (predicted)
            # print (labels)
        print('Train Accuracy %f ' % (100.0 * correct / total))

        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images).type(dtype)
            outputs = target_model(images)
            labels = labels.type(label_type)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Test Accuracy %f ' % (100.0 * correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_data_size', type=int, default=3000)  # number of data point used in target model
    parser.add_argument('--target_model', type=str, default='cnn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_learning_rate', type=float, default=0.001)
    parser.add_argument('--target_batch_size', type=int, default=100)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--target_epochs', type=int, default=20)
    parser.add_argument('--attack_epochs', type=int, default=500)
    parser.add_argument('--target_l2_ratio', type=float, default=5e-4)
    parser.add_argument('--shadow_data_size', type=int, default=30000)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model_number', type=int, default=10)
    parser.add_argument('--attack_times', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--membership_attack_number', type=int, default=500)
    parser.add_argument('--reference_number', type=int, default=1)
    parser.add_argument('--schedule', type=int, nargs='+', default=[80,120])
    ## for wrn , schedule = 60 120 160
    parser.add_argument('--save_exp_data',type=int,default=0)
    parser.add_argument('--model_name',type=str,default='alexnet')
    parser.add_argument('--alpha',type=float,default='1.0')
    parser.add_argument('--cutout',type=int,default=0)
    parser.add_argument('--n_holes',type=int,default=1)
    parser.add_argument('--length',type=int,default=16)
    parser.add_argument('--pretrained',type=int,default=0)
    parser.add_argument('--temperature_scaling',type=int,default=1)
    parser.add_argument('--freeze_layer',type=int,default=0)
    parser.add_argument('--num_gt',type=int,default=5)
    parser.add_argument('--mixup',type=int,default=0)
    # attack model configuration
    #    parser.add_argument('--attack_model', type=str, default='svm')
    #    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    #    parser.add_argument('--attack_batch_size', type=int, default=100)
    #    parser.add_argument('--attack_n_hidden', type=int, default=50)
    #    parser.add_argument('--attack_epochs', type=int, default=50)
    #    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)

    # parse configuration
    args = parser.parse_args()
    print (vars(args))
    main()
    print (vars(args))