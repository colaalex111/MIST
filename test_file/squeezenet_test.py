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

def main():
    #set parameters
    torch.set_printoptions(threshold=5000, edgeitems=20)
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

        target_model = models.squeezenet1_0(pretrained=False)
        target_model.classifier[-1] = nn.AvgPool2d(1, 1)

    if (args.dataset == 'cifar100'):
        train_data = torchvision.datasets.CIFAR100(root='/home/lijiacheng/torhvision_dataset/',train=True,
                                       transform=transform_train,target_transform=None,
                                       download=True)
        test_data = torchvision.datasets.CIFAR100(root='/home/lijiacheng/torchvision_dataset/',train=False,
                                        transform=transform_test,target_transform=None,
                                        download=True)

        target_model = models.squeezenet1_0(pretrained=False)
        target_model.classifier[1] = nn.Conv2d(512,10,kernel_size=1)
        target_model.classifier[-1] = nn.AvgPool2d(kernel_size=1)
        target_model.num_classes = 10

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=False,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=0)

    ## training
    dtype = torch.cuda.FloatTensor
    label_type = torch.cuda.LongTensor
    criterion = nn.CrossEntropyLoss().cuda()
    target_model.type(dtype)
    for epoch in range(epochs):
        # adjust learning rate
        optimizer = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
        if (epoch in args.schedule):
            learning_rate = learning_rate / 10
            print ("new learning rate = %f" % (learning_rate))
            optimizer = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).type(dtype)
            labels = Variable(labels).type(label_type)
            optimizer.zero_grad()
            #if (epoch == 0 and i==0):
            #    print (images[0])

            outputs = target_model(images)
            loss = criterion(outputs, labels)

            total_loss = loss
            total_loss.backward()
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
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=128)
    parser.add_argument('--target_epochs', type=int, default=20)
    parser.add_argument('--target_l2_ratio', type=float, default=5e-4)
    parser.add_argument('--schedule', type=int, nargs='+', default=[80,120])
    parser.add_argument('--dataset',type=str,default='cifar10')
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