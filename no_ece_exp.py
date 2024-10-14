from whitebox_attack import *
from blackbox_attack import *
import argparse
from data import dataset
from model import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from user import *
from data import *
from tqdm import tqdm
import copy
from fed_attack import *
from opacus import PrivacyEngine
from model_utils import *
from model_utils import _batchnorm_to_groupnorm_new
from diffmi_attack import diffmi_attack
from nasr_fed_attack import nasr_fed_attack
from multi_party_attack import *

import torch.backends.cudnn as cudnn
import os
import argparse

from torch.autograd import Variable
import numpy as np
import random

import sklearn
import argparse
from data import dataset
from model import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from user import *
from data import *
from tqdm import tqdm
import copy
from opacus import PrivacyEngine
from model_utils import _ECELoss
from model_utils import _batchnorm_to_groupnorm_new

parser = argparse.ArgumentParser(description='cSG-MCMC CIFAR10 Training')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch_size', type=int, default=100,
                    help='input batch size for training (default: 64)')
parser.add_argument('--alpha', type=int, default=1,
                    help='1: SGLD')
parser.add_argument('--device_id',type = int, help = 'device id to use',default=0)
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1./50000,
                    help='temperature (default: 1/dataset_size)')
parser.add_argument('--dataset',type=str,default='cifar100')

args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
sklearn.utils.check_random_state(args.seed)
model_name = 'resnet20'
target_dataset = dataset(dataset_name=args.dataset, gpu=1, membership_attack_number=0)
criterion = nn.CrossEntropyLoss()
net = ResNet18(10).cuda(device_id)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
datasize = 45000

def assign_part_dataset(dataset):
	if (dataset.dataset_name == 'cifar10' or dataset.dataset_name == 'cifar100'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),  ### totensor will perform the divide by 255 op
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		target_transform = transforms.ToTensor()
	
	train_data = np.load(f'./csgmcmc/{args.dataset}_{int(datasize / 10)}_train_data.npy')
	train_label = np.load(f'./csgmcmc/{args.dataset}_{int(datasize / 10)}_train_label.npy')
	test_data = np.load(f'./csgmcmc/{args.dataset}_{int(datasize / 10)}_test_data.npy')
	test_label = np.load(f'./csgmcmc/{args.dataset}_{int(datasize / 10)}_test_label.npy')
	
	print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)
	
	ece_data = np.concatenate((train_data,test_data),axis=0)
	ece_label = np.concatenate((train_label,test_label),axis=0)
	
	#### create dataset and dataloader
	train = part_pytorch_dataset(train_data, train_label, train=True, transform=transform_train,target_transform=target_transform)
	test = part_pytorch_dataset(test_data, test_label, train=False, transform=transform_test,target_transform=target_transform)
	train_eval= part_pytorch_dataset(train_data,train_label,train=False,transform=transform_test,target_transform=target_transform)
	ece = part_pytorch_dataset(ece_data,ece_label,train=False,transform=transform_test,target_transform=target_transform)
	
	train_data_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=1)
	test_data_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=1)
	train_eval_data_loader = torch.utils.data.DataLoader(train_eval, batch_size=args.batch_size, shuffle=False, num_workers=1)
	ece_loader = torch.utils.data.DataLoader(ece, batch_size=args.batch_size, shuffle=False, num_workers=1)
	
	return train_data_loader, test_data_loader,train_eval_data_loader,ece_loader

trainloader, testloader,train_eval_loader,ece_loader = assign_part_dataset(target_dataset)


def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets, _) in enumerate(trainloader):
		inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		
		train_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		
		if batch_idx % 200 == 0:
			print(' Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct.item() / total, correct, total))
	
	return train_loss / (batch_idx + 1), 100. * correct.item() / total


def test(net):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	
	ece_criterion = _ECELoss().cuda()
	logits_list = []
	labels_list = []
	
	with torch.no_grad():
		for batch_idx, (inputs, targets, _) in enumerate(testloader):
			if use_cuda:
				inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			
			test_loss += loss.data.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
			
			logits_list.append(outputs)
			labels_list.append(targets)
	
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss / len(testloader), correct,total, 100. * correct.item() / total))
	
	logits = torch.cat(logits_list).cuda()
	labels = torch.cat(labels_list).cuda()
	ece_loss = ece_criterion(logits, labels).detach().item()
	print(f"test ece loss {ece_loss}")
	#return test_loss / len(testloader), 100. * correct.item() / total, ece_loss
	
	ece_criterion = _ECELoss().cuda()
	logits_list = []
	labels_list = []
	
	with torch.no_grad():
		for batch_idx, (inputs, targets, _) in enumerate(trainloader):
			if use_cuda:
				inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			
			test_loss += loss.data.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
			
			logits_list.append(outputs)
			labels_list.append(targets)
	
	print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss / len(trainloader), correct,
	                                                                             total, 100. * correct.item() / total))
	
	logits = torch.cat(logits_list).cuda()
	labels = torch.cat(labels_list).cuda()
	ece_loss = ece_criterion(logits, labels).detach().item()
	print(f"train ece loss {ece_loss}")
	#return test_loss / len(testloader), 100. * correct.item() / total, ece_loss
	
def get_attack_input(net):
	confidences = []
	classes = []
	labels = []
	with torch.no_grad():
		for batch_idx, (inputs, targets, _) in enumerate(train_eval_loader):
			inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
			outputs = net(inputs)
			outputs = F.softmax(outputs,dim=1)
			confidences.append(outputs.cpu().numpy())
			classes.append(targets.cpu().numpy())
			labels.append(np.ones((len(targets))))
			if (batch_idx==99):
				break

	with torch.no_grad():
		for batch_idx, (inputs, targets, _) in enumerate(testloader):
			inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
			outputs = net(inputs)
			outputs = F.softmax(outputs,dim=1)
			confidences.append(outputs.cpu().numpy())
			classes.append(targets.cpu().numpy())
			labels.append(np.zeros((len(targets))))
			if (batch_idx==99):
				break
	
	labels = np.array(labels).flatten()
	classes = np.hstack(classes)
	#print (classes.shape,labels.shape)
	confidences = np.reshape(np.array(confidences),(len(labels),-1))
	
	return confidences,classes,labels

for epoch in range(args.epochs):
	train_loss, train_acc = train(epoch)
	if (epoch % 10 == 0):
		test(net)
	
test(net)
black_ref = blackbox_attack(10000, 'global_prob', num_classes=10)
total_confidences,total_classes,total_labels = get_attack_input(net)
black_ref.attack(total_confidences=total_confidences, total_classes=total_classes,total_labels=total_labels, output_file=None)

torch.save(net.state_dict(),'no_ece_model.pt')
### find a temperature that makes ECE_train = ECE_test
from temperatue_scaling import ModelWithTemperature
new_net = ModelWithTemperature(net)
new_net.set_temperature(ece_loader)
test(new_net)
black_ref = blackbox_attack(10000, 'global_prob', num_classes=10)
total_confidences,total_classes,total_labels = get_attack_input(new_net)
black_ref.attack(total_confidences=total_confidences, total_classes=total_classes,total_labels=total_labels, output_file=None)
