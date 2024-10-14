from __future__ import print_function

import argparse
import os
import shutil
import time
import random

from utils import *
from model_utils import _batchnorm_to_groupnorm_new
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from data import dataset, part_pytorch_dataset
from model import *
from model_utils import get_gpu_status
from model_utils import *


def get_naming_mid_str(setting_dict):
	# name_string_mid_str =  str(0) + '_' + str(0) + '_' + ('server_') + \
	#                      (0) + '_' + (0.0) + '_' + str(10) + '_' + str(setting_dict['num_step']) + '_' + str(0) + '_' + str(0.0) + \
	#                       '_' + str(0.0) + '_' + str(0) + '_' + str(0.0) + '_' + str(0) + '_'
	num_step = setting_dict['num_step']
	name_string_mid_str = f'0_0_server_0_0.0_10_{num_step}_0_0.0_0.0_0_0.0_0_'
	return name_string_mid_str


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable

def get_tpr(pred,label):
	# in this function, we assume that higher the pred is, higher chance it is a member.
	sorted_index = np.argsort(pred)[::-1]
	sorted_label = label[sorted_index]
	fpr_count = 10
	# we allow 10 false positive, because 1 false positive will be very unstable.
	tpr_count = 0
	for i in range(len(label)):
		if (sorted_label[i] == 0):
			fpr_count = fpr_count-1
			if (fpr_count == 0):
				break
		else:
			tpr_count+=1
	print (f" fpr {2*10/len(label)}, tpr {2*tpr_count/len(label)}") ## 2 is because the evaluation set is balanced
	return 2*tpr_count/(len(label))


class smallfc(nn.Module):
	def __init__(self, input_dim,num_neuron,output_dim):
		super().__init__()
		self.fc1 = nn.Linear(input_dim,num_neuron)
		self.fc2 = nn.Linear(num_neuron,output_dim)
	
	def forward(self, x):
		x = x.view(x.size(0),-1)
		x = F.relu(self.fc1(x))
		out = self.fc2(x)
		return out

class InvariantModel(nn.Module):
	def __init__(self, phi, rho):
		super().__init__()
		self.phi = phi
		self.rho = rho
	
	def forward(self, x):
		# compute the representation for each data point
		x = self.phi.forward(x)
		# sum up the representations
		# here I have assumed that x is 2D and the each row is representation of an input, so the following operation
		# will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
		x = torch.sum(x, dim=1, keepdim=True)
		# compute the output
		out = self.rho.forward(x)
		#print (out.size())
		
		return out


class SmallMNISTCNNPhi(nn.Module):
	def __init__(self, kernel_size=10, output_dim=100):
		super().__init__()
		self.output_dim = output_dim
		#self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=kernel_size,stride=1, padding=0)
		#self.conv2 = nn.Conv1d(10, 20, kernel_size=kernel_size,stride=1, padding=0)
		#self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(output_dim, 50)
		#self.fc1_drop = nn.Dropout2d()
		self.fc2 = nn.Linear(50, 10)
	
	def forward(self, x):
		#x = F.relu(self.conv1(x))
		#print (x.size())
		#x = self.conv2_drop(self.conv2(x))
		#x = F.relu(x)
		#x = x.view(x.size(0), -1)
		#print (x.size())
		#x = F.relu(self.fc1(x))
		#x = self.fc1_drop(x)
		#x = F.relu(self.fc2(x))
		#print (x.size())
		#return x
		x = x.view(x.size(0),-1)
		#print (x.size())
		#print (self.output_dim)
		x = self.fc1(x)
		x = self.fc2(x)
		return x

class SmallRho(nn.Module):
	def __init__(self, input_size=1, output_size=2,num_neuron=10):
		super().__init__()
		self.input_size = input_size
		self.output_size = output_size
		
		self.fc1 = nn.Linear(self.input_size, num_neuron)
		self.fc2 = nn.Linear(num_neuron, self.output_size)
	
	def forward(self, x):
		#print (x.size())
		x = F.relu(self.fc1(x))
		#print (x.size())
		x = self.fc2(x)
		return x


def read_all_comm_round_data(comm_round_list, prefix, mid_str, feature_str, final_str, have_label=True, label_str=''):
	all_data = []
	all_label = []
	
	if (len(comm_round_list) == 1):
		
		for comm_round_idx in comm_round_list:
			data_name = prefix + feature_str + mid_str + str(comm_round_idx) + final_str
			data = np.load(data_name, allow_pickle=True)
			all_data.append(data)
			
			label_name = prefix + label_str + mid_str + str(comm_round_idx) + final_str
			label = np.load(label_name)
			all_label.append(label)
		
		# print (data.shape, label.shape)
		
		all_data = np.array(all_data)
		all_label = np.array(all_label).flatten()
		
		# print (all_data.shape, all_label.shape)
		
		return np.squeeze(all_data), np.squeeze(all_label)


def transform_loss_to_prob(x):
	### this can be rewritten as a map function
	### we are using cross entropy loss for all tasks
	new_x = np.exp(-x)
	return new_x


def prepare_attack_data(epochs, prefix, mid_str, dataset, model, target_data_size, eval_data_size, num_user=10, comm_round_list=[], num_layer=5, best_layer=0,feature_name='cos'):
	num_valid_user = num_user
	total_data_num = (num_user * 2) * eval_data_size
	all_epoch_feature = np.zeros((total_data_num, num_valid_user, len(epochs)))
	
	for epoch_idx, epoch in enumerate(epochs):
		final_str = '_' + str(
			epoch + 1) + '_' + str(dataset) + '_' + str(target_data_size) + '_' + str(eval_data_size) + '_' + str(
			model) + '.npy'
		data, label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                                 feature_str='all_info_multi_party_member_attack_',
		                                                 final_str=final_str,
		                                                 label_str='all_label_multi_party_member_attack_')
		data = np.reshape(data, (-1, num_valid_user, num_layer, 7))
		if (feature_name == 'cos'):
			all_epoch_feature[:, :, epoch_idx] = copy.deepcopy(data[:,:,best_layer,0])
		elif (feature_name == 'grad-diff'):
			all_epoch_feature[:, :, epoch_idx] = copy.deepcopy(data[:,:,best_layer,1] - data[:,:,best_layer,2])
		
	
	all_epoch_feature = np.nan_to_num(all_epoch_feature)
	
	### for each user, we prepare attack_train_loader and attack_test_loader
	all_attack_train_loader = []
	all_attack_test_loader = []
	for this_user_idx in range(num_user):
		this_user_member_index = np.arange(total_data_num)[this_user_idx * eval_data_size:(this_user_idx + 1) * eval_data_size]
		this_user_nonmember_index = np.arange(total_data_num)[(this_user_idx + num_user) * eval_data_size:(this_user_idx + 1 + num_user) * eval_data_size]
		this_user_index = np.concatenate((this_user_member_index, this_user_nonmember_index))
		
		this_user_probs = all_epoch_feature[this_user_index, this_user_idx, :]
		this_user_probs = np.reshape(this_user_probs, (-1, len(epochs), 1))
		this_user_labels = np.concatenate((np.ones(len(this_user_member_index)), np.zeros((len(this_user_nonmember_index)))))
		
		this_user_train_index = np.random.choice(len(this_user_labels), int(0.5 * len(this_user_labels)), replace=False)
		this_user_test_index = np.setdiff1d(np.arange(len(this_user_labels)), this_user_train_index)
		
		#print (f"train data num {len(this_user_train_index)}, test data num{len(this_user_test_index)}")
		
		attack_train = part_pytorch_dataset(this_user_probs[this_user_train_index], this_user_labels[this_user_train_index], train=True, transform=None, target_transform=transforms.ToTensor())
		attack_train_loader = torch.utils.data.DataLoader(attack_train, batch_size=100, shuffle=True, num_workers=1)
		
		attack_test = part_pytorch_dataset(this_user_probs[this_user_test_index], this_user_labels[this_user_test_index], train=True, transform= None, target_transform=transforms.ToTensor())
		attack_test_loader = torch.utils.data.DataLoader(attack_test, batch_size=100, shuffle=True, num_workers=1)
		
		all_attack_train_loader.append(attack_train_loader)
		all_attack_test_loader.append(attack_test_loader)
	
	return all_attack_train_loader, all_attack_test_loader


def train(model, loader, optim):
	criterion = nn.CrossEntropyLoss()
	model.train().cuda()
	for _ in range(100):
		for x, y, _ in loader:
			# print (x.size())
			#print (x.size(),y.size())
			optim.zero_grad()
			x = x.cuda()
			pred = model(x)
			y = y.cuda()
			loss = criterion(pred, y)
			loss.backward()
			optim.step()
	#print("train finished")
	return model


def eval(model, loader, default_fpr=1e-3):
	model.eval().cuda()
	all_pred = []
	all_label = []
	for x, y, _ in loader:
		x = x.cuda()
		pred = model(x).detach().cpu().numpy()
		all_pred.append(pred)
		all_label.append(y.numpy())
	
	all_pred = np.reshape(all_pred, (-1, 2))
	all_label = np.array(all_label).flatten()
	
	# print (all_pred.shape,all_label.shape)
	### acc
	pred_label = np.argmax(all_pred, axis=1)
	cor = 0
	for i in range(len(all_label)):
		if (all_label[i] == pred_label[i]):
			cor += 1
	print(f"acc: {cor / len(pred_label) * 100:.2f}")
	acc = cor / len(pred_label) * 100
	
	all_pred = all_pred[:, 1]
	### calculate AUC score and TPR at 1e-3 FPR
	from sklearn.metrics import roc_auc_score, roc_curve
	print(f" AUC SCORE {roc_auc_score(all_label, all_pred)} ")
	auc = roc_auc_score(all_label, all_pred)
	return_tpr = get_tpr(pred=all_pred,label=all_label)
	return (acc, auc, return_tpr)

cifar10_densenet_dict = {'output_dim': 499, 'kernel_size': 166, 'target_data_size': 4000, 'epochs': 500, 'model': 'densenet_cifar', 'dataset': 'cifar10', 'num_classes': 10,
                         'eval_data_size': 1000, 'num_step': 40, 'num_layer':100, 'best_layer': 66}
cifar100_densenet_dict = {'output_dim': 499, 'kernel_size': 166, 'target_data_size': 4000, 'epochs': 500, 'model': 'densenet_cifar', 'dataset': 'cifar100', 'num_classes': 100,
                          'eval_data_size': 1000, 'num_step': 40, 'num_layer':100, 'best_layer': 66}

cifar10_alexnet_dict = {'output_dim': 299, 'kernel_size': 100, 'target_data_size': 4000, 'epochs': 300, 'model': 'alexnet', 'dataset': 'cifar10', 'num_classes': 10,
                        'eval_data_size': 1000, 'num_step': 40, 'num_layer':6, 'best_layer': 2}
cifar100_alexnet_dict = {'output_dim': 299, 'kernel_size': 100, 'target_data_size': 4000, 'epochs': 300, 'model': 'alexnet', 'dataset': 'cifar100', 'num_classes': 100,
                         'eval_data_size': 1000, 'num_step': 40, 'num_layer':6, 'best_layer': 4}
purchase_dict = {'output_dim': 99, 'kernel_size': 33, 'target_data_size': 4000, 'epochs': 100, 'model': 'purchase', 'dataset': 'purchase', 'num_classes': 100,
                 'eval_data_size': 1000, 'num_step': 40, 'num_layer':4, 'best_layer': 0}
kidney_dict = {'output_dim': 149, 'kernel_size': 50, 'target_data_size': 400, 'epochs': 150, 'model': 'kidney', 'dataset': 'kidney', 'num_classes': 4, 'eval_data_size': 400,
               'num_step': 4, 'num_layer':6, 'best_layer': 1}
retina_dict = {'output_dim': 99, 'kernel_size': 33, 'target_data_size': 3000, 'epochs': 100, 'model': 'retina', 'dataset': 'retina', 'num_classes': 4, 'eval_data_size': 1000,
               'num_step': 20, 'num_layer':6, 'best_layer': 1}
chest_dict = {'output_dim': 99, 'kernel_size': 33, 'target_data_size': 200, 'epochs': 100, 'model': 'chest', 'dataset': 'chest', 'num_classes': 2, 'eval_data_size': 200,
              'num_step': 2, 'num_layer':6, 'best_layer': 1}
skin_dict = {'output_dim': 149, 'kernel_size': 50, 'target_data_size': 500, 'epochs': 150, 'model': 'skin', 'dataset': 'skin', 'num_classes': 23, 'eval_data_size': 500,
             'num_step': 5, 'num_layer':6, 'best_layer': 4}
medical_mnist_dict = {'output_dim': 49,'kernel_size':16,'target_data_size':4000,'epochs':50,'model':'medical_mnist','dataset':'medical_mnist','num_classes':6,'eval_data_size':1000,'num_step':40,'num_layer':4, 'best_layer': 2}

setting_dict = cifar100_alexnet_dict
#attack_feature = 'grad-diff'
attack_feature = 'cos'

number_parties = 10
name_string_prefix = '/home/lijiacheng/whiteboxmi/new_expdata/'
name_string_mid_str = get_naming_mid_str(setting_dict)

all_attack_train_loader, all_attack_test_loader = prepare_attack_data(
	np.arange(setting_dict['epochs'] - 1) + 1,
	name_string_prefix,
	name_string_mid_str,
	setting_dict['dataset'],
	setting_dict['model'],
	setting_dict['target_data_size'],
	setting_dict['eval_data_size'],
	num_user=number_parties,
	comm_round_list=np.arange(1),
	num_layer=setting_dict['num_layer'],
	best_layer=setting_dict['best_layer'],
	feature_name=attack_feature)

all_result = []
for i in range(number_parties):
	#inferenece_model = attackmodel(setting_dict['output_dim'], setting_dict['kernel_size'])
	#phi = SmallMNISTCNNPhi(kernel_size=setting_dict['kernel_size'],output_dim=setting_dict['output_dim'])
	#rho = SmallRho(num_neuron=200)
	#inferenece_model = InvariantModel(phi,rho)
	inferenece_model = smallfc(input_dim=setting_dict['output_dim'],num_neuron=200,output_dim=2)
	inferenece_model = inferenece_model.type(torch.float32).cuda()
	optimizer_mem = optim.Adam(inferenece_model.parameters(), lr=0.001,weight_decay=5e-3)
	
	if (i == 0):
		pytorch_total_params = sum(p.numel() for p in inferenece_model.parameters())
		print(f"total params {pytorch_total_params}")
	
	train(inferenece_model, all_attack_train_loader[i], optimizer_mem)
	result = eval(inferenece_model, all_attack_test_loader[i], default_fpr=2 / setting_dict['eval_data_size'])
	all_result.append(result)

print(setting_dict)
print (attack_feature)
all_result = np.array(all_result)
print(f'avg: acc {np.average(all_result,axis=0)[0]}, auc {np.average(all_result,axis=0)[1]} , tpr {np.average(all_result,axis=0)[2]}')
print(f'std: acc {np.std(all_result,axis=0)[0]}, auc {np.std(all_result,axis=0)[1]} , tpr {np.std(all_result,axis=0)[2]}')