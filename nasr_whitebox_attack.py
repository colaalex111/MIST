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


def get_transformation(dataset):
	if (dataset.dataset_name == 'fashion_mnist'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(28),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
		])
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
	
	if (dataset.dataset_name == 'retina' or dataset.dataset_name == 'skin'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(64),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
		])
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
	
	# binary datasets
	elif (dataset.dataset_name == 'purchase' or dataset.dataset_name == 'texas' or dataset.dataset_name == 'location'):
		transform_train = None
		transform_test = None
		target_transform = None
	
	# cifar 10 / cifar100
	elif (dataset.dataset_name == 'cifar10' or dataset.dataset_name == 'cifar100'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		target_transform = transforms.ToTensor()
	
	return transform_train, transform_test, target_transform

## model definition
class InferenceAttack_HZ_FED(nn.Module):
	def __init__(self, num_classes, last_fc_shape):
		self.num_classes = num_classes
		super(InferenceAttack_HZ_FED, self).__init__()
		self.grads_conv = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Conv2d(1, 1000, kernel_size=(1, num_classes), stride=1),
			nn.ReLU(),
		)
		self.grads_linear = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear( last_fc_shape * 1000, 1024),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, 128),
			nn.ReLU(),
		)
		self.preds = nn.Sequential(
			nn.Linear(num_classes, 100),
			nn.ReLU(),
			nn.Linear(100, 64),
			nn.ReLU(),
		)
		self.correct = nn.Sequential(
			nn.Linear(1, num_classes),
			nn.ReLU(),
			nn.Linear(num_classes, 64),
			nn.ReLU(),
		)
		self.combine = nn.Sequential(
			nn.Linear(128 + 64 + 64 , 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)
		self.output = nn.Sigmoid()
		## init weights, no reason doing this, but default
		for key in self.state_dict():
			if key.split('.')[-1] == 'weight':
				nn.init.normal(self.state_dict()[key], std=0.01)
			elif key.split('.')[-1] == 'bias':
				self.state_dict()[key][...] = 0
	
	def forward(self, gradient, label, prediction):
		# only gradient / label / prediction is used here, no activation. Per my prev exp, using activation does nothing.
		# also, only the gradient of the last FC layer is used, per the paper. for better efficiency and same utility. .
		# process gradient
		#print (gradient.shape)
		gradient_out = self.grads_conv(gradient)
		#print (gradient_out.shape)
		gradient_out = self.grads_linear(gradient_out.view(100,-1))
		#print (gradient_out.shape)
		# process label
		label_out = self.correct(label.view(100,-1))
		# process prediction
		#print (prediction.shape)
		pred_out = self.preds(prediction.view(100,-1))
		#print (gradient_out.shape,label_out.shape,pred_out.shape)
		concat_input = torch.cat((gradient_out,label_out,pred_out),1)
		is_member = self.combine(concat_input)
		return self.output(is_member)

class nasr_whitebox_attack():
	
	def __init__(self,num_classes,last_fc_shape=256):
		self.attack_model = InferenceAttack_HZ_FED(num_classes,last_fc_shape=256)
		self.num_classes = num_classes
		self.last_fc_shape = last_fc_shape
		
	def prep_training_data(self,dataset,training_partition):
		criterion = nn.CrossEntropyLoss()
		self.target_model.eval()
		_,transform_test,target_transform = get_transformation(dataset)
		# prepare train_loader / test_loader
		train_index = training_partition
		test_index = np.arange(len(dataset.test_label))
		min_len = min(len(train_index),len(test_index))
		selected_train_index = np.random.choice(train_index,min_len,replace=False)
		selected_test_index = np.random.choice(test_index,min_len,replace=False)
		
		train_dataset = part_pytorch_dataset(dataset.train_data[selected_train_index], dataset.train_label[selected_train_index], train=False, transform=transform_test,
											 target_transform=target_transform)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
																	shuffle=False, num_workers=1)
		
		test_dataset = part_pytorch_dataset(dataset.test_data[selected_test_index], dataset.test_label[selected_test_index], train=False, transform=transform_test,
												  target_transform=target_transform)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
																shuffle=False, num_workers=1)
		num_classes = len(np.unique(dataset.train_label))
		attack_gradient = []
		attack_prediction = []
		attack_correct_label = []
		attack_membership_label = []
		
		for this_data, this_label,idx in train_loader:
			#print (idx)
			# to cuda()
			this_data = this_data.type(torch.float32)
			this_data = torch.autograd.Variable(this_data).cuda()
			this_label = torch.autograd.Variable(this_label).cuda()
			
			# compute prediction
			this_prediction = self.target_model(this_data)
			# this prediction needs to be softmaxed
			this_prediction = F.softmax(this_prediction, dim=1)
		
			# compute gradients
			self.target_model_optim.zero_grad()
			loss = criterion(this_prediction,this_label)
			loss.backward()
			# for alexnet last FC layer only. THIS NEEDS TO BE UPDATED FOR MORE MODELS.
			this_grad = copy.deepcopy(self.target_model.linear1.weight.grad).view([ 1, 256, num_classes])
			#print (this_data.shape,this_label.shape,this_prediction.shape,this_grad.shape)
			attack_gradient.append(this_grad)
			attack_prediction.append(torch.clone(this_prediction).detach())
			attack_correct_label.append(this_prediction[:,this_label].detach())
			attack_membership_label.append(torch.ones((len(this_label))))
		
		for this_data, this_label,_ in test_loader:
			# to cuda()
			this_data = this_data.type(torch.float32)
			this_data = torch.autograd.Variable(this_data).cuda()
			this_label = torch.autograd.Variable(this_label).cuda()
			
			# compute prediction
			this_prediction = self.target_model(this_data)
			# this prediction needs to be softmaxed
			this_prediction = F.softmax(this_prediction, dim=1)
			# compute gradients
			self.target_model_optim.zero_grad()
			loss = criterion(this_prediction, this_label)
			loss.backward()
			# for alexnet last FC layer only. THIS NEEDS TO BE UPDATED FOR MORE MODELS.
			this_grad = copy.deepcopy(self.target_model.linear1.weight.grad).view([ 1, 256, num_classes])
			#print (this_grad.requires_grad)
			
			attack_gradient.append(this_grad)
			attack_prediction.append(torch.clone(this_prediction).detach())
			attack_correct_label.append(this_prediction[:,this_label].detach())
			attack_membership_label.append(torch.zeros((len(this_label))))
		
		attack_gradient = torch.stack(attack_gradient)
		attack_prediction = torch.stack(attack_prediction)
		attack_correct_label = torch.stack(attack_correct_label)
		attack_membership_label = torch.stack(attack_membership_label)
		
		#print (attack_gradient.shape, attack_prediction.shape, attack_correct_label.shape, attack_membership_label.shape)
		
		self.attack_graident = attack_gradient
		self.attack_prediction = attack_prediction
		self.attack_correct_label = attack_correct_label
		self.attack_membership_label = attack_membership_label
		
		self.attack_train_index = np.random.choice(len(self.attack_membership_label),int(0.5*len(self.attack_membership_label)),replace=False)
		self.attack_test_index = np.setdiff1d(np.arange(len(self.attack_membership_label)),self.attack_train_index)
		
	def attack_model_train(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.attack_model.train().to(device)
		criterion = nn.CrossEntropyLoss()
		for e in range(self.train_epoch):
			batch_size = 100
			batch_num = len(self.attack_train_index) // batch_size
			for idx in range(batch_num):
				this_batch_index = self.attack_train_index[batch_size*idx:(idx+1)*batch_size]
				this_batch_gradient = self.attack_graident[this_batch_index].to(device)
				this_batch_prediction = self.attack_prediction[this_batch_index].to(device)
				this_batch_correct_label = self.attack_correct_label[this_batch_index].to(device)
				output = self.attack_model(this_batch_gradient,this_batch_correct_label,this_batch_prediction)
				this_batch_membership_label = self.attack_membership_label[this_batch_index].type(torch.LongTensor).to(device).view(-1)
				#print (this_batch_membership_label.shape)
				loss = criterion(output,this_batch_membership_label)
				if (idx == 0):
					print (f"epoch {e}, batch number {idx}, loss {loss}")
				self.attack_model_optim.zero_grad()
				loss.backward()
				self.attack_model_optim.step()
				#print (batch_num,e,idx)
		
	def attack_model_eval(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.attack_model.eval().to(device)
		batch_size = 100
		batch_num = len(self.attack_test_index) // batch_size
		all_output = []
		for idx in range(batch_num):
			this_batch_index = self.attack_test_index[batch_size*idx:(idx+1)*batch_size]
			this_batch_gradient = self.attack_graident[this_batch_index].to(device)
			this_batch_prediction = self.attack_prediction[this_batch_index].to(device)
			this_batch_correct_label = self.attack_correct_label[this_batch_index].to(device)
			output = self.attack_model(this_batch_gradient,this_batch_correct_label,this_batch_prediction).detach()
			output = F.softmax(output,dim=1)
			all_output.append(output)
			#print (idx)
			
		all_output = torch.stack(all_output).view(-1,2)
		#print (all_output.shape)
		
		self.summarize_result(all_output[:,1],self.attack_membership_label[self.attack_test_index])
		
	def summarize_result(self,pred,label):
		### need to eval the result using AUC, TPR @low fpr, eg 0.001
		#ACC
		corr = 0
		for i in range(len(pred)):
			if (pred[i]>0.5 and label[i] == 1):
				corr+=1
			if (pred[i]<0.5 and label[i] == 0):
				corr+=1
		print (f"whitebox attack acc {corr/len(pred)}")
		#AUC
		pred = pred.cpu().numpy()
		label = label.cpu().numpy()
		print (pred.shape,label.shape)
		from sklearn.metrics import roc_auc_score
		auc_score = roc_auc_score(label,pred)
		print (f"white box attack auc score {auc_score}")
		# PLR
		total_neg = 0
		neg_prob = []
		for i in range(len(label)):
			if (label[i] == 0):
				total_neg+=1
				neg_prob.append(pred[i])
		neg_prob = sorted(neg_prob)[::-1]
		fpr_prob_threshold = neg_prob[int(self.fpr_threshold*total_neg)]
		tpr_count = 0
		for i in range(len(label)):
			if (label[i]==1 and pred[i]>=fpr_prob_threshold):
				tpr_count+=1
		print (f"white box attack TPR {tpr_count/total_neg}, FPR {self.fpr_threshold}, PLR {tpr_count/(total_neg*self.fpr_threshold)}")
		
	def run_attack(self,target_model,target_model_optim,target_dataset,training_partition,train_epoch,fpr_threshold):
		
		### train attack model
		self.attack_model = self.attack_model.type(torch.float32).cuda()
		self.attack_model_optim = optim.Adam(self.attack_model.parameters(), lr=0.0005)
		self.target_model = target_model
		self.target_model_optim = target_model_optim
		self.train_epoch = train_epoch
		self.fpr_threshold = fpr_threshold
		
		self.prep_training_data(target_dataset,training_partition)
		self.attack_model_train()
		self.attack_model_eval()
		
		
	
