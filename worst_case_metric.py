## we have self influence function
## loss
## KNN in logit space
## privacy risk score --> very expensive and cannot be calculated on the fly
## we propose 3 different logit based metric

#! /usr/bin/env python3
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import numpy as np
import copy
from tqdm import tqdm
from scipy.optimize import fmin_ncg
import torch.nn as nn
import torch.nn.functional as F
import copy

def calculate_std(x):
	print (x.shape)
	return torch.std(x,dim=1)

def find_vulnerable_points(user_list, model, optimizer, mmd_train_loader, mmd_validation_loader,vul_metric,fpr_threshold=0.01,weight=0.1):
	if (vul_metric == 'loss'):
		return one_model_loss(model, mmd_train_loader, mmd_validation_loader,fpr_threshold=fpr_threshold)
	if (vul_metric == 'logit_diff'):
		return all_model_logits_diff(model,mmd_train_loader,mmd_validation_loader,fpr_threshold=fpr_threshold)
	if (vul_metric == 'cross_diff'):
		return all_model_cross_diff(user_list, fpr_threshold=fpr_threshold)
	return

def all_model_cross_diff(user_list,fpr_threshold=0.001):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.MSELoss(reduction='none').to(device)
	all_diff = []
	all_index = []
	for idx in range(len(user_list)):
		optimizer = user_list[idx].optim
		model = user_list[idx].model
		this_user_idx = user_list[idx].train_index
		for _, (images, labels, data_idx) in enumerate(user_list[idx].train_eval_data_loader):
			optimizer.zero_grad()
			images, labels = images.to(device), labels.to(device)
			train_log_probs = model(images)
			other_log_probs = torch.zeros((len(images),len(user_list)-1,len(train_log_probs[-1])))
			cnt = 0
			for other_idx in range(len(user_list)):
				if (other_idx!=idx):
					other_model = user_list[other_idx].model.to(device)
					other_log_probs[:,cnt,:] = other_model(images).detach()
					cnt+=1
			train_log_probs = F.softmax(train_log_probs,dim=1)
			train_log_probs = [train_log_probs[i,labels[i]] for i in range(len(labels))]
			train_log_probs = torch.stack(train_log_probs)
			other_log_probs = F.softmax(other_log_probs,dim=2)
			other_log_probs = torch.mean(other_log_probs,dim=1)
			other_log_probs = [other_log_probs[i,labels[i]] for i in range(len(labels))]
			other_log_probs = torch.stack(other_log_probs).to(device)
			loss = criterion(train_log_probs,other_log_probs).detach().cpu().numpy()
			all_diff.append(loss)
			all_index.append(np.array(this_user_idx[data_idx]))
			
	all_diff = np.array(all_diff).flatten()
	all_index = np.array(all_index).flatten()
	#print (all_index.shape,all_diff.shape)
	
	all_index = all_index[np.argsort(all_diff)[::-1]]
	#print (np.sort(all_diff)[::-1][:100])
	#print (all_index[:100])
	#print (len(all_index))
	threshold = int(fpr_threshold*len(all_diff)*10)
	#print (threshold)
	### should we just use this? like first 1000 instances?? but how can we select?
	### since nonmember cannot provide anything here... so FPR does not work.. or should we just manually select a threshold?

	#if (len(vulnerable) <= index):
	#	return 0 , None
	#else:
		#print(f"we have found {len(vulnerable) - index} vulenrable points")
		#return len(vulnerable) - index , vulnerable[:-index]
	return len(vulnerable) , vulnerable

def all_model_logits_diff(model,train_loader,validation_loader,fpr_threshold=0.001):
	## in this metric, we only calculate the std of logits for each train, then choose the records that have larger std
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(reduction='none').to(device)
	all_train_logits = []
	for images, labels, _ in train_loader:
		model.zero_grad()
		images = images.to(device)
		labels = labels.to(device)
		# print (images.size(),labels.size())
		outputs = model(images).detach()
		# print (loss,loss.size())
		all_train_logits.append(outputs)
	
	# print ("validation")
	all_validation_logits = []
	for images, labels, _ in validation_loader:
		model.zero_grad()
		images = images.to(device)
		labels = labels.to(device)
		# print (images.size(),labels.size())
		outputs = model(images).detach()
		# print (loss.size())
		all_validation_logits.append(outputs)
	
	model.zero_grad()
	# for x in all_train_loss:
	#		print (x.size())
	
	all_validation_logits = torch.cat(all_validation_logits, dim=0)
	all_train_logits = torch.cat(all_train_logits, dim=0)
	# print (all_validation_loss.size(),all_train_loss.size())
	### given a fixed fpr threshold, we just treat all the train loss that is higher than this threshold as vulnerable points

	all_train_logits_std = calculate_std(all_train_logits)
	all_validation_logits_std = calculate_std(all_validation_logits)
	min_len = min(len(all_train_logits_std),len(all_validation_logits_std))
	validation_index = np.random.choice(len(all_validation_logits_std),min_len,replace=False)
	train_index = np.random.choice(len(all_train_logits_std),min_len,replace=False)
	all_validation_logits_std = all_validation_logits_std[validation_index]
	all_train_logits_std = all_train_logits_std[train_index]
	print (len(all_train_logits_std),len(all_validation_logits_std))
	index = int(len(all_validation_logits)*1.0 * fpr_threshold)
	threshold = all_validation_logits_std[index]
	
	vulnerable = []
	for i in range(len(all_train_logits_std)):
		if (all_train_logits_std[i] < threshold):
			vulnerable.append(i)
	vulnerable = np.array(vulnerable)
	print (index,threshold)
	#if (len(vulnerable) <= index):
	#	return 0 , None
	#else:
		#print(f"we have found {len(vulnerable) - index} vulenrable points")
		#return len(vulnerable) - index , vulnerable[:-index]
	return len(vulnerable) , vulnerable
	# calculate the loss distribution for validation set, unlearn those with low prob from validation set
	
def one_model_validation_distribution_prob(model,train_loader,validation_loader,fpr_threshold=0.01):
	return
	
def one_model_validation_avg_kl(model,train_loader,validation_loader,fpr_threshold=0.01):
	return

def one_model_loss(model,train_loader,validation_loader,fpr_threshold=0.001):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(reduction='none').to(device)
	all_train_loss = []
	for images, labels, _ in train_loader:
		model.zero_grad()
		images = images.to(device)
		labels = labels.to(device)
		#print (images.size(),labels.size())
		outputs = model(images)
		loss = criterion(outputs,labels).detach()
		#print (loss,loss.size())
		all_train_loss.append(loss)
	
	#print ("validation")
	all_validation_loss = []
	for images,labels,_ in validation_loader:
		model.zero_grad()
		images = images.to(device)
		labels = labels.to(device)
		#print (images.size(),labels.size())
		outputs = model(images)
		loss = criterion(outputs,labels).detach()
		#print (loss.size())
		all_validation_loss.append(loss)
		
	model.zero_grad()
	#for x in all_train_loss:
	#		print (x.size())

	#print (len(all_train_loss),len(all_validation_loss))
	
	all_validation_loss = torch.cat(all_validation_loss,dim=0).flatten()
	all_train_loss = torch.cat(all_train_loss,dim=0).flatten()
	min_len = min(len(all_validation_loss),len(all_train_loss))
	validation_index = np.random.choice(len(all_validation_loss),min_len,replace=False)
	train_index = np.random.choice(len(all_train_loss),min_len,replace=False)
	all_validation_loss = all_validation_loss[validation_index]
	all_train_loss = all_train_loss[train_index]

	index = int(len(all_train_loss)*1.0 * fpr_threshold)
	threshold = torch.sort(all_validation_loss)[0][index]
	
	#print (len(all_train_loss),len(all_validation_loss))
	
	#print (torch.sort(all_train_loss)[0][:index])
	#print (torch.sort(all_validation_loss)[0][:index])
	#print(len(all_train_loss),fpr_threshold,int(len(all_train_loss)*1.0 * fpr_threshold))
	
	vulnerable = []
	for i in range(len(all_train_loss)):
		if (all_train_loss[i]<threshold):
			vulnerable.append(i)
	vulnerable = np.array(vulnerable)
	print (index,threshold,len(vulnerable))
	#index = int(len(all_validation_logits)*1.0 * fpr_threshold)
	#if (len(vulnerable) <= index):
	#	return 0 , None
	#else:
		#print(f"we have found {len(vulnerable) - index} vulenrable points")
		#return len(vulnerable) - index , vulnerable[:-index]
	return len(vulnerable) , vulnerable
	# calculate the loss distribution for validation set, unlearn those with low prob from validation set

def self_influence_function(model,train_loader,fpr_threshold=0.01):
	
	# use calc_self_influence function
	# this is going to be time consuming, since separate gradients need to be calculated for every instance
	# todo: understand calc_self_influence function and calc_influence_single function
	return
