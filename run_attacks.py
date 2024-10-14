import numpy as np
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import warnings
from data import *
from scipy.stats import norm
warnings.filterwarnings('ignore')
import torchvision.transforms as transforms
import torch
from model import *
import torch.nn as nn
import copy
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import math

def transform_loss_to_prob(x):
	### this can be rewritten as a map function
	### we are using cross entropy loss for all tasks
	new_x = np.exp(-x)
	return new_x


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
	#print (f" fpr {2*10/len(label)}, tpr {2*tpr_count/len(label)}") ## 2 is because the evaluation set is balanced
	return 2*tpr_count/(len(label))


def train_mlp_attack_model(train_data, train_label, test_data, test_label):
	# print (train_data.shape)
	### create data loader
	train = part_pytorch_dataset(train_data, train_label, train=True, transform=None,
	                             target_transform=None)
	test = part_pytorch_dataset(test_data, test_label, train=False, transform=None,
	                            target_transform=None)
	train_loader = torch.utils.data.DataLoader(train, batch_size=50, shuffle=True, num_workers=1)
	test_loader = torch.utils.data.DataLoader(test, batch_size=50, shuffle=False, num_workers=1)
	
	### setup constant
	num_classes = len(np.unique(train_label))
	num_features = train_data.shape[1]
	# print (f" num classes {num_classes}")
	dem0 = 50
	dem1 = 20
	dem2 = dem1 * num_features
	num_epochs = 50
	lr = 0.002
	
	model = simple_mlp_attacknet(train_data.shape[-1], dem0, dem1, dem2, num_classes=num_classes)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss().to(device)
	model.train().to(device)
	
	for iter in range(num_epochs):
		for batch_idx, (images, labels, _) in enumerate(train_loader):
			images, labels = images.to(device), labels.to(device)
			model.zero_grad()
			log_probs = model(images)
			loss = criterion(log_probs, labels)
			loss.backward()
			optimizer.step()
	
	model.eval()
	correct = 0.0
	total = 0.0
	for images, labels, _ in train_loader:
		images = images.to(device)
		outputs = model(images)
		labels = labels.to(device)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	train_acc = acc
	# print ("training accuracy %.2f" % (acc))
	### test the model
	correct = 0.0
	total = 0.0
	for images, labels, _ in test_loader:
		images = images.to(device)
		outputs = model(images)
		labels = labels.to(device)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	test_acc = acc
	# print ("testing accuracy %.2f" % (acc))
	
	return test_acc


def train_cnn_attack_model(train_data, train_label, test_data, test_label):
	### channel first
	# print (train_data.shape)
	if (len(train_data.shape) == 3):
		train_data = np.expand_dims(train_data, axis=1)
		test_data = np.expand_dims(test_data, axis=1)
	
	# print (train_data.shape)
	### create data loader
	train = part_pytorch_dataset(train_data, train_label, train=True, transform=None,
	                             target_transform=None)
	test = part_pytorch_dataset(test_data, test_label, train=False, transform=None,
	                            target_transform=None)
	train_loader = torch.utils.data.DataLoader(train, batch_size=50, shuffle=True, num_workers=1)
	test_loader = torch.utils.data.DataLoader(test, batch_size=50, shuffle=False, num_workers=1)
	
	### setup constant
	num_classes = len(np.unique(train_label))
	num_features = train_data.shape[2]
	lr = 0.001
	num_epochs = 50
	
	if (train_data.shape[-1] < 15):
		# 100 epoch
		#dem0 = (train_data.shape[-1] - 3 * 2) * 15 * num_features
		#kernel_size = 3
		
		# 50 epoch
		dem0 =  (train_data.shape[-1] - 3 * 1) * 15 * num_features
		kernel_size = 2
	
	else:
		dem0 = (train_data.shape[-1] - 15) * 15 * num_features
		kernel_size = 6
	
	model = simple_cnn_attacknet(in_channel=train_data.shape[1], dem0=dem0, num_classes=num_classes,
	                             kernel_size=kernel_size)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss().to(device)
	model.train().to(device)
	
	for iter in range(num_epochs):
		for batch_idx, (images, labels, _) in enumerate(train_loader):
			images, labels = images.to(device), labels.to(device)
			model.zero_grad()
			log_probs = model(images)
			loss = criterion(log_probs, labels)
			loss.backward()
			optimizer.step()
	
	model.eval()
	correct = 0.0
	total = 0.0
	for images, labels, _ in train_loader:
		images = images.to(device)
		outputs = model(images)
		labels = labels.to(device)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	train_acc = acc
	# print ("training accuracy %.2f" % (acc))
	### test the model
	correct = 0.0
	total = 0.0
	for images, labels, _ in test_loader:
		images = images.to(device)
		outputs = model(images)
		labels = labels.to(device)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	test_acc = acc
	# print ("testing accuracy %.2f" % (acc))
	
	return test_acc


def train_rnn_attack_model(train_data, train_label, test_data, test_label):
	# print (train_data.shape)
	
	### create data loader
	train = part_pytorch_dataset(train_data, train_label, train=True, transform=None,
	                             target_transform=None)
	test = part_pytorch_dataset(test_data, test_label, train=False, transform=None,
	                            target_transform=None)
	train_loader = torch.utils.data.DataLoader(train, batch_size=50, shuffle=True, num_workers=1)
	test_loader = torch.utils.data.DataLoader(test, batch_size=50, shuffle=False, num_workers=1)
	
	### setup constant
	num_classes = len(np.unique(train_label))
	num_features = train_data.shape[1]
	data_size = num_features
	hidden_size = 20
	output_size = num_classes
	num_epochs = 300
	lr = 0.001
	batch_size = 50
	
	model = simple_rnn_attacknet(data_size=data_size, hidden_size=hidden_size, output_size=output_size)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.NLLLoss().to(device)
	model.train().to(device)
	
	for iter in range(num_epochs):
		for batch_idx, (images, labels, _) in enumerate(train_loader):
			images, labels = images.to(device), labels.to(device)
			hidden = torch.zeros((batch_size, hidden_size)).to(device)
			
			# print (images.size(),hidden.size())
			
			model.zero_grad()
			for idx in range(images.size()[-1]):
				this_input = images[:, :, idx]
				# print (this_input.size())
				hidden, output = model(this_input, hidden)
			
			loss = criterion(output, labels)
			loss.backward()
			# Add parameters' gradients to their values, multiplied by learning rate
			for p in model.parameters():
				p.data.add_(p.grad.data, alpha=-lr)
	
	model.eval()
	
	### train acc
	correct = 0.0
	total = 0.0
	for images, labels, _ in train_loader:
		images, labels = images.to(device), labels.to(device)
		hidden = torch.zeros((batch_size, hidden_size)).to(device)
		model.zero_grad()
		for idx in range(images.size()[-1]):
			hidden, output = model(images[:, :, idx], hidden)
		
		_, predicted = torch.max(output.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	test_acc = acc
	# print ("training accuracy %.2f" % (acc))
	
	### test the model
	correct = 0.0
	total = 0.0
	for images, labels, _ in test_loader:
		images, labels = images.to(device), labels.to(device)
		hidden = torch.zeros((batch_size, hidden_size)).to(device)
		model.zero_grad()
		for idx in range(images.size()[-1]):
			hidden, output = model(images[:, :, idx], hidden)
		
		_, predicted = torch.max(output.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	test_acc = acc
	print ("RNN testing accuracy %.2f" % (acc))
	return test_acc


def report_acc(data, label, non_member_included=False):
	### available attack model: LR, RF, MLP, CNN and RNN
	
	default_runs = 1
	
	unique_labels = np.unique(label)
	train_index = []
	for this_label in unique_labels:
		this_class_index = np.arange(len(label))[label == this_label]
		this_class_train_index = np.random.choice(this_class_index, int(len(this_class_index) / 2), replace=False)
		train_index.append(this_class_train_index)
	train_index = np.reshape(np.array(train_index), (-1))
	test_index = np.setdiff1d(np.arange(len(label)), train_index)
	
	train_data = data[train_index]
	train_label = label[train_index]
	test_data = data[test_index]
	test_label = label[test_index]
	
	#print ("run attack debug")
	#print (np.bincount(train_label.astype(np.int64)))
	#print (np.bincount(test_label.astype(np.int64)))
	
	if (len(train_data.shape) >= 3):
		# FCNN
		acc3 = train_mlp_attack_model(train_data, train_label, test_data, test_label)
		# CNN
		acc4 = train_cnn_attack_model(train_data, train_label, test_data, test_label)
		# RNN
		# acc5 = train_rnn_attack_model(train_data,train_label,test_data,test_label,non_member_included)
		acc5 = 0
	else:
		acc3, acc4, acc5 = 0, 0, 0
	
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import balanced_accuracy_score
	from sklearn.metrics import roc_auc_score
	from sklearn.metrics import f1_score
	
	train_data = np.reshape(train_data, (len(train_label), -1))
	test_data = np.reshape(test_data, (len(test_label), -1))
	
	# LR
	sum_acc = 0
	for _ in range(default_runs):
		clf = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced')
		clf.fit(train_data, train_label)
		acc1 = balanced_accuracy_score(test_label, clf.predict(test_data)) * 100
		f1_1 = f1_score(test_label, clf.predict(test_data), average='weighted')
		sum_acc += acc1
	acc1 = sum_acc / default_runs
	
	'''
	# RF
	sum_acc = 0
	for _ in range(default_runs):
		from sklearn.ensemble import RandomForestClassifier
		clf = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=0, class_weight="balanced")
		clf.fit(train_data, train_label)
		acc2 = balanced_accuracy_score(test_label, clf.predict(test_data)) * 100
		f1_2 = f1_score(test_label, clf.predict(test_data), average='weighted')
		sum_acc += acc2
	acc2 = sum_acc / default_runs
	
	# MLP
	sum_acc = 0
	for _ in range(default_runs):
		from sklearn.neural_network import MLPClassifier
		clf = MLPClassifier(solver='sgd', alpha=0.01, hidden_layer_sizes=(100, len(unique_labels)), random_state=1)
		clf.fit(train_data, train_label)
		acc6 = balanced_accuracy_score(test_label, clf.predict(test_data)) * 100
		sum_acc += acc6
	acc6 = sum_acc / default_runs
	'''
	acc2 = 0
	acc6 = 0
	#print (acc1, acc2, acc3, acc4, acc5, acc6)
	return max(acc1, acc2, acc3, acc4, acc5, acc6)

def convert_raw_data_to_rank_based_on_validation(data,valid_data):
	rank_results = np.zeros_like(data)
	for i in range(data.shape[1]):
		this_epoch_valid_raw_data_sorted = np.sort(valid_data[:,i])
		for j in range(data.shape[0]):
			## find data[j,i]'s rank in this_epoch_valid_raw_data_sorted
			this_raw_data = data[j,i]
			for this_rank in range(len(this_epoch_valid_raw_data_sorted)):
				if (this_raw_data<this_epoch_valid_raw_data_sorted[this_rank]):
					break
			if (this_raw_data>this_epoch_valid_raw_data_sorted[-1]):
				this_rank+=1
			rank_results[j,i] = this_rank
	return rank_results

def convert_loss_to_logits(data):
	### this function is the same as the one used in the FIRST PRINCIPLE paper
	logits = np.zeros_like(data)
	
	for i in range(len(data)):
		for j in range(len(data[0])):
			logits[i,j] = np.exp(-data[i,j])
			logits[i,j] = np.log(logits[i,j]/(1-logits[i,j]+1e-5)) ### 1e-5 is for numerical stability
			
			if (np.isnan(logits[i,j]) or np.isinf(logits[i,j])):
				print (data[i,j],logits[i,j])
				#exit(0)
				
	#data = np.exp(-data) ### convert loss back to prob
	#data = np.log(data/(1-data)) ### convert prob to logit
	
	return data

def convert_raw_data_to_std_based_on_validation(data,valid_data):
	rank_results = np.zeros_like(data)
	for i in range(data.shape[1]):
		this_epoch_valid_data = valid_data[:,i]
		valid_mean = np.average(this_epoch_valid_data)
		valid_std = np.std(this_epoch_valid_data)
		for j in range(data.shape[0]):
			this_raw_data = data[j,i]
			rank_results[j,i] = (this_raw_data-valid_mean)/valid_std
			
	weights = [ np.std(valid_data[:,i]) for i in range(data.shape[1])]
	weights = np.array(weights)
	weight_sum = np.sum(weights)
	weights = np.array([weight_sum for i in range(data.shape[1])])/weights
	
	return rank_results, weights

def normal_cdf(z):
	return 0.5*(1+math.erf(z/sqrt(2)))

def convert_raw_data_to_log_prob_based_on_validation(data,valid_data):
	rank_results = np.zeros_like(data)
	for i in range(data.shape[1]):
		this_epoch_valid_data = valid_data[:,i]
		valid_mean = np.average(this_epoch_valid_data)
		valid_std = np.std(this_epoch_valid_data)
		for j in range(data.shape[0]):
			this_raw_data = data[j,i]
			#rank_results[j,i] = (this_raw_data-valid_mean)/valid_std
			rank_results[j,i] = np.log10(norm.cdf((this_raw_data-valid_mean)/valid_std))
			#rank_results[j,i] = np.log10(normal_cdf((this_raw_data-valid_mean)/valid_std))
			#if (np.isnan(rank_results[j,i]) or np.isinf(rank_results[j,i])):
				#print (this_raw_data,valid_mean,valid_std)
				#exit(0)
			#print (i,j)
			
	weights = np.ones((data.shape[1]))
	
	print ("CONVERSION DONE")
	
	return rank_results, weights
	

def convert_raw_data_to_rank(data):
	### we assume the data shape is # of instance, # of epochs,  1
	# print (data.shape)
	rank_results = np.zeros_like(data)
	for i in range(data.shape[1]):
		this_epoch_raw_data = data[:,i]
		this_argsort = np.argsort(this_epoch_raw_data)
		cur = 0
		for j in range(len(this_epoch_raw_data)):
			rank_results[this_argsort[j],i] = j
			if (j>=1 and this_epoch_raw_data[this_argsort[j]] == this_epoch_raw_data[this_argsort[j-1]]):
				rank_results[this_argsort[j],i] = cur
			else:
				cur = j
		
	return rank_results


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
			
			#print (data.shape, label.shape)
		
		all_data = np.array(all_data)
		all_label = np.array(all_label).flatten()
		
		#print (all_data.shape, all_label.shape)
		
		return np.squeeze(all_data), np.squeeze(all_label)
	
	
	elif (len(comm_round_list) > 1):
		### be careful, this only works for the one batch attack
		num_user = 10
		num_layer = 12
		feature_num = 7
		data_shape_len = 4
		
		user_member_data = [[] for _ in range(num_user)]
		user_member_label = [[] for _ in range(num_user)]
		user_nonmember_data = [[] for _ in range(num_user)]
		user_nonmember_label = [[] for _ in range(num_user)]
		
		### reshaping the data
		### for each comm round, we have member data 100*num_user, nonmember data 100*num_user
		### new data format: [user1_member_data,...,userN_member_data,user1_nonmember_data,userN_nonmember_data]
		### formula to calculate the location of each chunk:
		### member: user_id*eval_size+comm_round_idx*batch_size
		### nonmember: (num_user+user_id)*eval_size+comm_round_idx*batch_size
		
		for comm_round_idx in comm_round_list:
			data_name = prefix + feature_str + mid_str + str(comm_round_idx) + final_str
			data = np.load(data_name, allow_pickle=True)
			label_name = prefix + label_str + mid_str + str(comm_round_idx) + final_str
			label = np.load(label_name)
			
			# print (label)
			# print (data.shape)
			data = np.squeeze(data)
			data_shape_len = len(data.shape)
			# print (data.shape)
			
			if (len(data.shape) == 5):
				num_user = data.shape[2]
				num_layer = data.shape[3]
				feature_num = data.shape[4]
				data = np.reshape(data, (-1, num_user, num_layer, feature_num))
			elif (len(data.shape) == 3):
				data = np.reshape(data, (-1, num_user, 1))
			else:
				data = np.reshape(data, (-1))
			
			# print (data.shape)
			# print (label.shape)
			
			# for x in np.unique(label):
			#    print (np.where(label == x)[0])
			
			batch_size = int(len(label.flatten()) / (2 * num_user))
			
			for user_idx in range(num_user):
				member_data_start_index = batch_size * user_idx
				nonmember_data_start_index = batch_size * (user_idx + num_user)
				
				user_member_data[user_idx].append(data[member_data_start_index:member_data_start_index + batch_size])
				user_member_label[user_idx].append(label[member_data_start_index:member_data_start_index + batch_size])
				user_nonmember_data[user_idx].append(
					data[nonmember_data_start_index:nonmember_data_start_index + batch_size])
				user_nonmember_label[user_idx].append(
					label[member_data_start_index:member_data_start_index + batch_size])
				
				# print (user_idx,member_data_start_index,nonmember_data_start_index,user_member_label[user_idx][-1],user_nonmember_label[user_idx][-1])
			
			# break
		
		all_data = np.concatenate((np.array(user_member_data), np.array(user_nonmember_data)))
		all_label = np.concatenate((np.array(user_member_label), np.array(user_nonmember_label)))
		
		# print (all_data.shape,data_shape_len)
		
		if (data_shape_len == 5):
			all_data = np.reshape(all_data, (-1, num_user, num_layer, 7))
		elif (data_shape_len == 3):
			all_data = np.reshape(all_data, (-1, num_user, 1))
		elif (data_shape_len == 2):
			all_data = np.reshape(all_data, (-1, 1))
		
		all_label = np.reshape(all_label, (-1))
		
		# print (final_str,batch_size,all_data.shape,all_label.shape)
		
		return np.squeeze(all_data), np.squeeze(all_label)


def avg_roc_cal(metric_list):
	num_user = len(metric_list)
	# print(len(metric_list))
	### 10 users avg. we need to have a uniformed FPR
	uniform_fpr = [1e-5 * i for i in range(1, 10)] + [1e-4 * i for i in range(1, 10)] + [1e-3 * i for i in range(1, 10)]
	uniform_fpr = uniform_fpr + [1e-2 * i for i in range(1, 100)] #+ [1e-1 * i for i in range(1, 10)]
	
	# print (uniform_fpr)
	uniform_tpr = []
	
	for this_fpr in uniform_fpr:
		sum_tpr = 0
		for user_idx in range(num_user):
			this_user_roc_list = np.array(metric_list[user_idx])
			# print (this_user_roc_list.shape)
			idx = np.argmax(this_user_roc_list[0, :] > this_fpr)
			# print (idx)
			if (this_user_roc_list[0, idx] > this_fpr):
				idx -= 1
			this_tpr = this_user_roc_list[1, idx]
			# print (this_fpr,this_user_roc_list[0,idx],this_user_roc_list[1,idx])
			sum_tpr += this_tpr
		uniform_tpr.append(sum_tpr / num_user)
	
	uniform_fpr = np.array(uniform_fpr)
	uniform_tpr = np.array(uniform_tpr)
	return uniform_fpr, uniform_tpr

def read_valid_nonmember_data(comm_round_list,epochs,prefix,mid_str,dataset,model,target_data_size,eval_data_size,f,num_layers,num_user,validation_set_size=1000):
	total_data_num = validation_set_size
	num_valid_user = num_user
	all_epoch_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_layer_cos = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	all_epoch_layer_grad_diff = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	all_epoch_layer_grad_norm = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	
	for epoch_idx, epoch in enumerate(epochs):
		final_str = '_' + str(
			epoch + 1) + '_' + str(dataset) + '_' + str(target_data_size) + '_' + str(eval_data_size) + '_' + str(
			model) + '.npy'
		
		data, label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                       feature_str='valid_all_info_multi_party_member_attack_',
		                                       final_str=final_str, label_str='all_label_multi_party_member_attack_')

		loss_data, loss_label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                                 feature_str='valid_loss_info_multi_party_member_attack_',
		                                                 final_str=final_str,
		                                                 label_str='loss_label_multi_party_member_attack_')

		print (data.shape)
		data = np.reshape(data, (-1, num_valid_user, num_layers, 7)) ### need to change to 4.
		label = np.reshape(label, (-1))
		loss_data = np.reshape(loss_data, (-1, num_valid_user))
		
		all_epoch_layer_cos[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 0])
		all_epoch_layer_grad_diff[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 1] - data[:, :, :, 2])
		all_epoch_loss[:, :, epoch_idx] = copy.deepcopy(loss_data)
		all_epoch_layer_grad_norm[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 3])
	
	all_epoch_loss = np.nan_to_num(all_epoch_loss)
	all_epoch_layer_grad_norm = np.nan_to_num(all_epoch_layer_grad_norm)
	all_epoch_layer_grad_diff = np.nan_to_num(all_epoch_layer_grad_diff)
	all_epoch_layer_cos = np.nan_to_num(all_epoch_layer_cos)
	
	return all_epoch_layer_cos,all_epoch_layer_grad_diff,all_epoch_layer_grad_norm,all_epoch_loss
	

def all_analysis(epochs, prefix, mid_str, dataset, model, target_data_size, eval_data_size, f, special_layers=None,
                 num_layers=12, num_user=5, client_adversary=0, best_layer=0, comm_round_list=[],active_adversary=1,validation_set_size=1000):
	if (client_adversary):
		num_user -= 1
		num_valid_user = 1
	else:
		num_valid_user = num_user
	
	total_data_num = (num_user * 2) * eval_data_size
	#all_epoch_cos = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
	#all_epoch_target_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
	#all_epoch_grad_norm = np.zeros((total_data_num, num_valid_user, len(epochs)))
	#all_epoch_grad_diff = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_layer_cos = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	all_epoch_layer_grad_diff = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	#all_epoch_target_after_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_layer_grad_norm = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	#all_epoch_label = np.zeros(total_data_num)
	
	#this_best_layer = best_layer
	###
	# cifar10, alexnet : best layer 2
	# cifar100, alexnet : best layer 5
	# purchase: best layer 0
	# cifar10, densenet: best layer 30?
	# cifar100, densenet: best layer 30?
	# tinyimagenet, alexnet: best layer ?
	
	#print (f"this best layer:{this_best_layer}")
	
	# result_saving_file = prefix + mid_str + 'result_file.txt'
	# f = open(result_saving_file,"w")
	
	for epoch_idx, epoch in enumerate(epochs):
		#print (f"epoch idx {epoch_idx}")
		final_str = '_' + str(
			epoch + 1) + '_' + str(dataset) + '_' + str(target_data_size) + '_' + str(eval_data_size) + '_' + str(
			model) + '.npy'
		
		data, label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                       feature_str='all_info_multi_party_member_attack_',
		                                       final_str=final_str, label_str='all_label_multi_party_member_attack_')

		
		loss_data, loss_label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                                 feature_str='loss_info_multi_party_member_attack_',
		                                                 final_str=final_str,
		                                                 label_str='loss_label_multi_party_member_attack_')
		
		#target_loss_data, _ = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		#                                               feature_str='target_model_before_loss_info_multi_party_member_attack_',
		#                                               final_str=final_str,
		#                                               label_str='target_model_before_loss_label_multi_party_member_attack_')
		
		#target_after_loss_data, _ = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		#                                                     feature_str='target_model_after_loss_info_multi_party_member_attack_',
		#                                                     final_str=final_str,
		#                                                     label_str='target_model_after_loss_label_multi_party_member_attack_')
		
		
		data = np.reshape(data, (-1, num_valid_user, num_layers, 7))
		label = np.reshape(label, (-1))
		loss_data = np.reshape(loss_data, (-1, num_valid_user))
		loss_label = np.reshape(loss_label, (-1))
		#target_loss_data = np.reshape(target_loss_data, (-1, 1))
		#target_loss_data = np.tile(target_loss_data, (1, num_valid_user))
		#target_after_loss_data = np.reshape(target_after_loss_data, (-1, 1))
		#target_after_loss_data = np.tile(target_after_loss_data, (1, num_valid_user))
		
		# print (data.shape,label.shape,loss_data.shape,loss_label.shape,target_loss_data.shape)
		
		#all_epoch_cos[:, :, epoch_idx] = copy.deepcopy(data[:, :, this_best_layer, 0])
		#all_epoch_grad_diff[:, :, epoch_idx] = copy.deepcopy(data[:, :, this_best_layer, 1] - data[:, :, this_best_layer, 2])
		all_epoch_layer_cos[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 0])
		all_epoch_layer_grad_diff[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 1] - data[:, :, :, 2])
		all_epoch_loss[:, :, epoch_idx] = copy.deepcopy(loss_data)
		#all_epoch_target_loss[:, :, epoch_idx] = copy.deepcopy(target_loss_data)
		#all_epoch_target_after_loss[:, :, epoch_idx] = copy.deepcopy(target_after_loss_data)
		all_epoch_layer_grad_norm[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 3])
		#all_epoch_grad_norm[:, :, epoch_idx] = copy.deepcopy(data[:, :, this_best_layer, 3])
		all_epoch_label = label
	
	all_epoch_loss = np.nan_to_num(all_epoch_loss)
	#all_epoch_grad_diff = np.nan_to_num(all_epoch_grad_diff)
	#all_epoch_cos = np.nan_to_num(all_epoch_cos)
	#all_epoch_target_loss = np.nan_to_num(all_epoch_target_loss)
	#all_epoch_target_after_loss = np.nan_to_num(all_epoch_target_after_loss)
	all_epoch_layer_grad_norm = np.nan_to_num(all_epoch_layer_grad_norm)
	all_epoch_layer_grad_diff = np.nan_to_num(all_epoch_layer_grad_diff)
	all_epoch_layer_cos = np.nan_to_num(all_epoch_layer_cos)
	#all_epoch_grad_norm = np.nan_to_num(all_epoch_grad_norm)
	
	if (not active_adversary):
		#valid_all_epoch_cos,valid_all_epoch_grad_diff,valid_all_epoch_grad_norm,valid_all_epoch_loss = read_valid_nonmember_data(comm_round_list,epochs,
		#                                                                        prefix,mid_str,dataset,model,target_data_size,eval_data_size,f,num_layers,num_valid_user,validation_set_size=validation_set_size)
	
		for best_layer in range(num_layers):
			print (f"best layer {best_layer}")
			#print ("validation baesd rank")
			#all_epoch_analysis_all_adversary_auc_tpr_valid_based_rank(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
	         #                                    label, best_layer,eval_data_size, num_user, f, valid_all_epoch_cos,valid_all_epoch_grad_diff,valid_all_epoch_grad_norm,
	          #                                                   valid_all_epoch_loss)
			print ("set based rank")
			all_epoch_analysis_all_adversary_auc_tpr(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
			                                         label, best_layer, eval_data_size, num_user, f)
			
	else:
		for best_layer in range(num_layers):
			print(f"best layer {best_layer}")
			print("set based rank")
			all_epoch_analysis_all_adversary_auc_tpr(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
			                                         label, best_layer, eval_data_size, num_user, f)
	
	#if (client_adversary):
	#    return all_epoch_analysis_client_adversary(all_epoch_cos, all_epoch_grad_diff, all_epoch_loss,
	#                                                all_epoch_grad_norm, all_epoch_target_loss,
	#                                               all_epoch_target_after_loss, all_epoch_label,
	#                                               eval_data_size=eval_data_size, num_user=num_valid_user, f=f)
	#else:
	#    return all_epoch_analysis_server_adversary(all_epoch_cos, all_epoch_grad_diff, all_epoch_loss,
	#                                               all_epoch_grad_norm, all_epoch_target_loss,
	#                                               all_epoch_target_after_loss, all_epoch_label,
	#                                               eval_data_size=eval_data_size, num_user=num_valid_user, f=f)
	# per_layer_analysis(all_epoch_cos, all_epoch_grad_diff, all_epoch_loss, all_epoch_target_loss, all_epoch_label,
	#                   all_epoch_layer_cos, all_epoch_layer_grad_diff, member_index=num_user * eval_data_size)
	# loss_analysis(all_epoch_cos,all_epoch_grad_diff,all_epoch_loss,all_epoch_target_loss,all_epoch_target_after_loss,all_epoch_label,member_index = num_user*eval_data_size)

def all_analysis_layerwise(epochs, prefix, mid_str, dataset, model, target_data_size, eval_data_size, f, special_layers=None,
                 num_layers=12, num_user=5, client_adversary=0, best_layer=0, comm_round_list=[], active_adversary=1):
	if (client_adversary):
		num_user -= 1
		num_valid_user = 1
	else:
		num_valid_user = num_user
	
	if (not active_adversary):
		#alid_all_epoch_cos, valid_all_epoch_grad_diff, valid_all_epoch_grad_norm, valid_all_epoch_loss = read_valid_nonmember_data(comm_round_list, epochs,
		#                                                                                                                            prefix, mid_str, dataset, model,
		#                                                                                                                            target_data_size, eval_data_size, f, num_layers,
		#                                                                                                                            num_user)
		
		for best_layer in range(num_layers):
			total_data_num = (num_user * 2) * eval_data_size
			all_epoch_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
			all_epoch_layer_cos = np.zeros((total_data_num, num_valid_user, len(epochs),1))
			all_epoch_layer_grad_diff = np.zeros((total_data_num, num_valid_user, len(epochs),1))
			all_epoch_layer_grad_norm = np.zeros((total_data_num, num_valid_user, len(epochs),1))
			#best_layer = 66
			for epoch_idx, epoch in enumerate(epochs):
				#print(f"epoch idx {epoch_idx}")
				final_str = '_' + str(
					epoch + 1) + '_' + str(dataset) + '_' + str(target_data_size) + '_' + str(eval_data_size) + '_' + str(
					model) + '.npy'
		
				data, label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                       feature_str='all_info_multi_party_member_attack_',
		                                       final_str=final_str, label_str='all_label_multi_party_member_attack_')
		
				loss_data, loss_label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                                 feature_str='loss_info_multi_party_member_attack_',
		                                                 final_str=final_str,
		                                                 label_str='loss_label_multi_party_member_attack_')
		

				data = np.reshape(data, (-1, num_valid_user, num_layers, 7))
				label = np.reshape(label, (-1))
				loss_data = np.reshape(loss_data, (-1, num_valid_user))
				loss_label = np.reshape(loss_label, (-1))
				
				all_epoch_layer_cos[:, :, epoch_idx,0] = copy.deepcopy(data[:, :, best_layer, 0])
				all_epoch_layer_grad_diff[:, :, epoch_idx,0] = copy.deepcopy(data[:, :, best_layer, 1] - data[:, :, best_layer, 2])
				all_epoch_loss[:, :, epoch_idx] = copy.deepcopy(loss_data)
				all_epoch_layer_grad_norm[:, :, epoch_idx,0] = copy.deepcopy(data[:, :, best_layer, 3])
	
			all_epoch_loss = np.nan_to_num(all_epoch_loss)
			all_epoch_layer_grad_norm = np.nan_to_num(all_epoch_layer_grad_norm)
			all_epoch_layer_grad_diff = np.nan_to_num(all_epoch_layer_grad_diff)
			all_epoch_layer_cos = np.nan_to_num(all_epoch_layer_cos)
		
			print(f"best layer {best_layer}")
			#print("validation baesd rank")
			#all_epoch_analysis_all_adversary_auc_tpr_valid_based_rank(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
	       #                                                   label, 0, eval_data_size, num_user, f, valid_all_epoch_cos, valid_all_epoch_grad_diff,
	       #                                                   valid_all_epoch_grad_norm,
	        #                                                  valid_all_epoch_loss)
			print("set based rank")
			all_epoch_analysis_all_adversary_auc_tpr(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
	                                         label, 0, eval_data_size, num_user, f)
			
			#break
	


def all_epoch_analysis_all_adversary_auc_tpr(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
                                              label, best_layer, eval_data_size, num_user, f):
	auc_list = []
	acc_list = []
	tpr_list = []
	all_user_roc_list = []
	num_instance = all_epoch_layer_cos.shape[0]
	num_user = all_epoch_layer_cos.shape[1]
	num_layer = all_epoch_layer_cos.shape[3]
	num_epochs = all_epoch_layer_cos.shape[2]
	
	'''
	### specific for densnet-cifar100
	
	all_member_cos = []
	all_nonmember_cos = []

	for this_user_idx in range(num_user):

		this_user_member_index = np.arange(num_instance)[this_user_idx * eval_data_size:(this_user_idx + 1) * eval_data_size]
		this_user_nonmember_index = np.arange(num_instance)[(this_user_idx + num_user) * eval_data_size:(this_user_idx + 1 + num_user) * eval_data_size]
		this_user_index = np.concatenate((this_user_member_index, this_user_nonmember_index))

		#this_user_layer_cos = all_epoch_layer_grad_diff[this_user_index,this_user_idx,:,0]
		this_user_layer_cos = all_epoch_layer_cos[this_user_index,this_user_idx,:,0]

		this_user_member_cos = this_user_layer_cos[:int(len(this_user_layer_cos) / 2)]
		this_user_nonmember_cos = this_user_layer_cos[int(len(this_user_layer_cos) / 2):]

		all_member_cos.append(this_user_member_cos)
		all_nonmember_cos.append(this_user_nonmember_cos)

	all_member_cos = np.concatenate(all_member_cos)
	all_nonmember_cos = np.concatenate(all_nonmember_cos)
	print (all_member_cos.shape)
	epoch_avg_std = []
	for i in range(num_epochs):
		member_avg = np.average(all_member_cos[:, i])
		member_std = np.std(all_member_cos[:,i])
		nonmember_avg = np.average(all_nonmember_cos[:, i])
		nonmember_std = np.std(all_nonmember_cos[:,i])
		true_label = np.concatenate((np.ones((len(all_member_cos))),np.zeros((len(all_nonmember_cos)))))
		this_epoch_cos = np.concatenate((all_member_cos[:,i],all_nonmember_cos[:,i]))
		auc = roc_auc_score(true_label,this_epoch_cos)
		epoch_avg_std.append((member_avg, member_std, nonmember_avg, nonmember_std, auc))
	epoch_avg_std = np.array(epoch_avg_std)
	#np.save('/home/lijiacheng/whiteboxmi/densenet_passive_grad_diff_avg_std.npy', epoch_avg_std)
	np.save('/home/lijiacheng/whiteboxmi/densenet_passive_cos_avg_std.npy', epoch_avg_std)
	print("AVG STD SAVE FINISHED", epoch_avg_std.shape)
	exit(0)
	'''
	
	all_user_tpr_list = []
	
	for this_user_idx in range(num_user):
		
		if (num_user != 1):
			this_user_member_index = np.arange(num_instance)[this_user_idx * eval_data_size:(this_user_idx + 1) * eval_data_size]
			this_user_nonmember_index = np.arange(num_instance)[(this_user_idx + num_user) * eval_data_size:(this_user_idx + 1 + num_user) * eval_data_size]
			this_user_index = np.concatenate((this_user_member_index, this_user_nonmember_index))
			true_label = np.concatenate((np.ones((eval_data_size)), np.zeros((eval_data_size))))
		else:
			this_user_member_index = np.arange(int(num_instance / 2))
			this_user_nonmember_index = np.setdiff1d(np.arange(num_instance), this_user_member_index)
			this_user_index = np.concatenate((this_user_member_index, this_user_nonmember_index))
			true_label = np.concatenate((np.ones((len(this_user_member_index))), np.zeros((len(this_user_member_index)))))
		
		this_user_cos = all_epoch_layer_cos[this_user_index, this_user_idx, :, :]
		this_user_grad_diff = all_epoch_layer_grad_diff[this_user_index, this_user_idx, :, :]
		this_user_grad_norm = all_epoch_layer_grad_norm[this_user_index, this_user_idx, :, :]
		this_user_loss = all_epoch_loss[this_user_index, this_user_idx, :]
		#this_user_loss_reduction = all_epoch_target_loss[this_user_index, this_user_idx, :] - all_epoch_loss[this_user_index, this_user_idx, :]
		
		this_user_auc_list = []
		this_user_acc_list = []
		this_user_tpr_list = []
		
		### report layerwise AUC score for each sum
		for this_layer in [best_layer]:
			this_layer_cos = this_user_cos[:, :, this_layer]
			this_layer_grad_diff = this_user_grad_diff[:, :, this_layer]
			this_layer_loss = this_user_loss[:, :]
			# this_layer_grad_norm = np.sum(this_user_grad_norm[:,:,:],axis=-1)
			this_layer_grad_norm = this_user_grad_norm[:, :, this_layer]
			#this_layer_loss_reduction = this_user_loss_reduction[:, :]
			
			### convert raw data to rank
			#this_layer_cos = convert_raw_data_to_rank(this_layer_cos)
			#this_layer_grad_diff = convert_raw_data_to_rank(this_layer_grad_diff)
			#this_layer_loss = convert_raw_data_to_rank(this_layer_loss)
			#this_layer_grad_norm = convert_raw_data_to_rank(this_layer_grad_norm)
			#this_layer_loss_reduction = convert_raw_data_to_rank(this_layer_loss_reduction)
			
			## convert this layer cos to probability for each epoch
			# print (this_layer_cos.shape)
			# converted_this_layer_cos = np.transpose(np.array([convert_metric_to_prob(this_layer_cos[:,i],metric_name='cos') for i in range(num_epochs)]))
			# print (this_layer_cos.shape)
			
			# print (this_layer_cos.shape)
			# converted_this_layer_loss = np.transpose(np.array([convert_metric_to_prob(this_layer_loss[:,i],metric_name='loss') for i in range(num_epochs)]))
			# print (this_layer_cos.shape)
			
			if (this_user_idx < num_user):  # <10
				fpr, tpr, thresholds = metrics.roc_curve(true_label, np.sum(this_layer_cos, axis=1), pos_label=1)
				#fpr, tpr, thresholds = metrics.roc_curve(true_label, np.average(this_layer_cos,weights=cos_weights,axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(np.sum(this_layer_cos, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, np.sum(this_layer_grad_diff, axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(np.sum(this_layer_grad_diff, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * np.sum(this_layer_loss, axis=1), pos_label=1)
				#fpr, tpr, thresholds = metrics.roc_curve(true_label, -1*np.average(this_layer_loss,weights=loss_weights,axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*np.sum(this_layer_loss, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * np.sum(this_layer_grad_norm, axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*np.sum(this_layer_grad_norm, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * this_layer_loss[:, -1], pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*this_layer_loss[:, -1],true_label))
				all_user_tpr_list.append((fpr, tpr))
				
			cos_auc_score = roc_auc_score(true_label, np.sum(this_layer_cos, axis=1))
			grad_diff_auc_score = roc_auc_score(true_label, np.sum(this_layer_grad_diff, axis=1))
			loss_auc_score = roc_auc_score(true_label, -1 * np.sum(this_layer_loss, axis=1))
			grad_norm_auc_score = roc_auc_score(true_label, -1 * np.sum(this_layer_grad_norm, axis=1))
			last_epoch_loss_auc_score = roc_auc_score(true_label, -1 * this_layer_loss[:, -1])
			this_user_auc_list.append((cos_auc_score, grad_diff_auc_score, loss_auc_score, grad_norm_auc_score,last_epoch_loss_auc_score))
			
			cos_acc_score = report_acc(np.sum(this_layer_cos, axis=1), true_label)
			grad_diff_acc_score = report_acc(np.sum(this_layer_grad_diff, axis=1), true_label)
			loss_acc_score = report_acc(-1 * np.sum(this_layer_loss, axis=1), true_label)
			grad_norm_acc_score = report_acc(-1 * np.sum(this_layer_grad_norm, axis=1), true_label)
			last_epoch_loss_acc_score = report_acc(-1 * this_layer_loss[:, -1], true_label)
			this_user_acc_list.append((cos_acc_score, grad_diff_acc_score, loss_acc_score, grad_norm_acc_score,last_epoch_loss_acc_score))
		
		auc_list.append(this_user_auc_list)
		acc_list.append(this_user_acc_list)
		tpr_list.append(this_user_tpr_list)
	
	auc_list = np.array(auc_list)
	acc_list = np.array(acc_list)
	tpr_list = np.array(tpr_list)
	
	#all_user_tpr_list = np.array(all_user_tpr_list)
	
	#print("auc list:",auc_list)
	#print ("tpr list:",tpr_list)
	#print (tpr_list[:,-1])
	
	avg_auc = np.squeeze(np.average(auc_list, axis=0))
	avg_acc = np.squeeze(np.average(acc_list, axis=0))
	avg_tpr = np.squeeze(np.average(tpr_list, axis=0))
	
	std_auc = np.squeeze(np.std(auc_list,axis=0))
	std_tpr = np.squeeze(np.std(tpr_list, axis=0))
	
	print (f"best layer {best_layer}")
	
	print (f"avg over 10 users: cos AUC {avg_auc[0]:.4f}, grad diff AUC {avg_auc[1]:.4f}, loss AUC {avg_auc[2]:.4f}, grad norm AUC {avg_auc[3]:.4f}, last epoch loss AUC {avg_auc[4]:.4f}")
	
	print(
		f"std over 10 users: cos AUC {std_auc[0]:.4f}, grad diff AUC {std_auc[1]:.4f}, loss AUC {std_auc[2]:.4f}, grad norm AUC {std_auc[3]:.4f}, last epoch loss AUC {std_auc[4]:.4f}")
	
	#print (f"avg over 10 users: cos Acc {avg_acc[0]:.4f},  grad diff Acc {avg_acc[1]:.4f},  loss Acc {avg_acc[2]:.4f}, grad norm Acc {avg_acc[3]:.4f}, last epoch loss Acc {avg_acc[4]:.4f} ")
	
	print (f"avg over 10 users at fpr =1e-3 : cos tpr {avg_tpr[0]:.4f},  grad diff tpr {avg_tpr[1]:.4f},  loss tpr {avg_tpr[2]:.4f}, grad norm tpr {avg_tpr[3]:.4f}, last epoch loss tpr {avg_tpr[4]:.4f}")
	
	print(
		f"std over 10 users at fpr =1e-3 : cos tpr {std_tpr[0]:.4f},  grad diff tpr {std_tpr[1]:.4f},  loss tpr {std_tpr[2]:.4f}, grad norm tpr {std_tpr[3]:.4f}, last epoch loss tpr {std_tpr[4]:.4f}")
	
	return_tpr = []
	for i in range(5):
		fpr, tpr = avg_roc_cal([all_user_roc_list[j * 5 + i] for j in range(num_user)])
		return_tpr.append((fpr,tpr))
		#if (i==0):
		#	for j in range(len(tpr)):
		#		print(fpr[j],tpr[j])
	return_tpr = np.array(return_tpr)
	
	return avg_auc, avg_tpr,return_tpr


def all_epoch_analysis_all_adversary_auc_tpr_valid_based_rank(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
                                             label, best_layer, eval_data_size, num_user, f, valid_all_epoch_cos,valid_all_epoch_grad_diff,valid_all_epoch_grad_norm,
	                                                             valid_all_epoch_loss):
	auc_list = []
	acc_list = []
	tpr_list = []
	all_user_roc_list = []
	num_instance = all_epoch_layer_cos.shape[0]
	num_user = all_epoch_layer_cos.shape[1]
	num_layer = all_epoch_layer_cos.shape[3]
	num_epochs = all_epoch_layer_cos.shape[2]
	
	all_user_tpr_list = []
	
	for this_user_idx in range(num_user):
		
		if (num_user != 1):
			this_user_member_index = np.arange(num_instance)[this_user_idx * eval_data_size:(this_user_idx + 1) * eval_data_size]
			this_user_nonmember_index = np.arange(num_instance)[(this_user_idx + num_user) * eval_data_size:(this_user_idx + 1 + num_user) * eval_data_size]
			this_user_index = np.concatenate((this_user_member_index, this_user_nonmember_index))
			true_label = np.concatenate((np.ones((eval_data_size)), np.zeros((eval_data_size))))
		else:
			this_user_member_index = np.arange(int(num_instance / 2))
			this_user_nonmember_index = np.setdiff1d(np.arange(num_instance), this_user_member_index)
			this_user_index = np.concatenate((this_user_member_index, this_user_nonmember_index))
			true_label = np.concatenate((np.ones((len(this_user_member_index))), np.zeros((len(this_user_member_index)))))
		
		this_user_cos = all_epoch_layer_cos[this_user_index, this_user_idx, :, :]
		this_user_grad_diff = all_epoch_layer_grad_diff[this_user_index, this_user_idx, :, :]
		this_user_grad_norm = all_epoch_layer_grad_norm[this_user_index, this_user_idx, :, :]
		this_user_loss = all_epoch_loss[this_user_index, this_user_idx, :]
		
		this_user_auc_list = []
		this_user_acc_list = []
		this_user_tpr_list = []
		
		### report layerwise AUC score for each sum
		for this_layer in [best_layer]:
			this_layer_cos = this_user_cos[:, :, this_layer]
			this_layer_grad_diff = this_user_grad_diff[:, :, this_layer]
			this_layer_loss = this_user_loss[:, :]
			this_layer_grad_norm = this_user_grad_norm[:, :, this_layer]
			valid_layer_cos = valid_all_epoch_cos[:,this_user_idx,:,this_layer]
			valid_layer_grad_diff = valid_all_epoch_grad_diff[:,this_user_idx,:,this_layer]
			valid_layer_loss = valid_all_epoch_loss[:,this_user_idx,:]
			valid_layer_grad_norm = valid_all_epoch_grad_norm[:,this_user_idx,:,this_layer]
			
			### convert raw data to # of std
			#this_layer_cos,cos_weights = convert_raw_data_to_std_based_on_validation(this_layer_cos,valid_layer_cos)
			#this_layer_grad_diff,grad_diff_weights = convert_raw_data_to_std_based_on_validation(this_layer_grad_diff,valid_layer_grad_diff)
			#this_layer_loss = convert_loss_to_logits(this_layer_loss) ### convert loss to logits
			#this_layer_loss,loss_weights = convert_raw_data_to_std_based_on_validation(this_layer_loss,valid_layer_loss)
			#this_layer_grad_norm,grad_norm_weights = convert_raw_data_to_std_based_on_validation(this_layer_grad_norm,valid_layer_grad_norm)
			
			### convert_raw_data_to_log_prob_based_on_validation
			#this_layer_cos,cos_weights = convert_raw_data_to_log_prob_based_on_validation(this_layer_cos,valid_layer_cos)
			#this_layer_grad_diff,grad_diff_weights = convert_raw_data_to_log_prob_based_on_validation(this_layer_grad_diff,valid_layer_grad_diff)
			#this_layer_loss,loss_weights = convert_raw_data_to_log_prob_based_on_validation(this_layer_loss,valid_layer_loss)
			#this_layer_grad_norm,grad_norm_weights = convert_raw_data_to_log_prob_based_on_validation(this_layer_grad_norm,valid_layer_grad_norm)
			
			this_layer_cos = np.nan_to_num(this_layer_cos)
			this_layer_grad_diff = np.nan_to_num(this_layer_grad_diff)
			this_layer_loss = np.nan_to_num(this_layer_loss)
			this_layer_grad_norm = np.nan_to_num(this_layer_grad_norm)
			
			if (this_user_idx < num_user):  # <10
				fpr, tpr, thresholds = metrics.roc_curve(true_label, np.sum(this_layer_cos, axis=1), pos_label=1)
				#fpr, tpr, thresholds = metrics.roc_curve(true_label, np.average(this_layer_cos,weights=cos_weights,axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(np.sum(this_layer_cos, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, np.sum(this_layer_grad_diff, axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(np.sum(this_layer_grad_diff, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * np.sum(this_layer_loss, axis=1), pos_label=1)
				#fpr, tpr, thresholds = metrics.roc_curve(true_label, -1*np.average(this_layer_loss,weights=loss_weights,axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*np.sum(this_layer_loss, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * np.sum(this_layer_grad_norm, axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*np.sum(this_layer_grad_norm, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * this_layer_loss[:, -1], pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*this_layer_loss[:, -1],true_label))
				all_user_tpr_list.append((fpr, tpr))
			
			cos_auc_score = roc_auc_score(true_label, np.sum(this_layer_cos, axis=1))
			grad_diff_auc_score = roc_auc_score(true_label, np.sum(this_layer_grad_diff, axis=1))
			loss_auc_score = roc_auc_score(true_label, -1 * np.sum(this_layer_loss, axis=1))
			grad_norm_auc_score = roc_auc_score(true_label, -1 * np.sum(this_layer_grad_norm, axis=1))
			last_epoch_loss_auc_score = roc_auc_score(true_label, -1 * this_layer_loss[:, -1])
			this_user_auc_list.append((cos_auc_score, grad_diff_auc_score, loss_auc_score, grad_norm_auc_score,last_epoch_loss_auc_score))
			
			cos_acc_score = report_acc(np.sum(this_layer_cos, axis=1), true_label)
			grad_diff_acc_score = report_acc(np.sum(this_layer_grad_diff, axis=1), true_label)
			loss_acc_score = report_acc(-1 * np.sum(this_layer_loss, axis=1), true_label)
			grad_norm_acc_score = report_acc(-1 * np.sum(this_layer_grad_norm, axis=1), true_label)
			last_epoch_loss_acc_score = report_acc(-1 * this_layer_loss[:, -1], true_label)
			this_user_acc_list.append((cos_acc_score, grad_diff_acc_score, loss_acc_score, grad_norm_acc_score,last_epoch_loss_acc_score))
		
		auc_list.append(this_user_auc_list)
		acc_list.append(this_user_acc_list)
		tpr_list.append(this_user_tpr_list)
	
	auc_list = np.array(auc_list)
	acc_list = np.array(acc_list)
	tpr_list = np.array(tpr_list)
	
	# all_user_tpr_list = np.array(all_user_tpr_list)
	
	# print("auc list:",auc_list)
	# print ("tpr list:",tpr_list)
	#print(tpr_list[:, -1])
	
	avg_auc = np.squeeze(np.average(auc_list, axis=0))
	avg_acc = np.squeeze(np.average(acc_list, axis=0))
	avg_tpr = np.squeeze(np.average(tpr_list, axis=0))
	
	print(f"best layer {best_layer}")
	
	print(
		f"avg over 10 users: cos AUC {avg_auc[0]:.4f}, grad diff AUC {avg_auc[1]:.4f}, loss AUC {avg_auc[2]:.4f}, grad norm AUC {avg_auc[3]:.4f}, last epoch loss AUC {avg_auc[4]:.4f}")
	
	print(
		f"avg over 10 users: cos Acc {avg_acc[0]:.4f},  grad diff Acc {avg_acc[1]:.4f},  loss Acc {avg_acc[2]:.4f}, grad norm Acc {avg_acc[3]:.4f}, last epoch loss Acc {avg_acc[4]:.4f} ")
	
	print(
		f"avg over 10 users at fpr =1e-3 : cos tpr {avg_tpr[0]:.4f},  grad diff tpr {avg_tpr[1]:.4f},  loss tpr {avg_tpr[2]:.4f}, grad norm tpr {avg_tpr[3]:.4f}, last epoch loss tpr {avg_tpr[4]:.4f}")
	
	return_tpr = []
	for i in range(5):
		fpr, tpr = avg_roc_cal([all_user_roc_list[j * 5 + i] for j in range(num_user)])
		return_tpr.append((fpr, tpr))
	return_tpr = np.array(return_tpr)
	
	return avg_auc, avg_tpr, return_tpr


def all_epoch_analysis_client_adversary_acc(all_epoch_cosine, all_epoch_grad_diff, all_epoch_loss, all_epoch_grad_norm,
                                            all_epoch_target_loss,
                                            all_epoch_target_after_loss, label, eval_data_size, num_user, f):
	### for all epoch analysis and member-only case, we would like to see the following attacks:
	# 1. sum of loss, argmin
	# 2. sum of cosine, argmax
	# 3. sum of grad-diff, argmax
	# 4. all loss, NN
	# 5. all cosine, NN
	# 6. all grad-diff, NN
	# 7. sum of loss, sum of cosine, sum of grad-diff, sum of grad_norm, NN
	# 8. all loss, all cosine, all grad-diff, all grad_norm, NN
	# 9. sum of sign of loss reduction, argmax
	# 10. sum of loss, sum of cosine, sum of grad-diff, sum of sign of loss reduction, sum of grad_norm, NN
	# 11. sum of grad-norm, argmin
	# 12. all grad norm, NN
	import copy
	num_instance = all_epoch_cosine.shape[0]
	num_epochs = all_epoch_cosine.shape[2]
	
	# print (num_instance,eval_data_size)
	
	##### membership inference / binary case
	##### members come from all parties, so this is a generalized membership inference, not targeting any client
	max_acc = 0
	f.write("Membership binary classification\n")
	member_index = np.arange(int(num_instance / 2))
	nonmember_index = np.arange(int(num_instance / 2)) + int(num_instance / 2)
	combined_index = np.concatenate((member_index, nonmember_index), axis=0)
	binary_label = np.concatenate((np.ones(len(member_index)), np.zeros((len(nonmember_index)))))
	
	# 11
	metric = np.sum(all_epoch_grad_norm, axis=2)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of grad norm acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 1
	metric = np.sum(all_epoch_loss, axis=2)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of loss acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 2
	metric = np.sum(all_epoch_cosine, axis=2)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of cosine acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 3
	metric = np.sum(all_epoch_grad_diff, axis=2)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of grad diff acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 12
	metric = copy.deepcopy(all_epoch_grad_norm)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch grad norm acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 4
	metric = copy.deepcopy(all_epoch_loss)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch loss acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 5
	metric = copy.deepcopy(all_epoch_cosine)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch cosine acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 6
	metric = copy.deepcopy(all_epoch_grad_diff)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch grad diff acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 7
	metric = np.concatenate((np.sum(all_epoch_loss, axis=2),
	                         np.sum(all_epoch_cosine, axis=2),
	                         np.sum(all_epoch_grad_diff, axis=2),
	                         np.sum(all_epoch_grad_norm, axis=2)
	                         ), axis=1)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch all sum acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 8
	metric = np.zeros((num_instance, num_user * 4, num_epochs))
	for i in range(num_user):
		metric[:, 4 * i, :] = all_epoch_loss[:, i, :]
		metric[:, 4 * i + 1, :] = all_epoch_cosine[:, i, :]
		metric[:, 4 * i + 2, :] = all_epoch_grad_diff[:, i, :]
		metric[:, 4 * i + 3, :] = all_epoch_grad_norm[:, i, :]
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch loss+cosine+grad_diff+grad_norm acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)
	
	# 9
	loss_reduction = all_epoch_target_after_loss - all_epoch_loss
	# loss_reduction_sign_metric = np.sum(sign_loss_reduction,axis=1)
	# print (loss_reduction_sign_metric)
	metric = loss_reduction
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of loss reduction acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)
	
	# 10
	metric = np.zeros((num_instance, num_user * 5, num_epochs))
	for i in range(num_user):
		metric[:, 5 * i, :] = all_epoch_loss[:, i, :]
		metric[:, 5 * i + 1, :] = all_epoch_cosine[:, i, :]
		metric[:, 5 * i + 2, :] = all_epoch_grad_diff[:, i, :]
		metric[:, 5 * i + 3, :] = loss_reduction[:, i, :]
		metric[:, 5 * i + 4, :] = all_epoch_grad_norm[:, i, :]
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch loss+cosine+grad_diff+loss_reduction+grad_norm acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)
	
	f.write(f"binary membership inference: MAXIMUM ACC {max_acc:.2f}\n")
	
	f.close()
	
	return max_acc


def all_epoch_analysis_server_adversary_acc(all_epoch_cosine, all_epoch_grad_diff, all_epoch_loss, all_epoch_grad_norm,
                                            all_epoch_target_loss,
                                            all_epoch_target_after_loss, label, eval_data_size, num_user, f):
	### for all epoch analysis and member-only case, we would like to see the following attacks:
	# 1. sum of loss, argmin
	# 2. sum of cosine, argmax
	# 3. sum of grad-diff, argmax
	# 4. all loss, NN
	# 5. all cosine, NN
	# 6. all grad-diff, NN
	# 7. sum of loss, sum of cosine, sum of grad-diff, sum of grad_norm, NN
	# 8. all loss, all cosine, all grad-diff, all grad_norm, NN
	# 9. sum of sign of loss reduction, argmax
	# 10. sum of loss, sum of cosine, sum of grad-diff, sum of sign of loss reduction, sum of grad_norm, NN
	# 11. sum of grad-norm, argmin
	# 12. all grad norm, NN
	
	import copy
	num_instance = all_epoch_cosine.shape[0]
	num_epochs = all_epoch_cosine.shape[2]
	member_index_bound = eval_data_size * num_user
	
	# print (num_instance,eval_data_size)
	
	##### membership inference / binary case
	##### members come from all parties, so this is a generalized membership inference, not targeting any client
	max_acc = 0
	f.write("Membership binary classification\n")
	member_index = np.arange(int(num_instance / 2))
	nonmember_index = np.arange(int(num_instance / 2)) + int(num_instance / 2)
	
	# print (member_index)
	# print (nonmember_index)
	
	combined_index = np.concatenate((member_index, nonmember_index), axis=0)
	binary_label = np.concatenate((np.ones(len(member_index)), np.zeros((len(nonmember_index)))))
	
	# print (binary_label)
	
	# 11
	metric = np.sum(all_epoch_grad_norm, axis=2)
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of grad norm acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 1
	metric = np.sum(all_epoch_loss, axis=2)
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of loss acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 2
	metric = np.sum(all_epoch_cosine, axis=2)
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of cosine acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 3
	metric = np.sum(all_epoch_grad_diff, axis=2)
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of grad diff acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 12
	metric = copy.deepcopy(all_epoch_grad_norm)
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch grad norm acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 4
	metric = copy.deepcopy(all_epoch_loss)
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch loss acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 5
	metric = copy.deepcopy(all_epoch_cosine)
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch cosine acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 6
	metric = copy.deepcopy(all_epoch_grad_diff)
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch grad diff acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 7
	metric = np.concatenate((np.sum(all_epoch_loss, axis=2),
	                         np.sum(all_epoch_cosine, axis=2),
	                         np.sum(all_epoch_grad_diff, axis=2),
	                         np.sum(all_epoch_grad_norm, axis=2)
	                         ), axis=1)
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch all sum acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)
	
	# 8
	metric = np.zeros((num_instance, num_user * 4, num_epochs))
	for i in range(num_user):
		metric[:, 4 * i, :] = all_epoch_loss[:, i, :]
		metric[:, 4 * i + 1, :] = all_epoch_cosine[:, i, :]
		metric[:, 4 * i + 2, :] = all_epoch_grad_diff[:, i, :]
		metric[:, 4 * i + 3, :] = all_epoch_grad_norm[:, i, :]
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch loss+cosine+grad_diff+grad_norm acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)
	
	# 9'
	loss_reduction = all_epoch_target_after_loss - all_epoch_loss
	# loss_reduction_sign_metric = np.sum(sign_loss_reduction,axis=1)
	# print (loss_reduction_sign_metric)
	metric = loss_reduction
	# print (metric.shape)
	# print ("loss reduction shape",metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of after loss minus per client loss acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)
	
	# 9
	loss_reduction = all_epoch_target_loss - all_epoch_loss
	# loss_reduction_sign_metric = np.sum(sign_loss_reduction,axis=1)
	# print (loss_reduction_sign_metric)
	metric = loss_reduction
	# print (metric.shape)
	# print ("loss reduction shape",metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"sum of loss reduction - before loss minus client loss acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)
	
	# 10
	metric = np.zeros((num_instance, num_user * 5, num_epochs))
	for i in range(num_user):
		metric[:, 5 * i, :] = all_epoch_loss[:, i, :]
		metric[:, 5 * i + 1, :] = all_epoch_cosine[:, i, :]
		metric[:, 5 * i + 2, :] = all_epoch_grad_diff[:, i, :]
		metric[:, 5 * i + 3, :] = loss_reduction[:, i, :]
		metric[:, 5 * i + 4, :] = all_epoch_grad_norm[:, i, :]
	# print (metric.shape)
	this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
	f.write(f"all epoch loss+cosine+grad_diff+loss_reduction+grad_norm acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)
	
	f.write(f"binary membership inference: MAXIMUM ACC {max_acc}\n")
	
	iso_max_acc = max_acc
	
	##### membership inference for each client
	##### membership come from each client
	# sum_acc = 0
	acc_list = []
	# max_acc = 0
	attack_name = ['sum of grad norm', 'sum of loss', 'sum of cosine', 'sum of grad diff', \
	               'all epoch grad norm', 'all epoch loss', 'all epoch cosine', 'all epoch grad diff', \
	               'all epoch all sum', 'all epoch loss+cosine+grad_diff+grad_norm', 'all epoch loss reduction', \
	               'all epoch loss+cosine+grad_diff+loss_reduction+grad_norm', 'all epoch loss reduction+cosine']
	# print (attack_name)
	for this_idx in range(num_user):
		this_acc_list = []
		f.write(f"per client MI, client idx {this_idx}\n")
		max_acc = 0
		f.write("Membership binary classification\n")
		## we need to create a balanced test set, so
		nonmember_index = np.arange(num_instance)[
		                  (this_idx + num_user) * eval_data_size:(this_idx + 1 + num_user) * eval_data_size]
		member_index = np.arange(num_instance)[this_idx * eval_data_size:(this_idx + 1) * eval_data_size]
		combined_index = np.concatenate((member_index, nonmember_index), axis=0)
		binary_label = np.concatenate((np.ones(len(member_index)), np.zeros((len(nonmember_index)))))
		
		# print (this_idx,member_index,nonmember_index,num_instance,eval_data_size,num_user,binary_label)
		
		# 11
		metric = np.sum(all_epoch_grad_norm, axis=2)
		metric = metric[:, this_idx]
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"sum of grad norm acc {this_acc:.2f}\n")
		max_acc = max(this_acc, max_acc)
		this_acc_list.append(this_acc)
		
		# 1
		metric = np.sum(all_epoch_loss, axis=2)
		metric = metric[:, this_idx]
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"sum of loss acc {this_acc:.2f}\n")
		max_acc = max(this_acc, max_acc)
		this_acc_list.append(this_acc)
		
		# 2
		metric = np.sum(all_epoch_cosine, axis=2)
		metric = metric[:, this_idx]
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"sum of cosine acc {this_acc:.2f}\n")
		max_acc = max(this_acc, max_acc)
		this_acc_list.append(this_acc)
		
		# 3
		metric = np.sum(all_epoch_grad_diff, axis=2)
		metric = metric[:, this_idx]
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"sum of grad diff acc {this_acc:.2f}\n")
		max_acc = max(this_acc, max_acc)
		this_acc_list.append(this_acc)
		
		# 12
		metric = copy.deepcopy(all_epoch_grad_norm)
		metric = metric[:, this_idx]
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"all epoch grad_norm acc {this_acc:.2f}\n")
		max_acc = max(this_acc, max_acc)
		this_acc_list.append(this_acc)
		
		# 4
		# metric = copy.deepcopy(all_epoch_loss)
		# metric = metric[:,this_idx]
		# this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"all epoch loss acc {this_acc:.2f}\n")
		# max_acc = max(this_acc, max_acc)
		# this_acc_list.append(this_acc)
		
		# 4'
		metric = copy.deepcopy(all_epoch_loss)
		metric = metric[:, this_idx]
		### transform loss to prob
		new_metric = transform_loss_to_prob(metric)
		# print (new_metric.shape)
		this_acc = report_acc(new_metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"all epoch loss acc {this_acc:.2f}\n")
		max_acc = max(this_acc, max_acc)
		this_acc_list.append(this_acc)
		
		# 5
		metric = copy.deepcopy(all_epoch_cosine)
		metric = metric[:, this_idx]
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"all epoch cosine acc {this_acc:.2f}\n")
		max_acc = max(this_acc, max_acc)
		this_acc_list.append(this_acc)
		
		# 6
		metric = copy.deepcopy(all_epoch_grad_diff)
		metric = metric[:, this_idx]
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"all epoch grad diff acc {this_acc:.2f}\n")
		max_acc = max(this_acc, max_acc)
		this_acc_list.append(this_acc)
		
		# 7
		metric = np.concatenate((np.sum(all_epoch_loss, axis=2),
		                         np.sum(all_epoch_cosine, axis=2),
		                         np.sum(all_epoch_grad_diff, axis=2),
		                         np.sum(all_epoch_grad_norm, axis=2)
		                         ), axis=1)
		# print (metric.shape)
		new_metric = np.concatenate((np.expand_dims(metric[:, this_idx], axis=-1), \
		                             np.expand_dims(metric[:, num_user + this_idx], axis=-1), \
		                             np.expand_dims(metric[:, num_user * 2 + this_idx], axis=-1), \
		                             np.expand_dims(metric[:, num_user * 3 + this_idx], axis=-1)
		                             ), axis=1)
		# print (new_metric.shape)
		this_acc = report_acc(new_metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"all epoch all sum acc {this_acc:.2f}\n")
		max_acc = max(this_acc, max_acc)
		this_acc_list.append(this_acc)
		
		# 8
		metric = np.zeros((num_instance, num_user * 4, num_epochs))
		for i in range(num_user):
			metric[:, 4 * i, :] = all_epoch_loss[:, i, :]
			metric[:, 4 * i + 1, :] = all_epoch_cosine[:, i, :]
			metric[:, 4 * i + 2, :] = all_epoch_grad_diff[:, i, :]
			metric[:, 4 * i + 3, :] = all_epoch_grad_norm[:, i, :]
		metric = metric[:, 4 * this_idx:4 * (this_idx + 1), :]
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"all epoch loss+cosine+grad_diff+grad_norm acc {this_acc:.2f}\n")
		max_acc = max(max_acc, this_acc)
		this_acc_list.append(this_acc)
		
		# 9'
		loss_reduction = all_epoch_target_after_loss - all_epoch_loss
		# loss_reduction_sign_metric = np.sum(sign_loss_reduction,axis=1)
		# print (loss_reduction_sign_metric)
		metric = loss_reduction[:, this_idx, :]
		# print ("loss reduction shape", metric.shape)
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		f.write(f"sum of after loss minus per client loss acc {this_acc:.2f}\n")
		# max_acc = max(max_acc, this_acc)
		
		# 9
		loss_reduction = all_epoch_target_loss - all_epoch_loss
		# loss_reduction_sign_metric = np.sum(sign_loss_reduction,axis=1)
		# print (loss_reduction_sign_metric)
		metric = loss_reduction[:, this_idx, :]
		# print ("loss reduction shape", metric.shape)
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		# f.write(f"sum of loss reduction - before loss minus client loss acc {this_acc:.2f}\n")
		max_acc = max(max_acc, this_acc)
		this_acc_list.append(this_acc)
		
		# 10
		metric = np.zeros((num_instance, num_user * 5, num_epochs))
		for i in range(num_user):
			metric[:, 5 * i, :] = all_epoch_loss[:, i, :]
			metric[:, 5 * i + 1, :] = all_epoch_cosine[:, i, :]
			metric[:, 5 * i + 2, :] = all_epoch_grad_diff[:, i, :]
			metric[:, 5 * i + 3, :] = loss_reduction[:, i, :]
			metric[:, 5 * i + 4, :] = all_epoch_grad_norm[:, i, :]
		metric = metric[:, 5 * this_idx:5 * (this_idx + 1), :]
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		this_acc_list.append(this_acc)
		# f.write(f"all epoch loss+cosine+grad_diff+loss_reduction+grad_norm acc {this_acc:.2f}\n")
		max_acc = max(max_acc, this_acc)
		print(f"binary membership inference for client {this_idx}: ensemble ACC {this_acc}\n")
		
		return_val = this_acc
		
		# 10
		metric = np.zeros((num_instance, num_user * 2, num_epochs))
		for i in range(num_user):
			metric[:, 2 * i, :] = all_epoch_cosine[:, i, :]
			metric[:, 2 * i + 1, :] = loss_reduction[:, i, :]
		metric = metric[:, 2 * this_idx:2 * (this_idx + 1), :]
		# print (metric.shape)
		this_acc = report_acc(metric[combined_index], binary_label, non_member_included=True)
		this_acc_list.append(this_acc)
		max_acc = max(max_acc, this_acc)
		# f.write(f"binary membership inference for client {this_idx}: MAXIMUM ACC {max_acc}\n")
		acc_list.append(this_acc_list)
	
	acc_list = np.array(acc_list)
	print (acc_list.shape)
	acc_list = np.average(acc_list, axis=0)
	print (acc_list.shape)
	
	print (len(acc_list), len(attack_name))
	
	f.write(f"averaged binary membership inference for all clients:\n")
	
	for this_acc, this_attack_name in zip(acc_list, attack_name):
		f.write(f"{this_attack_name} has average accuracy {this_acc}\n")
		
		if (num_user == 1):
			# print ("single user case return")
			# print (iso_max_acc,sum_acc)
			f.close()
			return return_val
	
	print (f"return val {return_val}")
	
	max_acc = 0
	'''
	f.write("member only case\n")

	# 11
	metric = np.sum(all_epoch_grad_norm, axis=2)
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"sum of grad norm acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 1
	metric = np.sum(all_epoch_loss, axis=2)
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"sum of loss acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 2
	metric = np.sum(all_epoch_cosine, axis=2)
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"sum of cosine acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 3
	metric = np.sum(all_epoch_grad_diff, axis=2)
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"sum of grad diff acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 12
	metric = copy.deepcopy(all_epoch_grad_norm)
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"all epoch grad norm acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 4
	metric = copy.deepcopy(all_epoch_loss)
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"all epoch loss acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 5
	metric = copy.deepcopy(all_epoch_cosine)
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"all epoch cosine acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 6
	metric = copy.deepcopy(all_epoch_grad_diff)
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"all epoch grad diff acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 7
	metric = np.concatenate((np.sum(all_epoch_loss, axis=2),
							 np.sum(all_epoch_cosine, axis=2),
							 np.sum(all_epoch_grad_diff, axis=2),
							 np.sum(all_epoch_grad_norm,axis=2)
							 ), axis=1)
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"all epoch all sum acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 8
	metric = np.zeros((num_instance,num_user*4,num_epochs))
	for i in range(num_user):
		metric[:,4*i,:] = all_epoch_loss[:,i,:]
		metric[:,4*i+1,:] = all_epoch_cosine[:,i,:]
		metric[:,4*i+2,:] = all_epoch_grad_diff[:,i,:]
		metric[:,4*i+3,:] = all_epoch_grad_norm[:,i,:]
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"all epoch loss+cosine+grad_diff+grad_norm acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 9
	loss_reduction = all_epoch_target_after_loss - all_epoch_loss
	# loss_reduction_sign_metric = np.sum(sign_loss_reduction,axis=1)
	# print (loss_reduction_sign_metric)
	metric = loss_reduction
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"sum of loss reduction acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 10
	metric = np.zeros((num_instance,num_user*5,num_epochs))
	for i in range(num_user):
		metric[:,5*i,:] = all_epoch_loss[:,i,:]
		metric[:,5*i+1,:] = all_epoch_cosine[:,i,:]
		metric[:,5*i+2,:] = all_epoch_grad_diff[:,i,:]
		metric[:,5*i+3,:] = loss_reduction[:,i,:]
		metric[:,5*i+4,:] = all_epoch_grad_norm[:,i,:]
	this_acc = report_acc(metric[:member_index_bound], label[:member_index_bound])
	f.write(f"all epoch loss+cosine+grad_diff+loss_reduction+grad_norm acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	f.write (f"member only case: MAXIMUM ACC {max_acc}\n")

	max_acc = 0
	### for non-member including case, we need to consider the following attacks:
	# 1. sum loss
	# 2. all loss
	# 3. sum cos
	# 4. all cos
	# 5. sum grad diff
	# 6. all grad diff
	# 7. all feature
	# 8. sum grad_norm
	# 9. all grad_norm
	f.write("testing instances included\n")

	# 8
	metric = np.sum(all_epoch_grad_norm, axis=2)
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"sum of grad_norm acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)

	# 1
	metric = np.sum(all_epoch_loss, axis=2)
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"sum of loss acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)

	# 2
	metric = np.sum(all_epoch_cosine, axis=2)
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"sum of cosine acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)

	# 3
	metric = np.sum(all_epoch_grad_diff, axis=2)
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"sum of grad diff acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)

	# 9
	metric = copy.deepcopy(all_epoch_grad_norm)
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"all epoch grad norm acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)

	# 4
	metric = copy.deepcopy(all_epoch_loss)
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"all epoch loss acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)

	# 5
	metric = copy.deepcopy(all_epoch_cosine)
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"all epoch cosine acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)

	# 6
	metric = copy.deepcopy(all_epoch_grad_diff)
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"all epoch grad diff acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)

	# 7
	metric = np.concatenate((np.sum(all_epoch_loss, axis=2),
							 np.sum(all_epoch_cosine, axis=2),
							 np.sum(all_epoch_grad_diff, axis=2),
							 np.sum(all_epoch_grad_norm,axis=2)
							 ), axis=1)
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"all epoch all sum acc {this_acc:.2f}\n")
	max_acc = max(this_acc, max_acc)


	# 8
	metric = np.zeros((num_instance,num_user*4,num_epochs))
	for i in range(num_user):
		metric[:,4*i,:] = all_epoch_loss[:,i,:]
		metric[:,4*i+1,:] = all_epoch_cosine[:,i,:]
		metric[:,4*i+2,:] = all_epoch_grad_diff[:,i,:]
		metric[:,4*i+3,:] = all_epoch_grad_norm[:,i,:]
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write (f"all epoch loss+cosine+grad_diff+grad_norm acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 9
	loss_reduction = all_epoch_target_after_loss - all_epoch_loss
	# loss_reduction_sign_metric = np.sum(sign_loss_reduction,axis=1)
	# print (loss_reduction_sign_metric)
	metric = loss_reduction
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"sum of loss reduction acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	# 10
	metric = np.zeros((num_instance,num_user*5,num_epochs))
	for i in range(num_user):
		metric[:,5*i,:] = all_epoch_loss[:,i,:]
		metric[:,5*i+1,:] = all_epoch_cosine[:,i,:]
		metric[:,5*i+2,:] = all_epoch_grad_diff[:,i,:]
		metric[:,5*i+3,:] = loss_reduction[:,i,:]
		metric[:,5*i+4,:] = all_epoch_grad_norm[:,i,:]
	this_acc = report_acc(metric[:(member_index_bound+eval_data_size)], label[:(member_index_bound+eval_data_size)], non_member_included=True)
	f.write(f"all epoch loss+cosine+grad_diff+loss_reduction+grad_norm acc {this_acc:.2f}\n")
	max_acc = max(max_acc, this_acc)

	f.write (f"testing instance included: MAXIMUM ACC {max_acc}\n")
	'''
	f.close()
	
	return return_val

