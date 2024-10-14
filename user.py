import numpy as np

from whitebox_attack import *
from blackbox_attack import *
from data import dataset
from model import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from model_utils import *
from opacus import PrivacyEngine
import gc
from torch.distributions import Categorical


class User:
	def __init__(self, dataset, model_name, id):
		self.train_data = None
		self.train_label = None
		self.test_data = None
		self.test_label = None
		self.dataset = dataset
		self.train_dataset = None
		self.test_dataset = None
		self.train_eval_dataset = None
		self.test_eval_dataset = None
		# self.worker = sy.VirtualWorker(hook,id=id)
		self.train_data_loader = None
		self.test_data_loader = None
		self.train_eval_data_loader = None
		self.test_eval_data_loader = None
		self.train_gradient = None
		self.test_gradient = None
		self.train_grad_mean = None
		self.train_grad_var = None
		self.test_grad_mean = None
		self.test_grad_var = None
		self.train_loss_profile = []
		self.test_loss_profile = []
		self.optim = None
		self.model = None
		self.available_list = None
		self.target_transform = None
		self.train_transform = None
		self.test_transform = None
		self.class_weight = None
		self.scheduler = None
		self.valid_data = None
		self.valid_label = None
		self.validation_data_loader = None
		self.validation_data_set = None
		self.privacy_engine = None
		self.train_index = None
		self.hamp = False
	
	def create_new_train_data_loader(self, batch_size):
		
		if (self.available_list is None):
			self.available_list = np.arange(len(self.train_data))
			
		#print (len(self.train_data),len(self.train_index),len(self.train_label),len(self.available_list))
		if (self.hamp):
			new_train_dataset = part_pytorch_dataset(self.train_data[self.available_list], self.modified_train_label[self.available_list], train=True, transform=self.train_transform,
												 target_transform=self.target_transform,float_target=True)
		else:
			new_train_dataset = part_pytorch_dataset(self.train_data[self.available_list], self.train_label[self.available_list], train=True, transform=self.train_transform,
												 target_transform=self.target_transform)
		
		new_train_data_loader = torch.utils.data.DataLoader(new_train_dataset, batch_size=batch_size,shuffle=True, num_workers=1)
		
		return new_train_data_loader
	
	def update_ban_list(self, ban_list):
		left_index = []
		for i,x in enumerate(self.train_index):
			if (x in ban_list):
				continue
			left_index.append(i)
		# reset self train loader
		#print (f"ban {len(self.train_index) - len(left_index)} data")
		self.train_index = self.train_index[left_index]
		self.train_data = self.train_data[left_index]
		self.train_label = self.train_label[left_index]
		self.available_list = np.arange(len(left_index))
		#print (len(self.train_data))
		#self = self.create_new_train_data_loader(batch_size=100)

	def reset_ban_list(self):
		self.available_list = np.arange(len(self.train_data))
	
	def create_batch_attack_data_loader(self, data_index, batch_size):
		attack_dataset = part_pytorch_dataset(self.train_data[self.available_list[data_index]], self.train_label[self.available_list[data_index]],
											  train=False, transform=self.test_transform, target_transform=self.target_transform)
		attack_loader = torch.utils.data.DataLoader(attack_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
		return attack_loader
	

## utils for users

def std_loss(x):
	avg_std = torch.sum(torch.std(x, dim=1)) / len(x)
	return avg_std

#@profile
def update_weights(current_model_weights, model, optimizer, train_loader, local_epochs, mixup=0, selfswap=0, std_loss_lambda=0.1,class_weights=None,dataset_name=''):
	# Set mode to train model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	class_weights = torch.from_numpy(class_weights)
	#if (dataset_name == 'purchase' or dataset_name == 'texas'):
	# for base case we should comment this out
	#class_weights = class_weights.type(torch.float64)
	
	#class_weights = class_weights.type(torch.float32)
	#criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

	criterion = nn.CrossEntropyLoss().to(device)
	
	model.load_state_dict(current_model_weights)
	model.zero_grad(set_to_none=True)
	model.train().to(device)
	### for each user, we need to create a new dataloader, so we can avoid using instances that are used in previous steps, same epoch.
	### for single worker, it is not possible to play with sampler to achieve the above constraint.
	
	### mixup is added
	### std loss is added
	#all_data_idx = []
	
	for _ in range(local_epochs):
		for batch_idx, (images, labels, data_idx) in enumerate(train_loader):
			optimizer.zero_grad()
			#model.zero_grad()
			if (int(mixup) == 1):
				# print ('mixup is called!')
				inputs, targets_a, targets_b, lam = mixup_data(images, labels, 1)  ## set mixup.alpha = 1
				inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)
				outputs = model(inputs)
				loss_func = mixup_criterion(targets_a, targets_b, lam)
				loss = loss_func(criterion, outputs) + std_loss(outputs) * std_loss_lambda  ### std loss included
				loss.backward()
				optimizer.step()
			else:
				images, labels = images.to(device), labels.to(device)
				if (int(selfswap) == 1): ### self swap does not work...
					images = self_swap(images)
				log_probs = model(images)
				#print (log_probs.dtype,labels.dtype)
				loss = criterion(log_probs, labels)  + std_loss(log_probs) * std_loss_lambda
				loss.backward()
				optimizer.step()
				
			#all_data_idx.append(data_idx.detach().cpu().numpy())
	
	model.zero_grad(set_to_none=True)
	#all_data_idx = np.unique(np.array(all_data_idx).flatten())
	
	return model.state_dict()#,all_data_idx


class cross_diff_loss(torch.nn.Module):
	def __init__(self,type='l1'): ### default should be l1.
		super(cross_diff_loss, self).__init__()
		self.type = type
	def forward(self, outputs, labels):
		if (self.type == 'l1'):
		# L1 loss
			return torch.clip(outputs - labels, min=0).mean()
		if (self.type == 'l2'):
		# L2 loss
			weight = torch.clip(torch.sign(outputs-labels),min=0)
			loss = (outputs-labels)**2
			return (loss*weight).mean()

def update_weights_cross_loss_diff(user_list,loss_lambda=0.1,loss_name='l1'):
	# Set mode to train model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	if (loss_name == 'l1'):
		criterion = nn.L1Loss().to(device)
	elif (loss_name =='l2'):
		criterion = nn.MSELoss().to(device)
	elif (loss_name == 'l1t'):
		criterion = cross_diff_loss(type='l1').to(device)
	elif (loss_name == 'l2t'):
		criterion = cross_diff_loss(type='l2').to(device)
		
	for idx in range(len(user_list)):
	## for each user, we use the train data for this user.
	## for each instance, we look at the training prediction from this user
	## we also look at the averaged testing prediction from all other users
	## then we calculate a loss based on the diff between two predictions, using KL or just the prob of the correct class
		optimizer = user_list[idx].optim
		model = user_list[idx].model
		for _, (images, labels, _) in enumerate(user_list[idx].train_eval_data_loader):
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
					
			### KLDiv implementation
			'''
			kl_loss = nn.KLDivLoss().to(device)
			train_log_probs = F.softmax(train_log_probs,dim=1).to(device)
			other_log_probs = F.softmax(other_log_probs,dim=2)
			other_log_probs = torch.mean(other_log_probs,dim=1).to(device)
			loss = torch.tensor(loss_lambda).to(device) * kl_loss(train_log_probs,other_log_probs)
			loss.backward()
			optimizer.step()
			'''
			## we measure the prob of the correct class loss, l1 loss or l2 loss
			train_log_probs = F.softmax(train_log_probs,dim=1)
			train_log_probs = [train_log_probs[i,labels[i]] for i in range(len(labels))]
			#print (len(train_log_probs),train_log_probs)
			train_log_probs = torch.stack(train_log_probs)
			## now we have the correct prob for the correct labels for training set
			## get avg testing prob
			## shape should be [# of instance, # of testing models, # of classes]
			#print (other_log_probs.shape)
			other_log_probs = F.softmax(other_log_probs,dim=2)
			other_log_probs = torch.mean(other_log_probs,dim=1)
			#print (other_log_probs.shape)
			other_log_probs = [other_log_probs[i,labels[i]] for i in range(len(labels))]
			other_log_probs = torch.stack(other_log_probs).to(device)
			loss = loss_lambda * criterion(train_log_probs,other_log_probs)
			loss.backward()
			optimizer.step()
			
	## should return list of weights for all users' model
	return [user_list[i].model.state_dict() for i in range(len(user_list))]


def update_weights_mmd(model, user_list, optimizer, validation_loader, train_loader_in_order,validation_set, loss_lambda=0.1, starting_index=None,num_classes=100):
	# get training accuracy
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	correct = 0.0
	total = 0.0
	for images, labels, _ in train_loader_in_order:
		images = images.to(device)
		#print (images.size())
		outputs = model(images).detach()
		labels = labels.to(device)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	acc = correct.item()
	acc = acc / total
	this_training_acc = acc * 100.0
	#print ("MMD training accuracy %.2f" % (this_training_acc))
	#print (f"MMD {total} total, {correct.item()} correct")
	# get validation accuracy
	correct = 0.0
	total = 0.0
	for images, labels, _ in validation_loader:
		images = images.to(device)
		outputs = model(images)
		labels = labels.to(device)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	acc = correct.item()
	acc = acc / total
	this_validation_acc = acc * 100.0
	
	#print (this_training_acc,this_validation_acc)
	
	if (this_training_acc - this_validation_acc > 3):
		#### the gap between train/validation. the threshold here can be adjusted
		model.train()
		
		for loss_index, (train_images, train_labels, _) in enumerate(train_loader_in_order):
			model.zero_grad()
			### get the same number of images for each class
			valid_images = torch.zeros_like(train_images).type(torch.FloatTensor).to(device)
			valid_labels = torch.zeros_like(train_labels).type(torch.LongTensor).to(device)
			valid_index = 0
			for label_index, i in enumerate(torch.unique(train_labels)):
				this_frequency = torch.bincount(train_labels)[i].to(device)
				this_class_start = starting_index[i]
				## i is the current class label
				
				if (i < num_classes - 1):
					this_class_end = starting_index[i + 1] - 1
				else:
					this_class_end = validation_set.__len__() - 1
				
				for i in range(this_frequency):
					random_index = np.random.randint(this_class_start, this_class_end)
					new_images, new_labels, _ = ((validation_set).__getitem__(random_index))
					valid_images[valid_index] = new_images.to(device)
					valid_labels[valid_index] = new_labels.to(device)
					valid_index += 1
			
			train_images = train_images.to(device)  # .type(torch.float64)
			# train_labels = train_labels.to(device)
			outputs = model(train_images)
			all_train_outputs = F.softmax(outputs, dim=1)
			# all_train_outputs = all_train_outputs.view(-1, num_classes)
			# train_labels = train_labels.view(batch_num, 1)
			
			valid_images = valid_images.to(device)  # .type(torch.float64)
			# valid_labels = valid_labels.to(device)
			outputs = model(valid_images)
			all_valid_outputs = F.softmax(outputs, dim=1)
			all_valid_outputs = (all_valid_outputs).detach_()
			# valid_labels = valid_labels.view(batch_num, 1)
			
			# validation_label_in_training.append(valid_labels.cpu().data.numpy())
			# validation_confidence_in_training.append(all_valid_outputs.cpu().data.numpy())
			
			mmd_loss = mix_rbf_mmd2(all_train_outputs, all_valid_outputs, sigma_list=[1]) * loss_lambda
			mmd_loss.backward()
			optimizer.step()
			
	return model.state_dict()


def update_weights_hamp(current_model_weights, model, optimizer, train_loader, local_epochs, hamp_weight =0.01, mixup=0, selfswap=0, std_loss_lambda=0.1,class_weights=None,dataset_name=''):
	# switch to train mode
	model.train()
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss().to(device)
	
	model.load_state_dict(current_model_weights)
	model.zero_grad(set_to_none=True)
	model.train().to(device)
	### for each user, we need to create a new dataloader, so we can avoid using instances that are used in previous steps, same epoch.
	### for single worker, it is not possible to play with sampler to achieve the above constraint.
	for _ in range(local_epochs):
		for batch_idx, (inputs, targets,batch_idx) in enumerate(train_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			optimizer.zero_grad()
			outputs = model(inputs)
			entropy = Categorical(probs=F.softmax(outputs, dim=1)).entropy()
			loss1 = F.kl_div(F.log_softmax(outputs, dim=1), targets)
			loss2 = -1 * hamp_weight * torch.mean(entropy) # 0.01 for purchase and texas, 0.001 for densenet
			loss = loss1 + loss2
			loss.backward()
			optimizer.step()
	return model.state_dict()


def update_weights_add_grad(model,target_model_optim,vulnerable_points_set_loader,weight=1):
	# Set mode to train model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss().to(device)  # weight=class_weights
	model.train().to(device)
	model.zero_grad(set_to_none=True)
	
	for _ in range(1):
		for batch_idx, (images, labels, _) in enumerate(vulnerable_points_set_loader):
			images, labels = images.to(device), labels.to(device)
			model.zero_grad()
			log_probs = model(images)
			loss = criterion(log_probs, labels) * -1 * weight # essentially we are doing gradient ascent here
			loss.backward()
			target_model_optim.step()
			
	model.zero_grad(set_to_none=True)
	return model.state_dict()

def average_weights(w, weight=None):
	"""
	Returns the average of the weights and the param diff from each user.
	"""
	# print (len(w))
	if (len(w) == 1):
		return w[0]
	if (weight is None):
		weight = torch.ones((len(w))) / len(w)
	else:
		weight = torch.from_numpy(weight)
	
	# print(weight)
	# w_avg = copy.deepcopy(w[0])
	w_avg = copy.deepcopy(w[0])
	
	for key, val in w_avg.items():
		w_avg[key] = val * weight[0]
	
	for key in w_avg.keys():
		for i in range(1, len(w)):
			w_avg[key] += w[i][key] * weight[i]
	# w_avg[key] +=w[i][key]
	# w_avg[key] = torch.div(w_avg[key], len(w))
	return w_avg


def active_attacker_mislabel(model, optimizer, user_list, local_epochs=1, batch_size=100, client_adversary=0, lr_multiplier=0.1, target_label=0, num_classes=10):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
	model.train().to(device)
	# print (f"client adversary {client_adversary}, multiplier {lr_multiplier}")
	# print ("mislabel active attack")
	# print (f"target_label{target_label}")
	default_backup_target_label = 5
	
	for user_idx in range(len(user_list)):
		loader = torch.utils.data.DataLoader(user_list[user_idx].evaluation_member_dataset, batch_size=50,
											 shuffle=False)
		for iter in range(local_epochs):
			for batch_idx, (images, labels, _) in enumerate(loader):
				images, labels = images.to(device), labels.to(device)
				
				if (target_label != -1):
					### create wrong labels / target fix label
					temp_new_labels = torch.ones_like(labels) * target_label
					new_labels = torch.zeros_like(temp_new_labels)
					for idx, (old_label, new_label) in enumerate(zip(labels, temp_new_labels)):
						if (old_label == new_label):
							new_labels[idx] = default_backup_target_label
						else:
							new_labels[idx] = new_label
				else:
					### create wrong labels / target random label
					random_labels = torch.from_numpy(np.random.randint(0, num_classes, labels.size(0))).to(device)
					new_labels = torch.zeros_like(labels)
					for idx, (old_label, random_label) in enumerate(zip(labels, random_labels)):
						if (old_label == random_label):
							new_labels[idx] = (random_label + 1) % num_classes
						else:
							new_labels[idx] = random_label
				
				# print (torch.unique(labels).size(),torch.unique(new_labels).size())
				# print (torch.unique(new_labels))
				
				model.zero_grad()
				log_probs = model(images)
				loss = -1 * lr_multiplier * criterion(log_probs, new_labels.to(device))
				loss.backward()
				optimizer.step()
	# print (batch_idx)
	
	for user_idx in range(len(user_list)):
		loader = torch.utils.data.DataLoader(user_list[user_idx].evaluation_non_member_dataset, batch_size=50,
											 shuffle=False)
		for iter in range(local_epochs):
			for batch_idx, (images, labels, _) in enumerate(loader):
				images, labels = images.to(device), labels.to(device)
				if (target_label != -1):
					### create wrong labels / target fix label
					temp_new_labels = torch.ones_like(labels) * target_label
					new_labels = torch.zeros_like(temp_new_labels)
					for idx, (old_label, new_label) in enumerate(zip(labels, temp_new_labels)):
						if (old_label == new_label):
							new_labels[idx] = default_backup_target_label
						else:
							new_labels[idx] = new_label
				else:
					### create wrong labels / target random label
					random_labels = torch.from_numpy(np.random.randint(0, num_classes, labels.size(0))).to(device)
					new_labels = torch.zeros_like(labels)
					for idx, (old_label, random_label) in enumerate(zip(labels, random_labels)):
						if (old_label == random_label):
							new_labels[idx] = (random_label + 1) % num_classes
						else:
							new_labels[idx] = random_label
				
				model.zero_grad()
				log_probs = model(images)
				loss = -1 * lr_multiplier * criterion(log_probs, new_labels)
				loss.backward()
				optimizer.step()
	# print (batch_idx)
	
	return model.state_dict()


def active_attacker_gradient_ascent(model, optimizer, user_list, local_epochs=1, ga_adversary=1, param_search=False, class_weights=None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
	model.train().to(device)
	sum_loss = 0
	
	### if ga_adversary == 1, this means we are doing GA attack
	### if ga_adversary == -1, this means we are doing GD attack
	
	for user_idx in range(len(user_list)):
		loader = torch.utils.data.DataLoader(user_list[user_idx].evaluation_member_dataset, batch_size=50, shuffle=False)
		for iter in range(local_epochs):
			for batch_idx, (images, labels, _) in enumerate(loader):
				images, labels = images.to(device), labels.to(device)
				optimizer.zero_grad()
				log_probs = model(images)
				loss = criterion(log_probs, labels) * (-1 * ga_adversary)
				sum_loss += loss.cpu().item()
				loss.backward()
				### before step, we need to check the previous param, current grad
				# print (model.fc1.weight[0,0])
				# print (model.fc1.weight.grad[0,0])
				optimizer.step()
	# print (batch_idx)
	# print (model.fc1.weight[0,0])
	
	for user_idx in range(len(user_list)):
		loader = torch.utils.data.DataLoader(user_list[user_idx].evaluation_non_member_dataset, batch_size=50, shuffle=False)
		for iter in range(local_epochs):
			for batch_idx, (images, labels, _) in enumerate(loader):
				images, labels = images.to(device), labels.to(device)
				optimizer.zero_grad()
				log_probs = model(images)
				loss = criterion(log_probs, labels) * (-1 * ga_adversary)
				sum_loss += loss.cpu().item()
				loss.backward()
				optimizer.step()
	# print (batch_idx)
	
	# print (f"sumloss{sum_loss}")
	sum_loss = sum_loss / ((len(user_list)) * 2)  ## caclculate the avg loss of targeted instances
	
	return model.state_dict(), sum_loss


# for key,val in model.state_dict().items():
#	print (key,val.size())
# for name,param in model.named_parameters():
#   print (name,param.size(),param.grad.size())
# active_attacker_gradient = copy.deepcopy([param.grad for param in model.parameters()]) ## this implementation would occur errors for BN layer

# print(active_grad_magnitude)
# active_attacker_gradient_dict = {}
# for idx,(key,val) in enumerate(model.state_dict().items()):
#	active_attacker_gradient_dict[key] =

# print (f"sumloss{sum_loss}")
# sum_loss = sum_loss/((len(user_list))*2) ## caclculate the avg loss of targeted instances

# active_grad_magnitude = 0
# for this_grad in active_attacker_gradient:
# print (this_grad.size())
#	active_grad_magnitude+=torch.norm(torch.flatten(this_grad),p=1)
# return active_attacker_gradient_dict,active_grad_magnitude,sum_loss

def get_user_update_list(model_dict, user_dict_list, learning_rate, num_batches):
	# for key,val in model_dict.items():
	#    print (key,val.size())
	# print (f"user update num batches {num_batches}, lr {learning_rate}")
	user_update_list = []
	for user_idx in range(len(user_dict_list)):
		this_user_update_list = []
		for param1, param2 in zip(model_dict.values(), user_dict_list[user_idx].values()):
			# if (user_idx == 0):
			#	print (param1.size(),param1.requires_grad,param2.size())
			# print (f"user update param shape{param1.size()}")
			
			# if (len(param1.shape)>1): ### avoid params from batchnorm / bias.
			this_user_update_list.append((param1 - param2) / (learning_rate * num_batches))  ### be careful with the order here, new-param = old-param - gradient
		### also, dividing by learning rate means each user update is sum of gradient for all batches
		user_update_list.append(this_user_update_list)  ### this is the average gradient of all training samples. NEED to be scaled by # of instances when do Grad-diff.
	
	# if (user_idx == 0):
	#	print ("require grads list")
	#	for p in this_user_update_list:
	#		print (p.size())
	#	exit(0)
	
	return user_update_list


def simplex_uniform_sampling(num):
	sampling = np.zeros((num))
	
	# we sample num-1 numbers from (0,1), then add 0 and 1 to the array, so we have num+1 numbers. After sorting
	# the gap between each adjacent pairs is the probability we use.
	
	array = []
	for _ in range(num - 1):
		# print (np.random.get_state())
		array.append(np.random.uniform(low=0, high=1))
	array.append(0)
	array.append(1)
	array = np.sort(np.array(array))
	for i in range(len(array) - 1):
		sampling[i] = array[i + 1] - array[i]
	
	# print (array,sampling,np.sum(sampling))
	
	return sampling


def user_update_list_sanity_check(user_list, user_update_list, get_gradient_func, model):
	### this is to check how different the user update is from the sum of gradients
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cos = nn.CosineSimilarity(dim=0, eps=1e-6)
	
	all_info = []
	
	for user_idx in range(len(user_list)):
		
		#print(f"for user {user_idx}:")
		
		this_user_info = []
		
		train_eval_dataset = user_list[user_idx].train_eval_dataset
		data_iter = torch.utils.data.DataLoader(train_eval_dataset, batch_size=100, shuffle=False,
												num_workers=1)
		
		acc_batch_grad = []
		batch_count = 0
		for image, label, _ in data_iter:
			this_batch_grad = get_gradient_func(model, image.to(device), label.to(device))
			
			# for param in this_batch_grad:
			#    print (param.size())
			
			if (batch_count == 0):
				for param in this_batch_grad:
					acc_batch_grad.append(torch.zeros_like(param))
			
			for layer_idx in range(len(acc_batch_grad)):
				acc_batch_grad[layer_idx] += this_batch_grad[layer_idx]
			
			batch_count += 1
		
		for layer_idx in range(len(acc_batch_grad)):
			acc_batch_grad[layer_idx] = acc_batch_grad[layer_idx] / batch_count
		
		print(f"total batch number {batch_count}")
		
		### now we compare user update and param
		
		for layer_idx, (param1, param2) in enumerate(zip(acc_batch_grad, user_update_list[user_idx])):
			param1 = torch.flatten(param1)
			param2 = torch.flatten(param2)
			
			print(f"for layer {layer_idx}:")
			
			print(f"norm of batch grad {torch.norm(param1, p=1).item()}")
			print(f"norm of user epoch grad {torch.norm(param2, p=1).item()}")
			print(f"norm of grad diff {torch.norm(param1 - param2, p=1).item()}")
			print(f"cosine similarity {cos(param1, param2).item()}")
			
			this_user_info.append([torch.norm(param1, p=1).item(),
								   torch.norm(param2, p=1).item(),
								   torch.norm(param1 - param2, p=1).item(),
								   cos(param1, param2).item()])
		
		all_info.append(this_user_info)
	
	print(np.array(all_info).shape)
	
	return np.array(all_info)
