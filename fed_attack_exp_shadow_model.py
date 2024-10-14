from whitebox_attack import *
from blackbox_attack import *
import argparse
from data import dataset
from model import *
from model import alexnet_tinyimagenet
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
from model_utils import get_train_loss
from diffmi_attack import diffmi_attack
from nasr_fed_attack import nasr_fed_attack
from multi_party_attack import *
import datetime
import gc
from run_attacks import *

def select_model(target_dataset):
	if (target_dataset.dataset_name == 'cifar100'):
		args.num_classes = 100
		if (args.model_name == 'resnet18'):
			target_model = convert_batchnorm_modules(ResNet18(num_classes=100),
			                                         converter=_batchnorm_to_groupnorm_new)
		if (args.model_name == 'resnet20'):
			### here, if we replace the layers, the train / test acc curve is different from the original.
			### the training will slow down
			# target_model = resnet(depth=20,num_classes=100)
			target_model = convert_batchnorm_modules(resnet(depth=20, num_classes=100),
			                                         converter=_batchnorm_to_groupnorm_new)
		if (args.model_name == 'densenet_cifar'):
			# print (densenet(depth=100,num_classes=100))
			target_model = convert_batchnorm_modules(densenet(depth=100, num_classes=100),
			                                         converter=_batchnorm_to_groupnorm_new)
		# print (target_model)
		if (args.model_name == 'alexnet'):
			target_model = alexnet(num_classes=100)
	
	elif (target_dataset.dataset_name == 'cifar10'):
		args.num_classes = 10
		if (args.model_name == 'resnet18'):
			target_model = convert_batchnorm_modules(ResNet18(num_classes=10),
			                                         converter=_batchnorm_to_groupnorm_new)
		if (args.model_name == 'resnet20'):
			# target_model = resnet(depth=20,num_classes=10)
			target_model = convert_batchnorm_modules(resnet(depth=20, num_classes=10),
			                                         converter=_batchnorm_to_groupnorm_new)
		if (args.model_name == 'densenet_cifar'):
			target_model = convert_batchnorm_modules(densenet(depth=100, num_classes=10),
			                                         converter=_batchnorm_to_groupnorm_new)
		if (args.model_name == 'alexnet'):
			target_model = alexnet(num_classes=10)
	elif (target_dataset.dataset_name == 'tinyimagenet'):
		target_model = alexnet_tinyimagenet(n_class=200)
	elif (args.model_name == 'mnist_special'):
		target_model = mnist_special('mnist')
	else:
		target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
		if (target_dataset.dataset_name == 'texas'):
			args.num_classes = 100
		if (target_dataset.dataset_name == 'purchase'):
			args.num_classes = 100
			
	# pytorch_total_params = sum(p.numel() for p in target_model.parameters())
	# print(f"total params {pytorch_total_params}")
	return target_model

def get_naming_mid_str(shadow_model_idx):
	name_string_mid_str = 'shadow_'+ str(shadow_model_idx) + '_' + str(args.random_seed) + '_' + str(args.noniid) + '_' + ('client_' if (args.client_adversary) else 'server_') + \
	                      (str(args.active_attacker)) + '_' + (str(args.active_attacker_lr_multiplier)) + '_' + \
	                      str(args.user_number) + '_' + str(args.num_step) + '_' + str(args.dpsgd) + '_' + str(
		args.noise_scale) + '_' + str(args.grad_norm) + '_' + str(args.mmd) + '_' + str(
		args.mmd_loss_lambda) + '_' + str(args.mixup) + '_'
	
	return name_string_mid_str

def get_gradient(target_model, data, label):  ### using gpu 1
	import time
	from opacus import GradSampleModule
	# device = torch.device("cuda",1) # use extra gpu if time is more important so we can process larger batches
	device = torch.device("cuda", 0)
	criterion = nn.CrossEntropyLoss().to(device)
	target_model = target_model.to(device)
	if (args.dpsgd == 0):
		local_model = copy.deepcopy(target_model)
		local_model = GradSampleModule(local_model)
		local_model.zero_grad()
	else:
		### one issue is, deepcopy cannot copy p._forward_counter for some reasons, so we need to add _forward_counter and grad_sample by hand..
		local_model = target_model
		for _, p in local_model.named_parameters():
			p._forward_counter = 0
			p.grad_sample = None
	
	### parallel implementation based on opacus
	data = data.to(device)
	label = label.to(device)
	prediction = local_model(data)
	loss = criterion(prediction, label)
	loss.backward()
	grad_sample = copy.deepcopy([param.grad_sample.detach() for param in local_model.parameters()])
	local_model.zero_grad()
	del local_model
	gc.collect()
	return grad_sample

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
			
		all_data = np.array(all_data)
		all_label = np.array(all_label).flatten()
		return np.squeeze(all_data), np.squeeze(all_label)

def eval_feature(feature,label,feature_name=None):
	from run_attacks import convert_loss_to_logits
	### eval auc, tpr at low fpr for one feature, using shadow model as nonmember set?
	converted_features = []
	for data_idx in range(feature.shape[0]):
		this_instance_label = label[:,data_idx]
		this_instance_feature = feature[data_idx]
		
		### reformat this instace data:
		### we need to select some nonmember cases to form a distribution, all 10 clients are treated as different models.
		### we are still not sure how to combine information from different epochs though.
		### we need to assume that the non-member distribution is normal, which is not fully correct for logits.
		### then, for each evaluation sample, we calculate a probability, it is essentially norm.cdf(...).
		### this converting procedure is done for each instance
	
	### after finishing conversion, we now have
	

	

def shadow_model_evaluation(membership_labels,epochs, prefix, dataset, model, target_data_size, eval_data_size,
                 num_layers=12, num_user=5,best_layer=0, comm_round_list=[]):
	# expected data size: num_epoch (300) * training_data_size (50000) * num_features (8) * num_users (10) * num_layers (6)
	# overall around 30 GB
	# if we do all client MI, then we should use max? or sum would also work I guess, then we reduce the data size by 10
	# now we have 3 GB
	# 50 shadow models mean 150GB
	# so, we need to do the attack evaluation per layer, then we reduce the data size by 6
	# in total 25GB per layer
	### maybe we should do the attack per 10 epochs, so we can have all 10 users kept..
	num_shadow_models = membership_labels.shape[0]
	for best_layer in range(num_layers):
		print (f"current layer {best_layer}")
		num_valid_user = num_user
		total_data_num = membership_labels.shape[1]
		all_epoch_loss = np.zeros((num_shadow_models,total_data_num, num_valid_user, len(epochs)))
		all_epoch_layer_cos = np.zeros((num_shadow_models,total_data_num, num_valid_user, len(epochs)))
		all_epoch_layer_grad_diff = np.zeros((num_shadow_models,total_data_num, num_valid_user, len(epochs)))
		all_epoch_layer_grad_norm = np.zeros((num_shadow_models,total_data_num, num_valid_user, len(epochs)))
		
		for shadow_model_idx in range(num_shadow_models):
			mid_str = get_naming_mid_str(shadow_model_idx)
			for epoch_idx, epoch in enumerate(epochs):
			# print (f"epoch idx {epoch_idx}")
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
				loss_data = np.reshape(loss_data, (-1, num_valid_user))
		
				all_epoch_layer_cos[shadow_model_idx,:, :, epoch_idx] = copy.deepcopy(data[:, :, best_layer, 0])
				all_epoch_layer_grad_diff[shadow_model_idx,:, :, epoch_idx] = copy.deepcopy(data[:, :, best_layer, 1] - data[:, :, best_layer, 2])
				all_epoch_loss[shadow_model_idx,:, :, epoch_idx] = copy.deepcopy(loss_data)
				all_epoch_layer_grad_norm[shadow_model_idx,:, :, epoch_idx] = copy.deepcopy(data[:, :, best_layer, 3])
			
		all_epoch_loss = np.nan_to_num(all_epoch_loss,nan=1e-6)
		all_epoch_layer_grad_norm = np.nan_to_num(all_epoch_layer_grad_norm)
		all_epoch_layer_grad_diff = np.nan_to_num(all_epoch_layer_grad_diff)
		all_epoch_layer_cos = np.nan_to_num(all_epoch_layer_cos)
		
		eval_feature(all_epoch_layer_cos,membership_labels,feature_name='cos')
		eval_feature(all_epoch_layer_grad_diff,membership_labels,feature_name='grad-diff')
		eval_feature(all_epoch_layer_grad_norm,membership_labels,feature_name='grad-norm')
		eval_feature(all_epoch_loss,membership_labels,feature_name='loss')
	
def assign_part_dataset(dataset, user_list=[]):
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
	
	if (dataset.dataset_name == 'tinyimagenet'):
		norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		train_trans = [transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
		val_trans = [transforms.ToTensor(), norm]
		transform_train = transforms.Compose(train_trans + [norm])
		transform_test = transforms.Compose(val_trans)
		target_transform = transforms.ToTensor()
	
	if (dataset.dataset_name == 'mnist' or dataset.dataset_name == 'fashion_mnist' or (
			'celeb' in dataset.dataset_name) or dataset.dataset_name == 'retina'):
		transform_train = transforms.ToTensor()
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
	
	if (dataset.dataset_name == 'purchase' or dataset.dataset_name == 'texas'):
		transform_train = None
		transform_test = None
		target_transform = None
		print("1D dataset!")
		
	### choose training data
	all_training_index = np.random.choice(len(dataset.train_label),args.target_data_size*args.user_number,replace=False)
	this_round_instance_membership_label = np.zeros((len(dataset.train_label)))
	#this_round_instance_membership_label[all_training_index] = 1
	### assign training data to all users
	for i in range(args.user_number):
		this_user = user_list[i]
		this_user.target_transform = target_transform
		this_user.train_transform = transform_train
		this_user.test_transform = transform_test
		this_user.train_data = dataset.train_data[all_training_index[i*args.target_data_size:(i+1)*args.target_data_size]]
		this_user.train_label = dataset.train_label[all_training_index[i*args.target_data_size:(i+1)*args.target_data_size]]
		this_user.test_data = dataset.test_data
		this_user.test_label = dataset.test_label
		
		this_round_instance_membership_label[all_training_index[i*args.target_data_size:(i+1)*args.target_data_size]] = i
		
		this_user.class_weight = np.ones((len(np.unique(dataset.train_label)))) * args.target_data_size / (len(np.unique(dataset.train_label)) * (np.bincount(this_user.train_label) + 1))
		train = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=True, transform=transform_train,
		                             target_transform=target_transform)
		test = part_pytorch_dataset(this_user.test_data, this_user.test_label, train=False, transform=transform_test,
		                            target_transform=target_transform)
		this_user.train_dataset = train
		this_user.test_dataset = test
		this_user.train_data_loader = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,
		                                                          shuffle=True, num_workers=1)
		this_user.test_data_loader = torch.utils.data.DataLoader(test, batch_size=args.target_batch_size, shuffle=False,
		                                                         num_workers=1)
		
	### create a data loader for collecting membership attack information
	eval_dataset = part_pytorch_dataset(dataset.train_data,dataset.train_label,train=False,transform=transform_test,
	                                    target_transform=target_transform)
	user_list[0].evaluation_member_dataset = eval_dataset
	#membership_eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=args.target_data_size,shuffle=False,num_workers=1)
	
	return this_round_instance_membership_label
	### IGNORE VALIDATION SET FOR NOW..
	### check remaining unassigned data
	#dataset.remaining_index = index_left
	### we select some data as validation set
	#validation_data_index = np.random.choice(index_left, len(index_left),replace=True)  ### this should be false, but just for the sake of # of user exp
	#validation_data = dataset.train_data[validation_data_index]
	#validation_label = dataset.train_label[validation_data_index]
	#dataset.remaining_index = np.setdiff1d(index_left, validation_data_index)
	#for user_idx in range(num_users):
	#	this_user = user_list[user_idx]
	#	this_user.eval_validation_data = validation_data
	#	this_user.eval_validation_label = validation_label
	#	### processing validation set for MMD defense
	#	### sort the validation data according to the class index
	#	sorted_index = np.argsort(this_user.eval_validation_label)
	#	this_user.eval_validation_data = this_user.eval_validation_data[sorted_index]
	#	this_user.eval_validation_label = this_user.eval_validation_label[sorted_index]
		### create an index list for starting index of each class
	#	this_user.starting_index = []
	#	for i in np.unique(this_user.eval_validation_label):
	#		for j in range(len(this_user.eval_validation_label)):
	#			if (this_user.eval_validation_label[j] == i):
	#				this_user.starting_index.append(j)
	#				break
	#	this_user.validation_dataset = part_pytorch_dataset(validation_data, validation_label, train=False,
	#	                                                    transform=transform_test,
	#	                                                    target_transform=target_transform)
	#	this_user.validation_data_loader = torch.utils.data.DataLoader(this_user.validation_dataset,
	#	                                                               batch_size=args.target_batch_size, shuffle=False,
	#	                                                               num_workers=1)
		


def run_multi_party_attacks(user_list, target_model, epoch, user_update_list, user_model_list, ori_model_weight_dict,
                            server_attacker=False, attack_loader_list=[], comm_round_idx=0, best_layer=None,shadow_idx=0):
	naming_str = get_naming_mid_str(shadow_idx) + str(comm_round_idx) + '_' + str(epoch + 1) + '_' + str(
		args.dataset) + '_' + str(args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(
		args.model_name) + '.npy'
	
	print(naming_str)
	
	if (server_attacker):
		
		all_info, all_label = multi_party_member_attack(user_list, target_model, batch_size=args.target_batch_size,
		                                                user_update_list=user_update_list,
		                                                get_gradient_func=get_gradient,
		                                                attack_loader_list=attack_loader_list, user_total_instance=args.num_step * args.target_batch_size,
		                                                max_instance_per_batch=args.max_instance_per_batch, best_layer=best_layer, whole_nn=args.whole_nn)
		np.save('./new_expdata/all_info_multi_party_member_attack_' + naming_str, all_info)
		np.save('./new_expdata/all_label_multi_party_member_attack_' + naming_str, all_label)
		
		loss_info, loss_label = multi_party_member_loss_attack(user_list, target_model,
		                                                       batch_size=args.target_batch_size,
		                                                       user_update_list=user_update_list,
		                                                       get_gradient_func=get_gradient,
		                                                       user_model_list=user_model_list,
		                                                       attack_loader_list=attack_loader_list,
		                                                       max_instance_per_batch=args.max_instance_per_batch)
		np.save('./new_expdata/loss_info_multi_party_member_attack_' + naming_str, loss_info)
		np.save('./new_expdata/loss_label_multi_party_member_attack_' + naming_str, loss_label)
		
def train_models(user_list, target_model, learning_rate, decay, epochs,target_dataset=None,shadow_idx=0):
	num_users = len(user_list)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_model.to(device)
	momentum = 0.9
	### notice that in pytorch, momentum etc. is bound with optimizer, so we need to initialize the optimizer/model for each user
	for user_idx in range(num_users):
		user_list[user_idx].model = copy.deepcopy(target_model)
		if (args.dataset == 'purchase' or args.dataset == 'texas'):
			this_optim = torch.optim.Adam(user_list[user_idx].model.parameters(), lr=learning_rate, weight_decay=decay)
		else:
			this_optim = torch.optim.SGD(user_list[user_idx].model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
		### using sgd actually leads to 3x memory consumption.. this is very tricky
		user_list[user_idx].optim = this_optim
		
		### specific procedure for dpsgd
		if (args.dpsgd):
			user_list[user_idx].privacy_engine = PrivacyEngine()
			user_list[user_idx].model, user_list[user_idx].optim, user_list[user_idx].train_data_loader = user_list[user_idx].privacy_engine.make_private(
				module=user_list[user_idx].model,
				optimizer=user_list[user_idx].optim,
				data_loader=user_list[user_idx].train_data_loader,
				noise_multiplier=args.noise_scale,  ### sigma
				max_grad_norm=args.grad_norm)  ### this is from dp-sgd paper)
	
	### for dpsgd case.. just to make sure the name of parameters for target model is the same as other private models,
	if (args.dpsgd):
		print("DPSGD ACTIVATED")
		target_model_privacy_engine = PrivacyEngine()
		if (args.dataset == 'purchase' or args.dataset == 'texas'):
			target_model_optim = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=decay)
		else:
			target_model_optim = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
		
		train = part_pytorch_dataset(target_dataset.train_data, target_dataset.train_label, train=True, transform=None, target_transform=None)
		target_model_train_loader = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size, shuffle=True, num_workers=1)
		target_model, target_model_optim, target_model_train_loader = target_model_privacy_engine.make_private(
			module=target_model,
			optimizer=target_model_optim,
			data_loader=target_model_train_loader,
			noise_multiplier=args.noise_scale,  ### sigma
			max_grad_norm=args.grad_norm)  ### this is from dp-sgd paper)
	
	### start training
	for epoch in tqdm(range(epochs)):
		ori_target_model_state_dict = target_model.state_dict()
		## LR schedule
		if (epoch in args.schedule):
			learning_rate = learning_rate / 10
			print("new learning rate = %f" % (learning_rate))
			### set up new learning rate for each user
			for user_idx in range(num_users):
				for param_group in user_list[user_idx].optim.param_groups:
					param_group['lr'] = learning_rate
	
		#### we need to know the # of steps, # of training set size to perform fed sgd
		comm_round_per_epoch = int(args.target_data_size / (args.target_batch_size * args.num_step))
		for comm_round_idx in range(comm_round_per_epoch):
			local_weights = []
			for idx in range(len(user_list)):
				train_data_loader = user_list[idx].create_new_train_data_loader(batch_size=args.target_batch_size)
				if (args.mmd):
					_ = update_weights_mmd(current_model_weights=ori_target_model_state_dict,
					                       model=user_list[idx].model, optimizer=user_list[idx].optim,
					                       train_loader=user_list[idx].train_data_loader,
					                       test_loader=user_list[idx].test_data_loader,
					                       local_epochs=args.local_epochs, mixup=args.mixup,
					                       validation_set=user_list[idx].validation_dataset,
					                       loss_lambda=args.mmd_loss_lambda,
					                       train_loader_in_order=user_list[idx].train_loader_in_order,
					                       num_classes=len(np.unique(user_list[idx].train_label)),
					                       starting_index=user_list[idx].starting_index,
					                       class_weights=user_list[idx].class_weight)
				else:
					ban_list = update_weights(current_model_weights=ori_target_model_state_dict,
					                          model=user_list[idx].model,
					                          optimizer=user_list[idx].optim, train_loader=train_data_loader,
					                          local_epochs=args.local_epochs, mixup=args.mixup,
					                          num_step=args.num_step,
					                          class_weights=user_list[idx].class_weight, unequal=args.unequal)
					user_list[idx].update_ban_list(ban_list) ### this is for 1 step per communication round.. not useful here
					torch.cuda.empty_cache()
				
				local_weights.append((user_list[idx].model.state_dict()))
			
			
			if (epoch % args.attacker_epoch_gap == 0 and comm_round_idx + 1 == comm_round_per_epoch and epoch > 0):
				### this atack requires # of steps = # of batches
				num_batches = int(args.target_data_size / args.target_batch_size)
				if (args.num_step != num_batches):
					print("epoch attack: num steps not equal to num batches")
					exit(0)
		
				user_update_list = get_user_update_list(ori_target_model_state_dict, local_weights, learning_rate=learning_rate, num_batches=num_batches)
				local_model = copy.deepcopy(target_model)
				run_multi_party_attacks([user_list[0]], local_model, epoch, user_update_list,
				                        user_model_list=local_weights,
				                        server_attacker=1 - args.client_adversary,
				                        ori_model_weight_dict=ori_target_model_state_dict,
				                        attack_loader_list=[],
				                        comm_round_idx=comm_round_idx, best_layer=args.best_layer,shadow_idx=shadow_idx)
				print(f"finish attacks for {epoch} epochs")
				del user_update_list, local_model
			
			### update global weights
			global_weights = average_weights(local_weights)
			del local_weights
			target_model.load_state_dict(global_weights)

	train_acc, test_acc = get_train_test_acc(user_list, target_model)
	print(f"train acc {train_acc},test acc {test_acc}")
	return target_model, train_acc, test_acc


def attack_experiment():
	import warnings
	warnings.filterwarnings("ignore")
	np.random.seed(seed=12345)
	torch.set_printoptions(threshold=5000, edgeitems=20)
	
	#### here we assume that when we load the target dataset every single time, the ordering is fixed.
	### the index of a particular instance is always the same
	
	target_dataset = dataset(dataset_name=args.dataset, gpu=args.gpu,
	                         membership_attack_number=0,
	                         cutout=args.cutout, n_holes=args.n_holes, length=args.length)
	
	### assume we only do this attack on CIFAR
	instance_membership_label = np.zeros((args.num_shadow,len(target_dataset.train_label)))
	
	### the total size of training set is 50000, we have 10 users, 10000 instances in training and 40000 instances not in training
	### we also assume we train 50 models in total, so, each instance has 0.2 prob to be selected as member.
	### therefore in average, we have 10 members cases in evaluation, and we can have 10 nonmember cases in evaluation,
	### 30 nonmember cases are used to create the non-member distribution. (30 -> 300 nonmembers, if we use all 10 clients as different models)
	### We can only perform ALL clients MI, because it is not possible to do shadow models for per-client MI?
	
	for shadow_idx in range(args.num_shadow):
	
		### dataset && membership inference data
		membership_attack_number = args.membership_attack_number
		target_model = select_model(target_dataset)

		user_list = [User(dataset=args.dataset, model_name=args.target_model, id=i) for i in range(args.user_number)]
		this_round_instance_membership_label = assign_part_dataset(target_dataset, user_list)
		instance_membership_label[shadow_idx] = this_round_instance_membership_label
		
		train_models(user_list, target_model, args.target_learning_rate,decay=args.target_l2_ratio, epochs=args.target_epochs,target_dataset=target_dataset,shadow_idx=shadow_idx)
	
	### shadow model evaluation

	name_string_prefix = '/home/lijiacheng/whiteboxmi/new_expdata/'
	epochs = ((np.arange(int(args.target_epochs / args.attacker_epoch_gap) - 1) + 1) * args.attacker_epoch_gap)
	num_layers_dict = {'alexnet': 6, 'densenet_cifar': 100, 'purchase': 4, 'texas': 4, 'fashion_mnist': 4,
	                   'retina': 6,'mnist':4,'mnist_special':2}
	num_layers = num_layers_dict[args.model_name]
	exit(0)
	shadow_model_evaluation(instance_membership_label,epochs, name_string_prefix, args.dataset, args.model_name,
		                                                   int(args.target_data_size), int(args.eval_data_size), num_layers=num_layers,
		                                                   num_user=args.user_number, best_layer=args.best_layer,  # this should be args.best_layer
		                                                   comm_round_list=np.arange(int(args.target_data_size / (args.num_step * args.target_batch_size))))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--target_data_size', type=int, default=3000)
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
	parser.add_argument('--num_classes', type=int, default=10)
	# parser.add_argument('--attack_times', type=int, default=1)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--membership_attack_number', type=int, default=0)
	parser.add_argument('--reference_number', type=int, default=1)
	
	parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120])
	parser.add_argument('--model_name', type=str, default='alexnet')
	parser.add_argument('--pretrained', type=int, default=0)
	parser.add_argument('--temperature_scaling', type=int, default=0)
	parser.add_argument('--early_stopping', type=int, default=0)
	parser.add_argument('--alpha', type=float, default='1.0')
	parser.add_argument('--mixup', type=int, default=0)
	parser.add_argument('--label_smoothing', type=float, default=0)
	# parser.add_argument('--dropout', type=int, default=0)
	parser.add_argument('--cutout', type=int, default=0)
	parser.add_argument('--n_holes', type=int, default=1)
	parser.add_argument('--length', type=int, default=16)
	
	parser.add_argument('--num_step', type=int, default=20)
	
	parser.add_argument('--whitebox', type=int, default=0)
	parser.add_argument('--middle_output', type=int, default=0)
	parser.add_argument('--middle_gradient', type=int, default=0)
	parser.add_argument('--save_exp_data', type=int, default=1)
	parser.add_argument('--test', type=int, default=0)
	
	### fed params
	parser.add_argument('--local_epochs', type=int, default=1)
	parser.add_argument('--user_number', type=int, default=2)
	parser.add_argument('--client_adversary', type=int, default=0)
	parser.add_argument('--noniid', type=int, default=0)
	## if adversary_client is 1, then one client is the adversary, otherwise the server is the adversary
	
	### dpsgd params
	parser.add_argument('--dpsgd', type=int, default=0)
	parser.add_argument('--grad_norm', type=float, default=1e10)
	parser.add_argument('--noise_scale', type=float, default=1e-7)
	
	### MMD params
	parser.add_argument('--mmd', type=int, default=0)
	parser.add_argument('--mmd_loss_lambda', type=float, default=0.1)
	
	### attacker params
	parser.add_argument('--active_attacker', type=int, default=0)
	parser.add_argument('--active_attacker_mislabel', type=int, default=0)
	parser.add_argument('--eval_data_size', type=int, default=100)
	parser.add_argument('--aux_data_size', type=int, default=200)
	parser.add_argument('--active_attacker_epoch', type=int, default=1)
	parser.add_argument('--attacker_epoch_gap', type=int, default=10)
	parser.add_argument('--active_attacker_lr_multiplier', type=float, default=0.1)
	parser.add_argument('--mislabeling_target_label', type=int, default=-1)
	parser.add_argument('--pre_gd_epochs', type=int, default=10)
	# if mislabeling_target_label == -1, random label
	parser.add_argument('--mislabeling_target_class', type=int, default=-1)
	parser.add_argument('--best_layer', type=int, default=-1)
	parser.add_argument('--unequal', type=int, default=0)
	parser.add_argument('--max_instance_per_batch', type=int, default=100)  # 100
	parser.add_argument('--active_attack_param_search', type=int, default=0)
	parser.add_argument('--whole_nn', type=int, default=0)
	
	parser.add_argument('--track_loss', type=int, default=0)
	parser.add_argument('--random_seed', type=int, default=0)
	parser.add_argument('--repartition', type=int, default=0)
	parser.add_argument('--num_shadow',type=int,default=1)
	args = parser.parse_args()
	print(vars(args))
	
	if (args.dataset == 'texas'):
		args.max_instance_per_batch = 10
	if (args.model_name == 'densenet_cifar'):
		args.max_instance_per_batch = 200  ### maybe 10. not sure.
	args.max_instance_per_batch = min(args.max_instance_per_batch, args.eval_data_size)
	
	random_seed_list = [args.random_seed]
	avg_tpr_list = []
	avg_auc_list = []
	for this_seed in random_seed_list:
		import torch
		torch.manual_seed(this_seed)
		import numpy as np
		np.random.seed(this_seed)
		import sklearn
		sklearn.utils.check_random_state(this_seed)
		attack_experiment()
