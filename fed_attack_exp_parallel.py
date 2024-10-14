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
from model_utils import get_train_loss
from diffmi_attack import diffmi_attack
from nasr_fed_attack import nasr_fed_attack
from multi_party_attack import *
import datetime
import gc


def get_naming_mid_str():
	name_string_mid_str = str(args.noniid) + '_' + ('client_' if (args.client_adversary) else 'server_') + \
	                      (str(args.active_attacker)) + '_' + (str(args.active_attacker_lr_multiplier)) + '_' + \
	                      str(args.user_number) + '_' + str(args.num_step) + '_' + str(args.dpsgd) + '_' + str(
		args.noise_scale) + '_' + str(args.grad_norm) + '_' + str(args.mmd) + '_' + str(
		args.mmd_loss_lambda) + '_' + str(args.mixup) + '_'
	return name_string_mid_str


def non_idd_assign_part_dataset(dataset, user_list=[]):
	non_iid_dict = {
		'cifar10': 4,
		'cifar100': 20,
		'purchase': 20,
		'texas': 5
	}
	
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
	
	if (dataset.dataset_name == 'mnist' or ('celeb' in dataset.dataset_name)):
		transform_train = transforms.ToTensor()
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
	
	if (dataset.dataset_name == 'purchase' or dataset.dataset_name == 'texas'):
		transform_train = None
		transform_test = None
		target_transform = None
	
	# print (dataset.dataset_name)
	
	num_users = len(user_list)
	num_classes = len(np.unique(dataset.train_label))
	# print(f"num_classes{num_classes}")
	num_classes_non_iid_per_user = non_iid_dict[args.dataset]
	
	### generating train / test indices for each user
	
	## for CIFAR-10 / CIFAR-100, we have 50000/10000 train/test data,
	## then each user should share the test data and we need to split the training data
	## for purchase and texas, we have enough data
	
	## in our case, we choose 3 classes and choose some data from each class for one user to achieve non-iid setting
	
	training_set_size = args.target_data_size
	index_left = np.arange(len(dataset.train_label))  # the # of data left for generating new split of training data
	
	assigned_index = []
	
	# print (np.bincount(dataset.train_label))
	
	for i in range(num_users):
		this_user = user_list[i]
		
		this_user.target_transform = target_transform
		this_user.train_transform = transform_train
		this_user.test_transform = transform_test
		
		### perform non-iid training data selection
		assigned_classes = np.random.choice(num_classes, num_classes_non_iid_per_user, replace=False)
		
		# print(f"user {i} assigned classes:{assigned_classes}")
		
		class_size = int(training_set_size / num_classes_non_iid_per_user)
		this_user_train_index = []
		for this_class in assigned_classes:
			this_class_remaining_index = index_left[
				np.arange(len(index_left))[dataset.train_label[index_left] == this_class]]
			this_user_this_class_train_index = np.random.choice(this_class_remaining_index, class_size)
			this_user_train_index.append(this_user_this_class_train_index)
			index_left = np.setdiff1d(index_left, this_user_this_class_train_index)
		
		this_user_train_index = np.array(this_user_train_index).flatten()
		
		# print(f"user {i} has classes:{np.bincount(dataset.train_label[this_user_train_index])}")
		
		this_user.train_data = dataset.train_data[this_user_train_index]
		this_user.train_label = dataset.train_label[this_user_train_index]
		
		this_user.test_data = dataset.test_data
		this_user.test_label = dataset.test_label
		assigned_index.append(this_user_train_index)
		
		### take a fraction of training data to be used as MI evaluation data (this is the member part of evaluation data)
		# when active attacker is not activated, here the eval_data_size == target_train_size
		# when active attacker is activated, here the eval_data_size == 100 or 50
		eval_data_index = np.random.choice(len(this_user_train_index), args.eval_data_size, replace=False)
		
		evaluation_data = copy.deepcopy(this_user.train_data[eval_data_index])
		evaluation_label = copy.deepcopy(this_user.train_label[eval_data_index])
		
		#### create dataset and dataloader
		train = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=True, transform=transform_train,
		                             target_transform=target_transform)
		test = part_pytorch_dataset(this_user.test_data, this_user.test_label, train=False, transform=transform_test,
		                            target_transform=target_transform)
		### training data processing includes adding noise, but when evaluating, we should remove the effect of noise
		### the data in train_eval set is the same as the train set, but the preprocessing part is removed
		train_eval = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=False,
		                                  transform=transform_test, target_transform=target_transform)
		
		this_user.train_dataset = train
		this_user.test_dataset = test
		this_user.train_eval_dataset = train_eval
		
		this_user.train_data_loader = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,
		                                                          shuffle=False, num_workers=1)
		this_user.test_data_loader = torch.utils.data.DataLoader(test, batch_size=args.target_batch_size, shuffle=False,
		                                                         num_workers=1)
		this_user.train_eval_data_loader = torch.utils.data.DataLoader(train_eval, batch_size=1, shuffle=False,
		                                                               num_workers=1)
		this_user.train_loader_in_order = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,
		                                                              shuffle=False, num_workers=1)
		
		evaluation = part_pytorch_dataset(evaluation_data, evaluation_label, train=False, transform=transform_test,
		                                  target_transform=target_transform)
		this_user.evaluation_member_dataset = evaluation
		
		### we use testing data as 'eval_non_member' set
		non_member_index = np.random.choice(len(this_user.test_label), args.eval_data_size, replace=False)
		evaluation_non_member = part_pytorch_dataset(copy.deepcopy(this_user.test_data[non_member_index]),
		                                             copy.deepcopy(this_user.test_label[non_member_index]), train=False,
		                                             transform=transform_test,
		                                             target_transform=target_transform)
		this_user.evaluation_non_member_dataset = evaluation_non_member
	
	### check remaining unassigned data
	dataset.remaining_index = index_left
	print(len(index_left))
	
	### we select some data as validation set
	validation_data_index = np.random.choice(index_left, training_set_size, replace=False)
	validation_data = dataset.train_data[validation_data_index]
	validation_label = dataset.train_label[validation_data_index]
	dataset.remaining_index = np.setdiff1d(index_left, validation_data_index)
	
	for user_idx in range(num_users):
		this_user = user_list[user_idx]
		this_user.eval_validation_data = validation_data
		this_user.eval_validation_label = validation_label
		this_user.validation_dataset = part_pytorch_dataset(validation_data, validation_label, train=False,
		                                                    transform=transform_test,
		                                                    target_transform=target_transform)
		this_user.validation_data_loader = torch.utils.data.DataLoader(this_user.validation_dataset,
		                                                               batch_size=args.target_batch_size, shuffle=False,
		                                                               num_workers=1)
		### processing validation set for MMD defense
		
		### sort the validation data according to the class index
		sorted_index = np.argsort(this_user.eval_validation_label)
		this_user.eval_validation_data = this_user.eval_validation_data[sorted_index]
		this_user.eval_validation_label = this_user.eval_validation_label[sorted_index]
		
		### create a index list for starting index of each class
		this_user.starting_index = []
		# print ("starting index",self.starting_index)
		for i in np.unique(this_user.eval_validation_label):
			for j in range(len(this_user.eval_validation_label)):
				if (this_user.eval_validation_label[j] == i):
					# print ("class %d index %d "%(i,j))
					this_user.starting_index.append(j)
					break
	
	print(len(index_left))


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
	
	num_users = len(user_list)
	
	### generating train / test indices for each user
	
	## for CIFAR-10 / CIFAR-100, we have 50000/10000 train/test data,
	## then each user should share the test data and we need to split the training data
	
	## for purchase and texas, we have enough data
	
	## the maximum number of user we can support is 10
	training_set_size = args.target_data_size
	index_left = np.arange(len(dataset.train_label))  # the # of data left for generating new split of training data
	
	assigned_index = []
	
	'''
	attacker_evaluation_data = []
	attacker_evaluation_label = []
	attacker_evaluation_data_index = []
	#### do we need to include non-member data in this? do we need to do gradient ascent for non-members?

	### active attackers may attack in the following 2 different ways.
	### 2 cases:
	# 1: do gradient ascent for members and non-members. the grad of members should be smaller than non-members.
	# but the grad of members might be close to the grad of normal testing samples.
	# the grad of non-members would be super large.

	# 2: do gradient ascent for members only. here the goal is to tell the members apart from normal testing samples.
	# the expected result is that the grad norm of members should be super large, even larger than normal testing samples.
	# but this is actually depending on the testing accuracy. and the grad norm of members might not be that large.

	In our current experiment, we choose method 1, i.e. do gradient ascent for both members and non-members.
	We need to explicitly find a targeted set to do gradient ascent. The server is the attacker to do gradient ascent.
	'''
	
	### for # of instance exp. each client has diff # of instance.
	if (args.unequal):
		training_set_size_list = [200, 600, 1000, 1400, 1800, 2200, 2600, 3000, 3400, 3800]
	
	for i in range(num_users):
		
		if (args.unequal):
			training_set_size = training_set_size_list[i]
		else:
			training_set_size = args.target_data_size
		
		this_user = user_list[i]
		this_user.target_transform = target_transform
		this_user.train_transform = transform_train
		this_user.test_transform = transform_test
		
		this_user_index = np.random.choice(len(index_left), training_set_size, replace=False)
		this_user_train_index = index_left[this_user_index]
		new_index_left = np.setdiff1d(np.arange(len(index_left)), this_user_index)
		index_left = index_left[new_index_left]
		
		this_user.train_data = dataset.train_data[this_user_train_index]
		this_user.train_label = dataset.train_label[this_user_train_index]
		
		print(f"user {i} has classes:{np.bincount(dataset.train_label[this_user_train_index])}")
		# print (np.bincount(this_user.train_label))
		
		this_user.class_weight = np.ones((len(np.unique(dataset.train_label)))) * training_set_size / (
				len(np.unique(dataset.train_label)) * (np.bincount(this_user.train_label) + 1))
		
		# this_user.class_weight = np.ones((len(np.unique(this_user.train_label)))) * training_set_size / (len(np.unique(this_user.train_label)) * (np.bincount(this_user.train_label)))
		print("class weight:", this_user.class_weight)
		
		this_user.test_data = dataset.test_data
		this_user.test_label = dataset.test_label
		assigned_index.append(this_user_train_index)
		
		### take a fraction of training data to be used as MI evaluation data (this is the member part of evaluation data)
		# when active attacker is not activated, here the eval_data_size == target_train_size
		# when active attacker is activated, here the eval_data_size == 100 or 50
		### if we are going to do targeted class GA mislabeling, then we need to make sure eval set in a specific class
		
		if (args.active_attacker_mislabel == 1 and args.mislabeling_target_class != -1):
			targeted_class = args.mislabeling_target_class
			this_class_index_in_train = np.arange(len(this_user.train_label))[this_user.train_label == targeted_class]
			eval_data_index = np.random.choice(this_class_index_in_train, args.eval_data_size, replace=False)
			print(f"mislabel active attack eval data size {len(eval_data_index)}")
		else:
			eval_data_index = np.random.choice(len(this_user_train_index), args.eval_data_size, replace=False)
		
		evaluation_data = copy.deepcopy(this_user.train_data[eval_data_index])
		evaluation_label = copy.deepcopy(this_user.train_label[eval_data_index])
		
		#### create dataset and dataloader
		train = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=True, transform=transform_train,
		                             target_transform=target_transform)
		test = part_pytorch_dataset(this_user.test_data, this_user.test_label, train=False, transform=transform_test,
		                            target_transform=target_transform)
		### training data processing includes adding noise, but when evaluating, we should remove the effect of noise
		### the data in train_eval set is the same as the train set, but the preprocessing part is removed
		train_eval = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=False,
		                                  transform=transform_test, target_transform=target_transform)
		
		this_user.train_dataset = train
		this_user.test_dataset = test
		this_user.train_eval_dataset = train_eval
		
		this_user.train_data_loader = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,
		                                                          shuffle=True, num_workers=1)
		this_user.test_data_loader = torch.utils.data.DataLoader(test, batch_size=args.target_batch_size, shuffle=False,
		                                                         num_workers=1)
		this_user.train_eval_data_loader = torch.utils.data.DataLoader(train_eval, batch_size=1, shuffle=False,
		                                                               num_workers=1)
		this_user.train_loader_in_order = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,
		                                                              shuffle=False, num_workers=1)
		
		evaluation = part_pytorch_dataset(evaluation_data, evaluation_label, train=False, transform=transform_test,
		                                  target_transform=target_transform)
		this_user.evaluation_member_dataset = evaluation
		
		### we use testing data as 'eval_non_member' set
		# if (args.active_attacker_mislabel==1 and args.mislabeling_target_class!=-1):
		#    targeted_class = args.mislabeling_target_class
		#    this_class_index_in_test = np.arange(len(this_user.test_label))[this_user.test_label == targeted_class]
		#    non_member_index = np.random.choice(this_class_index_in_test, args.eval_data_size, replace=False)
		#    print (f"mislabel active attack eval data size {len(non_member_index)}")
		# else:
		non_member_index = np.random.choice(len(this_user.test_label), args.eval_data_size, replace=False)
		evaluation_non_member = part_pytorch_dataset(copy.deepcopy(this_user.test_data[non_member_index]),
		                                             copy.deepcopy(this_user.test_label[non_member_index]), train=False,
		                                             transform=transform_test,
		                                             target_transform=target_transform)
		this_user.evaluation_non_member_dataset = evaluation_non_member
	
	# print("eval set balance test:", np.bincount(evaluation_label),np.bincount(this_user.test_label[non_member_index]))
	# print("train test set balance test:", np.bincount(this_user.train_label), np.bincount(this_user.test_label))
	
	### check remaining unassigned data
	dataset.remaining_index = index_left
	
	print(len(index_left))
	print(index_left)
	### we select some data as validation set
	validation_data_index = np.random.choice(index_left, len(index_left),
	                                         replace=True)  ### this should be false, but just for the sake of # of user exp
	validation_data = dataset.train_data[validation_data_index]
	validation_label = dataset.train_label[validation_data_index]
	dataset.remaining_index = np.setdiff1d(index_left, validation_data_index)
	
	for user_idx in range(num_users):
		this_user = user_list[user_idx]
		this_user.eval_validation_data = validation_data
		this_user.eval_validation_label = validation_label
		this_user.validation_dataset = part_pytorch_dataset(validation_data, validation_label, train=False,
		                                                    transform=transform_test,
		                                                    target_transform=target_transform)
		this_user.validation_data_loader = torch.utils.data.DataLoader(this_user.validation_dataset,
		                                                               batch_size=args.target_batch_size, shuffle=False,
		                                                               num_workers=1)
		### processing validation set for MMD defense
		
		### sort the validation data according to the class index
		sorted_index = np.argsort(this_user.eval_validation_label)
		this_user.eval_validation_data = this_user.eval_validation_data[sorted_index]
		this_user.eval_validation_label = this_user.eval_validation_label[sorted_index]
		
		### create an index list for starting index of each class
		this_user.starting_index = []
		# print ("starting index",self.starting_index)
		for i in np.unique(this_user.eval_validation_label):
			for j in range(len(this_user.eval_validation_label)):
				if (this_user.eval_validation_label[j] == i):
					# print ("class %d index %d "%(i,j))
					this_user.starting_index.append(j)
					break
	
	print(len(index_left))
	
	'''
	### select some aux data for the attacker
	if (args.aux_data_size == 0):
		args.aux_data_size = args.target_data_size
	aux_data_index = np.random.choice(len(dataset.test_label),args.aux_data_size,replace=False)
	### aux data is not used in the training
	attacker_aux_data = dataset.test_data[aux_data_index]
	attacker_aux_label = dataset.test_label[aux_data_index]

	if (dataset.dataset_name == 'mnist'):
		attacker_evaluation_data = np.reshape(np.array(attacker_evaluation_data),(len(user_list)*args.target_set_size,28,28,1))
	if (dataset.dataset_name == 'cifar10' or dataset.dataset_name == 'cifar100'):
		attacker_evaluation_data = np.reshape(np.array(attacker_evaluation_data),(len(user_list)*args.target_set_size,32,32,3))
	attacker_evaluation_label = np.array(attacker_evaluation_label).flatten()
	#print (np.array(attacker_evaluation_data).shape)

	attacker.evaluation_dataset = part_pytorch_dataset(attacker_evaluation_data,attacker_evaluation_label,train=False,transform=transform_test,target_transform=target_transform)
	attacker.aux_dataset = part_pytorch_dataset(attacker_aux_data,attacker_aux_label,train=False,transform=transform_test,target_transform=target_transform)

	return None,None,None,np.array(attacker_evaluation_data_index)
	'''


def run_blackbox_attacks(user_list, target_model, num_classes, output_file):
	# num_classes = 10 if (args.dataset == 'cifar10' or args.dataset == 'mnist') else 100
	acc = 0
	for user_idx in range(len(user_list)):
		black_ref = blackbox_attack(args.eval_data_size, 'global_prob', num_classes=num_classes)
		total_confidences, total_classes, total_labels = black_ref.get_attack_input(target_model, user_list[user_idx])
		acc += black_ref.attack(total_confidences=total_confidences, total_classes=total_classes,
		                        total_labels=total_labels, output_file=output_file)  ### labels here is the true label
	# print ("blackbox attack finished")
	return acc / len(user_list)


def run_multi_party_attacks(user_list, target_model, epoch, user_update_list, user_model_list, ori_model_weight_dict,
                            server_attacker=False, attack_loader_list=[], comm_round_idx=0, best_layer=None):
	naming_str = get_naming_mid_str() + str(comm_round_idx) + '_' + str(epoch + 1) + '_' + str(
		args.dataset) + '_' + str(args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(
		args.model_name) + '.npy'
	
	print(naming_str)
	
	if (server_attacker):
		all_info, all_label = multi_party_member_attack(user_list, target_model, batch_size=args.target_batch_size,
		                                                user_update_list=user_update_list,
		                                                get_gradient_func=get_gradient,
		                                                attack_loader_list=attack_loader_list, user_total_instance=args.num_step * args.target_batch_size,
		                                                max_instance_per_batch=args.max_instance_per_batch, best_layer=best_layer)
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
		
		target_loss_info, target_loss_label = multi_party_member_loss_attack(user_list, target_model,
		                                                                     batch_size=args.target_batch_size,
		                                                                     user_update_list=user_update_list,
		                                                                     get_gradient_func=get_gradient,
		                                                                     user_model_list=[ori_model_weight_dict],
		                                                                     attack_loader_list=attack_loader_list,
		                                                                     max_instance_per_batch=args.max_instance_per_batch)
		np.save('./new_expdata/target_model_before_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
		np.save('./new_expdata/target_model_before_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)
		
		weights_after_avg = average_weights(user_model_list)
		target_loss_info, target_loss_label = multi_party_member_loss_attack(user_list, target_model,
		                                                                     batch_size=args.target_batch_size,
		                                                                     user_update_list=user_update_list,
		                                                                     get_gradient_func=get_gradient,
		                                                                     user_model_list=[weights_after_avg],
		                                                                     attack_loader_list=attack_loader_list,
		                                                                     max_instance_per_batch=args.max_instance_per_batch)
		np.save('./new_expdata/target_model_after_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
		np.save('./new_expdata/target_model_after_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)
		
		print("server adversary - attack finished")
		# print (naming_str)
		return
	
	if (not server_attacker):
		### if the client is the adversary, we need to change the user_update_list to make it the sum of updates from all other parties
		### here we assume that the last party is the adversary
		sum_user_update_list = []
		for param in user_update_list[0]:
			sum_user_update_list.append(torch.zeros_like(param))
		
		for user_idx in range(len(user_list) - 1):
			for idx, param in enumerate(user_update_list[user_idx]):
				sum_user_update_list[idx] = sum_user_update_list[idx] + param
		
		for param in sum_user_update_list:
			param = param / (len(user_list) - 1)
		
		### for the loss attack, the available model is the model updated with sum of updates from all other parties
		temp_sum_weights = average_weights(user_model_list[:-1])
		
		all_info, all_label = multi_party_member_attack(user_list[:-1], target_model, batch_size=args.target_batch_size,
		                                                user_update_list=[sum_user_update_list],
		                                                get_gradient_func=get_gradient,
		                                                attack_loader_list=attack_loader_list,
		                                                user_total_instance=args.num_step * args.target_batch_size * (len(user_list) - 1),
		                                                max_instance_per_batch=args.max_instance_per_batch, best_layer=best_layer)
		np.save('./new_expdata/all_info_multi_party_member_attack_' + naming_str, all_info)
		np.save('./new_expdata/all_label_multi_party_member_attack_' + naming_str, all_label)
		
		loss_info, loss_label = multi_party_member_loss_attack(user_list[:-1], target_model,
		                                                       batch_size=args.target_batch_size,
		                                                       user_update_list=[sum_user_update_list],
		                                                       get_gradient_func=get_gradient,
		                                                       user_model_list=[temp_sum_weights],
		                                                       attack_loader_list=attack_loader_list,
		                                                       max_instance_per_batch=args.max_instance_per_batch)
		np.save('./new_expdata/loss_info_multi_party_member_attack_' + naming_str, loss_info)
		np.save('./new_expdata/loss_label_multi_party_member_attack_' + naming_str, loss_label)
		
		target_loss_info, target_loss_label = multi_party_member_loss_attack(user_list[:-1], target_model,
		                                                                     batch_size=args.target_batch_size,
		                                                                     user_update_list=[sum_user_update_list],
		                                                                     get_gradient_func=get_gradient,
		                                                                     user_model_list=[ori_model_weight_dict],
		                                                                     attack_loader_list=attack_loader_list,
		                                                                     max_instance_per_batch=args.max_instance_per_batch)
		np.save('./new_expdata/target_model_before_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
		np.save('./new_expdata/target_model_before_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)
		
		weights_after_avg = average_weights(user_model_list)
		target_loss_info, target_loss_label = multi_party_member_loss_attack(user_list[:-1], target_model,
		                                                                     batch_size=args.target_batch_size,
		                                                                     user_update_list=[sum_user_update_list],
		                                                                     get_gradient_func=get_gradient,
		                                                                     user_model_list=[weights_after_avg],
		                                                                     attack_loader_list=attack_loader_list,
		                                                                     max_instance_per_batch=args.max_instance_per_batch)
		np.save('./new_expdata/target_model_after_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
		np.save('./new_expdata/target_model_after_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)
		print("client adversary - attack finished")
		return


def proxy_metric_calculation(targeted_loss):
	targeted_loss = torch.log(torch.exp(-1 * targeted_loss) / (1 - torch.exp(-1 * targeted_loss) + 1e-8))
	targeted_loss, _ = torch.sort(targeted_loss)
	
	'''
	# simple half division
	half = int(len(targeted_loss) / 2)
	avg1 = torch.mean(targeted_loss[:half]).cpu().item()
	avg2 = torch.mean(targeted_loss[half:]).cpu().item()
	std1 = torch.std(targeted_loss[:half]).cpu().item()
	std2 = torch.std(targeted_loss[half:]).cpu().item()
	metric = np.abs(avg1 - avg2) / (std1 + std2)
	'''
	
	# gaussian mixture fitting
	half = int(len(targeted_loss) / 2)
	avg1 = torch.mean(targeted_loss[:half]).cpu().item()
	avg2 = torch.mean(targeted_loss[half:]).cpu().item()
	targeted_loss = targeted_loss.cpu().numpy()
	targeted_loss = np.reshape(targeted_loss, (-1, 1))
	from sklearn.mixture import GaussianMixture
	
	means_init = np.reshape(np.array([avg1, avg2]), (-1, 1))
	gm = GaussianMixture(n_components=2, random_state=0, covariance_type='spherical', init_params='random', means_init=means_init)
	gm.fit(targeted_loss)
	means = gm.means_
	covariances = gm.covariances_
	metric = np.abs(means[0][0] - means[1][0]) / (covariances[0] + covariances[1])
	print(f"means {means},cov {covariances}")
	return metric


def ga_active_attacker_param_search(user_list, target_model, ori_target_model_state_dict, gradient_ascent_weight, global_weights, scheduling=1):
	alpha_list = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
	best_alpha = 0
	target_model.load_state_dict(global_weights)
	loss_metric = torch.mean(get_active_loss(user_list, target_model)).cpu().item()
	_, _, _, _, _, validation_loss_metric = get_train_test_acc(user_list, target_model, return_validation_result=True, print_option=False)
	max_metric = -1
	logs = []
	
	# for utility-preserving attacker, we need to keep the validation loss not increasing and find the min loss metric
	# for utility-ignoring attacker, we just need to find the min loss metric
	# this min loss metric can be replaced with better metric.
	print(f"initial validation loss {validation_loss_metric}, initial loss metric {loss_metric}")
	for alpha in alpha_list:
		param_search_weight_dict = {}
		for (key1, val1), (key2, val2), (key3, val3) in zip(ori_target_model_state_dict.items(), gradient_ascent_weight.items(), global_weights.items()):
			param_search_weight_dict[key2] = val3 + (val2 - val1) * alpha
		# print (torch.isnan(val2))
		target_model.load_state_dict(param_search_weight_dict)
		train_acc, test_acc, valid_acc, _, _, valid_loss = get_train_test_acc(user_list, target_model, return_validation_result=True, print_option=False)
		targeted_loss = get_active_loss(user_list, target_model)
		avg_targeted_loss = torch.mean(targeted_loss).cpu().item()
		
		### we need to transform loss (-logp) to log(p/1-p)
		metric = proxy_metric_calculation(targeted_loss)
		print(f"targeted avg loss {avg_targeted_loss}, validation loss {valid_loss}, train_acc {train_acc}, test_acc {test_acc},simple division metric {metric}")
		logs.append((alpha, avg_targeted_loss, valid_loss, train_acc, test_acc))
		
		if (max_metric < metric):
			best_alpha = alpha
			max_metric = metric
	
	return best_alpha, logs


def gd_active_attacker_param_search(user_list, target_model, ori_target_model_state_dict, gradient_ascent_weight, global_weights, scheduling=1):
	l = 0
	r = 2 / (pow(10, scheduling))  ## this is to make the param search faster for alexnet with scheduling..
	best_alpha = 0
	target_model.load_state_dict(global_weights)
	loss_metric = torch.mean(get_active_loss(user_list, target_model)).cpu().item()
	_, _, _, _, _, validation_loss_metric = get_train_test_acc(user_list, target_model, return_validation_result=True, print_option=False)
	
	logs = []
	
	# for utility-preserving attacker, we need to keep the validation loss not increasing and find the min loss metric
	# for utility-ignoring attacker, we just need to find the min loss metric
	# this min loss metric can be replaced with better metric.
	print(f"initial validation loss {validation_loss_metric}, initial loss metric {loss_metric}")
	while (l < r):
		alpha = (l + r) / 2
		param_search_weight_dict = {}
		for (key1, val1), (key2, val2), (key3, val3) in zip(ori_target_model_state_dict.items(), gradient_ascent_weight.items(), global_weights.items()):
			param_search_weight_dict[key2] = val3 + (val2 - val1) * alpha
		target_model.load_state_dict(param_search_weight_dict)
		train_acc, test_acc, valid_acc, _, _, valid_loss = get_train_test_acc(user_list, target_model, return_validation_result=True, print_option=False)
		targeted_loss = get_active_loss(user_list, target_model)
		avg_targeted_loss = torch.mean(targeted_loss).cpu().item()
		metric = proxy_metric_calculation(targeted_loss)
		print(f"targeted avg loss {avg_targeted_loss}, validation loss {valid_loss}, train_acc {train_acc}, test_acc {test_acc},simple division metric {metric}")
		logs.append((alpha, avg_targeted_loss, valid_loss, train_acc, test_acc))
		
		# this is utility preserving implementation.
		'''
		if (valid_loss > validation_loss_metric) or (avg_targeted_loss>loss_metric):
			r = alpha - 0.05
		else:
			l = alpha + 0.05
			best_alpha = alpha
			loss_metric = avg_targeted_loss
		'''
		# this is utility ignoring implementation.
		if ((avg_targeted_loss > loss_metric)):
			r = alpha - 0.05
		else:
			l = alpha + 0.05
			best_alpha = alpha
			loss_metric = avg_targeted_loss
	
	print(f"alpha {best_alpha}, loss metric {loss_metric}")
	# print (f"alpha {best_alpha}, validation loss metric {validation_loss_metric}")
	
	return best_alpha, logs


def train_models(user_list, target_model, learning_rate, decay, epochs, class_weights=None): ### this function needs to be reimplemented.
	# print (target_model)
	num_users = len(user_list)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_model.type(torch.FloatTensor)
	target_model.to(device)
	momentum = 0.9
	# beta1 = 0.9
	# beta2 = 0.999
	
	# active_attack_multiplier_initial_val = -0.1
	# active_attack_multiplier_step_size = -0.025
	# active_attack_multiplier = args.active_attacker_lr_multiplier
	
	grad_norm_results = []
	all_acc_results = []
	all_loss_results = []
	all_active_results = []
	lr_scheduling_count = 0
	
	### notice that in pytorch, momentum etc. is bound with optimizer, so we need to initialize the optimizer/model for each user
	for user_idx in range(num_users):
		user_list[user_idx].model = copy.deepcopy(target_model)
		if (args.dataset == 'purchase' or args.dataset == 'texas'):
			# this_optim = torch.optim.SGD(user_list[user_idx].model.parameters(), lr=learning_rate, momentum=momentum,weight_decay=decay)
			this_optim = torch.optim.Adam(user_list[user_idx].model.parameters(), lr=learning_rate, weight_decay=decay)
		
		else:
			this_optim = torch.optim.SGD(user_list[user_idx].model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
		### using sgd actually leads to 3x memory consumption.. this is very tricky
		
		user_list[user_idx].optim = this_optim
		
		# print (this_optim.state_dict())
		
		if (args.dpsgd):
			### adding dp components
			privacy_engine = PrivacyEngine(
				module=user_list[user_idx].model,
				optimizer=user_list[user_idx].optim,
				data_loader=user_list[user_idx].train_data_loader,
				# sample_rate=(args.target_batch_size / args.target_data_size),
				# alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),  ### params for renyi dp
				noise_multiplier=args.noise_scale,  ### sigma
				max_grad_norm=args.grad_norm)  ### this is from dp-sgd paper)
	# privacy_engine.attach(user_list[user_idx].optim)
	
	if (args.active_attacker):
		attacker_model_copy = copy.deepcopy(target_model)
		'''
		if (args.dataset == 'purchase' or args.dataset == 'texas'):
				# attacker_optim = torch.optim.SGD(attacker_model_copy.parameters(), lr=learning_rate, momentum=momentum,weight_decay=decay)
			attacker_optim = torch.optim.Adam(attacker_model_copy.parameters(), lr=learning_rate, weight_decay=decay)
		else:
			attacker_optim = torch.optim.SGD(attacker_model_copy.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
		'''
	
	### start training
	for epoch in tqdm(range(epochs)):
		# get_live_objects()
		ori_target_model_state_dict = target_model.state_dict()  ### should we copy this?
		## LR schedule
		if (epoch in args.schedule):
			learning_rate = learning_rate / 10
			print("new learning rate = %f" % (learning_rate))
			### set up new learning rate for each user
			for user_idx in range(num_users):
				for param_group in user_list[user_idx].optim.param_groups:
					param_group['lr'] = learning_rate
			
			lr_scheduling_count += 1
		
		## active attacker gradient ascent
		if (args.active_attacker):
			attacker_model_copy.load_state_dict(ori_target_model_state_dict)
			
			if (args.dataset == 'purchase' or args.dataset == 'texas'):
				# attacker_optim = torch.optim.SGD(attacker_model_copy.parameters(), lr=learning_rate, momentum=momentum,weight_decay=decay)
				attacker_optim = torch.optim.Adam(attacker_model_copy.parameters(), lr=learning_rate, weight_decay=decay)
			else:
				attacker_optim = torch.optim.SGD(attacker_model_copy.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
			
			# gradient implementation
			# gradient_ascent_gradient, active_magnitude,active_loss = active_attacker_gradient_ascent(attacker_model_copy,attacker_optim,user_list,
			# local_epochs=args.local_epoch,batch_size=50,client_adversary=args.client_adversary,
			# lr_multiplier=args.active_attacker_lr_multiplier,param_search=args.active_attack_param_search)
			
			# optim implementation
			gradient_ascent_weight, active_loss = active_attacker_gradient_ascent(attacker_model_copy, attacker_optim, user_list,
			                                                                      ga_adversary=args.active_attacker_lr_multiplier, local_epochs=1)
			#### calculate magnitude
			active_magnitude = calculate_param_change_magnitude(ori_target_model_state_dict, gradient_ascent_weight)
		
		#### we need to know the # of steps, # of training set size to perform fed sgd
		comm_round_per_epoch = int(args.target_data_size / (args.target_batch_size * args.num_step))
		print(f"comm round per epoch{comm_round_per_epoch}")
		
		for comm_round_idx in range(comm_round_per_epoch):
			local_weights = []
			attack_loader_list = []
			for idx in range(len(user_list)):
				# create a new dataloader
				train_data_loader = user_list[idx].create_new_train_data_loader(batch_size=args.target_batch_size)
				### defense
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
					# print ("no defense applied")
					ban_list = update_weights(current_model_weights=ori_target_model_state_dict,
					                          model=user_list[idx].model,
					                          optimizer=user_list[idx].optim, train_loader=train_data_loader,
					                          local_epochs=args.local_epochs, mixup=args.mixup,
					                          num_step=args.num_step,
					                          class_weights=user_list[idx].class_weight, unequal=args.unequal)
					
					attack_loader_list.append(user_list[idx].create_batch_attack_data_loader(ban_list))
					user_list[idx].update_ban_list(ban_list)
					torch.cuda.empty_cache()
				
				local_weights.append((user_list[idx].model.state_dict()))
			
			print(f" TEST:num step{args.num_step}, epoch{epoch}, attacker_epoch_gap {args.attacker_epoch_gap},comm_round {comm_round_idx}, "
			      f"comm_round_per_epoch {comm_round_per_epoch}")
			
			### batch level attack
			if (args.num_step == 1 and epoch % args.attacker_epoch_gap == 0 and epoch > 0):
				num_batches = 1
				print(f"epoch {epoch}, comm_round_idx {comm_round_idx}, num step {args.num_step}")
				user_update_list = get_user_update_list(ori_target_model_state_dict, local_weights,
				                                        learning_rate=learning_rate, num_batches=num_batches)
				# _, _ = get_train_test_acc(user_list, target_model)
				local_model = copy.deepcopy(target_model)
				run_multi_party_attacks(user_list, local_model, epoch, user_update_list,
				                        user_model_list=local_weights,
				                        server_attacker=1 - args.client_adversary,
				                        ori_model_weight_dict=ori_target_model_state_dict,
				                        attack_loader_list=attack_loader_list,
				                        comm_round_idx=comm_round_idx, best_layer=args.best_layer)
				# print (f"finish batch level attacks")
				del user_update_list, local_model
			
			### epoch level attack
			elif (epoch % args.attacker_epoch_gap == 0 and comm_round_idx + 1 == comm_round_per_epoch):  # and epoch > 0 NEED TO RESTORE
				### this atack requires # of steps = # of batches
				num_batches = int(args.target_data_size / args.target_batch_size)
				if (args.num_step != num_batches):
					print("epoch attack: num steps not equal to num batches")
					exit(0)
				
				print(f"EPOCH LEVEL ATTACK:epoch {epoch}, comm_round_idx {comm_round_idx}, num step {args.num_step}")
				
				user_update_list = get_user_update_list(ori_target_model_state_dict, local_weights, learning_rate=learning_rate, num_batches=num_batches)
				
				local_model = copy.deepcopy(target_model)
				run_multi_party_attacks(user_list, local_model, epoch, user_update_list,
				                        user_model_list=local_weights,
				                        server_attacker=1 - args.client_adversary,
				                        ori_model_weight_dict=ori_target_model_state_dict,
				                        attack_loader_list=[],
				                        comm_round_idx=comm_round_idx, best_layer=args.best_layer)
				print(f"finish attacks for {epoch} epochs")
				del user_update_list, local_model
			
			### update global weights
			global_weights = average_weights(local_weights)
			del local_weights
			
			if (args.active_attacker):
				# active_attack_multiplier = -0.05 * (normal_magnitude*100/ (learning_rate*args.user_number*args.target_data_size)) / (active_magnitude * 50 / (2*args.eval_data_size*args.user_number))
				# active_attack_multiplier = args.active_attacker_lr_multiplier*-1
				# if (args.active_attacker_lr_multiplier > 0):
				#	ga_constant = -1
				#	active_attack_multiplier = active_attack_multiplier * ga_constant
				
				normal_magnitude = calculate_param_change_magnitude(ori_target_model_state_dict, global_weights)
				normal_loss = get_train_loss(user_list, copy.deepcopy(target_model), ori_target_model_state_dict)
				avg_train_norm = (normal_magnitude.cpu().item() * 100) / (learning_rate * args.user_number * args.target_data_size)
				avg_attack_norm = active_magnitude.cpu().item() * 50 / (args.user_number * 2 * args.eval_data_size)
				
				# if (args.active_attacker_lr_multiplier>0): ### GA case
				#	active_attack_multiplier = 1 * normal_loss / active_loss *avg_train_norm / avg_attack_norm ### 1 is required for GA.
				# else:### GD case
				#	active_attack_multiplier = -1 * active_loss / active_loss*avg_train_norm / avg_attack_norm  ### -1 is required for GD. We add gradient to model.
				# print(f"ratio {active_attack_multiplier} ")
				
				### we need to perform a param search here.
				
				if (args.active_attacker_lr_multiplier < 0):
					alpha, log = gd_active_attacker_param_search(user_list, copy.deepcopy(target_model), ori_target_model_state_dict, gradient_ascent_weight, global_weights,
					                                             scheduling=lr_scheduling_count)
				else:
					alpha, log = ga_active_attacker_param_search(user_list, copy.deepcopy(target_model), ori_target_model_state_dict, gradient_ascent_weight, global_weights,
					                                             scheduling=lr_scheduling_count)
				
				if (alpha < 0):
					alpha = 0
				active_attack_multiplier = alpha
				all_active_results.append(np.array(log))
				print(f"this epoch {epoch}, multiplier {active_attack_multiplier}")
				# all_active_results.append((alpha_list,this_epoch_valid_loss))
				
				new_weight_dict = {}
				# optim implementation
				for (key1, val1), (key2, val2), (key3, val3) in zip(ori_target_model_state_dict.items(), gradient_ascent_weight.items(), global_weights.items()):
					if (alpha != 0):
						new_weight_dict[key2] = val3 + (val2 - val1) * active_attack_multiplier
					else:
						new_weight_dict[key2] = val3  ## this is to deal with numerical problems when gradient ascent weight is NaN (only for GA)
				
				# gradient implementation
				# for (key1,val1),(key2,val2),(key3,val3) in zip(ori_target_model_state_dict.items(),gradient_ascent_gradient.items(),global_weights.items()):
				# active_attack_multiplier = torch.nan_to_num(torch.norm(torch.flatten(val1-val3),p=1)/torch.norm(torch.flatten(val2),p=1))
				# new_weight_dict[key2] = val3 + val2*active_attack_multiplier
				
				# optim implementation
				cos_result = calculate_param_cosine_similarity(gradient_ascent_weight, global_weights, ori_target_model_state_dict)
				
				# gradient implementation
				# cos_result = -1 * calculate_param_cosine_similarity(copy.deepcopy(gradient_ascent_gradient),copy.deepcopy(global_weights),copy.deepcopy(ori_target_model_state_dict))
				
				grad_norm_results.append((avg_train_norm, avg_attack_norm, normal_loss, active_loss, cos_result))
				print("grad norm results", grad_norm_results[-1])
				target_model.load_state_dict(new_weight_dict)
			
			# print(get_train_test_acc(user_list, target_model))
			
			else:
				target_model.load_state_dict(global_weights)
			
			if (args.track_loss):
				train_acc, test_acc, val_acc, train_loss, test_loss, val_loss = get_train_test_acc(user_list, target_model, print_option=True, return_loss=True,
				                                                                                   return_validation_result=True)
				all_loss_results.append((train_loss, val_loss, test_loss))
				all_acc_results.append((train_acc, test_acc, val_acc))
	
	'''
	grad_norm_str = './new_expdata/grad_norm_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
		args.target_data_size) + '_' + str(args.model_name) + '.npy'
	grad_norm_results = np.array(grad_norm_results)
	np.save(grad_norm_str, grad_norm_results)
	'''
	all_active_str = './new_expdata/all_active_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
		args.target_data_size) + '_' + str(args.model_name) + '.npy'
	all_acc_results = np.array(all_acc_results)
	np.save(all_active_str, all_acc_results)
	
	if (args.track_loss):
		loss_str = './new_expdata/track_loss_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
			args.target_data_size) + '_' + str(args.model_name) + '.npy'
		all_loss_results = np.array(all_loss_results)
		np.save(loss_str, all_loss_results)
		
		acc_str = './new_expdata/track_acc_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
			args.target_data_size) + '_' + str(args.model_name) + '.npy'
		all_acc_results = np.array(all_acc_results)
		np.save(acc_str, all_acc_results)
	
	train_acc, test_acc = get_train_test_acc(user_list, target_model)
	print(f"train acc {train_acc},test acc {test_acc}")
	return target_model, train_acc, test_acc


def get_gradient(target_model, data, label):  ### using gpu 1
	import time
	device = torch.device("cuda", 1)
	criterion = nn.CrossEntropyLoss().to(device)
	target_model = target_model.to(device)
	local_model = copy.deepcopy(target_model)
	local_model.zero_grad()
	
	# print (data.size(),label.size())
	# print (type(target_model))
	'''
	grad_list = []
	### one by one implementation
	start_time = time.time()
	for this_data,this_label in zip(data,label):
		this_data = this_data.to(device)
		this_label = this_label.to(device)
		this_data = torch.unsqueeze(this_data,dim=0)
		this_label = torch.unsqueeze(this_label,dim=0)
		prediction = local_model(this_data)
		loss = criterion(prediction, this_label)
		loss.backward()
		this_grad = copy.deepcopy([param.grad.detach() for param in local_model.parameters()])
		grad_list.append(this_grad)
		local_model.zero_grad()

	print (len(grad_list))
	end_time = time.time()
	print (f"one by one implementation takes {end_time-start_time}") ## 0.1869s
	# return grad_list
	'''
	
	### parallel implementation based on opacus
	start_time = time.time()
	from opacus import GradSampleModule
	local_model = GradSampleModule(local_model)
	data = data.to(device)
	label = label.to(device)
	prediction = local_model(data)
	loss = criterion(prediction, label)
	loss.backward()
	grad_sample = copy.deepcopy([param.grad_sample.detach() for param in local_model.parameters()])
	end_time = time.time()
	local_model.zero_grad()  ### ?
	# print (f"batch implementation takes {end_time-start_time}") ## 0.0090s
	### we also need to check if gradient is calculated correctly..
	# sum1 = [this_grad[0] for this_grad in grad_list]
	# sum2 = grad_sample[0]
	# sum1 = torch.stack(sum1)
	# print (sum1.size(),sum2.size())
	# avg1 = torch.mean(sum1,dim=0)
	# avg2 = torch.mean(sum2,dim=0)
	# print (avg1.size(),avg2.size())
	# print (torch.equal(avg1,avg2))
	# print (torch.allclose(avg1,avg2))
	# print (torch.nonzero(avg1-avg2))
	# print ((avg1-avg2)[0,:20])
	# print (torch.sum(avg1-avg2))
	return grad_sample


def attack_experiment():
	import warnings
	warnings.filterwarnings("ignore")
	np.random.seed(seed=12345)
	torch.set_printoptions(threshold=5000, edgeitems=20)
	
	### dataset && membership inference data
	membership_attack_number = args.membership_attack_number
	target_dataset = dataset(dataset_name=args.dataset, gpu=args.gpu,
	                         membership_attack_number=membership_attack_number,
	                         cutout=args.cutout, n_holes=args.n_holes, length=args.length)
	
	# print ("Data Check")
	# print (np.amax(target_dataset.train_data[0]),np.amin(target_dataset.train_data[0]))
	print(f"class distribution:{np.bincount(target_dataset.label)}")
	
	user_list = [User(dataset=args.dataset, model_name=args.target_model, id=i) for i in range(args.user_number)]
	
	if (args.noniid):
		non_idd_assign_part_dataset(target_dataset, user_list)
	else:
		assign_part_dataset(target_dataset, user_list)
	
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
	elif ('celeba' in target_dataset.dataset_name):
		args.num_classes = 2
		target_model = celeba_model()
	else:
		target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
		if (target_dataset.dataset_name == 'texas'):
			args.num_classes = 100
		if (target_dataset.dataset_name == 'purchase'):
			args.num_classes = 100
	
	# print (target_model)
	pytorch_total_params = sum(p.numel() for p in target_model.parameters())
	print(f"total params {pytorch_total_params}")
	
	target_model, train_acc, test_acc = train_models(user_list, target_model, learning_rate=args.target_learning_rate,
	                                                 decay=args.target_l2_ratio, epochs=args.target_epochs)
	
	### doing final attacks evaluation
	
	name_string_prefix = '/home/lijiacheng/whiteboxmi/new_expdata/'
	name_string_mid_str = get_naming_mid_str()
	print(name_string_mid_str)
	result_file = name_string_prefix + name_string_mid_str + str(args.dataset) + '_' + str(args.model_name) + '_' + str(
		args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(args.best_layer) + '_' + 'result_file.txt'
	f = open(result_file, 'w')
	
	### diff MI attack
	### for this attack, we have no AUC score to report, only accuracy or F1-score. (all_info should contain m_pred and m_true)
	
	# _,_ = diffmi_attack(user_list,target_model,batch_size=20,output_file=f)
	# np.save('./new_expdata/diffmi_attack'+get_naming_mid_str()+'_m_true_20.npy',m_true)
	# np.save('./new_expdata/diffmi_attack'+get_naming_mid_str()+'_m_pred_20.npy',m_pred)
	# print ("diffmi attack batch_size 20 finished")
	# _,_ = diffmi_attack(user_list,target_model,batch_size=2*args.target_batch_size,output_file=f)
	# np.save('./new_expdata/diffmi_attack'+get_naming_mid_str()+'_m_true_200.npy',m_true)
	# np.save('./new_expdata/diffmi_attack'+get_naming_mid_str()+'_m_pred_200.npy',m_pred)
	# print ("diffmi attack batch_size 200 finished")
	
	### nasr fed attack
	### for this attack, we can report AUC score since the output is a probability.
	# print ("nasr attack finished")
	
	### blackbox attack
	run_blackbox_attacks(user_list, target_model, num_classes=len(np.unique(target_dataset.train_label)), output_file=f)
	
	### whitebox attack
	from run_attacks import all_analysis
	# args.target_epochs = 400
	epochs = ((np.arange(int(args.target_epochs / args.attacker_epoch_gap) - 1) + 1) * args.attacker_epoch_gap)
	
	# print (epochs)
	
	num_layers_dict = {'alexnet': 12, 'densenet_cifar': 299, 'purchase': 8, 'texas': 8, 'fashion_mnist': 8,
	                   'retina': 12}
	num_layers = num_layers_dict[args.model_name]
	# if (args.dataset == 'purchase' or args.dataset == 'texas'):
	#    num_layers = 8
	
	binary_acc = all_analysis(epochs, name_string_prefix, name_string_mid_str, args.dataset, args.model_name,
	                          int(args.target_data_size), int(args.eval_data_size), f=f, num_layers=num_layers,
	                          num_user=args.user_number,
	                          client_adversary=args.client_adversary, best_layer=args.best_layer,
	                          comm_round_list=np.arange(
		                          int(args.target_data_size / (args.num_step * args.target_batch_size))))
	
	f.close()
	f = open(result_file, 'r')
	for lines in f.readlines():
		print(lines)
	f.close()
	print(f"binary_acc{binary_acc}")
	# return train_acc,test_acc,binary_acc
	
	return train_acc, test_acc, binary_acc


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
	parser.add_argument('--max_instance_per_batch', type=int, default=200)
	parser.add_argument('--active_attack_param_search', type=int, default=0)
	
	parser.add_argument('--track_loss', type=int, default=0)
	parser.add_argument('--random_seed', type=int, default=123)
	args = parser.parse_args()
	print(vars(args))
	
	if (args.num_step == 1):
		args.eval_data_size = args.target_data_size
	
	if (args.dataset == 'texas'):
		args.max_instance_per_batch = 50
	if (args.model_name == 'densenet_cifar'):
		args.max_instance_per_batch = 20  ### maybe 10. not sure.
	
	args.max_instance_per_batch = min(args.max_instance_per_batch, args.eval_data_size)
	
	# random_seed_list = [1,123,1234,123321,12345,123456]
	# random_seed_list = [0,1,123]
	random_seed_list = [args.random_seed]
	# if ('celeba' in args.dataset):
	# args.attacker_epoch_gap = 300 ### get rid of wb attack
	# mislabeling_target_label_list = [0,1,2,3,4,5,6,7,8,9]
	
	# lr_list = [100,10,1,0.1,0.01,0.001,0.0001,0.00001]
	
	acc_list = []
	for this_seed in random_seed_list:
		import torch
		
		torch.manual_seed(this_seed)
		import numpy as np
		
		np.random.seed(this_seed)
		import sklearn
		
		sklearn.utils.check_random_state(this_seed)
		
		acc_list.append(attack_experiment())
	
	print(vars(args))
	acc_list = np.array(acc_list)
	print(acc_list)
	print(f"train acc - avg:{np.average(acc_list[:, 0])},var:{np.var(acc_list[:, 0])}")
	print(f"test acc - avg:{np.average(acc_list[:, 1])},var:{np.var(acc_list[:, 1])}")
	print(f"binary membership acc - avg:{np.average(acc_list[:, 2])},var:{np.var(acc_list[:, 2])}")

# print ("train / test")
# print (train_test)
