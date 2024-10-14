import model
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
from opacus import PrivacyEngine
from model_utils import *
from model_utils import _batchnorm_to_groupnorm_new
from model_utils import get_train_loss
from opacus.validators import ModuleValidator
from worst_case_metric import find_vulnerable_points
from sklearn.metrics import roc_auc_score
import math


# from line_profiler import LineProfiler
# profiler = LineProfiler()

def get_naming_mid_str():
	name_string_mid_str = str(args.dataset) + '_' + str(args.model_name) + '_' + str(args.user_number) + '_' + str(args.num_step) \
						  + '_' + str(args.dpsgd) + '_' + str(args.noise_scale) + '_' + str(args.grad_norm) \
						  + '_' + str(args.mmd) + '_' + str(args.mmd_loss_lambda) + '_' + str(args.mixup) \
						  + '_' + str(args.std_loss_lambda) + '_' + str(args.vul_metric) + '_' + str(args.grad_addback_weight) + '_' \
						  + str(args.cross_loss_diff_lambda) + '_' + str(args.shadow_model_number) + '_'
	return name_string_mid_str


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
	elif (dataset.dataset_name == 'purchase' or dataset.dataset_name == 'texas'):
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


def assign_part_dataset(dataset, user_list=[]):
	# hard medical datasets
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
	elif (dataset.dataset_name == 'purchase' or dataset.dataset_name == 'texas'):
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
	
	## we need to first divide the training set into train + validation
	
	num_users = len(user_list)
	training_partition = np.random.choice(len(dataset.train_label), num_users * args.target_data_size, replace=False)
	validation_partition = np.setdiff1d(np.arange(len(dataset.train_label)), training_partition)
	
	### these two loaders are for MMD loss
	overall_train = part_pytorch_dataset(dataset.train_data[training_partition], dataset.train_label[training_partition], train=False, transform=transform_test,
										 target_transform=target_transform)
	overall_train_loader_in_order = torch.utils.data.DataLoader(overall_train, batch_size=args.target_batch_size,
																shuffle=False, num_workers=1)
	overall_validation = part_pytorch_dataset(dataset.train_data[validation_partition], dataset.train_label[validation_partition], train=False, transform=transform_test,
											  target_transform=target_transform)
	overall_validation_loader = torch.utils.data.DataLoader(overall_validation, batch_size=args.target_batch_size,
															shuffle=False, num_workers=1)
	
	## then we divide the training set into # of users' partitions
	index_left = training_partition  # the # of data left for generating new split of training data
	assigned_index = []
	for i in range(num_users):
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
		this_user.train_index = this_user_train_index
		
		this_user.class_weights = np.ones((len(np.unique(dataset.train_label)))) * training_set_size / (
			len(np.unique(dataset.train_label)) * (np.bincount(this_user.train_label, minlength=len(np.unique(dataset.train_label))) + 10))
		
		this_user.test_data = dataset.test_data
		this_user.test_label = dataset.test_label
		assigned_index.append(this_user_train_index)
		
		train = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=True, transform=transform_train,
									 target_transform=target_transform)
		test = part_pytorch_dataset(this_user.test_data, this_user.test_label, train=False, transform=transform_test,
									target_transform=target_transform)
		train_eval = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=False, transform=transform_test,
										  target_transform=target_transform)
		this_user.train_dataset = train
		this_user.test_dataset = test
		this_user.train_eval_dataset = train_eval
		this_user.train_data_loader = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,
																  shuffle=True, num_workers=1)
		this_user.train_eval_data_loader = torch.utils.data.DataLoader(train_eval, batch_size=args.target_batch_size,
																	   shuffle=False, num_workers=1)
		this_user.test_data_loader = torch.utils.data.DataLoader(test, batch_size=args.target_batch_size, shuffle=False,
																 num_workers=1)
	
	### assign validation set
	validation_data = dataset.train_data[validation_partition]
	validation_label = dataset.train_label[validation_partition]
	
	for user_idx in range(num_users):
		this_user = user_list[user_idx]
		this_user.eval_validation_data = validation_data
		this_user.eval_validation_label = validation_label
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
					this_user.starting_index.append(j)
					break
		
		this_user.validation_dataset = part_pytorch_dataset(validation_data, validation_label, train=False,
															transform=transform_test,
															target_transform=target_transform)
		this_user.validation_data_loader = torch.utils.data.DataLoader(this_user.validation_dataset,
																	   batch_size=args.target_batch_size, shuffle=False,
																	   num_workers=1)
	
	return training_partition, validation_partition, overall_train_loader_in_order, overall_validation_loader, overall_validation


def simplex_sampling(user_list, target_model, repeat_count=100):
	if (args.user_number == 1):
		repeat_count = 1
	all_train_acc = []
	all_test_acc = []
	model_weight_list = [user_list[i].model.state_dict() for i in range(len(user_list))]
	for _ in range(repeat_count):
		this_weight = simplex_uniform_sampling(num=len(user_list))
		this_model_weight = average_weights(model_weight_list, this_weight)
		target_model.load_state_dict(this_model_weight)
		train_acc, test_acc = get_train_test_acc(user_list, target_model)
		all_train_acc.append(train_acc)
		all_test_acc.append(test_acc)
	
	all_train_acc = np.array(all_train_acc)
	all_test_acc = np.array(all_test_acc)
	print(f"test acc avg {np.average(all_test_acc)},test acc std {np.std(all_test_acc)}")
	return all_train_acc, all_test_acc


def train_models(target_dataset, user_list, target_model, learning_rate, decay, epochs, mmd_validation_loader=None, mmd_train_loader=None, validation_set=None, training_index=None,
				 validation_index=None):
	num_users = len(user_list)
	num_classes = len(np.unique(target_dataset.train_label))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_model.to(device)
	momentum = 0.9  # should be 0.9
	#simplex_result = []
	validation_acc_list = []
	reduce_point = []
	last_reduce_epoch = -10
	this_num_all_val = []
	prev_avg = 0
	
	if (args.dataset == 'purchase' or args.dataset == 'texas'):
		# target_model_optim = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=decay)
		target_model_optim = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
	
	else:
		target_model_optim = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
	
	### notice that in pytorch, momentum etc. is bound with optimizer, so we need to initialize the optimizer/model for each user
	for user_idx in range(num_users):
		user_list[user_idx].model = copy.deepcopy(target_model)
		if (args.dataset == 'purchase' or args.dataset == 'texas'):  #
			# this_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, user_list[user_idx].model.parameters()), lr=learning_rate, weight_decay=decay)
			this_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, user_list[user_idx].model.parameters()), lr=learning_rate, momentum=momentum, weight_decay=decay)
		else:
			this_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, user_list[user_idx].model.parameters()), lr=learning_rate, momentum=momentum, weight_decay=decay)
		
		user_list[user_idx].optim = this_optim
		
	### start training
	for epoch in tqdm(range(epochs)):
		# print (f"{epoch} epoch")
		if (args.repartition):
			repartition_dataset(args.dataset, user_list)
		ori_target_model_state_dict = target_model.state_dict()
		if (epoch in args.schedule):
			learning_rate = learning_rate / 10
			print("new learning rate = %f" % (learning_rate))
			for user_idx in range(num_users):
				for param_group in user_list[user_idx].optim.param_groups:
					param_group['lr'] = learning_rate
		
		comm_round_per_epoch = int(args.target_data_size / (args.target_batch_size * args.num_step))
		
		for _ in range(comm_round_per_epoch):
			local_weights = []
			for idx in range(len(user_list)):
				# create a new dataloader
				train_data_loader = user_list[idx].create_new_train_data_loader(batch_size=args.target_batch_size)
				
				new_state_dict = update_weights(current_model_weights=ori_target_model_state_dict,
												model=user_list[idx].model,
												optimizer=user_list[idx].optim, train_loader=train_data_loader,
												local_epochs=args.local_epochs, mixup=args.mixup,
												selfswap=args.self_swap,
												class_weights=user_list[idx].class_weights,
												std_loss_lambda=args.std_loss_lambda, dataset_name=target_dataset.dataset_name)  ### model name is specifically for inception..
				local_weights.append((new_state_dict))
			
			global_weights = average_weights(local_weights)
			target_model.load_state_dict(global_weights)

			if (epoch!=epochs-1 and args.cross_loss_diff_lambda!=0):
				weight_after_cross_loss_diff = average_weights(update_weights_cross_loss_diff(user_list,loss_lambda=args.cross_loss_diff_lambda,loss_name=args.cross_loss))
				target_model.load_state_dict(weight_after_cross_loss_diff)
				print ('cross diff done')
				
			### get validation accuracy, if no improvement, reduce the num of users
			train_acc, test_acc,validation_acc,_,_,_ = get_train_test_acc(user_list, target_model,return_validation_result=True)
			print (f"epoch {epoch}, validation acc {validation_acc}, num user {len(user_list)}")
			
			this_num_all_val.append(validation_acc)
			most_recent_10_val = np.average(np.array(this_num_all_val)[-10:])
			if (most_recent_10_val >validation_acc and last_reduce_epoch+10<epoch):
				## this is the point where we need to change the number of users
				half = int(len(this_num_all_val)*0.5)
				first_avg = np.average(np.array(this_num_all_val[:half]))
				last_avg = np.average(np.array(this_num_all_val[half:]))
				if (first_avg > last_avg or len(user_list) == 2): ##
					## this is the time to increase
					user_list = increase_user_num_by_one(user_list)
					reduce_point.append(-1)
					last_reduce_epoch = epoch
					this_num_all_val = [validation_acc]
				else:
					### or reduce
					user_list = reduce_user_num_by_one(user_list)
					reduce_point.append(1)
					last_reduce_epoch = epoch
					this_num_all_val = [validation_acc]
			else:
				reduce_point.append(0)
			validation_acc_list.append(validation_acc)
	
	file_name = './find_best_user_' + get_naming_mid_str() + '.npz'
	print (f"log save file name {file_name}")
	np.savez(file_name, np.array(validation_acc_list), np.array(reduce_point))
			
	return target_model, train_acc, test_acc


def increase_user_num_by_one(user_list):
	dataset = args.dataset
	### change the training set
	if (dataset == 'purchase' or dataset == 'texas'):
		transform_train = None
		transform_test = None
		target_transform = None
	# cifar 10 / cifar100
	elif (dataset == 'cifar10' or dataset == 'cifar100'):
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
	
	### gather all current training data and label
	all_train_data = [user_list[i].train_data for i in range(len(user_list))]
	all_train_label = [user_list[i].train_label for i in range(len(user_list))]
	all_train_data = np.vstack(all_train_data)
	# print (all_train_data.shape)
	all_train_label = np.concatenate(all_train_label).flatten()
	# print (all_train_label.shape)
	user_list = user_list + [copy.deepcopy(user_list[-1])]
	### reassigning data to each user
	reassign_index = np.random.choice(np.arange(len(all_train_label)), len(all_train_label), replace=False)
	cnt = 0
	target_data_size = int(len(reassign_index)/len(user_list))
	for i in range(len(user_list)):
		this_user = user_list[i]
		this_user_index = reassign_index[cnt: cnt + target_data_size]
		cnt += target_data_size
		this_user.train_data = all_train_data[this_user_index]
		this_user.train_label = all_train_label[this_user_index]
		this_user.available_list = np.arange(len(this_user.train_data))
		train = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=True, transform=transform_train,
									 target_transform=target_transform)
		this_user.train_dataset = train
		this_user.train_data_loader = torch.utils.data.DataLoader(train, batch_size=100,
																  shuffle=True, num_workers=1)
	return user_list


def reduce_user_num_by_one(user_list):
	dataset = args.dataset
	### change the training set
	if (dataset == 'purchase' or dataset == 'texas'):
		transform_train = None
		transform_test = None
		target_transform = None
	# cifar 10 / cifar100
	elif (dataset == 'cifar10' or dataset == 'cifar100'):
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
	
	### gather all current training data and label
	all_train_data = [user_list[i].train_data for i in range(len(user_list))]
	all_train_label = [user_list[i].train_label for i in range(len(user_list))]
	all_train_data = np.vstack(all_train_data)
	# print (all_train_data.shape)
	all_train_label = np.concatenate(all_train_label).flatten()
	# print (all_train_label.shape)
	#user_list = user_list + [copy.deepcopy(user_list[-1])]
	user_list = user_list[:-1]
	### reassigning data to each user
	reassign_index = np.random.choice(np.arange(len(all_train_label)), len(all_train_label), replace=False)
	cnt = 0
	target_data_size = int(len(reassign_index) / len(user_list))
	for i in range(len(user_list)):
		this_user = user_list[i]
		this_user_index = reassign_index[cnt: cnt + target_data_size]
		cnt += target_data_size
		this_user.train_data = all_train_data[this_user_index]
		this_user.train_label = all_train_label[this_user_index]
		this_user.available_list = np.arange(len(this_user.train_data))
		train = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=True, transform=transform_train,
									 target_transform=target_transform)
		this_user.train_dataset = train
		this_user.train_data_loader = torch.utils.data.DataLoader(train, batch_size=100,
																  shuffle=True, num_workers=1)
	return user_list


def get_all_prob(model, dataset, training_index, validation_index):
	## create training / testing data loader / validation loader
	_, transform_test, target_transform = get_transformation(dataset)
	overall_train = part_pytorch_dataset(dataset.train_data[training_index], dataset.train_label[training_index], train=False, transform=transform_test,
										 target_transform=target_transform)
	train_loader = torch.utils.data.DataLoader(overall_train, batch_size=args.target_batch_size, shuffle=False, num_workers=1)
	
	overall_test = part_pytorch_dataset(dataset.test_data, dataset.test_label, train=False, transform=transform_test,
										target_transform=target_transform)
	test_loader = torch.utils.data.DataLoader(overall_test, batch_size=args.target_batch_size, shuffle=False, num_workers=1)
	
	overall_validation = part_pytorch_dataset(dataset.train_data[validation_index], dataset.train_label[validation_index], train=False, transform=transform_test,
											  target_transform=target_transform)
	validation_loader = torch.utils.data.DataLoader(overall_validation, batch_size=args.target_batch_size, shuffle=False, num_workers=1)
	## gather all loss
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(reduction='none').to(device)
	all_train_prediction = []
	all_train_loss = []
	for images, labels, _ in train_loader:
		model.zero_grad()
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		pred = F.softmax(outputs, dim=1).detach()
		# print (outputs.shape)
		# print (pred[0],torch.sum(pred[0]))
		# loss = criterion(outputs, labels).detach()
		## instead of using loss, we use prob
		this_batch_prob = torch.tensor([pred[i][labels[i]] for i in range(len(labels))])
		# print (this_batch_prob.shape)
		all_train_loss.append(this_batch_prob)
		all_train_prediction.append(pred)
	all_test_prediction = []
	all_test_loss = []
	for images, labels, _ in test_loader:
		model.zero_grad()
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		pred = F.softmax(outputs, dim=1).detach()
		# loss = criterion(outputs, labels).detach()
		## instead of using loss, we use prob
		this_batch_prob = torch.tensor([pred[i][labels[i]] for i in range(len(labels))])
		all_test_loss.append(this_batch_prob)
		all_test_prediction.append(pred)
	all_validation_prediction = []
	all_validation_loss = []
	for images, labels, _ in validation_loader:
		model.zero_grad()
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		pred = F.softmax(outputs, dim=1).detach()
		# loss = criterion(outputs, labels).detach()
		## instead of using loss, we use prob
		this_batch_prob = torch.tensor([pred[i][labels[i]] for i in range(len(labels))])
		all_validation_loss.append(this_batch_prob)
		all_validation_prediction.append(pred)
	
	model.zero_grad()
	
	all_test_loss = torch.cat(all_test_loss, dim=0).flatten().cpu().numpy()
	all_train_loss = torch.cat(all_train_loss, dim=0).flatten().cpu().numpy()
	all_validation_loss = torch.cat(all_validation_loss, dim=0).flatten().cpu().numpy()
	
	all_test_prediction = torch.cat(all_test_prediction, dim=0).cpu().numpy()
	all_train_prediction = torch.cat(all_train_prediction, dim=0).cpu().numpy()
	all_validation_prediction = torch.cat(all_validation_prediction, dim=0).cpu().numpy()
	
	# print(all_train_loss.shape, all_test_loss.shape,all_validation_loss.shape)
	# print (all_train_loss.dtype)
	# print (np.sort(all_train_loss)[::-1][:100])
	# print (np.sort(all_test_loss)[::-1][:100])
	# print (np.sort(all_validation_loss)[::-1][:100])
	# print(all_train_prediction.shape, all_test_prediction.shape,all_validation_prediction.shape)
	
	return (all_train_prediction, all_validation_prediction, all_test_prediction), (all_train_loss, all_validation_loss, all_test_loss)


def attack_experiment():
	all_training_partition = []
	all_validation_partition = []
	all_class_label = []
	all_prob = []
	all_member_loss = []
	all_nonmember_loss = []
	all_loss_auc = []
	all_loss_plr = []
	
	for shadow_model_index in range(args.shadow_model_number):
		print("shadow model index ", shadow_model_index)
		target_dataset = dataset(dataset_name=args.dataset)
		num_classes = len(np.unique(target_dataset.train_label))
		user_list = [User(dataset=args.dataset, model_name=args.model_name, id=i) for i in range(args.user_number)]
		
		# training_partition,validation_partition,overall_train_loader_in_order,overall_validation_loader,overall_validation
		# this is needed to perform MI attack
		# training_partition,validation_partition,mmd_train_loader,mmd_validation_loader,validation_set = assign_part_dataset(target_dataset, user_list)
		
		training_partition, validation_partition, mmd_train_loader, mmd_validation_loader, validation_set = assign_part_dataset(target_dataset, user_list)
		
		all_training_partition.append(training_partition)
		all_validation_partition.append(validation_partition)
		
		if (args.model_name == 'resnet18'):
			target_model = ResNet18(num_classes=num_classes)
			target_model = ModuleValidator.fix(target_model)
		elif (args.model_name == 'alexnet'):
			target_model = alexnet(num_classes=num_classes)
		elif (args.model_name == 'densenet'):
			target_model = densenet(num_classes=num_classes)
			target_model = ModuleValidator.fix(target_model)
		else:
			target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
		
		# print(target_model)
		
		# _,_,_ = new_train_models(user_list,target_model,learning_rate=args.target_learning_rate,
		#													 decay=args.target_l2_ratio,
		#												 epochs=args.target_epochs,target_dataset=target_dataset)
		
		target_model, train_acc, test_acc = train_models(target_dataset, user_list, target_model, learning_rate=args.target_learning_rate,
														 decay=args.target_l2_ratio,
														 epochs=args.target_epochs,
														 mmd_train_loader=mmd_train_loader, mmd_validation_loader=mmd_validation_loader,
														 validation_set=validation_set, training_index=training_partition,
														 validation_index=validation_partition)
		# all_models.append(target_model)
		# get blackbox baseline auc and tpr
		
		all_pred, all_loss = get_all_prob(target_model, target_dataset, training_index=training_partition, validation_index=validation_partition)
		auc, plr, save_neg_loss, save_neg_label = get_blackbox_auc_no_shadow(all_loss[0], all_loss[2], fpr_threshold=args.fpr_threshold)
		### for debug
		dis_name = './expdata/' + get_naming_mid_str() + 'loss_distribution.npy'
		np.save(dis_name, save_neg_loss)
		label_name = './expdata/' + get_naming_mid_str() + 'loss_label.npy'
		np.save(label_name, save_neg_label)
		###
		all_loss_auc.append(auc)
		all_loss_plr.append(plr)
		all_member_loss.append(all_loss[0])
		all_nonmember_loss.append(all_loss[2])
		all_prob.append(np.concatenate((all_pred[0], all_pred[1]), axis=0))
		this_run_label = np.concatenate((target_dataset.train_label[training_partition], target_dataset.train_label[validation_partition]), axis=0)
		all_class_label.append(this_run_label)
	
	all_training_partition = np.stack(all_training_partition)
	all_validation_partition = np.stack(all_validation_partition)
	all_loss_auc = np.array(all_loss_auc).flatten()
	all_loss_plr = np.array(all_loss_plr).flatten()
	all_prob = np.stack(all_prob)
	all_class_label = np.array(all_class_label)
	all_member_loss = np.array(all_member_loss).flatten()
	all_nonmember_loss = np.array(all_nonmember_loss).flatten()
	
	print(all_training_partition.shape, all_validation_partition.shape, all_prob.shape, all_class_label.shape, all_member_loss.shape, all_nonmember_loss.shape)
	
	print(f"loss based attack, avg auc {np.average(all_loss_auc)}, std auc {np.std(all_loss_auc)},"
		  f"avg plr {np.average(all_loss_plr)}, std plr {np.std(all_loss_plr)}")
	
	auc, plr, all_blackbox_loss_val, all_blackbox_loss_label = get_blackbox_auc_no_shadow(all_member_loss, all_nonmember_loss, fpr_threshold=args.fpr_threshold)
	print(f"loss based attack, putting all data together, auc {auc}, plr {plr}")
	
	if (args.test_result == 0):
		all_name = './expdata/' + get_naming_mid_str() + 'all_info.npz'
		np.savez(all_name, all_prob, all_training_partition, all_validation_partition, all_class_label, all_blackbox_loss_val, all_blackbox_loss_label)
		print(all_name)
	else:
		all_name = './expdata/' + get_naming_mid_str() + 'all_info.npz'
		print(all_name)
		data = np.load(all_name)
		all_prob = data['arr_0']
		all_training_partition = data['arr_1']
		all_validation_partition = data['arr_2']
		all_class_label = data['arr_3']
		all_loss = data['arr_4']
		all_label = data['arr_5']
	
	# get shadow models and run LIRA
	if (args.shadow_model_number > 1):
		all_prob_dis, all_label, auc, plr = get_blackbox_auc_lira(all_prob, all_training_partition, all_validation_partition, all_class_label=all_class_label)
		dis_name = './expdata/' + get_naming_mid_str() + 'metric_distribution.npy'
		np.save(dis_name, all_prob_dis)
		label_name = './expdata/' + get_naming_mid_str() + 'metric_label.npy'
		np.save(label_name, all_label)
		print(f"LIRA attack, auc {auc}, plr {plr}")


# compare with other defenses
# here we compare with MMD+MIXUP for a single model
# we also compare with DPSGD for single model


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--target_data_size', type=int, default=3000)
	parser.add_argument('--target_learning_rate', type=float, default=0.01)
	parser.add_argument('--target_batch_size', type=int, default=100)
	parser.add_argument('--target_epochs', type=int, default=20)
	parser.add_argument('--target_l2_ratio', type=float, default=5e-4)
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('--num_classes', type=int, default=10)
	parser.add_argument('--validation_set_size', type=int, default=1000)
	parser.add_argument('--model_name', type=str, default='alexnet')
	parser.add_argument('--alpha', type=float, default='1.0')
	parser.add_argument('--mixup', type=int, default=0)
	parser.add_argument('--num_step', type=int, default=20)
	parser.add_argument('--self_swap', type=int, default=0)
	### fed params
	parser.add_argument('--local_epochs', type=int, default=1)
	parser.add_argument('--user_number', type=int, default=2)
	parser.add_argument('--schedule', type=int, nargs='+', default=[100])
	parser.add_argument('--unequal', type=int, default=0)
	### dpsgd params
	parser.add_argument('--dpsgd', type=int, default=0)
	parser.add_argument('--grad_norm', type=float, default=0)  # 1e10
	parser.add_argument('--noise_scale', type=float, default=0)  # 1e-7
	### MMD params
	parser.add_argument('--mmd', type=int, default=0)
	parser.add_argument('--mmd_loss_lambda', type=float, default=0)
	parser.add_argument('--random_seed', type=int, default=12345)
	parser.add_argument('--repartition', type=int, default=0)
	### std loss params
	parser.add_argument('--std_loss_lambda', type=float, default=0)
	### vulnerable metric params
	parser.add_argument('--vul_metric', type=str, default='loss')
	parser.add_argument('--grad_addback_weight', type=float, default=0.1)
	parser.add_argument('--fpr_threshold', type=float, default=0.001)
	parser.add_argument('--cross_loss_diff_lambda', type=float, default=0.01)
	# Lira params
	parser.add_argument('--shadow_model_number', type=int, default=1)
	
	# precision param
	parser.add_argument('--set_double', type=int, default=0)
	
	# cross diff loss param
	parser.add_argument('--cross_loss', type=str, default='l1')
	
	# simplex param
	parser.add_argument('--simplex', type=int, default=0)
	
	# test param
	parser.add_argument('--test_result', type=int, default=0)
	
	args = parser.parse_args()
	print(vars(args))
	random_seed_list = [args.random_seed]
	import warnings
	
	warnings.filterwarnings("ignore")
	torch.set_printoptions(threshold=5000, edgeitems=20)
	
	for this_seed in random_seed_list:
		import torch
		
		torch.manual_seed(this_seed)
		import numpy as np
		
		np.random.seed(this_seed)
		import sklearn
		
		sklearn.utils.check_random_state(this_seed)
		
		# lp_wrapper = profiler(attack_experiment)
		# lp_wrapper()
		# profiler.print_stats()
		attack_experiment()
	
	print(vars(args))