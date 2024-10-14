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
from model_utils import _ECELoss
from decisionboundaryattack import DecisionBlackBoxAttack
from hsja import HSJAttack

#from line_profiler import LineProfiler
#profiler = LineProfiler()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def entropy(preds, axis=0):
	logp = np.log(preds)
	entropy = np.sum(-preds * logp, axis=axis)
	return entropy


def get_top1(num_classes, entropy_threshold, reduced_prob=0.01):
	# reduced_prob : for reducing top-1 class's probability
	true_target = 1
	preds = np.zeros(num_classes)
	preds[true_target] = 1.
	while (True):
		preds[true_target] -= reduced_prob
		preds[:true_target] += reduced_prob / (num_classes - 1)
		preds[true_target + 1:] += reduced_prob / (num_classes - 1)
		if (entropy(preds) >= entropy_threshold):
			break
	return preds[true_target], preds[true_target + 1]


def get_soft_labels(train_label, num_classes, top1, uniform_non_top1):
	new_soft_label = np.zeros((train_label.shape[0], num_classes))
	for i in range(train_label.shape[0]):
		new_soft_label[i][train_label[i]] = top1
		new_soft_label[i][:train_label[i]] = uniform_non_top1
		new_soft_label[i][train_label[i] + 1:] = uniform_non_top1
	print(new_soft_label[0], train_label[0], np.argmax(new_soft_label[0]))
	print('top1 %.6f | non-top1 %.6f' % (np.max(new_soft_label[0]), np.min(new_soft_label[0])))
	return new_soft_label


def get_naming_mid_str():
	name_string_mid_str =  str(args.dataset) + '_' + str(args.model_name) + '_' + str(args.user_number) + '_' + str(args.num_step)  \
						   + '_' + str(args.dpsgd) + '_' + str(args.noise_scale) + '_' + str(args.grad_norm)  \
						   + '_' + str(args.mmd) + '_' + str(args.mmd_loss_lambda) + '_' + str(args.mixup) \
						   + '_' + str(args.std_loss_lambda) + '_' + str(args.vul_metric) + '_' + str(args.grad_addback_weight) + '_' \
						   + str(args.cross_loss_diff_lambda) + '_' + str(args.shadow_model_number) + '_'
	if (args.hamp):
		# for alexnet running case, just hamp or no hamp
		# for new densenet cifar10 case, whole string
		name_string_mid_str = 'hamp_' + str(args.entropy_percentile) + '_' + str(args.hamp_weight) + '_' + name_string_mid_str
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
		
	return transform_train,transform_test,target_transform

def assign_part_dataset(dataset, user_list=[],ban_list=[]):
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
		
		
	## we need to first divide the training set into train + validation
	
	num_users = len(user_list)
	print (len(dataset.train_label),num_users*args.target_data_size)
	
	### if case of ban list, we need to remove the ban list from training set.
	
	if (ban_list is not None):
		rest_index = np.setdiff1d(np.arange(len(dataset.train_label)),ban_list)
		training_partition = np.random.choice(rest_index, num_users * args.target_data_size, replace=False)
	else:
		training_partition = np.random.choice(len(dataset.train_label),num_users*args.target_data_size,replace=False)
	validation_partition = np.setdiff1d(np.arange(len(dataset.train_label)),training_partition)
	
	### these two loaders are for MMD loss
	overall_train =  part_pytorch_dataset(dataset.train_data[training_partition], dataset.train_label[training_partition], train=False, transform=transform_test,
										  target_transform=target_transform)
	overall_train_loader_in_order =  torch.utils.data.DataLoader(overall_train, batch_size=args.target_batch_size,
																 shuffle=False, num_workers=1)
	overall_validation =  part_pytorch_dataset(dataset.train_data[validation_partition], dataset.train_label[validation_partition], train=False, transform=transform_test,
											   target_transform=target_transform)
	overall_validation_loader =  torch.utils.data.DataLoader(overall_validation, batch_size=args.target_batch_size,
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
		
		if (args.hamp):
		### for hamp, we need to modify the training data
			this_user.hamp = True
			num_classes = len(np.unique(dataset.train_label))
			preds = np.ones(num_classes)
			preds /= float(num_classes)
			highest_entropy = entropy(preds)
			# assign uniform class prob for all the non-top-1 classes
			top1, uniform_non_top1 = get_top1(num_classes, highest_entropy * args.entropy_percentile)
			print("Highest entropy {:.4f} | entropy_percentile {:.4f} | entropy threshold {:.4f}".format(highest_entropy, args.entropy_percentile,
																									 highest_entropy * args.entropy_percentile))
			train_label_modified = get_soft_labels(this_user.train_label, num_classes, top1, uniform_non_top1)
			this_user.modified_train_label = train_label_modified
			#print (train_label_modified.shape)
			#print (train_label_modified[0],this_user.train_label[0])
		
		this_user.train_index = this_user_train_index
		
		this_user.class_weights = np.ones((len(np.unique(dataset.train_label)))) * training_set_size / (len(np.unique(dataset.train_label)) * (np.bincount(this_user.train_label,minlength=len(np.unique(dataset.train_label))) + 10))
		
		#this_user.class_weights = np.ones((len(np.unique(dataset.train_label)))) * training_set_size / (len(np.unique(dataset.train_label)) * (np.bincount(this_user.train_label)))
		
		this_user.test_data = dataset.test_data
		this_user.test_label = dataset.test_label
		assigned_index.append(this_user_train_index)
		
		if (args.hamp):
			train = part_pytorch_dataset(this_user.train_data, this_user.modified_train_label, train=True, transform=transform_train,
										 target_transform=target_transform,float_target=True)
			#print ("HAMP dataset")
		else:
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
		this_user.train_eval_data_loader =  torch.utils.data.DataLoader(train_eval, batch_size=args.target_batch_size,
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
	
	
	return training_partition,validation_partition,overall_train_loader_in_order,overall_validation_loader,overall_validation

def simplex_sampling(user_list,target_model,repeat_count=60):
	if (args.user_number == 1):
		repeat_count = 1
	all_train_acc = []
	all_test_acc = []
	model_weight_list = [user_list[i].model.state_dict() for i in range(len(user_list))]
	for _ in range(repeat_count):
		this_weight = simplex_uniform_sampling(num=len(user_list))
		this_model_weight = average_weights(model_weight_list,this_weight)
		target_model.load_state_dict(this_model_weight)
		train_acc, test_acc = get_train_test_acc(user_list, target_model)
		all_train_acc.append(train_acc)
		all_test_acc.append(test_acc)
	
	all_train_acc = np.array(all_train_acc)
	all_test_acc = np.array(all_test_acc)
	print (f"test acc avg {np.average(all_test_acc)},test acc std {np.std(all_test_acc)}")
	return all_train_acc,all_test_acc


def train_models(target_dataset,user_list, target_model, learning_rate, decay, epochs,mmd_validation_loader=None,mmd_train_loader=None,validation_set=None,training_index=None,validation_index=None):
	num_users = len(user_list)
	num_classes = len(np.unique(target_dataset.train_label))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_model.to(device)
	momentum = 0.9 # should be 0.9
	#if (args.model_name =='densenet' or args.model_name =='alexnet' or args.dataset =='purchase' or args.dataset =='texas' or args.dataset =='location'):
	if (args.hamp and args.model_name=='densenet'):
		momentum = 0.99
		
	simplex_result = []
	loss_dis_before = []
	loss_dis_after = []
	best_model_state_dict = target_model.state_dict()
	best_val = 0.0
	if (args.dataset == 'purchase' or args.dataset == 'texas' or args.dataset == 'location'):
		#target_model_optim = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=decay)
		target_model_optim = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
	
	else:
		target_model_optim = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
	
	### notice that in pytorch, momentum etc. is bound with optimizer, so we need to initialize the optimizer/model for each user
	for user_idx in range(num_users):
		user_list[user_idx].model = copy.deepcopy(target_model)
		if (args.dataset == 'purchase' or args.dataset == 'texas' or args.dataset == 'location'):  #
			#this_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, user_list[user_idx].model.parameters()), lr=learning_rate, weight_decay=decay)
			this_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, user_list[user_idx].model.parameters()), lr=learning_rate, momentum=momentum, weight_decay=decay)
		else:
			this_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, user_list[user_idx].model.parameters()), lr=learning_rate, momentum=momentum, weight_decay=decay)
		
		user_list[user_idx].optim = this_optim
		
		if (args.dpsgd):
			### adding dp components
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
		#print (f"{epoch} epoch")
		if (args.repartition):
			repartition_dataset(args.dataset,user_list)
		ori_target_model_state_dict = target_model.state_dict()
		if (epoch in args.schedule):
			learning_rate = learning_rate / 10
			print("new learning rate = %f" % (learning_rate))
			for user_idx in range(num_users):
				for param_group in user_list[user_idx].optim.param_groups:
					param_group['lr'] = learning_rate
		
		comm_round_per_epoch = int(args.target_data_size / (args.target_batch_size * args.num_step))
		
		#print (f" comm round per epoch {comm_round_per_epoch}")
		
		for _ in range(comm_round_per_epoch):
			local_weights = []
			for idx in range(len(user_list)):
				# create a new dataloader
				train_data_loader = user_list[idx].create_new_train_data_loader(batch_size=args.target_batch_size)
				#print (len(train_data_loader))
				### defense
				
				if (args.hamp):
					new_state_dict = update_weights_hamp(current_model_weights=ori_target_model_state_dict,
													model=user_list[idx].model,
													optimizer=user_list[idx].optim, train_loader=train_data_loader,
													local_epochs=args.local_epochs, mixup=args.mixup,
													selfswap=args.self_swap,
													class_weights=user_list[idx].class_weights,
													std_loss_lambda=args.std_loss_lambda,
												    dataset_name=target_dataset.dataset_name,
													hamp_weight = args.hamp_weight)
					#print (f"hamp activated!")
				else:
					new_state_dict = update_weights(current_model_weights=ori_target_model_state_dict,
										  model=user_list[idx].model,
										  optimizer=user_list[idx].optim, train_loader=train_data_loader,
										  local_epochs=args.local_epochs, mixup=args.mixup,
										  selfswap=args.self_swap,
										  class_weights=user_list[idx].class_weights,
										  std_loss_lambda=args.std_loss_lambda,dataset_name=target_dataset.dataset_name)  ### model name is specifically for inception..
				#user_list[idx].update_ban_list(ban_list)
				local_weights.append((new_state_dict))
			
			### update global weights
			
			if (args.simplex):
				simplex_result.append(simplex_sampling(user_list,target_model))
			
			global_weights = average_weights(local_weights)
			target_model.load_state_dict(global_weights)
			
			#train_acc, test_acc = get_train_test_acc(user_list, target_model)
			### apply MMD regularization
			if (epoch!=epochs-1 and args.mmd_loss_lambda!=0):
				weight_after_mmd = update_weights_mmd(target_model, user_list, target_model_optim, validation_loader=mmd_validation_loader,
													  train_loader_in_order=mmd_train_loader,
													  validation_set=validation_set,num_classes=num_classes,
													  loss_lambda=args.mmd_loss_lambda, starting_index=user_list[0].starting_index)
				target_model.load_state_dict(weight_after_mmd)
				#print ("MMD done")
			
			# (all_train_prediction, all_validation_prediction, all_test_prediction), (all_train_loss, all_validation_loss, all_test_loss)
			
			#if (epoch == 60):
			#	_,loss_dis_before = get_all_prob(target_model,target_dataset,training_index=training_index,validation_index=validation_index)
			
			if (epoch!=epochs-1 and args.cross_loss_diff_lambda!=0):
				weight_after_cross_loss_diff = average_weights(update_weights_cross_loss_diff(user_list,loss_lambda=args.cross_loss_diff_lambda,loss_name=args.cross_loss))
				target_model.load_state_dict(weight_after_cross_loss_diff)
				#print ("CROSS LOSS DIFF done")

			#if (epoch == 60):
			#	_,loss_dis_after = get_all_prob(target_model, target_dataset, training_index=training_index, validation_index=validation_index)
			#	np.savez('./expdata/cross-diff-eff.npz',loss_dis_before,loss_dis_after)
			
			### remove impacts of vulnerable points
			## we find vulnerable points first
			## for loss we can start from the beginning
			## for logits diff, we should let the model train first and then start to do removing when it starts to overfit, like in the middle
			if (args.grad_addback_weight!=0):
				length,vulnerable_points_set = find_vulnerable_points(user_list,target_model,target_model_optim, mmd_train_loader, mmd_validation_loader,vul_metric=args.vul_metric,fpr_threshold=args.fpr_threshold)
				#print ("vulnerable point set length", length)
				if (length>0):
					## create vulnerable points set loader
					_,transform_test,target_transform = get_transformation(target_dataset)
					vulnerable_dataset = part_pytorch_dataset(target_dataset.train_data[vulnerable_points_set],target_dataset.train_label[vulnerable_points_set],
															  train=False, transform=transform_test,target_transform=target_transform)
					vulnerable_points_set_loader = torch.utils.data.DataLoader(vulnerable_dataset,batch_size=args.target_batch_size, shuffle=False,num_workers=1)
					## given those vulnerable points, we add those gradient back to the central model
					weight_after_adding = update_weights_add_grad(target_model,target_model_optim,vulnerable_points_set_loader,weight=args.grad_addback_weight)
					target_model.load_state_dict(weight_after_adding)
					#print ("weight added back")
					## ban these vulnerable points from training, since the last schedule ???
					#print (epoch,args.schedule[-1])
					if (epoch>= args.schedule[-1] or (epoch>=50)):  ## 0 or -1?
						for this_user in user_list:
							this_user.update_ban_list(vulnerable_points_set)
	#block for HAMP
			if (args.hamp):
				train_acc,test_acc,val_acc,_,_,_ = get_train_test_acc(user_list,target_model,return_validation_result=True)
				if (val_acc > best_val+1):
					best_val = val_acc
					best_model_state_dict = target_model.state_dict()
					print (f"best validation acc {best_val}")

	if (args.hamp):
		target_model.load_state_dict(best_model_state_dict)
	
	
	train_acc, test_acc = get_train_test_acc(user_list, target_model)
	print(f"train acc {train_acc},test acc {test_acc}")
	#
	if (args.simplex):
		simplex_result_name = './expdata/' + get_naming_mid_str() + 'simplex_result.npy'
		np.save(simplex_result_name,np.array(simplex_result))
		print (np.array(simplex_result).shape)
		
	## need to eval on 500 vul set if ban list
	vul_set_acc = eval_on_ban_list(target_model,target_dataset)
	return target_model, train_acc, test_acc,target_model_optim,vul_set_acc
	
def eval_on_ban_list(model,dataset):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	vulnerable_index = np.load(f'./{args.dataset}_top500_vul_index.npy')
	vul_data = dataset.train_data[vulnerable_index]
	vul_label = dataset.train_label[vulnerable_index]
	_,transform_test,target_transform = get_transformation(dataset)
	vul_dataset = part_pytorch_dataset(vul_data, vul_label, train=False, transform=transform_test,
										  target_transform=target_transform)
	vul_dataloader =   torch.utils.data.DataLoader(vul_dataset, batch_size=args.target_batch_size,
																 shuffle=False, num_workers=1)
	
	correct = 0.0
	total = 0.0
	with torch.no_grad():
		model.eval()
		for images, labels, _ in vul_dataloader:
			model.zero_grad()
			images = images.to(device)
			outputs = model(images)
			labels = labels.to(device)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
	
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	print (f"evaluation on ban list: testing accuracy {acc}")
	return acc


def get_all_prob(model,dataset,training_index,validation_index):
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
		pred = F.softmax(outputs,dim=1).detach()
		if (args.post_def):
			random_input = torch.randn_like(images)
			random_pred = F.softmax(model(random_input),dim=1).detach()
			# sort random_pred based on pred
			sorted_random_pred  = []
			for this_pred,this_random_pred in zip(pred,random_pred):
				argsort_index = torch.argsort(this_pred)
				this_random_pred,_ = torch.sort(this_random_pred)
				new_pred = torch.zeros_like(this_pred)
				for index in range(len(new_pred)):
					new_pred[argsort_index[index]] = this_random_pred[index]
				sorted_random_pred.append(new_pred)
			pred = torch.stack(sorted_random_pred)
			#print (pred.shape)
			
		#print (outputs.shape)
		#print (pred[0],torch.sum(pred[0]))
		#loss = criterion(outputs, labels).detach()
		## instead of using loss, we use prob
		this_batch_prob = torch.tensor([pred[i][labels[i]] for i in range(len(labels))])
		#print (this_batch_prob.shape)
		all_train_loss.append(this_batch_prob)
		all_train_prediction.append(pred)
	all_test_prediction = []
	all_test_loss = []

	logits_list = []
	labels_list = []
	for images, labels, _ in test_loader:
		model.zero_grad()
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		pred = F.softmax(outputs,dim=1).detach()
		
		labels_list.append(labels.detach())
		
		if (args.post_def):
			random_input = torch.randn_like(images)
			random_pred = F.softmax(model(random_input),dim=1).detach()
			# sort random_pred based on pred
			sorted_random_pred  = []
			for this_pred,this_random_pred in zip(pred,random_pred):
				argsort_index = torch.argsort(this_pred)
				this_random_pred,_ = torch.sort(this_random_pred)
				new_pred = torch.zeros_like(this_pred)
				for index in range(len(new_pred)):
					new_pred[argsort_index[index]] = this_random_pred[index]
				sorted_random_pred.append(new_pred)
			pred = torch.stack(sorted_random_pred)
			
		logits_list.append(pred)
		#loss = criterion(outputs, labels).detach()
		## instead of using loss, we use prob
		this_batch_prob = torch.tensor([pred[i][labels[i]] for i in range(len(labels))])
		all_test_loss.append(this_batch_prob)
		all_test_prediction.append(pred)
	
	logits = torch.cat(logits_list).cuda()
	labels = torch.cat(labels_list).cuda()
	ece_criterion = _ECELoss().cuda()
	ece_loss = ece_criterion(logits, labels,islogit=False).detach().item()
	print(f"POST DEF ECE LOSS:{ece_loss}")
	
	all_validation_prediction = []
	all_validation_loss = []
	for images, labels, _ in validation_loader:
		model.zero_grad()
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		pred = F.softmax(outputs,dim=1).detach()
		if (args.post_def):
			random_input = torch.randn_like(images)
			random_pred = F.softmax(model(random_input),dim=1).detach()
			# sort random_pred based on pred
			sorted_random_pred  = []
			for this_pred,this_random_pred in zip(pred,random_pred):
				argsort_index = torch.argsort(this_pred)
				this_random_pred,_ = torch.sort(this_random_pred)
				new_pred = torch.zeros_like(this_pred)
				for index in range(len(new_pred)):
					new_pred[argsort_index[index]] = this_random_pred[index]
				sorted_random_pred.append(new_pred)
			pred = torch.stack(sorted_random_pred)
		
		#loss = criterion(outputs, labels).detach()
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
	
	#print(all_train_loss.shape, all_test_loss.shape,all_validation_loss.shape)
	#print (all_train_loss.dtype)
	#print (np.sort(all_train_loss)[::-1][:100])
	#print (np.sort(all_test_loss)[::-1][:100])
	#print (np.sort(all_validation_loss)[::-1][:100])
	#print(all_train_prediction.shape, all_test_prediction.shape,all_validation_prediction.shape)
	
	return (all_train_prediction,all_validation_prediction,all_test_prediction), (all_train_loss,all_validation_loss,all_test_loss)

# do label only attack for all def.
def label_only_attack(model,dataset,training_index,validation_index):
	### do HSJA for each instance and calculate the distance
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	training_data = dataset.train_data[training_index]
	training_label = dataset.train_label[training_index]
	test_data = dataset.test_data
	test_label = dataset.test_label
	### balance evaluation set
	min_len = min(len(test_label),len(training_label))
	#min_len = 200 # for testing if correct
	training_index =np.random.choice(len(training_label),min_len,replace=False)
	testing_index = np.random.choice(len(test_label),min_len,replace=False)
	
	_, transform_test, target_transform = get_transformation(dataset)
	overall_train = part_pytorch_dataset(training_data[training_index], training_label[training_index], train=False, transform=transform_test,
										 target_transform=target_transform)
	train_loader = torch.utils.data.DataLoader(overall_train, batch_size=1, shuffle=False, num_workers=1)
	
	overall_test = part_pytorch_dataset(test_data[testing_index], test_label[testing_index], train=False, transform=transform_test,
										target_transform=target_transform)
	test_loader = torch.utils.data.DataLoader(overall_test, batch_size=1, shuffle=False, num_workers=1)
	
	attack_method = HSJAttack(model=model)
	### get train data distance
	train_distance = []
	for this_data,this_label,_ in tqdm(train_loader):
		#this_adv_data,_ = attack_method._perturb(this_data,this_label)
		#print (this_adv_data.shape,this_data.shape)
		#this_distance = torch.linalg.norm(torch.abs(this_adv_data-this_data))
		this_data = this_data.to(device)
		this_pred = torch.argmax(model(this_data),dim=1)
		#print (this_pred.shape)
		this_label = this_label.to(device)
		if (torch.squeeze(this_pred)!=this_label):
			this_distance = 0
		else:
			this_distance = attack_method._perturb(this_data,this_label)
		train_distance.append(this_distance)
		
	### get test data distance
	test_distance = []
	for this_data,this_label,_ in tqdm(test_loader):
		this_data = this_data.to(device)
		this_pred = torch.argmax(model(this_data),dim=1)
		this_label = this_label.to(device)
		if (torch.squeeze(this_pred)!=this_label):
			this_distance = 0
		else:
			this_distance = attack_method._perturb(this_data,this_label)
		test_distance.append(this_distance)
		
	train_distance = np.array(train_distance)
	test_distance = np.array(test_distance)
	
	print (f"train distance {train_distance}")
	print (f"test distance {test_distance}")
	
	### report AUC score and TPR when FPR @ 0.1%
	label = np.concatenate((np.ones((len(train_distance))), np.zeros((len(test_distance)))))
	auc = roc_auc_score(label, np.concatenate((train_distance, test_distance), axis=0))
	print(f"AUC score {auc}")
	fpr_threshold = 0.001
	threshold = np.sort(test_distance)[::-1][int(len(test_distance) * fpr_threshold)]
	print(f"distance threshold {threshold}")
	cnt = 0
	for i in range(len(train_distance)):
		if (train_distance[i] >= threshold):
			cnt += 1
	print(f"TPR {cnt / min_len}, FPR {fpr_threshold}, PLR {cnt / (min_len * fpr_threshold)}")


def attack_experiment():
	
	all_training_partition = []
	all_validation_partition = []
	all_class_label = []
	all_prob = []
	all_member_loss = []
	all_nonmember_loss = []
	all_loss_auc = []
	all_loss_plr = []
	all_vul_set_test_acc = []
	
	### outlier exp comments:
	### shadow model number is 1
	### the goal of this exp is that, we train: 1. 20000 base case model, 2. 20000 mist model, 3. 19500 base case model. 4. 20000 for all other defs.
	### we evaluate the following metrics: 1. train acc. 2. test acc on testing set. 3. test acc on 500 vul set. (diff) [need to implement this part].
	
	### for the 500 vul set, we need to show the attack effectiveness reduction (20000 base vs 20000 mist vs 20000 other defs).
	### we could say, for 1%FPR, what is the number of this set can be correctly identified? also try other FPR. (.5%, .1%). can be done locally. we need to download all alxenet data.
	### we also need the combined PLR at these FPRs. just for reference.
	### if we remove this 500 vul set, there is no way for us to attack this set, then here we need to show the onion effect, where
	### there is going to be other vul instances. we could show that at (1%,.5%,.1% FPR), there is still a large portion of TPR. (even though not this vul set).
	### so the metric to compare here is still the PLR at 1, 0.5, 0.1 % (20000 mist vs 20000 other defs vs 19500 base) for combined PLR.
	### here we want to show that PLR is still way less, even though we consider the 500 vul set is already in.
	### we just need to use LIRA here, because using canary is going to be harder to run exp. (figure out a reason for canary not applicable here..)
	
	### in the end, we may need other exps to confirm the finding from the onion effect paper? and we may need results from canary as well.
	### the main goal of this exp is to show that MIST is necessary because of onion effect, simply removing it would not work.
	
	for shadow_model_index in range(args.shadow_model_number):
		if (args.test_result):
			break
		print ("shadow model index ",shadow_model_index)
		target_dataset = dataset(dataset_name=args.dataset)
		num_classes = len(np.unique(target_dataset.train_label))
		user_list = [User(dataset=args.dataset, model_name=args.model_name, id=i) for i in range(args.user_number)]
	
		# training_partition,validation_partition,overall_train_loader_in_order,overall_validation_loader,overall_validation
		# this is needed to perform MI attack
		#training_partition,validation_partition,mmd_train_loader,mmd_validation_loader,validation_set = assign_part_dataset(target_dataset, user_list)
	
		## need to use preassigned index to perform exp.
		#rest_index = np.load('./left19500_sample_index.npy')
		#training_partition = np.concatenate((vulnerable_index,rest_index))
		#validation_partition = np.setdiff1d(np.arange(50000),training_partition)
		if (args.ban_list):
			vulnerable_index = np.load(f'./{args.dataset}_top500_vul_index.npy')
			training_partition, validation_partition, mmd_train_loader, mmd_validation_loader, validation_set = assign_part_dataset(target_dataset, user_list,ban_list = vulnerable_index)
		else:
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
			#target_model = densenet(num_classes=num_classes, depth=100, growthRate=12, compressionRate=2, dropRate=0).to(device)
			target_model = ModuleValidator.fix(target_model)
		else:
			target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
	
	#print(target_model)
		print (count_parameters(target_model))
	#_,_,_ = new_train_models(user_list,target_model,learning_rate=args.target_learning_rate,
	#													 decay=args.target_l2_ratio,
	#												 epochs=args.target_epochs,target_dataset=target_dataset)
	
		target_model, train_acc, test_acc,target_model_optim,vul_set_acc = train_models(target_dataset,user_list, target_model, learning_rate=args.target_learning_rate,
													 decay=args.target_l2_ratio,
													 epochs=args.target_epochs,
													 mmd_train_loader = mmd_train_loader, mmd_validation_loader = mmd_validation_loader,
													 validation_set = validation_set,training_index=training_partition,
													 validation_index=validation_partition)
		all_vul_set_test_acc.append(vul_set_acc)
		#all_models.append(target_model)
		
		#from nasr_whitebox_attack import nasr_whitebox_attack
		#nasr_whitebox_attack = nasr_whitebox_attack(num_classes)
		#nasr_whitebox_attack.run_attack(target_model=target_model,target_model_optim=target_model_optim,target_dataset=target_dataset,training_partition=training_partition,train_epoch=20,fpr_threshold=0.001)
		
		### do label only attack
		if (args.label_only_attack):
			label_only_attack(target_model,target_dataset,training_index=training_partition,validation_index=validation_partition)
		
		### save all info for canary attack
		keep =training_partition
		keep_bool = np.full((len(target_dataset.train_data)), False)
		keep_bool[keep] = True
		target_model.eval()
		state = {"model": target_model.state_dict(),
				 "in_data": keep,
				 "keep_bool": keep_bool,
				 "model_arch": args.model_name}
		os.makedirs('saved_models/' + args.model_name, exist_ok=True)
		### include def in path name if def is applied
		torch.save(state, './saved_models/' + get_naming_mid_str() + str(shadow_model_index) + '.pth')
		
		print (f"save model:{get_naming_mid_str() + str(shadow_model_index) + '.pth'}")
		
		all_pred,all_loss = get_all_prob(target_model,target_dataset,training_index=training_partition,validation_index=validation_partition)
		auc,plr,save_neg_loss,save_neg_label = get_blackbox_auc_no_shadow(all_loss[0],all_loss[2],fpr_threshold=args.fpr_threshold)
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
		all_prob.append(np.concatenate((all_pred[0],all_pred[1]),axis=0))
		this_run_label = np.concatenate((target_dataset.train_label[training_partition],target_dataset.train_label[validation_partition]),axis=0)
		all_class_label.append(this_run_label)
		

	if (args.test_result == 0):
		all_training_partition = np.stack(all_training_partition)
		all_validation_partition = np.stack(all_validation_partition)
		all_loss_auc = np.array(all_loss_auc).flatten()
		all_loss_plr = np.array(all_loss_plr).flatten()
		all_prob = np.stack(all_prob)
		all_class_label = np.array(all_class_label)
		all_member_loss = np.array(all_member_loss).flatten()
		all_nonmember_loss = np.array(all_nonmember_loss).flatten()
		all_name = './expdata/' + get_naming_mid_str() + 'all_info.npz'
		
		print(f"loss based attack, avg auc {np.average(all_loss_auc)}, std auc {np.std(all_loss_auc)},"
			  f"avg plr {np.average(all_loss_plr)}, std plr {np.std(all_loss_plr)}")
		
		auc, plr, all_blackbox_loss_val, all_blackbox_loss_label = get_blackbox_auc_no_shadow(all_member_loss, all_nonmember_loss, fpr_threshold=args.fpr_threshold)
		print(f"loss based attack, putting all data together, auc {auc}, plr {plr}")
		
		np.savez(all_name,all_prob,all_training_partition,all_validation_partition,all_class_label,all_blackbox_loss_val,all_blackbox_loss_label)
		print (all_name)
	else:
		all_name = './expdata/' + get_naming_mid_str() + 'all_info.npz'
		print (all_name)
		data = np.load(all_name)
		all_prob = data['arr_0']
		all_training_partition = data['arr_1']
		all_validation_partition = data['arr_2']
		all_class_label = data['arr_3']
		all_loss = data['arr_4']
		all_label = data['arr_5']
	
	#print (all_training_partition.shape,all_validation_partition.shape,all_prob.shape,all_class_label.shape,all_member_loss.shape,all_nonmember_loss.shape)
	
	auc, plr = get_blackbox_auc_class_nn(all_prob, all_training_partition, all_validation_partition, all_class_label=all_class_label,fpr_threshold=args.fpr_threshold)
	print (f"classwise nn attack, putting all data together, auc {auc}, plr {plr}")
	
	# get shadow models and run LIRA
	if (args.shadow_model_number>1):
		all_prob_dis,all_label,auc,plr,_,_ = get_blackbox_auc_lira(all_prob,all_training_partition,all_validation_partition,all_class_label=all_class_label,fpr_threshold_list=[0.01,0.005,0.001])
		dis_name = './expdata/' + get_naming_mid_str() + 'metric_distribution.npy'
		np.save(dis_name,all_prob_dis)
		label_name = './expdata/' + get_naming_mid_str() + 'metric_label.npy'
		np.save(label_name,all_label)
		print (f"LIRA attack, auc {auc}, plr {plr}")
		
	print (f"all vul set test acc avg{np.average(np.array(all_vul_set_test_acc))}")
	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--target_data_size', type=int, default=3000)
	parser.add_argument('--target_learning_rate', type=float, default=0.01)
	parser.add_argument('--target_batch_size', type=int, default=100)
	parser.add_argument('--target_epochs', type=int, default=20)
	parser.add_argument('--target_l2_ratio', type=float, default=5e-4)
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('--num_classes', type=int, default=10)
	parser.add_argument('--validation_set_size', type=int, default=100)
	parser.add_argument('--model_name', type=str, default='alexnet')
	parser.add_argument('--alpha', type=float, default='1.0')
	parser.add_argument('--mixup', type=int, default=0)
	parser.add_argument('--num_step', type=int, default=20)
	parser.add_argument('--self_swap',type=int,default=0)
	### fed params
	parser.add_argument('--local_epochs', type=int, default=1)
	parser.add_argument('--user_number', type=int, default=2)
	parser.add_argument('--schedule',type=int,nargs='+',default=[100])
	parser.add_argument('--unequal',type=int,default=0)
	### dpsgd params
	parser.add_argument('--dpsgd', type=int, default=0)
	parser.add_argument('--grad_norm', type=float, default=0)  # 1e10
	parser.add_argument('--noise_scale', type=float, default=0)  # 1e-7
	### MMD params
	parser.add_argument('--mmd', type=int, default=0)
	parser.add_argument('--mmd_loss_lambda', type=float, default=0)
	parser.add_argument('--random_seed', type=int, default=1)
	parser.add_argument('--repartition', type=int, default=0)
	### std loss params
	parser.add_argument('--std_loss_lambda', type=float, default=0)
	### vulnerable metric params
	parser.add_argument('--vul_metric',type=str,default='loss')
	parser.add_argument('--grad_addback_weight',type=float,default=0)
	parser.add_argument('--fpr_threshold',type=float,default=0.001)
	parser.add_argument('--cross_loss_diff_lambda',type=float,default=0.01)
	# Lira params
	parser.add_argument('--shadow_model_number',type=int,default=1)
	
	# precision param
	parser.add_argument('--set_double',type=int,default=0)
	
	# cross diff loss param
	parser.add_argument('--cross_loss',type=str,default='l1')
	
	#simplex param
	parser.add_argument('--simplex',type=int,default=0)
	
	#test param
	parser.add_argument('--test_result',type=int,default=0)
	
	## hamp param
	parser.add_argument('--hamp',type=int,default=0)
	parser.add_argument('--hamp_weight',type=float,default=0.001)
	parser.add_argument('--entropy_percentile',type=float,default=0.95)
	parser.add_argument('--post_def',type=int,default=0)
	parser.add_argument('--ban_list',type=int,default=0)
	
	## label only attack param
	parser.add_argument('--label_only_attack',type=int,default=0)
	
	args = parser.parse_args()
	print(vars(args))
	random_seed_list = [args.random_seed]
	import warnings
	warnings.filterwarnings("ignore")
	torch.set_printoptions(threshold=5000, edgeitems=20)
	
	args.fpr_threshold = 0.005
	
	for this_seed in random_seed_list:
		import torch
		torch.manual_seed(this_seed)
		import numpy as np
		np.random.seed(this_seed)
		import sklearn
		sklearn.utils.check_random_state(this_seed)
		
		#if (args.dataset=='cifar10'):
		#	torch.set_default_dtype(torch.float64)
		#	torch.set_default_tensor_type(torch.DoubleTensor)
		#	args.set_double=1
		#	print ('set type!')
		
		#lp_wrapper = profiler(attack_experiment)
		#lp_wrapper()
		#profiler.print_stats()
		
		
		#w_list = [13,13.5,14,14.5,15]
		#w_list = [6,6.5,7,7.5,8]
		#w_list = [10,20,30,40,50]
		#w_list = [2,3,4,5,6,7,8,9,10]
		#for w in w_list:
		#	args.user_number = int(w)
		#	args.target_data_size = int(40000/args.user_number)
		#	args.num_step = int(args.target_data_size/100)
		#	args.cross_loss_diff_lambda = w
		print(vars(args))
		attack_experiment()
		print(vars(args))
		
		#n_list = [5,6,7,8,9,10,11,12,13,14]
		#for n in n_list:
		#	args.cross_loss_diff_lambda = n
		
		#e_list = [0.001,0.005,0.01,0.05,0.1]
		#w_list = [6,6.5,7,7.5,8,8.5,9,9.5]
		#w_list = [10,20,30,40,50,60,70,80,90,100]
		#for w in w_list:
		#	args.cross_loss_diff_lambda = w
		
	
	