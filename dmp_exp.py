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


# from line_profiler import LineProfiler
# profiler = LineProfiler()

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_naming_mid_str():
	name_string_mid_str = 'dmp_' + str(args.dataset) + '_' + str(args.model_name) + '_' + str(args.user_number) \
						  + '_' + str(args.shadow_model_number) + '_'
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
	print(len(dataset.train_label), num_users * args.target_data_size)
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
		
		# this_user.class_weights = np.ones((len(np.unique(dataset.train_label)))) * training_set_size / (len(np.unique(dataset.train_label)) * (np.bincount(this_user.train_label)))
		
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




def train_models(target_dataset, user_list, target_model, learning_rate, decay, epochs, mmd_validation_loader=None, mmd_train_loader=None, validation_set=None,
				 training_index=None,
				 validation_index=None):
	num_users = len(user_list)
	num_classes = len(np.unique(target_dataset.train_label))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_model.to(device)
	momentum = 0.9
	target_learning_rate = learning_rate
	target_model_optim = torch.optim.SGD(target_model.parameters(), lr=target_learning_rate, momentum=momentum, weight_decay=decay)
	criterion = nn.CrossEntropyLoss().to(device)  #
	
	final_model = copy.deepcopy(target_model)
	momentum = 0.9
	final_learning_rate = learning_rate
	final_model_optim = torch.optim.SGD(final_model.parameters(), lr=final_learning_rate, momentum=momentum, weight_decay=decay)
	
	### start training
	train_loader = user_list[0].train_data_loader
	for epoch in tqdm(range(epochs)):
		# print (f"{epoch} epoch")
		if (epoch in args.schedule):
			target_learning_rate = target_learning_rate / 10
			print("new learning rate = %f" % (target_learning_rate))
			for param_group in target_model_optim.param_groups:
					param_group['lr'] = target_learning_rate
		
		for batch_idx, (images, labels, data_idx) in enumerate(train_loader):
			target_model_optim.zero_grad()
			images, labels = images.to(device), labels.to(device)
			log_probs = target_model(images)
			loss = criterion(log_probs, labels)
			loss.backward()
			target_model_optim.step()

	train_acc, test_acc = get_train_test_acc(user_list, target_model)
	print(f"unprotected model: train acc {train_acc},test acc {test_acc}")
	
	### generate reference data.. ### we should use generator..
	
	if (args.dataset == 'cifar10' or args.dataset == 'cifar100'):
		ref_data = np.load('./syn_data.npy')
		
		#print (np.max(ref_data),np.min(ref_data),np.max(target_dataset.test_data),np.min(target_dataset.test_data))
		
		#print (ref_data.shape)
		ref_data = np.moveaxis(ref_data,1,3)
		### we need to rescale ref data to 255-0.
		ref_data = ((ref_data - ref_data.min()) * (1 / (ref_data.max() - ref_data.min()) * 255)).astype('uint8')
		
		#print (ref_data.shape)
		#print (target_dataset.train_data.shape)
		ref_label = np.zeros((20000)).astype(np.int64)
		
		#print (np.max(ref_data),np.min(ref_data),np.max(target_dataset.test_data),np.min(target_dataset.test_data))
		
		ref_dataset = part_pytorch_dataset(ref_data, ref_label, train=True, transform=user_list[0].train_transform,
										   target_transform=user_list[0].target_transform)
		ref_data_loader = torch.utils.data.DataLoader(ref_dataset, batch_size=args.target_batch_size,
													  shuffle=False, num_workers=1)
	else:
		#print (len(validation_index),args.target_data_size)
		if (len(validation_index)<args.target_data_size):
			sampled_validation_index = validation_index
		sampled_validation_index = np.random.choice(validation_index,min(args.target_data_size,len(validation_index)),replace=False)
		ref_data = target_dataset.train_data[sampled_validation_index]
		ref_true_label = target_dataset.train_label[sampled_validation_index]
		ref_dataset = part_pytorch_dataset(ref_data,ref_true_label,train=True, transform=user_list[0].train_transform,
										 target_transform=user_list[0].target_transform)
		ref_data_loader =  torch.utils.data.DataLoader(ref_dataset, batch_size=args.target_batch_size,
																	   shuffle=False, num_workers=1)

	t_softmax = 0.5
	if (args.dataset == 'cifar100'):
		t_softmax = 4
		
	ref_label = []
	for this_batch_data,_,_ in ref_data_loader:
		this_batch_data = this_batch_data.to(device)
		this_batch_pred = F.softmax(target_model(this_batch_data)/t_softmax,dim=1)
		ref_label.append(this_batch_pred.detach().cpu().numpy())
	ref_label = np.vstack(ref_label)
	#print (ref_label.shape,ref_data.shape)
	
	### train final distilled model
	ref_dataset = part_pytorch_dataset(ref_data,ref_label,train=True, transform=user_list[0].train_transform,
										 target_transform=user_list[0].target_transform,float_target=True)
	ref_data_loader =  torch.utils.data.DataLoader(ref_dataset, batch_size=args.target_batch_size,
																	   shuffle=True, num_workers=1)
	#print (final_learning_rate)
	for epoch in tqdm(range(epochs)):
		# print (f"{epoch} epoch")
		if (epoch in args.schedule):
			final_learning_rate = final_learning_rate / 10
			print("new learning rate = %f" % (final_learning_rate))
			for param_group in final_model_optim.param_groups:
					param_group['lr'] = final_learning_rate
		
		for batch_idx, (images, labels, data_idx) in enumerate(ref_data_loader):
			final_model_optim.zero_grad()
			images, labels = images.to(device), labels.to(device)
			log_probs = final_model(images)
			#print (log_probs[0],labels[0])
			loss = F.kl_div(F.log_softmax(log_probs / t_softmax, dim=1), labels)
			#print(loss)
			loss.backward()
			final_model_optim.step()
	
	train_acc, test_acc = get_train_test_acc(user_list, final_model)
	print(f"final model: train acc {train_acc},test acc {test_acc}")
	
	return final_model, train_acc, test_acc


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
		## instead of using loss, we use prob
		this_batch_prob = torch.tensor([pred[i][labels[i]] for i in range(len(labels))])
		# print (this_batch_prob.shape)
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
		pred = F.softmax(outputs, dim=1).detach()
		
		labels_list.append(labels.detach())

		
		logits_list.append(pred)
		# loss = criterion(outputs, labels).detach()
		## instead of using loss, we use prob
		this_batch_prob = torch.tensor([pred[i][labels[i]] for i in range(len(labels))])
		all_test_loss.append(this_batch_prob)
		all_test_prediction.append(pred)
	
	logits = torch.cat(logits_list).cuda()
	labels = torch.cat(labels_list).cuda()
	ece_criterion = _ECELoss().cuda()
	ece_loss = ece_criterion(logits, labels, islogit=False).detach().item()
	print(f"POST DEF ECE LOSS:{ece_loss}")
	
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


# do label only attack for all def.
def label_only_attack(model, dataset, training_index, validation_index):
	### do HSJA for each instance and calculate the distance
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	training_data = dataset.train_data[training_index]
	training_label = dataset.train_label[training_index]
	test_data = dataset.test_data
	test_label = dataset.test_label
	### balance evaluation set
	min_len = min(len(test_label), len(training_label))
	# min_len = 200 # for testing if correct
	training_index = np.random.choice(len(training_label), min_len, replace=False)
	testing_index = np.random.choice(len(test_label), min_len, replace=False)
	
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
	for this_data, this_label, _ in tqdm(train_loader):
		# this_adv_data,_ = attack_method._perturb(this_data,this_label)
		# print (this_adv_data.shape,this_data.shape)
		# this_distance = torch.linalg.norm(torch.abs(this_adv_data-this_data))
		this_data = this_data.to(device)
		this_pred = torch.argmax(model(this_data), dim=1)
		# print (this_pred.shape)
		this_label = this_label.to(device)
		if (torch.squeeze(this_pred) != this_label):
			this_distance = 0
		else:
			this_distance = attack_method._perturb(this_data, this_label)
		train_distance.append(this_distance)
	
	### get test data distance
	test_distance = []
	for this_data, this_label, _ in tqdm(test_loader):
		this_data = this_data.to(device)
		this_pred = torch.argmax(model(this_data), dim=1)
		this_label = this_label.to(device)
		if (torch.squeeze(this_pred) != this_label):
			this_distance = 0
		else:
			this_distance = attack_method._perturb(this_data, this_label)
		test_distance.append(this_distance)
	
	train_distance = np.array(train_distance)
	test_distance = np.array(test_distance)
	
	print(f"train distance {train_distance}")
	print(f"test distance {test_distance}")
	
	### report AUC score and TPR when FPR @ 0.1%
	label = np.concatenate((np.ones((len(train_distance))), np.zeros((len(test_distance)))))
	auc = roc_auc_score(label, np.concatenate((train_distance, test_distance), axis=0))
	print(f"AUC score {auc}")
	fpr_threshold = 0.001
	threshold = np.sort(test_distance)[int(len(test_distance) * fpr_threshold)]
	print(f"distance threshold {threshold}")
	cnt = 0
	for i in range(len(train_distance)):
		if (train_distance[i] <= threshold):
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
	
	for shadow_model_index in range(args.shadow_model_number):
		print("shadow model index ", shadow_model_index)
		target_dataset = dataset(dataset_name=args.dataset)
		num_classes = len(np.unique(target_dataset.train_label))
		user_list = [User(dataset=args.dataset, model_name=args.model_name, id=i) for i in range(args.user_number)]
		
		#if (args.test_result):
	#		break
		
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
			# target_model = densenet(num_classes=num_classes, depth=100, growthRate=12, compressionRate=2, dropRate=0).to(device)
			target_model = ModuleValidator.fix(target_model)
		else:
			target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
		
		# print(target_model)
		print(count_parameters(target_model))
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

		label_only_attack(target_model, target_dataset, training_index=training_partition, validation_index=validation_partition)
		exit(0)
		### save all info for canary attack
		keep = training_partition
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
		
		print(f"save model:{get_naming_mid_str() + str(shadow_model_index) + '.pth'}")
		
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
	
	#print(f"loss based attack, avg auc {np.average(all_loss_auc)}, std auc {np.std(all_loss_auc)},"
	#	  f"avg plr {np.average(all_loss_plr)}, std plr {np.std(all_loss_plr)}")
	
	#auc, plr, all_blackbox_loss_val, all_blackbox_loss_label = get_blackbox_auc_no_shadow(all_member_loss, all_nonmember_loss, fpr_threshold=args.fpr_threshold)
	#print(f"loss based attack, putting all data together, auc {auc}, plr {plr}")
	
	auc, plr = get_blackbox_auc_class_nn(all_prob, all_training_partition, all_validation_partition, all_class_label=all_class_label)
	# print (f"classwise nn attack, putting all data together, auc {auc}, plr {plr}")
	
	# get shadow models and run LIRA
	if (args.shadow_model_number > 1):
		all_prob_dis, all_label, auc, plr, _, _ = get_blackbox_auc_lira(all_prob, all_training_partition, all_validation_partition, all_class_label=all_class_label)
		dis_name = './expdata/' + get_naming_mid_str() + 'metric_distribution.npy'
		np.save(dis_name, all_prob_dis)
		label_name = './expdata/' + get_naming_mid_str() + 'metric_label.npy'
		np.save(label_name, all_label)
		print(f"LIRA attack, auc {auc}, plr {plr}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--target_data_size', type=int, default=20000)
	parser.add_argument('--target_learning_rate', type=float, default=0.01)
	parser.add_argument('--target_batch_size', type=int, default=100)
	parser.add_argument('--target_epochs', type=int, default=120)
	parser.add_argument('--target_l2_ratio', type=float, default=5e-4)
	parser.add_argument('--dataset', type=str, default='cifar10')
	parser.add_argument('--num_classes', type=int, default=10)
	parser.add_argument('--validation_set_size', type=int, default=40000)
	parser.add_argument('--model_name', type=str, default='alexnet')
	parser.add_argument('--user_number', type=int, default=1)
	parser.add_argument('--schedule', type=int, nargs='+', default=[100,250])
	### vulnerable metric params
	parser.add_argument('--fpr_threshold', type=float, default=0.001)
	# Lira params
	parser.add_argument('--shadow_model_number', type=int, default=10)
	# cross diff loss param
	parser.add_argument('--cross_loss', type=str, default='l1')
	# test param
	parser.add_argument('--test_result', type=int, default=1)
	
	args = parser.parse_args()
	print(vars(args))
	random_seed_list = [1]
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
		
		print(vars(args))
		attack_experiment()
		print(vars(args))

