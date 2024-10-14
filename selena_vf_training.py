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

### 25 models in total
### 10 nonmember models
### resnet 18

total_model_number = 25  ## 25
nonmember_model_number = 10  ## 10
dataset_name = 'cifar100'
num_classes = 100
# target_model = ResNet18(num_classes=num_classes)
# target_model = ModuleValidator.fix(target_model)
model_name = 'alexnet'
target_model = alexnet(num_classes=num_classes)
target_dataset = dataset(dataset_name=dataset_name)
# target_model = TargetNet(dataset_name, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
num_classes = len(np.unique(target_dataset.label))
training_set_size = 20000
target_dataset.train_data = target_dataset.train_data
target_dataset.train_label = target_dataset.train_label
epochs = 120  # for texas, 100
target_batch_size = 100
init_learning_rate = 0.01
momentum = 0.9
decay = 1e-5
# schedule = [80,100]
schedule = [100]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fpr_threshold = 0.001
shadow_model_number = 100  # 40
if (dataset_name == 'purchase' or dataset_name == 'texas'):
	transform_train = None
	transform_test = None
	target_transform = None
# cifar 10 / cifar100
if (dataset_name == 'cifar10' or dataset_name == 'cifar100'):
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


def selena_training():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	##todo: combine this selena with our vfd
	
	## divide train / validation
	training_partition = np.random.choice(len(target_dataset.train_label), training_set_size, replace=False)
	validation_partition = np.setdiff1d(np.arange(len(target_dataset.train_label)), training_partition)
	current_round_train_data = target_dataset.train_data[training_partition]
	current_round_train_label = target_dataset.train_label[training_partition]
	current_round_validation_data = target_dataset.train_data[validation_partition]
	current_round_validation_label = target_dataset.train_label[validation_partition]
	
	## data assignment
	index_assignment = np.zeros((total_model_number, training_set_size))
	for i in range(training_set_size):
		## we randomly select nonmember set
		nonmember_set = np.random.choice(total_model_number, nonmember_model_number, replace=False)
		this_index = np.ones((total_model_number))
		this_index[nonmember_set] = 0
		index_assignment[:, i] = this_index
	
	## train models
	all_model_list = []
	for i in tqdm(range(total_model_number)):
		this_model = copy.deepcopy(target_model)
		this_model_train_data = current_round_train_data[np.arange(training_set_size)[index_assignment[i] == 1]]
		this_model_train_label = current_round_train_label[np.arange(training_set_size)[index_assignment[i] == 1]]
		class_weights = np.ones((len(np.unique(this_model_train_label)))) * training_set_size / (
			len(np.unique(this_model_train_label)) * (np.bincount(this_model_train_label, minlength=len(np.unique(target_dataset.train_label))) + 10))
		
		this_model_train_dataset = part_pytorch_dataset(this_model_train_data, this_model_train_label, train=True, transform=transform_train,
														target_transform=target_transform)
		this_model_train_loader = torch.utils.data.DataLoader(this_model_train_dataset, batch_size=target_batch_size,
															  shuffle=True, num_workers=1)
		class_weights = torch.from_numpy(class_weights)
		class_weights = class_weights.type(torch.float32)
		criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
		this_model.zero_grad(set_to_none=True)
		this_model.train().to(device)
		learning_rate = init_learning_rate
		optimizer = torch.optim.SGD(this_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
		for epoch in range(epochs):
			if (epoch in schedule):
				learning_rate = learning_rate / 10
				for param_group in optimizer.param_groups:
					param_group['lr'] = learning_rate
			for _, (images, labels, data_idx) in enumerate(this_model_train_loader):
				optimizer.zero_grad()
				images, labels = images.to(device), labels.to(device)
				log_probs = this_model(images)
				loss = criterion(log_probs, labels)
				loss.backward()
				optimizer.step()
		all_model_list.append(this_model)
		
	## gather training predictions for self distillation
	soft_label = {i: [] for i in range(training_set_size)}
	for i in range(total_model_number):
		## fina all data indexes that need prediction from this model
		this_model = all_model_list[i]
		all_nonmember_index = np.array(index_assignment[i]).flatten()
		all_index = np.arange(training_set_size)[all_nonmember_index == 0]
		this_model_test_data = current_round_train_data[all_index]
		this_model_test_label = current_round_train_label[all_index]
		this_model_test_dataset = part_pytorch_dataset(this_model_test_data, this_model_test_label, train=False, transform=transform_test,
													   target_transform=target_transform)
		this_model_test_loader = torch.utils.data.DataLoader(this_model_test_dataset, batch_size=target_batch_size,
															 shuffle=False, num_workers=1)
		this_model_pred = []
		for images, labels, data_idx in this_model_test_loader:
			images, labels = images.to(device), labels.to(device)
			# print (images.shape,labels.shape)
			log_probs = this_model(images)
			pred = F.softmax(log_probs, dim=1)
			this_model_pred.append(pred.detach())
		
		this_model_pred = torch.vstack(this_model_pred)
		# print(this_model_pred.shape)
		## after getting all predictions, we put these predictions to soft labels
		for j in range(len(all_index)):
			soft_label[all_index[j]].append(this_model_pred[j])
	
	### average the predictions and get soft labels
	avg_soft_label = []
	for k, v in soft_label.items():
		# print (k,v,len(v))
		this_all_soft_label = torch.vstack(v)
		# print (this_all_soft_label.shape)
		this_avg = torch.mean(this_all_soft_label, dim=0)
		# print (this_avg.shape)
		avg_soft_label.append(this_avg)
	avg_soft_label = torch.vstack(avg_soft_label).cpu().numpy()
	# print (avg_soft_label.shape)
	
	## self distillation
	final_model = copy.deepcopy(target_model)
	this_model_train_data = current_round_train_data
	this_model_train_label = avg_soft_label
	this_model_test_data = target_dataset.test_data
	this_model_test_label = target_dataset.test_label
	class_weights = np.ones((len(np.unique(current_round_train_label)))) * training_set_size / (
		len(np.unique(current_round_train_label)) * (np.bincount(current_round_train_label, minlength=len(np.unique(target_dataset.train_label))) + 10))
	
	this_model_train_dataset = part_pytorch_dataset(this_model_train_data, this_model_train_label, train=True, transform=transform_train,
													target_transform=target_transform, float_target=True)
	this_model_train_eval_dataset = part_pytorch_dataset(this_model_train_data, current_round_train_label, train=True, transform=transform_test,
														 target_transform=target_transform, float_target=True)
	this_model_test_dataset = part_pytorch_dataset(this_model_test_data, this_model_test_label, train=False, transform=transform_test,
												   target_transform=target_transform)
	this_model_train_loader = torch.utils.data.DataLoader(this_model_train_dataset, batch_size=target_batch_size,
														  shuffle=True, num_workers=1)
	this_model_train_eval_loader = torch.utils.data.DataLoader(this_model_train_eval_dataset, batch_size=target_batch_size,
															   shuffle=False, num_workers=1)
	this_model_test_loader = torch.utils.data.DataLoader(this_model_test_dataset, batch_size=target_batch_size,
														 shuffle=False, num_workers=1)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	class_weights = torch.from_numpy(class_weights)
	class_weights = class_weights.type(torch.float32)
	criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
	final_model.zero_grad(set_to_none=True)
	final_model.train().to(device)
	learning_rate = init_learning_rate
	optimizer = torch.optim.SGD(final_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
	for epoch in range(epochs):
		if (epoch in schedule):
			learning_rate = learning_rate / 10
			for param_group in optimizer.param_groups:
				param_group['lr'] = learning_rate
		for _, (images, labels, data_idx) in enumerate(this_model_train_loader):
			optimizer.zero_grad()
			images, labels = images.to(device), labels.to(device)
			log_probs = final_model(images)
			loss = criterion(log_probs, labels)
			loss.backward()
			optimizer.step()
	
	### calculate train / test accuracy
	total = 0.0
	correct = 0.0
	with torch.no_grad():
		final_model.eval()
		for images, labels, _ in this_model_train_eval_loader:
			final_model.zero_grad()
			images = images.to(device)
			outputs = final_model(images)
			labels = labels.to(device)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	print(f"train accuracy {acc}")
	
	total = 0.0
	correct = 0.0
	with torch.no_grad():
		final_model.eval()
		for images, labels, _ in this_model_test_loader:
			final_model.zero_grad()
			images = images.to(device)
			outputs = final_model(images)
			labels = labels.to(device)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	print(f"test accuracy {acc}")
	
	### get the final model and test it against MI attacks
	all_train_loss = []
	all_validation_loss = []
	all_train_pred = []
	all_validation_pred = []
	this_model_train_data = current_round_train_data
	this_model_train_label = current_round_train_label
	this_model_train_dataset = part_pytorch_dataset(this_model_train_data, this_model_train_label, train=True, transform=transform_test,
													target_transform=target_transform)
	this_model_train_loader = torch.utils.data.DataLoader(this_model_train_dataset, batch_size=target_batch_size,
														  shuffle=False, num_workers=1)
	this_model_test_dataset = part_pytorch_dataset(current_round_validation_data, current_round_validation_label, train=False, transform=transform_test,
												   target_transform=target_transform)
	this_model_test_loader = torch.utils.data.DataLoader(this_model_test_dataset, batch_size=target_batch_size,
														 shuffle=False, num_workers=1)
	final_model.eval()
	for _, (images, labels, data_idx) in enumerate(this_model_train_loader):
		images, labels = images.to(device), labels.to(device)
		log_probs = final_model(images)
		pred = F.softmax(log_probs, dim=1).detach()
		this_batch_prob = torch.tensor([pred[i][labels[i]] for i in range(len(labels))])
		all_train_loss.append(this_batch_prob.detach().cpu().numpy())
		all_train_pred.append(pred)
	
	for _, (images, labels, data_idx) in enumerate(this_model_test_loader):
		images, labels = images.to(device), labels.to(device)
		log_probs = final_model(images)
		pred = F.softmax(log_probs, dim=1).detach()
		this_batch_prob = torch.tensor([pred[i][labels[i]] for i in range(len(labels))])
		all_validation_loss.append(this_batch_prob.detach().cpu().numpy())
		all_validation_pred.append(pred)
	
	all_train_loss = np.array(all_train_loss).flatten()
	all_validation_loss = np.array(all_validation_loss).flatten()
	all_train_pred = torch.vstack(all_train_pred).cpu().numpy()
	all_validation_pred = torch.vstack(all_validation_pred).cpu().numpy()
	# print (all_train_pred.shape,all_validation_pred.shape)
	get_blackbox_auc_no_shadow(all_train_loss=all_train_loss, all_test_loss=all_validation_loss, fpr_threshold=fpr_threshold)
	
	return final_model, np.concatenate((all_train_pred, all_validation_pred), axis=0), training_partition, validation_partition, target_dataset.train_label


all_pred = []
all_train_index = []
all_validation_index = []
all_class_label = []
for shadow_idx in range(shadow_model_number):
	this_shadow_model, this_all_pred, this_all_train_index, this_all_validation_index, this_all_class_label = selena_training()
	all_pred.append(this_all_pred)
	all_train_index.append(this_all_train_index)
	all_validation_index.append(this_all_validation_index)
	all_class_label.append(this_all_class_label)
	
	### save all info for canary attack
	keep = this_all_train_index
	keep_bool = np.full((len(target_dataset.train_data)), False)
	keep_bool[keep] = True
	state = {"model": this_shadow_model.state_dict(),
			 "in_data": keep,
			 "keep_bool": keep_bool,
			 "model_arch": model_name}
	os.makedirs('saved_models/selena/', exist_ok=True)
	### include def in path name if def is applied
	torch.save(state, './saved_models/selena/' + 'selena_' + model_name + '_' + dataset_name + '_' + str(shadow_idx) + '.pth')
	print(f"save model:{'./saved_models/selena/' + 'selena_' + model_name + '_' + dataset_name + '_' + str(shadow_idx) + '.pth'}")

all_train_index = np.stack(all_train_index)
all_validation_index = np.stack(all_validation_index)
all_pred = np.stack(all_pred)
all_class_label = np.array(all_class_label)
all_name = f'./expdata/selena_{model_name}_{dataset_name}_{shadow_model_number}_all_info.npz'
np.savez(all_name, all_pred, all_train_index, all_validation_index, all_class_label)
print(all_name)

get_blackbox_auc_lira(all_pred, all_train_index, all_validation_index, all_class_label, fpr_threshold=fpr_threshold)




