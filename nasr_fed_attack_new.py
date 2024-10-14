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


cifar_alexnet_dict = {'output_dim':512,'kernel_size':100,'target_data_size':4000,'epochs':300,'model':'alexnet','dataset':'cifar100','num_classes':100,'eval_data_size':1000,'num_step':40,'target_epochs':[100, 150, 200, 250, 300],'last_fc_shape':256}
purchase_dict = {'output_dim':3*256,'kernel_size':33,'target_data_size':4000,'epochs':100,'model':'purchase','dataset':'purchase','num_classes':100,'eval_data_size':1000,'num_step':40,'target_epochs':[60,70,80,90,100],'last_fc_shape':256}
cifar_densenet_dict =  {'output_dim':4*256,'kernel_size':166,'target_data_size':4000,'epochs':500,'model':'densenet_cifar','dataset':'cifar100','num_classes':100,'eval_data_size':1000,'num_step':40,'target_epochs':[300,350,400,450,500],'last_fc_shape':342}
kidney_dict = {'output_dim':512,'kernel_size':50,'target_data_size':400,'epochs':150,'model':'kidney','dataset':'kidney','num_classes':4,'eval_data_size':400,'num_step':4,'target_epochs':[110,120,130,140,150],'last_fc_shape':1024}
retina_dict = {'output_dim':3*256,'kernel_size':33,'target_data_size':2000,'epochs':100,'model':'retina','dataset':'retina','num_classes':4,'eval_data_size':1000,'num_step':20,'target_epochs':[60,70,80,90,100],'last_fc_shape':1024}
chest_dict = {'output_dim':3*256,'kernel_size':33,'target_data_size':200,'epochs':100,'model':'chest','dataset':'chest','num_classes':2,'eval_data_size':200,'num_step':2,'target_epochs':[110,120,130,140,150],'last_fc_shape':1024}
skin_dict = {'output_dim':512,'kernel_size':50,'target_data_size':500,'epochs':150,'model':'skin','dataset':'skin','num_classes':23,'eval_data_size':500,'num_step':5,'target_epochs':[110,120,130,140,150],'last_fc_shape':1024}
medical_mnist_dict = {'output_dim':512,'kernel_size':50,'target_data_size':4000,'epochs':50,'model':'medical_mnist','dataset':'medical_mnist','num_classes':6,'eval_data_size':1000,'num_step':40,'target_epochs':[10,20,30,40,50],'last_fc_shape':84}

# NOTE: change setting dict
setting_dict = cifar_alexnet_dict

role = 'server'
batch_privacy = 32 ## should be 32.
checkpoint_path = './checkpoints/'
number_parties = 10
cudnn.benchmark = True
dataset = setting_dict['dataset']
model = setting_dict['model']
training_data_size = setting_dict['target_data_size']
num_classes = setting_dict['num_classes']
input_feature_num = 600 ## for purchase



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
	print (f" fpr {2*10/len(label)}, tpr {2*tpr_count/len(label)}") ## 2 is because the evaluation set is balanced
	return 2*tpr_count/(len(label))


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
	    
manualSeed = 0
random.seed(manualSeed)
torch.manual_seed(manualSeed)
import warnings
warnings.filterwarnings("ignore")

class InferenceAttack_HZ_FED(nn.Module):
	def __init__(self, num_classes, num_feds,last_fc_shape):
		self.num_classes = num_classes
		self.num_feds = num_feds
		super(InferenceAttack_HZ_FED, self).__init__()
		self.grads_conv = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Conv2d(1, 1000, kernel_size=(1, num_classes), stride=1),
			nn.ReLU(),
		)
		self.grads_linear = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(last_fc_shape * 1000, 1024),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, 128),
			nn.ReLU(),
		)
		self.labels = nn.Sequential(
			nn.Linear(num_classes, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
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
			nn.Linear(64 * 4 * self.num_feds, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)
		for key in self.state_dict():
			#print(key)
			if key.split('.')[-1] == 'weight':
				nn.init.normal(self.state_dict()[key], std=0.01)
				#print(key)
			elif key.split('.')[-1] == 'bias':
				self.state_dict()[key][...] = 0
		self.output = nn.Sigmoid()
	
	def forward(self, gs, ls, cs, os):
		
		
		for i in range(self.num_feds):
			out_g = self.grads_conv(gs[i]).view([gs[i].size()[0], -1])
			out_g = self.grads_linear(out_g)
			out_c = self.correct(cs[i])
			out_o = self.preds(os[i])
			if i == 0:
				com_inp = torch.cat((out_g, out_c, out_o), 1)
			else:
				com_inp = torch.cat((out_g, out_c, out_o, com_inp), 1)
			#get_gpu_status()
		is_member = self.combine(com_inp)
		return self.output(is_member)

def privacy_train_fed(trainloader, testloader, models, inference_model, classifier_criterion, criterion, classifier_optimizers, optimizer, num_batchs=40):
	global best_acc
	
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	inference_model.train()
	for model in models:
		model.eval()
	# switch to evaluate mode
	
	end = time.time()
	for batch_idx, ((tr_input, tr_target,_), (te_input, te_target,_)) in enumerate(zip(trainloader, testloader)):
		# measure data loading time
		if batch_idx > num_batchs:
			break
		data_time.update(time.time() - end)
		tr_input = tr_input.type(torch.float32)
		te_input = te_input.type(torch.float32)
		tr_target = tr_target
		te_target = te_target
		

		v_tr_input = torch.autograd.Variable(tr_input).cuda()
		v_te_input = torch.autograd.Variable(te_input).cuda()
		v_tr_target = torch.autograd.Variable(tr_target).cuda()
		v_te_target = torch.autograd.Variable(te_target).cuda()
		
		# compute output
		model_input = torch.cat((v_tr_input, v_te_input))
		pred_outputs = []
		for i in range(len(models)):
			pred_outputs.append(models[i](model_input))
		
		infer_input = torch.cat((v_tr_target, v_te_target))
		
		one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), num_classes)))).type(torch.FloatTensor)
		target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.LongTensor).view([-1, 1]).data, 1)
		infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr).cuda()
		
		models_outputs = []
		correct_labels = []
		model_grads = []
		
		for m_n in range(len(models)):
			
			correct = torch.sum(pred_outputs[m_n] * infer_input_one_hot, dim=1)
			grads = torch.zeros(0)
			
			for i in range(2 * batch_privacy):
				loss_classifier = classifier_criterion(pred_outputs[m_n][i].view([1, -1]), infer_input[i].view([-1]))
				classifier_optimizers[m_n].zero_grad()
				if i == (2 * batch_privacy) - 1:
					loss_classifier.backward(retain_graph=False)
				else:
					loss_classifier.backward(retain_graph=True)
				
				#print (f"model {model}")
				## NOTE: CHANGE THIS
				if (setting_dict['model'] == 'alexnet'):
				# alexnet
					g = models[m_n].linear1.weight.grad.view([1, 1, 256, num_classes])
				elif (setting_dict['model'] == 'densenet_cifar'):
				## densenet
					g = models[m_n].fc.weight.grad.view([1, 1, 342, num_classes])
				elif (dataset == 'purchase'):
					g = models[m_n].fc4.weight.grad.view([1, 1, 256, 100])
				elif (dataset == 'kidney' or dataset == 'skin' or dataset =='chest' or dataset == 'retina'):
					g = models[m_n].linear1.weight.grad.view([1, 1, 1024, num_classes])
				
				if grads.size()[0] != 0:
					grads = torch.cat((grads, g))
				else:
					grads = g
			
			grads = torch.autograd.Variable(torch.from_numpy(grads.data.cpu().numpy())).cuda()
			c = torch.autograd.Variable(torch.from_numpy(correct.view([-1, 1]).data.cpu().numpy())).cuda()
			preds = torch.autograd.Variable(torch.from_numpy(pred_outputs[m_n].data.cpu().numpy())).cuda()
			models_outputs.append(preds)
			correct_labels.append(c)
			model_grads.append(grads)
		
		model_grads = torch.stack(model_grads)
		correct_labels = torch.stack(correct_labels)
		models_outputs = torch.stack(models_outputs)
		member_output = inference_model(model_grads, infer_input_one_hot, correct_labels, models_outputs)
		
		is_member_labels = torch.from_numpy(np.reshape(np.concatenate((np.ones(v_tr_input.size(0)), np.zeros(v_te_input.size(0)))), [-1, 1]))
		v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.float32).cuda()
		
		loss = criterion(member_output, v_is_member_labels)
		
		# measure accuracy and record loss
		prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
		losses.update(loss.item(), model_input.size(0))
		top1.update(prec1, model_input.size(0))
		
		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		
		# plot progress
		#if batch_idx % 10 == 0:
		#	print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
		#		batch=batch_idx,
		#		size=len(trainloader),
		#		data=data_time.avg,
		#		bt=batch_time.avg,
		#		loss=losses.avg,
		#		top1=top1.avg,
		#	))
	
	return (losses.avg, top1.avg)


def privacy_test_fed(trainloader, testloader, models, inference_model, classifier_criterion, criterion, classifier_optimizers, optimizer, num_batchs=1000):
	global best_acc
	
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	
	total_pred = []
	total_true_label = []
	
	inference_model.eval()
	for model in models:
		model.eval()
	# switch to evaluate mode
	
	end = time.time()
	for batch_idx, ((tr_input, tr_target,_), (te_input, te_target,_)) in enumerate(zip(trainloader, testloader)):
		# measure data loading time
		if batch_idx > num_batchs:
			break
		data_time.update(time.time() - end)
		tr_input = tr_input.type(torch.float32)
		te_input = te_input.type(torch.float32)
		tr_target = tr_target
		te_target = te_target
		
		v_tr_input = torch.autograd.Variable(tr_input).cuda()
		v_te_input = torch.autograd.Variable(te_input).cuda()
		v_tr_target = torch.autograd.Variable(tr_target).cuda()
		v_te_target = torch.autograd.Variable(te_target).cuda()
		
		# compute output
		model_input = torch.cat((v_tr_input, v_te_input))
		pred_outputs = []
		for i in range(len(models)):
			pred_outputs.append(models[i](model_input))
		
		infer_input = torch.cat((v_tr_target, v_te_target))
		
		one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), num_classes)))).type(torch.FloatTensor)
		target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.LongTensor).view([-1, 1]).data, 1)
		infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr).cuda()
		
		models_outputs = []
		correct_labels = []
		model_grads = []
		
		for m_n in range(len(models)):
			
			correct = torch.sum(pred_outputs[m_n] * infer_input_one_hot, dim=1)
			grads = torch.zeros(0)
			
			for i in range(2 * batch_privacy):
				loss_classifier = classifier_criterion(pred_outputs[m_n][i].view([1, -1]), infer_input[i].view([-1]))
				classifier_optimizers[m_n].zero_grad()
				if i == (2 * batch_privacy) - 1:
					loss_classifier.backward(retain_graph=False)
				else:
					loss_classifier.backward(retain_graph=True)
				# NOTE: need to change the gradient size.
				if (setting_dict['model'] == 'alexnet'):
				# alexnet
					g = models[m_n].linear1.weight.grad.view([1, 1, 256, num_classes])
				elif (setting_dict['model'] == 'densenet_cifar'):
				## densenet
					g = models[m_n].fc.weight.grad.view([1, 1, 342, num_classes])
				elif (dataset == 'purchase'):
					g = models[m_n].fc4.weight.grad.view([1, 1, 256, 100])
				elif (dataset == 'kidney' or dataset == 'skin' or dataset =='chest' or dataset == 'retina'):
					g = models[m_n].linear1.weight.grad.view([1, 1, 1024, num_classes])
				
				
				if grads.size()[0] != 0:
					
					grads = torch.cat((grads, g))
				
				else:
					grads = g
			
			grads = torch.autograd.Variable(torch.from_numpy(grads.data.cpu().numpy())).cuda()
			c = torch.autograd.Variable(torch.from_numpy(correct.view([-1, 1]).data.cpu().numpy())).cuda()
			preds = torch.autograd.Variable(torch.from_numpy(pred_outputs[m_n].data.cpu().numpy())).cuda()
			models_outputs.append(preds)
			correct_labels.append(c)
			model_grads.append(grads)
		member_output = inference_model(model_grads, infer_input_one_hot, correct_labels, models_outputs)
		
		is_member_labels = torch.from_numpy(np.reshape(np.concatenate((np.ones(v_tr_input.size(0)), np.zeros(v_te_input.size(0)))), [-1, 1]))
		
		v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.float32).cuda()
		
		loss = criterion(member_output, v_is_member_labels)
		
		
		total_pred.append(np.squeeze(member_output.detach().cpu().numpy()))
		total_true_label.append(np.squeeze(v_is_member_labels.cpu().numpy()))
		
		# measure accuracy and record loss
		prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
		losses.update(loss.item(), model_input.size(0))
		top1.update(prec1, model_input.size(0))
		
		# compute gradient and do SGD step
		
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		
		# plot progress
		#if batch_idx % 10 == 0:
		#	print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
		#		batch=batch_idx,
		#		size=len(trainloader),
		#		data=data_time.avg,
		#		bt=batch_time.avg,
		#		loss=losses.avg,
		#		top1=top1.avg,
		#	))
	total_pred = np.array(total_pred).flatten()
	total_true_label = np.array(total_true_label).flatten()
	
	#print (total_pred.shape,total_true_label.shape)
	#print (total_true_label)
	# 0 means member, 1 means non-member..
	from sklearn.metrics import roc_auc_score,roc_curve
	fpr, tpr, thresholds = roc_curve(total_true_label, total_pred, pos_label=1)
	return_tpr = get_tpr(total_pred,total_true_label)
	
	return (losses.avg, top1.avg,roc_auc_score(total_true_label,total_pred),fpr,tpr,return_tpr)

def assign_part_dataset(dic,user_idx):
	dataset = dic['dataset']
	model = dic['model']
	training_data_size = dic['target_data_size']
	#eval_data_size = dic['eval_data_size']
	
	if (dataset == 'cifar10' or dataset == 'cifar100'):
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
	elif (dataset == 'purchase'):
		transform_test = None
		target_transform = transforms.ToTensor()
	else:
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
		
	
	train_data = np.load(f'./checkpoints/{role}_{dataset}_{model}_{training_data_size}_train_data_{user_idx}.npy')
	train_label = np.load(f'./checkpoints/{role}_{dataset}_{model}_{training_data_size}_train_label_{user_idx}.npy')
	test_data = np.load(f'./checkpoints/{role}_{dataset}_{model}_{training_data_size}_test_data_{user_idx}.npy')
	test_label = np.load(f'./checkpoints/{role}_{dataset}_{model}_{training_data_size}_test_label_{user_idx}.npy')
	
	min_len = min(len(train_label),len(test_label))
	min_len = min(min_len,setting_dict['eval_data_size'])
	print (f"min len {min_len}, train len {len(train_label)}, test len {len(test_label)}")
	train_selected_index = np.random.choice(np.arange(len(train_label)),min_len,replace=False)
	test_selected_index = np.random.choice(np.arange(len(test_label)),min_len,replace=False)
	train_data = train_data[train_selected_index]
	train_label = train_label[train_selected_index]
	test_data = test_data[test_selected_index]
	test_label = test_label[test_selected_index]

	half_min_len = int(0.5*min_len)
	train_intrain_index = np.random.choice(np.arange(min_len),half_min_len,replace=False)
	train_intest_index = np.random.choice(np.arange(min_len),half_min_len,replace=False)
	test_intrain_index = np.setdiff1d(np.arange(training_data_size),train_intrain_index)
	test_intest_index = np.setdiff1d(np.arange(training_data_size),train_intest_index)
	
	#print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)
	#### create dataset and dataloader
	train_intrain = part_pytorch_dataset(train_data[train_intrain_index], train_label[train_intrain_index], train=False, transform=transform_test,
	                             target_transform=target_transform)
	train_intest = part_pytorch_dataset(test_data[train_intest_index], test_label[train_intest_index], train=False, transform=transform_test,
	                             target_transform=target_transform)
	test_intrain = part_pytorch_dataset(train_data[test_intrain_index], train_label[test_intrain_index], train=False, transform=transform_test,
	                             target_transform=target_transform)
	test_intest = part_pytorch_dataset(test_data[test_intest_index], test_label[test_intest_index], train=False, transform=transform_test,
	                             target_transform=target_transform)
	
	train_intrain_data_loader = torch.utils.data.DataLoader(train_intrain, batch_size=batch_privacy, shuffle=False, num_workers=1,drop_last=True )
	test_intest_data_loader = torch.utils.data.DataLoader(test_intest, batch_size=batch_privacy, shuffle=False, num_workers=1,drop_last=True )
	train_intest_data_loader = torch.utils.data.DataLoader(train_intest, batch_size=batch_privacy, shuffle=False, num_workers=1,drop_last=True )
	test_intrain_data_loader = torch.utils.data.DataLoader(test_intrain, batch_size=batch_privacy, shuffle=False, num_workers=1,drop_last=True )
	
	return train_intrain_data_loader,train_intest_data_loader,test_intrain_data_loader,test_intest_data_loader


print (f"dataset {dataset}")

all_results = []

for user_idx in range(number_parties):
	
	this_user_result = (0,0,0)

	criterion = nn.CrossEntropyLoss()
	criterion_attack = nn.MSELoss()
	inferenece_model = InferenceAttack_HZ_FED(num_classes, 5, last_fc_shape= setting_dict['last_fc_shape'])
	inferenece_model = inferenece_model.type(torch.float32).cuda()
	optimizer_mem = optim.Adam(inferenece_model.parameters(), lr=0.0001)
	
	if (user_idx == 0):
		pytorch_total_params = sum(p.numel() for p in inferenece_model.parameters())
		print(f"total params {pytorch_total_params}")

	nets = []
	optims = []
	for i in setting_dict['target_epochs']: ## 30/30, 10/20
		state_dict_file = f'./checkpoints/{role}_{dataset}_{model}_{training_data_size}_{i}.pt'
		state_dict = torch.load(state_dict_file)
		
		if (model == 'alexnet'):
			net_main =  alexnet(num_classes=100)
			optim_main = optim.SGD(net_main.parameters(),lr=0.01)
		elif (model == 'densenet_cifar'):
			net_main = convert_batchnorm_modules(densenet(depth=100, num_classes=num_classes),converter=_batchnorm_to_groupnorm_new)
			optim_main = optim.SGD(net_main.parameters(), lr=0.1)
		elif (dataset == 'purchase'):
			net_main = TargetNet(dataset,input_feature_number=600)
			optim_main = optim.Adam(net_main.parameters(), lr=0.0001)
		else:
			net_main = TargetNet(dataset)
			optim_main = optim.SGD(net_main.parameters(),lr=0.01)
	
		net_main = net_main.cuda()
		for k,v in state_dict['model_state_dict'].items():
			state_dict['model_state_dict'][k] = v.type(torch.float32)
		net_main.load_state_dict(state_dict['model_state_dict'])
		optim_main.load_state_dict(state_dict['optimizer_state_dict'])
		nets.append(net_main)
		optims.append(optim_main)

	private_trainloader_intrain,private_trainloader_intest,private_testloader_intrain,private_testloader_intest = assign_part_dataset(setting_dict,user_idx)

	for i in range(100):
		privacy_train_fed(private_trainloader_intrain, private_trainloader_intest, nets, inferenece_model, criterion, criterion_attack, optims,
                  optimizer_mem)
		bb = privacy_test_fed(private_testloader_intrain, private_testloader_intest, nets, inferenece_model, criterion, criterion_attack, optims,
                      optimizer_mem)
		print (f"epoch {i}, avg test loss{bb[0]}, avg test acc {bb[1]}, auc score {bb[2]}, tpr at fpr=1e-3 {bb[-1]}")
		
		if (bb[1]>this_user_result[0]):
			this_user_result = (bb[1],bb[2],bb[3])
	
		save_name = f'{dataset}_{model}_{training_data_size}_nasr_result.npy'
		np.save(save_name,bb)
		
	all_results.append(this_user_result)
		
print ( f'{role}_{dataset}_{model}_{training_data_size}')
all_results = np.array(all_results)
all_results = np.average(all_results,axis=0)
print (f"avg result:acc {all_results[0]}, auc {all_results[1]},tpr {all_results[2]}")