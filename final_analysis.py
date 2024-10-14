import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
import copy
# warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch
import sklearn

sklearn.utils.check_random_state(12345)
from scipy.spatial import distance
from scipy.stats import norm
from scipy import stats
import numpy as np
from sklearn import metrics

prefix = './new_expdata/'
np.set_printoptions(suppress=True)


def report_acc(data, label):
	default_runs = 1
	acc = 0
	for _ in range(default_runs):
		acc += sub_report_acc(data, label)
	
	return acc / default_runs


def convert_count_to_acc_count(count):
	acc_count = [count[0]]
	for i in range(1, len(count)):
		acc_count.append(acc_count[-1] + count[i])
	return np.array(acc_count) / np.sum(count)


def sub_report_acc(data, label):
	### for one number feature, we can find the best threshold by just iterating through
	if (len(np.squeeze(data).shape) == 1):
		best_acc = 0
		data = np.squeeze(data)
		sorted_idx = np.argsort(data)
		data = data[sorted_idx]
		label = label[sorted_idx]
		for i in range(len(label)):
			
			while (i < len(label) - 1):
				if (data[i] == data[i + 1]):
					i = i + 1
				else:
					break
			
			predicted = np.array([0 for j in range(i)] + [1 for j in range(len(label) - i)]).astype('int64')
			this_acc = len(np.arange(len(label))[label == predicted]) / len(label)
			best_acc = max(best_acc, this_acc)
		return best_acc


def avg_roc_cal(metric_list):
	num_user = len(metric_list)
	# print(len(metric_list))
	### 10 users avg. we need to have a uniformed FPR
	uniform_fpr = [1e-5 * i for i in range(1, 10)] + [1e-4 * i for i in range(1, 10)] + [1e-3 * i for i in range(1, 10)]
	uniform_fpr = uniform_fpr + [1e-2 * i for i in range(1, 10)] + [1e-1 * i for i in range(1, 10)]
	
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


# def normal_prob_cal(avg,std,x):
# p =  1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-avg)/std)*((x-avg)/std))
# return p


def convert_metric_to_prob(metric, metric_name='loss'):
	metric = np.nan_to_num(metric)
	## this only works for cosine and loss
	if (metric_name == 'loss'):
		metric = np.log((np.exp(-1 * metric) / (1 - np.exp(-1 * metric) + 1e-8)))
	
	## we need to know a non-member distribution (assumed to be normal), and then calculate the probability for each instance
	nonmember_metric = metric[int(len(metric) / 2):]  # we assume the remaining half is nonmember
	## estimate the value of normal distribution
	avg = np.average(nonmember_metric)
	std = np.std(nonmember_metric)
	
	from scipy.stats import norm
	this_norm = norm.fit(nonmember_metric)
	
	prob_list = [norm.cdf(this_metric) for this_metric in metric]
	
	# print (avg,std)
	# print (metric)
	# print (prob_list)
	if (metric_name == 'loss'):
		return np.nan_to_num(np.array(prob_list)) * (-1)
	
	return np.nan_to_num(np.array(prob_list))


def read_all_comm_round_data(comm_round_list, prefix, mid_str, feature_str, final_str, num_layers, num_user, have_label=True, label_str=''):
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
	
	
	elif (len(comm_round_list) > 1):
		### be careful, this only works for the one batch attack
		feature_num = 14
		all_data = []
		all_label = []
		
		for comm_round_idx in comm_round_list:
			data_name = prefix + feature_str + mid_str + str(comm_round_idx) + final_str
			data = np.load(data_name, allow_pickle=True)
			label_name = prefix + label_str + mid_str + str(comm_round_idx) + final_str
			label = np.load(label_name)
			
			all_data.append(data)
			all_label.append(label)
		
		all_data = np.squeeze(np.array(all_data))
		all_label = np.array(all_label)
		
		print(all_data.shape, all_label.shape)
		
		new_all_data = np.zeros_like(np.squeeze(data))
		new_all_label = label
		
		if (len(all_data.shape) == 5):
			### for all feature case
			for data_idx in range(all_data.shape[1]):
				for user_idx in range(10):
					for layer_idx in range(num_layers):
						this_cos = np.amax(all_data[:, data_idx, user_idx, layer_idx, 0])
						this_grad_diff = np.amax(all_data[:, data_idx, user_idx, layer_idx, 1] - all_data[:, data_idx, user_idx, layer_idx, 2])
						this_grad_norm = np.amin(all_data[:, data_idx, user_idx, layer_idx, 3])
						new_all_data[data_idx, user_idx, layer_idx, 0] = this_cos
						new_all_data[data_idx, user_idx, layer_idx, 1] = this_grad_diff
						new_all_data[data_idx, user_idx, layer_idx, 3] = this_grad_norm
						
						if (data_idx == 0 and user_idx == 0 and layer_idx == 0):
							print(all_data[:, data_idx, user_idx, layer_idx, 0])
		
		
		elif (len(all_data.shape) == 3):
			### for all loss case
			for data_idx in range(all_data.shape[1]):
				for user_idx in range(10):
					this_loss = np.amin(all_data[:, data_idx, user_idx])
					new_all_data[data_idx, user_idx] = this_loss
		
		else:
			### for single loss case
			for data_idx in range(all_data.shape[1]):
				this_loss = np.amin(all_data[:, data_idx])
				new_all_data[data_idx] = this_loss
		
		print(new_all_data.shape, new_all_label.shape)
		
		return np.squeeze(new_all_data), np.squeeze(new_all_label)


def all_analysis(epochs, prefix, mid_str, dataset, model, target_data_size, eval_data_size, special_layers=None,
                 num_layers=12, num_user=5, client_adversary=0, best_layer=0, comm_round_list=[]):
	if (client_adversary):
		num_user -= 1
		num_valid_user = 1
	else:
		num_valid_user = num_user
	
	total_data_num = (num_user * 2) * eval_data_size
	# if (eval_data_size == target_data_size):
	#    total_data_num = (num_user * 2) * 100
	
	# print (f"total data num {total_data_num}")
	
	all_epoch_cos = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_target_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_grad_norm = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_grad_diff = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_layer_cos = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	all_epoch_layer_grad_diff = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	all_epoch_target_after_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_layer_grad_norm = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	all_epoch_label = np.zeros(total_data_num)
	
	best_layer_dict = {'cifar100': {10: {}, 4: {}}, 'cifar10': {10: {}, 4: {}}, 'texas': {}, 'purchase': {},
	                   'fashion_mnist': {}}
	best_layer_dict['cifar10'][10]['alexnet'] = 4
	best_layer_dict['cifar10'][4]['alexnet'] = 4
	best_layer_dict['cifar100'][10]['alexnet'] = 8
	best_layer_dict['cifar100'][4]['alexnet'] = 6
	best_layer_dict['cifar10'][10]['densenet_cifar'] = 99
	best_layer_dict['cifar100'][10]['densenet_cifar'] = 99
	best_layer_dict['texas'][10] = 0
	best_layer_dict['texas'][4] = 0
	best_layer_dict['purchase'][10] = 0
	best_layer_dict['purchase'][4] = 0
	best_layer_dict['fashion_mnist'][4] = 0
	best_layer_dict['fashion_mnist'][10] = 0
	
	### densenet cifar10 layer 99
	### densenet cifar100 layer 99
	
	if (best_layer == -1):
		this_best_layer = best_layer_dict[dataset][num_user]
	else:
		this_best_layer = best_layer
	
	# print (f"this best layer:{this_best_layer}")
	
	# result_saving_file = prefix + mid_str + 'result_file.txt'
	# f = open(result_saving_file,"w")
	
	for epoch_idx, epoch in enumerate(epochs):
		final_str = '_' + str(
			epoch + 1) + '_' + str(dataset) + '_' + str(target_data_size) + '_' + str(eval_data_size) + '_' + str(
			model) + '.npy'
		
		data, label = read_all_comm_round_data(comm_round_list, prefix, mid_str, num_layers=num_layers, num_user=num_user,
		                                       feature_str='all_info_multi_party_member_attack_',
		                                       final_str=final_str, label_str='all_label_multi_party_member_attack_')
		
		loss_data, loss_label = read_all_comm_round_data(comm_round_list, prefix, mid_str, num_layers=num_layers, num_user=num_user,
		                                                 feature_str='loss_info_multi_party_member_attack_',
		                                                 final_str=final_str,
		                                                 label_str='loss_label_multi_party_member_attack_')
		
		target_loss_data, _ = read_all_comm_round_data(comm_round_list, prefix, mid_str, num_layers=num_layers, num_user=num_user,
		                                               feature_str='target_model_before_loss_info_multi_party_member_attack_',
		                                               final_str=final_str,
		                                               label_str='target_model_before_loss_label_multi_party_member_attack_')
		
		target_after_loss_data, _ = read_all_comm_round_data(comm_round_list, prefix, mid_str, num_layers=num_layers, num_user=num_user,
		                                                     feature_str='target_model_after_loss_info_multi_party_member_attack_',
		                                                     final_str=final_str,
		                                                     label_str='target_model_after_loss_label_multi_party_member_attack_')
		
		####print (data.shape)
		
		data = np.reshape(data, (-1, num_valid_user, num_layers, 14))
		label = np.reshape(label, (-1))
		loss_data = np.reshape(loss_data, (-1, num_valid_user))
		loss_label = np.reshape(loss_label, (-1))
		target_loss_data = np.reshape(target_loss_data, (-1, 1))
		target_loss_data = np.tile(target_loss_data, (1, num_valid_user))
		target_after_loss_data = np.reshape(target_after_loss_data, (-1, 1))
		target_after_loss_data = np.tile(target_after_loss_data, (1, num_valid_user))
		
		####print (data.shape,label.shape,loss_data.shape,loss_label.shape,target_loss_data.shape)
		
		all_epoch_cos[:, :, epoch_idx] = copy.deepcopy(data[:, :, this_best_layer, 0])
		all_epoch_grad_diff[:, :, epoch_idx] = copy.deepcopy(data[:, :, this_best_layer, 1] - data[:, :, this_best_layer, 2])
		all_epoch_layer_cos[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 0])
		all_epoch_layer_grad_diff[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 1] - data[:, :, :, 2])
		all_epoch_loss[:, :, epoch_idx] = copy.deepcopy(loss_data)
		all_epoch_target_loss[:, :, epoch_idx] = copy.deepcopy(target_loss_data)
		all_epoch_target_after_loss[:, :, epoch_idx] = copy.deepcopy(target_after_loss_data)
		all_epoch_layer_grad_norm[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 3])
		all_epoch_grad_norm[:, :, epoch_idx] = copy.deepcopy(data[:, :, this_best_layer, 3])
		all_epoch_label = label
	
	all_epoch_loss = np.nan_to_num(all_epoch_loss)
	all_epoch_grad_diff = np.nan_to_num(all_epoch_grad_diff)
	all_epoch_cos = np.nan_to_num(all_epoch_cos)
	all_epoch_target_loss = np.nan_to_num(all_epoch_target_loss)
	all_epoch_target_after_loss = np.nan_to_num(all_epoch_target_after_loss)
	all_epoch_layer_grad_norm = np.nan_to_num(all_epoch_layer_grad_norm)
	all_epoch_grad_norm = np.nan_to_num(all_epoch_grad_norm)
	all_epoch_layer_cos = np.nan_to_num(all_epoch_layer_cos)
	all_epoch_layer_grad_diff = np.nan_to_num(all_epoch_layer_grad_diff)
	all_epoch_layer_grad_norm = np.nan_to_num(all_epoch_layer_grad_norm)
	
	# if (eval_data_size == target_data_size): ## special for one batch case
	#    eval_data_size = 100
	
	return layer_analysis(all_epoch_loss, all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_layer_grad_norm, special_layers=[best_layer], eval_data_size=eval_data_size,
	                      all_epoch_target_loss=all_epoch_target_loss)


def layer_analysis(all_epoch_loss, all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_layer_grad_norm, special_layers=[], eval_data_size=50, all_epoch_target_loss=None):
	num_instance = all_epoch_layer_cos.shape[0]
	num_user = all_epoch_layer_cos.shape[1]
	num_layer = all_epoch_layer_cos.shape[3]
	num_epochs = all_epoch_layer_cos.shape[2]
	
	auc_list = []
	acc_list = []
	tpr_list = []
	all_user_roc_list = []
	
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
		this_user_loss_reduction = all_epoch_target_loss[this_user_index, this_user_idx, :] - all_epoch_loss[this_user_index, this_user_idx, :]
		
		# print (all_epoch_layer_cos.shape,this_user_cos.shape,true_label.shape,eval_data_size)
		
		this_user_auc_list = []
		this_user_acc_list = []
		this_user_tpr_list = []
		
		### report layerwise AUC score for each sum
		for this_layer in special_layers:
			this_layer_cos = this_user_cos[:, :, this_layer]
			this_layer_grad_diff = this_user_grad_diff[:, :, this_layer]
			this_layer_loss = this_user_loss[:, :]
			# this_layer_grad_norm = np.sum(this_user_grad_norm[:,:,:],axis=-1)
			this_layer_grad_norm = this_user_grad_norm[:, :, this_layer]
			this_layer_loss_reduction = this_user_loss_reduction[:, :]
			
			## convert this layer cos to probability for each epoch
			# print (this_layer_cos.shape)
			# converted_this_layer_cos = np.transpose(np.array([convert_metric_to_prob(this_layer_cos[:,i],metric_name='cos') for i in range(num_epochs)]))
			# print (this_layer_cos.shape)
			
			# print (this_layer_cos.shape)
			# converted_this_layer_loss = np.transpose(np.array([convert_metric_to_prob(this_layer_loss[:,i],metric_name='loss') for i in range(num_epochs)]))
			# print (this_layer_cos.shape)
			
			if (this_user_idx < 10):  # <10
				### show distribution
				# fpr_threshold = 0.001
				# plt.plot(np.arange(11)/10000,np.arange(11)/10000)
				# fpr_offset = fpr_threshold / 300
				start_offset = 1e-5
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, np.sum(this_layer_cos, axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				# print (np.stack((fpr,tpr)).shape)
				for i in range(len(fpr)):
					if (fpr[i] == 0.0):
						fpr[i] = start_offset
					else:
						break
				log_fpr = np.log10(fpr)
				log_tpr = np.log10(tpr)
				
				for idx in range(len(fpr)):
					if (fpr[idx] == 1e-3):
						break
					elif (fpr[idx] > 1e-3):
						idx -= 1
						break
				this_user_tpr_list.append(tpr[idx])
				# print (fpr[idx],tpr[idx])
				
				'''
				sum_auc = 0
				for i in range(this_layer_loss.shape[1]):
					member_avg = np.average(this_layer_cos[:int(len(this_layer_cos)/2),i])
					nonmember_avg = np.average(this_layer_cos[int(len(this_layer_cos)/2):,i])
					this_auc = roc_auc_score(true_label,this_layer_cos[:,i])
					sum_auc+=this_auc
					#print (f"max nonmember cos {np.amax(this_layer_cos[int(len(this_layer_cos)/2):,i])}")
					#print (f"user {this_user_idx}, epoch {i}, member avg cos {member_avg:.4f}, nonmemberavg cos{nonmember_avg:.4f}, auc {this_auc:.3f}")
				print (f"sum auc {sum_auc}")
				'''
				'''
				if (this_user_idx==0):
					loss = np.sum(this_layer_cos,axis=1)
					member_loss = loss[:int(len(loss)/2)]
					nonmember_loss = loss[int(len(loss)/2):]

					bins = np.linspace(np.amin(loss),np.amax(loss),20)
					member_counts,bins = np.histogram(member_loss,bins)
					nonmember_counts,_ = np.histogram(nonmember_loss,bins)

					fig = plt.figure(figsize=(5,5))
					plt.plot(bins[1:],member_counts)
					plt.plot(bins[1:],nonmember_counts)
					plt.legend(['members','nonmembers'])
					plt.show()
				'''
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, np.sum(this_layer_grad_diff, axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				for i in range(len(fpr)):
					if (fpr[i] == 0.0):
						fpr[i] = start_offset
					else:
						break
				log_fpr = np.log10(fpr)
				log_tpr = np.log10(tpr)
				
				for idx in range(len(fpr)):
					if (fpr[idx] == 1e-3):
						break
					elif (fpr[idx] > 1e-3):
						idx -= 1
						break
				this_user_tpr_list.append(tpr[idx])
				# print ("grad diff",fpr[idx],tpr[idx])
				# print (fpr[:20],tpr[:20])
				# print ("grad diff",fpr,tpr)
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * np.sum(this_layer_loss, axis=1), pos_label=1)
				
				for i in range(this_layer_loss.shape[1]):
					member_avg = np.average(this_layer_loss[:int(len(this_layer_loss) / 2), i])
					nonmember_avg = np.average(this_layer_loss[int(len(this_layer_loss) / 2):, i])
					this_auc = roc_auc_score(true_label, -1 * this_layer_loss[:, i])
					# print (f"epoch {i}, auc {this_auc},  nonmemberavg {nonmember_avg}")
					#print(f" epoch {i}, auc {this_auc}, member avg {member_avg}, nonmemberavg {nonmember_avg}")
				# print (f"min member loss {np.amin(this_layer_loss[:int(len(this_layer_loss)/2),i])}, min nonmember loss {np.amin(this_layer_loss[int(len(this_layer_loss)/2):,i])}")
				
				all_user_roc_list.append(np.stack((fpr, tpr)))
				for i in range(len(fpr)):
					if (fpr[i] == 0.0):
						fpr[i] = start_offset
					else:
						break
				log_fpr = np.log10(fpr)
				log_tpr = np.log10(tpr)
				
				for idx in range(len(fpr)):
					if (fpr[idx] == 1e-3):
						break
					elif (fpr[idx] > 1e-3):
						idx -= 1
						break
				this_user_tpr_list.append(tpr[idx])
				# print (fpr[idx],tpr[idx])
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * np.sum(this_layer_grad_norm, axis=1), pos_label=1)
				
				all_user_roc_list.append(np.stack((fpr, tpr)))
				for i in range(len(fpr)):
					if (fpr[i] == 0.0):
						fpr[i] = start_offset
					else:
						break
				log_fpr = np.log10(fpr)
				log_tpr = np.log10(tpr)
				
				for idx in range(len(fpr)):
					if (fpr[idx] == 1e-3):
						break
					elif (fpr[idx] > 1e-3):
						idx -= 1
						break
				this_user_tpr_list.append(tpr[idx])
				# print (fpr[idx],tpr[idx])
				
				'''
				#fpr, tpr, thresholds = metrics.roc_curve(true_label,np.sum(this_layer_loss_reduction,axis=1), pos_label=1)
				#all_user_roc_list.append(np.stack((fpr,tpr)))
				for i in range(len(fpr)):
					if (fpr[i] == 0.0):
						fpr[i] = start_offset
					else:
						break
				log_fpr = np.log10(fpr)
				log_tpr = np.log10(tpr)

				for idx in range(len(fpr)):
					if (fpr[idx]==1e-3):
						break
					elif (fpr[idx]>1e-3):
						idx-=1
						break
				this_user_tpr_list.append(tpr[idx])
				#print (fpr[idx],tpr[idx])
				'''
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * this_layer_loss[:, -1], pos_label=1)
				
				all_user_roc_list.append(np.stack((fpr, tpr)))
				for i in range(len(fpr)):
					if (fpr[i] == 0.0):
						fpr[i] = start_offset
					else:
						break
				log_fpr = np.log10(fpr)
				log_tpr = np.log10(tpr)
				
				for idx in range(len(fpr)):
					if (fpr[idx] == 1e-3):
						break
					elif (fpr[idx] > 1e-3):
						idx -= 1
						break
				this_user_tpr_list.append(tpr[idx])
			# print (fpr[idx],tpr[idx])
			
			cos_auc_score = roc_auc_score(true_label, np.sum(this_layer_cos, axis=1))
			# cos_auc_score = roc_auc_score(true_label,this_layer_cos[:,-1])
			grad_diff_auc_score = roc_auc_score(true_label, np.sum(this_layer_grad_diff, axis=1))
			loss_auc_score = roc_auc_score(true_label, -1 * np.sum(this_layer_loss, axis=1))
			grad_norm_auc_score = roc_auc_score(true_label, -1 * np.sum(this_layer_grad_norm, axis=1))
			loss_reduction_auc_score = roc_auc_score(true_label, np.sum(this_layer_loss_reduction, axis=1))
			last_epoch_loss_auc_score = roc_auc_score(true_label, -1 * this_layer_loss[:, -1])
			# converted_cos_auc_score = roc_auc_score(true_label,np.sum(converted_this_layer_cos,axis=1))
			# converted_loss_auc_score = roc_auc_score(true_label,-1*np.sum(converted_this_layer_loss,axis=1))
			# print (f"layer {this_layer}: cos AUC {cos_auc_score:.4f}, grad diff AUC {grad_diff_auc_score:.4f}, loss AUC {loss_auc_score:.4f}, grad norm AUC {grad_norm_auc_score:.4f}")
			this_user_auc_list.append((cos_auc_score, grad_diff_auc_score, loss_auc_score, grad_norm_auc_score,
			                           loss_reduction_auc_score, last_epoch_loss_auc_score))
			
			### calculate min var for cosine
			# std_list = [np.std(this_layer_cos[int(len(this_layer_cos)/2):,i]) for i in range(num_epochs)]
			# best_epoch = np.argmin(np.array(std_list))
			# best_epoch_auc_score = roc_auc_score(true_label,this_layer_cos[:,best_epoch])
			# print (f"AUC use best epoch {best_epoch_auc_score}")
			
			cos_acc_score = report_acc(np.sum(this_layer_cos, axis=1), true_label)
			# cos_acc_score = report_acc(this_layer_cos[:,-1],true_label)
			grad_diff_acc_score = report_acc(np.sum(this_layer_grad_diff, axis=1), true_label)
			
			# print (np.sum(this_layer_grad_diff,axis=1),true_label)
			
			loss_acc_score = report_acc(-1 * np.sum(this_layer_loss, axis=1), true_label)
			grad_norm_acc_score = report_acc(-1 * np.sum(this_layer_grad_norm, axis=1), true_label)
			loss_reduction_acc_score = report_acc(np.sum(this_layer_loss_reduction, axis=1), true_label)
			last_epoch_loss_acc_score = report_acc(-1 * this_layer_loss[:, -1], true_label)
			# converted_cos_acc_score = report_acc(np.sum(converted_this_layer_cos,axis=1),true_label)
			# converted_loss_acc_score = report_acc(-1*np.sum(converted_this_layer_loss,axis=1),true_label)
			# print (f"layer {this_layer}: cos AUC {cos_auc_score:.4f}, grad diff AUC {grad_diff_auc_score:.4f}, loss AUC {loss_auc_score:.4f}, grad norm AUC {grad_norm_auc_score:.4f}")
			this_user_acc_list.append((cos_acc_score, grad_diff_acc_score, loss_acc_score, grad_norm_acc_score,
			                           loss_reduction_acc_score, last_epoch_loss_acc_score))
		
		auc_list.append(this_user_auc_list)
		acc_list.append(this_user_acc_list)
		tpr_list.append(this_user_tpr_list)
	
	auc_list = np.array(auc_list)
	acc_list = np.array(acc_list)
	tpr_list = np.array(tpr_list)
	# print (tpr_list.shape)
	# print (auc_list.shape)
	# print (tpr_list.shape)
	# print (tpr_list)
	
	# print (np.average(auc_list,axis=0))
	avg_auc = np.squeeze(np.average(auc_list, axis=0))
	avg_acc = np.squeeze(np.average(acc_list, axis=0))
	avg_tpr = np.squeeze(np.average(tpr_list, axis=0))
	# print (avg_auc.shape)
	# print (avg_auc)
	# print (f"avg over 10 users: cos AUC {avg_auc[0]:.4f}, grad diff AUC {avg_auc[1]:.4f}, loss AUC {avg_auc[2]:.4f}, grad norm AUC {avg_auc[3]:.4f}, last epoch loss AUC {avg_auc[5]:.4f}")
	
	# print (f"avg over 10 users: cos Acc {avg_acc[0]:.4f},  grad diff Acc {avg_acc[1]:.4f},  loss Acc {avg_acc[2]:.4f}, grad norm Acc {avg_acc[3]:.4f}, last epoch loss Acc {avg_acc[5]:.4f} ")
	
	# print (f"avg over 10 users at fpr =1e-3 : cos tpr {avg_tpr[0]:.4f},  grad diff tpr {avg_tpr[1]:.4f},  loss tpr {avg_tpr[2]:.4f}, grad norm tpr {avg_tpr[3]:.4f}, last epoch loss tpr {avg_tpr[4]:.4f}")
	
	for i in range(5):
		fpr, tpr = avg_roc_cal([all_user_roc_list[j * 5 + i] for j in range(num_user)])
	# plt.plot(np.log10(fpr),np.log10(tpr))
	# print (fpr.shape)
	# print(fpr)
	# print(tpr)
	# plt.plot(np.log10([1e-5,1e-4,1e-3,1e-2,1e-1,1]),np.log10([1e-5,1e-4,1e-3,1e-2,1e-1,1]))
	
	# print (tpr_list)
	
	return avg_auc, avg_acc, avg_tpr

print ("CIFAR10")
epochs = np.arange(60)+1
print("ISO ONLY CASE:")
seed_list = [0]
auc_list = []
acc_list = []
tpr_list = []
for this_seed in seed_list:
	# print (f"random seed {this_seed}")
	mid_str = str(this_seed) + '_0_server_0_0.0_1_40_0_0.0_0.0_0_0.0_0_'
	auc, acc, tpr = all_analysis(epochs, prefix, mid_str, 'cifar10', 'alexnet', eval_data_size=1000, num_user=1, target_data_size=4000, client_adversary=0, num_layers=12,
	                             best_layer=4, comm_round_list=np.arange(1))
	auc_list.append(auc)
	acc_list.append(acc)
	tpr_list.append(tpr)

auc_list = np.array(auc_list)
acc_list = np.array(acc_list)
tpr_list = np.array(tpr_list)

avg_auc = np.squeeze(np.average(auc_list, axis=0))
avg_acc = np.squeeze(np.average(acc_list, axis=0))
avg_tpr = np.squeeze(np.average(tpr_list, axis=0))
print(
	f"avg over 10 users: cos AUC {avg_auc[0]:.4f}, grad diff AUC {avg_auc[1]:.4f}, loss AUC {avg_auc[2]:.4f}, grad norm AUC {avg_auc[3]:.4f}, last epoch loss AUC {avg_auc[5]:.4f}")
print(
	f"avg over 10 users: cos Acc {avg_acc[0]:.4f},  grad diff Acc {avg_acc[1]:.4f},  loss Acc {avg_acc[2]:.4f}, grad norm Acc {avg_acc[3]:.4f}, last epoch loss Acc {avg_acc[5]:.4f} ")
print(
	f"avg over 10 users at fpr =1e-3 : cos tpr {avg_tpr[0]:.4f},  grad diff tpr {avg_tpr[1]:.4f},  loss tpr {avg_tpr[2]:.4f}, grad norm tpr {avg_tpr[3]:.4f}, last epoch loss tpr {avg_tpr[4]:.4f}")


print ("CIFAR100")
epochs = np.arange(60)+1
print("ISO ONLY CASE:")
seed_list = [0]
auc_list = []
acc_list = []
tpr_list = []
for this_seed in seed_list:
	# print (f"random seed {this_seed}")
	mid_str = str(this_seed) + '_0_server_0_0.0_1_40_0_0.0_0.0_0_0.0_0_'
	auc, acc, tpr = all_analysis(epochs, prefix, mid_str, 'cifar100', 'alexnet', eval_data_size=1000, num_user=1, target_data_size=4000, client_adversary=0, num_layers=12,
	                             best_layer=8, comm_round_list=np.arange(1))
	auc_list.append(auc)
	acc_list.append(acc)
	tpr_list.append(tpr)

auc_list = np.array(auc_list)
acc_list = np.array(acc_list)
tpr_list = np.array(tpr_list)

avg_auc = np.squeeze(np.average(auc_list, axis=0))
avg_acc = np.squeeze(np.average(acc_list, axis=0))
avg_tpr = np.squeeze(np.average(tpr_list, axis=0))
print(
	f"avg over 10 users: cos AUC {avg_auc[0]:.4f}, grad diff AUC {avg_auc[1]:.4f}, loss AUC {avg_auc[2]:.4f}, grad norm AUC {avg_auc[3]:.4f}, last epoch loss AUC {avg_auc[5]:.4f}")
print(
	f"avg over 10 users: cos Acc {avg_acc[0]:.4f},  grad diff Acc {avg_acc[1]:.4f},  loss Acc {avg_acc[2]:.4f}, grad norm Acc {avg_acc[3]:.4f}, last epoch loss Acc {avg_acc[5]:.4f} ")
print(
	f"avg over 10 users at fpr =1e-3 : cos tpr {avg_tpr[0]:.4f},  grad diff tpr {avg_tpr[1]:.4f},  loss tpr {avg_tpr[2]:.4f}, grad norm tpr {avg_tpr[3]:.4f}, last epoch loss tpr {avg_tpr[4]:.4f}")

print ("purchase")
epochs = np.arange(20)+1
print("ISO ONLY CASE:")
seed_list = [0]
auc_list = []
acc_list = []
tpr_list = []
for this_seed in seed_list:
	# print (f"random seed {this_seed}")
	mid_str = str(this_seed) + '_0_server_0_0.0_1_40_0_0.0_0.0_0_0.0_0_'
	auc, acc, tpr = all_analysis(epochs, prefix, mid_str, 'purchase', 'purchase', eval_data_size=1000, num_user=1, target_data_size=4000, client_adversary=0, num_layers=8,
	                             best_layer=0, comm_round_list=np.arange(1))
	auc_list.append(auc)
	acc_list.append(acc)
	tpr_list.append(tpr)

auc_list = np.array(auc_list)
acc_list = np.array(acc_list)
tpr_list = np.array(tpr_list)

avg_auc = np.squeeze(np.average(auc_list, axis=0))
avg_acc = np.squeeze(np.average(acc_list, axis=0))
avg_tpr = np.squeeze(np.average(tpr_list, axis=0))
print(
	f"avg over 10 users: cos AUC {avg_auc[0]:.4f}, grad diff AUC {avg_auc[1]:.4f}, loss AUC {avg_auc[2]:.4f}, grad norm AUC {avg_auc[3]:.4f}, last epoch loss AUC {avg_auc[5]:.4f}")
print(
	f"avg over 10 users: cos Acc {avg_acc[0]:.4f},  grad diff Acc {avg_acc[1]:.4f},  loss Acc {avg_acc[2]:.4f}, grad norm Acc {avg_acc[3]:.4f}, last epoch loss Acc {avg_acc[5]:.4f} ")
print(
	f"avg over 10 users at fpr =1e-3 : cos tpr {avg_tpr[0]:.4f},  grad diff tpr {avg_tpr[1]:.4f},  loss tpr {avg_tpr[2]:.4f}, grad norm tpr {avg_tpr[3]:.4f}, last epoch loss tpr {avg_tpr[4]:.4f}")
