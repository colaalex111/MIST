import copy

import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
# from pytorch2keras.converter import pytorch_to_keras
import torchvision
import cv2 as cv
import tensorflow as tf
from functools import partial
from torch.nn import functional as F
from data import part_pytorch_dataset
from sklearn.metrics import roc_auc_score
import math
from sklearn import metrics
import torchvision.transforms as transforms


def skin_special(test_rank, num_classes):
	target_model = torchvision.models.mobilenet_v3_small(num_classes=num_classes)
	## replace the first fc layer with 2 fc layers.
	target_model.classifier = nn.Sequential(
		nn.Linear(576, test_rank),
		nn.Linear(test_rank, 1024),
		nn.Hardswish(inplace=True),
		nn.Dropout(p=0.2, inplace=True),
		nn.Linear(1024, num_classes),
	)
	return target_model


def _batchnorm_to_groupnorm_new(module):
	# print (module)
	# print (module.num_features)
	return nn.GroupNorm(num_groups=module.num_features, num_channels=module.num_features, affine=True)


'''
def convert_model_from_pytorch_to_tensorflow(model, test_error=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
    input_var = Variable(torch.FloatTensor(input_np).to(device))
    output = model(input_var)

    k_model = pytorch_to_keras(model, input_var, (3, 32, 32,), verbose=False, change_ordering=True)

    # k_model.summary()

    return k_model
'''


def sobel(img_set):
	ret = np.empty(img_set.shape)
	for i, img in enumerate(img_set):
		grad_x = cv.Sobel(np.float32(img), cv.CV_32F, 1, 0)
		grad_y = cv.Sobel(np.float32(img), cv.CV_32F, 0, 1)
		gradx = cv.convertScaleAbs(grad_x)
		grady = cv.convertScaleAbs(grad_y)
		gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
		ret[i, :] = gradxy
	return ret


def compute_pairwise_distances(x, y):
	"""Computes the squared pairwise Euclidean distances between x and y.
	Args:
	  x: a tensor of shape [num_x_samples, num_features]
	  y: a tensor of shape [num_y_samples, num_features]
	Returns:
	  a distance matrix of dimensions [num_x_samples, num_y_samples].
	Raises:
	  ValueError: if the inputs do no matched the specified dimensions.
	"""
	
	if not len(x.get_shape()) == len(y.get_shape()) == 2:
		raise ValueError('Both inputs should be matrices.')
	
	if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
		raise ValueError('The number of features should be the same.')
	
	norm = lambda x: tf.reduce_sum(tf.square(x), 1)
	
	return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
	r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
	We create a sum of multiple gaussian kernels each having a width sigma_i.
	Args:
	  x: a tensor of shape [num_samples, num_features]
	  y: a tensor of shape [num_samples, num_features]
	  sigmas: a tensor of floats which denote the widths of each of the
		gaussians in the kernel.
	Returns:
	  A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
	"""
	beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
	
	dist = compute_pairwise_distances(x, y)
	
	s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
	
	return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
	'''
	Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
	Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
	the distributions of x and y. Here we use the kernel two sample estimate
	using the empirical mean of the two distributions.
	MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
				= \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
	where K = <\phi(x), \phi(y)>,
	  is the desired kernel function, in this case a radial basis kernel.
	Args:
		x: a tensor of shape [num_samples, num_features]
		y: a tensor of shape [num_samples, num_features]
		kernel: a function which computes the kernel in MMD. Defaults to the
				GaussianKernelMatrix.
	Returns:
		a scalar denoting the squared maximum mean discrepancy loss.
	'''
	with tf.name_scope('MaximumMeanDiscrepancy'):
		# \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
		cost = tf.reduce_mean(kernel(x, x))
		cost += tf.reduce_mean(kernel(y, y))
		cost -= 2 * tf.reduce_mean(kernel(x, y))
		
		# We do not allow the loss to become negative.
		cost = tf.where(cost > 0, cost, 0, name='value')
	return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
	"""Adds a similarity loss term, the MMD between two representations.
	This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
	different Gaussian kernels.
	Args:
	  source_samples: a tensor of shape [num_samples, num_features].
	  target_samples: a tensor of shape [num_samples, num_features].
	  weight: the weight of the MMD loss.
	  scope: optional name scope for summary tags.
	Returns:
	  a scalar tensor representing the MMD loss value.
	"""
	sigmas = [
		1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
		1e3, 1e4, 1e5, 1e6
	]
	gaussian_kernel = partial(
		gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
	
	loss_value = maximum_mean_discrepancy(
		source_samples, target_samples, kernel=gaussian_kernel)
	loss_value = tf.maximum(1e-4, loss_value) * weight
	
	return loss_value


def get_gpu_status():
	import nvidia_smi
	nvidia_smi.nvmlInit()
	deviceCount = nvidia_smi.nvmlDeviceGetCount()
	for i in range(deviceCount):
		if (i == 0):
			handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
			info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
			print(
				"Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100 * info.free / info.total, info.total,
																								 info.free, info.used))
	
	nvidia_smi.nvmlShutdown()


def get_live_objects():
	import torch
	import gc
	counts = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
				# if (obj.shape == (1024,6169)):
				# print(type(obj), obj.size())
				if (obj.shape == (1024, 6169)):
					counts += 1
		except:
			pass
	print(f"model counts {counts}")


def get_active_loss(user_list, model):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(reduction='none').to(device)
	
	loss_list = []
	with torch.no_grad():
		for user_idx in range(len(user_list)):
			loader = torch.utils.data.DataLoader(user_list[user_idx].evaluation_member_dataset, batch_size=50, shuffle=False)
			for batch_idx, (images, labels, _) in enumerate(loader):
				images, labels = images.to(device), labels.to(device)
				model.zero_grad()
				log_probs = model(images)
				loss = criterion(log_probs, labels)
				loss_list.append(loss)
	
	with torch.no_grad():
		for user_idx in range(len(user_list)):
			loader = torch.utils.data.DataLoader(user_list[user_idx].evaluation_non_member_dataset, batch_size=50, shuffle=False)
			for batch_idx, (images, labels, _) in enumerate(loader):
				images, labels = images.to(device), labels.to(device)
				model.zero_grad()
				log_probs = model(images)
				loss = criterion(log_probs, labels)
				loss_list.append(loss)
	
	return torch.flatten(torch.vstack(loss_list))


def get_train_test_acc(user_list, target_model, print_option=True, return_loss=False, return_validation_result=False, return_ece_loss=False):
	#### get the train/test accuracy after training
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
	
	train_acc = 0
	test_acc = 0
	
	train_loss = 0
	test_loss = 0
	
	idxs_users = len(user_list)
	correct = 0.0
	total = 0.0
	with torch.no_grad():
		for idx in range(idxs_users):
			target_model.eval()
			for images, labels, _ in user_list[idx].train_eval_data_loader:  ### should we use train_eval or train?
				# user_list[idx].train_eval_data_loader or user_list[idx].train_data_loader
				target_model.zero_grad()
				images = images.to(device)
				outputs = target_model(images)
				labels = labels.to(device)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum()
				
				loss = criterion(outputs, labels).detach().item()
				train_loss += loss
	
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	
	train_acc = acc
	train_loss = train_loss / total
	
	if (print_option):
		print("training accuracy %.2f" % (acc))
		#print(f"{total} total, {correct.item()} correct")
	
	correct = 0.0
	total = 0.0
	ece_criterion = _ECELoss().cuda()
	logits_list = []
	labels_list = []
	
	with torch.no_grad():
		for images, labels, _ in user_list[idx].test_data_loader:
			target_model.zero_grad()
			images = images.to(device)
			outputs = target_model(images)
			labels = labels.to(device)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
			
			loss = criterion(outputs, labels).detach().item()
			test_loss += loss
			
			logits_list.append(outputs.detach())
			labels_list.append(labels.detach())
		
		logits = torch.cat(logits_list).cuda()
		labels = torch.cat(labels_list).cuda()
		ece_loss = ece_criterion(logits, labels).detach().item()
	
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	
	test_acc = acc
	test_loss = test_loss / total
	
	if (print_option):
		print("testing accuracy %.2f" % (acc))
		#print(f"ECE LOSS {ece_loss}")
	
	if (return_validation_result):
		val_acc = 0
		val_loss = 0
		
		correct = 0.0
		total = 0.0
		target_model.eval()
		
		with torch.no_grad():
			for images, labels, _ in user_list[0].validation_data_loader:
				target_model.zero_grad()
				images = images.to(device)
				outputs = target_model(images)
				labels = labels.to(device)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum()
				
				loss = criterion(outputs, labels).detach().item()
				val_loss += loss
		
		acc = correct.item()
		acc = acc / total
		acc = acc * 100.0
		
		val_acc = acc
		val_loss = val_loss / total
		
		return train_acc, test_acc, val_acc, train_loss, test_loss, val_loss
	
	if (return_loss):
		# print (train_acc,test_acc,train_loss,test_loss)
		if (return_ece_loss):
			return train_acc, test_acc, train_loss, test_loss, ece_loss
		else:
			return train_acc, test_acc, train_loss, test_loss
	
	return train_acc, test_acc


def get_train_loss(user_list, model, model_weight=None):
	if (model_weight != None):
		model.load_state_dict(model_weight)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss().to(device)
	sum_loss = 0
	
	with torch.no_grad():
		for idx in range(len(user_list)):
			model.eval()
			for images, labels, _ in user_list[idx].train_data_loader:
				model.zero_grad()
				images = images.to(device)
				outputs = model(images)
				labels = labels.to(device)
				loss = criterion(outputs, labels).detach().item()
				sum_loss += loss
	
	# print (f"sumloss{sum_loss}")
	# print (len(user_list[0].train_data_loader))
	sum_loss = sum_loss / (len(user_list) * len(user_list[0].train_data_loader))  ## calculate the avg loss
	
	return sum_loss


def calculate_volume(model_weight_list, prev_weight_list, lr=0, num_step=1, max_dim=None):
	# print (len(model_weight_list),len(prev_weight_list))
	
	flattened_model_param = []
	for this_model_param in model_weight_list:
		all_tensors = []
		for key, val in this_model_param.items():
			all_tensors.append(val.flatten())
		# print (val.size())
		cat_all_tensors = torch.cat(all_tensors)
		# print (cat_all_tensors.size())
		flattened_model_param.append(cat_all_tensors)
	
	prev_flattened_model_param = []
	for this_model_param in prev_weight_list:
		all_tensors = []
		for key, val in this_model_param.items():
			all_tensors.append(val.flatten())
		cat_all_tensors = torch.cat(all_tensors)
		prev_flattened_model_param.append(cat_all_tensors)
	
	# print (len(flattened_model_param),len(prev_flattened_model_param))
	
	if (len(prev_flattened_model_param) == 1):
		### learning rate adjustment
		for idx in range(len(flattened_model_param)):
			flattened_model_param[idx] = prev_flattened_model_param[0] + (
				flattened_model_param[idx] - prev_flattened_model_param[0]) / (lr)  ### should we add num step here? so we are actually calculating
	else:
		for idx in range(len(flattened_model_param)):
			flattened_model_param[idx] = prev_flattened_model_param[idx] + (flattened_model_param[idx] - prev_flattened_model_param[idx]) / (lr)
	
	matrix_a = torch.stack(flattened_model_param)
	# print (matrix_a.size()) ### this should be # of points * # of params
	matrix_a = matrix_a.cpu().numpy()  #### gpu memory is not enough to do SVD
	
	from sklearn.decomposition import PCA
	# print (f"n_components {matrix_a.shape[0]-1}")
	if (max_dim != None):
		pca = PCA(n_components=max_dim)
	else:
		pca = PCA(n_components=matrix_a.shape[0] - 1)
	points = pca.fit_transform(matrix_a)
	# print (points.shape)
	# print (pca.singular_values_)
	
	from scipy.spatial import ConvexHull, convex_hull_plot_2d
	hull = ConvexHull(points)
	# print (hull.area,hull.volume)
	return (hull.area, hull.volume)


def flatten_weight_dict(weight):
	all_tensors = []
	for key, val in weight.items():
		all_tensors.append(val.flatten())
		# print (val.size())
	cat_all_tensors = torch.cat(all_tensors)
	return cat_all_tensors


def calculate_param_cosine_similarity(param1, param2, param3):
	# idx = 0
	# for (key1,val1),(key2,val2),(key3,val3) in zip(param1.items(),param2.items(),param3.items()):
	#    if (idx == best_layer):
	#        layer1 = val1
	#        layer2 = val2
	#    idx+=1
	
	param1 = flatten_weight_dict(param1)
	param2 = flatten_weight_dict(param2)
	param3 = flatten_weight_dict(param3)
	cos = nn.CosineSimilarity(dim=0, eps=1e-6)
	
	## gradient implementation
	cos_result = cos(torch.flatten(param1) - torch.flatten(param3), torch.flatten(param2) - torch.flatten(param3))
	
	## optim implementation
	# cos_result = cos(torch.flatten(param1)-torch.flatten(param3),torch.flatten(param2)-torch.flatten(param3))
	# cos_result = cos(torch.flatten(layer1),torch.flatten(layer2))
	return cos_result.cpu().item()


def calculate_param_change_magnitude(weight_before, weight_after):
	flattened_weight_before = flatten_weight_dict(weight_before)
	flattened_weight_after = flatten_weight_dict(weight_after)
	
	# print (flattened_weight_after.size())
	# print (torch.sum(torch.abs(flattened_weight_after)))
	# print (torch.norm(flattened_weight_after,p=1))
	
	magnitude = torch.norm((flattened_weight_before - flattened_weight_after), p=1)
	
	return magnitude


def calculate_avg_cosine_similarity_normalized(user_list, target_model, get_gradient_func=None, target_layer=0):
	### just use user 0
	result = []
	
	all_grad = []
	for i in range(len(user_list)):
		member_one_instance_loader = torch.utils.data.DataLoader(user_list[i].evaluation_member_dataset, batch_size=100, shuffle=True)
		for (x1, y1, idx1) in member_one_instance_loader:
			grad1 = get_gradient_func(target_model, x1, y1)
			flattened_grad1 = torch.stack([torch.cat([torch.flatten(x[i]) for x in grad1]) for i in range(100)])
			all_grad.append(flattened_grad1)
	all_grad = torch.cat(all_grad)
	std, mean = torch.std_mean(all_grad, dim=0)
	print(all_grad.size())
	print(mean)
	print(std)
	print(torch.var(all_grad, dim=0))
	print(torch.sum(torch.var(all_grad, dim=0)))
	print(f" sqrt(sum(var)) {torch.sqrt(torch.sum(torch.var(all_grad, dim=0)))}")
	
	for i in range(len(user_list)):
		for j in range(i + 1, len(user_list)):
			
			member_one_instance_loader = torch.utils.data.DataLoader(user_list[i].evaluation_member_dataset, batch_size=100, shuffle=True)
			nonmember_one_instance_loader = torch.utils.data.DataLoader(user_list[j].evaluation_member_dataset, batch_size=100, shuffle=True)
			
			for (x1, y1, idx1) in member_one_instance_loader:
				for (x2, y2, idx2) in nonmember_one_instance_loader:
					
					grad1 = get_gradient_func(target_model, x1, y1)
					grad2 = get_gradient_func(target_model, x2, y2)
					
					flattened_grad1 = torch.stack([torch.cat([torch.flatten(x[i]) for x in grad1]) for i in range(100)])
					flattened_grad2 = torch.stack([torch.cat([torch.flatten(x[i]) for x in grad2]) for i in range(100)])
					
					# flattened_grad1 = (flattened_grad1-mean)/(std+1e-8)
					# flattened_grad2 = (flattened_grad2-mean)/(std+1e-8)
					
					# print (torch.count_nonzero(flattened_grad1)/100)
					# print (torch.count_nonzero(flattened_grad2)/100)
					
					for k in range(100):
						repeat_grad2_k = torch.stack([flattened_grad2[k] for _ in range(100)])
						this_cos = F.cosine_similarity(flattened_grad1, repeat_grad2_k, dim=1).cpu().numpy()
						# print (flattened_grad1.size(),repeat_grad2_j.size(),this_cos.shape)
						# print (this_cos)
						result.append(this_cos)
	
	result = np.array(result).flatten()
	print(result.shape, np.average(result), np.std(result))
	return np.array(result)


def calculate_avg_cosine_similarity(user_list, target_model, get_gradient_func=None, target_layer=0):
	member_member_result = []
	
	for i in range(len(user_list)):
		for j in range(i + 1, len(user_list)):
			member_one_instance_loader = torch.utils.data.DataLoader(user_list[i].evaluation_member_dataset, batch_size=100, shuffle=True)
			nonmember_one_instance_loader = torch.utils.data.DataLoader(user_list[j].evaluation_member_dataset, batch_size=100, shuffle=True)
			
			for (x1, y1, idx1) in member_one_instance_loader:
				for (x2, y2, idx2) in nonmember_one_instance_loader:
					grad1 = get_gradient_func(target_model, x1, y1)
					grad2 = get_gradient_func(target_model, x2, y2)
					
					flattened_grad1 = torch.stack([torch.cat([torch.flatten(x[i]) for x in grad1]) for i in range(100)])
					flattened_grad2 = torch.stack([torch.cat([torch.flatten(x[i]) for x in grad2]) for i in range(100)])
					
					this_cos = F.cosine_similarity(flattened_grad1, flattened_grad2, dim=1).detach().cpu().numpy()
					member_member_result.append(this_cos)
	
	member_member_result = np.array(member_member_result).flatten()
	print(member_member_result.shape)
	print(" member member results:", member_member_result.shape, np.average(member_member_result), np.std(member_member_result))
	
	member_nonmember_result = []
	
	for i in range(len(user_list)):
		for j in range(i + 1, len(user_list)):
			member_one_instance_loader = torch.utils.data.DataLoader(user_list[i].evaluation_member_dataset, batch_size=100, shuffle=True)
			nonmember_one_instance_loader = torch.utils.data.DataLoader(user_list[j].evaluation_non_member_dataset, batch_size=100, shuffle=True)
			
			for (x1, y1, idx1) in member_one_instance_loader:
				for (x2, y2, idx2) in nonmember_one_instance_loader:
					grad1 = get_gradient_func(target_model, x1, y1)
					grad2 = get_gradient_func(target_model, x2, y2)
					
					flattened_grad1 = torch.stack([torch.cat([torch.flatten(x[i]) for x in grad1]) for i in range(100)])
					flattened_grad2 = torch.stack([torch.cat([torch.flatten(x[i]) for x in grad2]) for i in range(100)])
					
					this_cos = F.cosine_similarity(flattened_grad1, flattened_grad2, dim=1).detach().cpu().numpy()
					member_nonmember_result.append(this_cos)
	
	member_nonmember_result = np.array(member_nonmember_result).flatten()
	print(" member nonmember results:", member_nonmember_result.shape, np.average(member_nonmember_result), np.std(member_nonmember_result))
	
	nonmember_nonmember_result = []
	
	for i in range(len(user_list)):
		for j in range(i + 1, len(user_list)):
			member_one_instance_loader = torch.utils.data.DataLoader(user_list[i].evaluation_non_member_dataset, batch_size=100, shuffle=True)
			nonmember_one_instance_loader = torch.utils.data.DataLoader(user_list[j].evaluation_non_member_dataset, batch_size=100, shuffle=True)
			
			for (x1, y1, idx1) in member_one_instance_loader:
				for (x2, y2, idx2) in nonmember_one_instance_loader:
					grad1 = get_gradient_func(target_model, x1, y1)
					grad2 = get_gradient_func(target_model, x2, y2)
					
					flattened_grad1 = torch.stack([torch.cat([torch.flatten(x[i]) for x in grad1]) for i in range(100)])
					flattened_grad2 = torch.stack([torch.cat([torch.flatten(x[i]) for x in grad2]) for i in range(100)])
					
					this_cos = F.cosine_similarity(flattened_grad1, flattened_grad2, dim=1).detach().cpu().numpy()
					nonmember_nonmember_result.append(this_cos)
	
	nonmember_nonmember_result = np.array(nonmember_nonmember_result).flatten()
	print(" nonmember nonmember results:", nonmember_nonmember_result.shape, np.average(nonmember_nonmember_result), np.std(nonmember_nonmember_result))
	
	all_results = np.array([member_member_result, member_nonmember_result, nonmember_nonmember_result])
	
	return all_results


def volume_dim_test(model_weight_list, prev_weight_list, lr=0):
	volume_dim_result = []
	
	flattened_model_param = []
	for this_model_param in model_weight_list:
		all_tensors = []
		for key, val in this_model_param.items():
			all_tensors.append(val.flatten())
		# print (val.size())
		cat_all_tensors = torch.cat(all_tensors)
		# print (cat_all_tensors.size())
		flattened_model_param.append(cat_all_tensors)
	
	prev_flattened_model_param = []
	for this_model_param in prev_weight_list:
		all_tensors = []
		for key, val in this_model_param.items():
			all_tensors.append(val.flatten())
		cat_all_tensors = torch.cat(all_tensors)
		prev_flattened_model_param.append(cat_all_tensors)
	
	print(len(flattened_model_param), len(prev_flattened_model_param))
	
	if (len(prev_flattened_model_param) == 1):
		### learning rate adjustment
		for idx in range(len(flattened_model_param)):
			flattened_model_param[idx] = prev_flattened_model_param[0] + (
				flattened_model_param[idx] - prev_flattened_model_param[0]) / (
											 lr)  ### should we add num step here? so we are actually calculating
	else:
		for idx in range(len(flattened_model_param)):
			flattened_model_param[idx] = prev_flattened_model_param[idx] + (
				flattened_model_param[idx] - prev_flattened_model_param[idx]) / (lr)
	
	matrix_a = torch.stack(flattened_model_param)
	# print (matrix_a.size()) ### this should be # of points * # of params
	matrix_a = matrix_a.cpu().numpy()  #### gpu memory is not enough to do SVD
	matrix_a = np.nan_to_num(matrix_a)
	from sklearn.decomposition import PCA
	
	num_points = matrix_a.shape[0]
	print(f"total user:{num_points}")
	for dim in range(2, num_points):
		print(f"current dim {dim}")
		pca = PCA(n_components=dim)
		points = pca.fit_transform(copy.deepcopy(matrix_a))
		from scipy.spatial import ConvexHull, convex_hull_plot_2d
		hull = ConvexHull(points)
		volume_dim_result.append((hull.area, hull.volume))
		print(hull.area, hull.volume)
	return volume_dim_result


class _ECELoss(nn.Module):
	"""
	Calculates the Expected Calibration Error of a model.
	(This isn't necessary for temperature scaling, just a cool metric).

	The input to this loss is the logits of a model, NOT the softmax scores.

	This divides the confidence outputs into equally-sized interval bins.
	In each bin, we compute the confidence gap:

	bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

	We then return a weighted average of the gaps, based on the number
	of samples in each bin

	See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
	"Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
	2015.
	"""
	
	def __init__(self, n_bins=15):
		"""
		n_bins (int): number of confidence interval bins
		"""
		super(_ECELoss, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]
		
	def forward(self, logits, labels, islogit=True):
		if (islogit):
			softmaxes = F.softmax(logits, dim=1)
		else:
			softmaxes = logits  ### the other case is just for softmaxed output, i.e. probs
		confidences, predictions = torch.max(softmaxes, 1)
		accuracies = predictions.eq(labels)
			
		ece = torch.zeros(1, device=logits.device)
		
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			# Calculated |confidence - accuracy| in each bin
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
		
		return ece


def repartition_dataset(dataset, user_list):
	### init transforms
	# hard medical datasets
	
	if (dataset == 'fashion_mnist'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(28),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
		])
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
	
	if (dataset == 'retina' or dataset == 'skin'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(64),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
		])
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
	# binary datasets
	elif (dataset == 'purchase' or dataset == 'texas' or dataset == 'location'):
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
	
	### reassigning data to each user
	reassign_index = np.random.choice(np.arange(len(all_train_label)), len(all_train_label), replace=False)
	cnt = 0
	for i in range(len(user_list)):
		this_user = user_list[i]
		target_data_size = len(user_list[i].train_label)
		this_user_index = reassign_index[cnt: cnt + target_data_size]
		cnt += target_data_size
		this_user.train_data = all_train_data[this_user_index]
		this_user.train_label = all_train_label[this_user_index]
		train = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=True, transform=transform_train,
									 target_transform=target_transform)
		this_user.train_dataset = train
		this_user.train_data_loader = torch.utils.data.DataLoader(train, batch_size=100,
																  shuffle=True, num_workers=1)


def non_idd_assign_part_dataset(dataset, user_list=[]):
	non_iid_dict = {
		'cifar10': 4,
		'cifar100': 10,
		'purchase': 20,
		'texas': 5
	}
	
	if (dataset.dataset_name == 'cifar10' or dataset.dataset_name == 'cifar100'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(32),
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
		norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		train_trans = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
		val_trans = [transforms.ToTensor(), norm]
		transform_train = transforms.Compose(train_trans + [norm])
		transform_test = transforms.Compose(val_trans)
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
	avaliable_classes = np.arange(num_classes)
	
	### generating train / test indices for each user
	
	## for CIFAR-10 / CIFAR-100, we have 50000/10000 train/test data,
	## then each user should share the test data and we need to split the training data
	## for purchase and texas, we have enough data
	
	## in our case, we choose 3 classes and choose some data from each class for one user to achieve non-iid setting
	
	training_set_size = args.target_data_size
	index_left = np.arange(len(dataset.train_label))  # the # of data left for generating new split of training data
	
	assigned_index = []
	
	all_assigned_classes = []
	# print (np.bincount(dataset.train_label))
	
	for i in range(num_users):
		this_user = user_list[i]
		
		this_user.target_transform = target_transform
		this_user.train_transform = transform_train
		this_user.test_transform = transform_test
		
		### perform non-iid training data selection
		# print (len(avaliable_classes))
		assigned_classes = np.random.choice(avaliable_classes, num_classes_non_iid_per_user, replace=False)
		all_assigned_classes.append(assigned_classes)
		
		# print(f"user {i} assigned classes:{assigned_classes}")
		this_user_class_count = np.zeros((len(np.unique(dataset.train_label))))
		class_size = int(training_set_size / num_classes_non_iid_per_user)
		this_user_train_index = []
		for this_class in assigned_classes:
			
			this_class_remaining_index = index_left[np.arange(len(index_left))[dataset.train_label[index_left] == this_class]]
			# print(f"user id {i}, this class {this_class}, instance left {len(this_class_remaining_index)}")
			if (len(this_class_remaining_index) - class_size < class_size):
				avaliable_classes = avaliable_classes[avaliable_classes != this_class]
			this_user_this_class_train_index = np.random.choice(this_class_remaining_index, class_size, replace=False)
			this_user_train_index.append(this_user_this_class_train_index)
			index_left = np.setdiff1d(index_left, this_user_this_class_train_index)
			this_user_class_count[this_class] = class_size
		
		this_user_train_index = np.array(this_user_train_index).flatten()
		# print(f"user {i} has classes:{np.bincount(dataset.train_label[this_user_train_index])}")
		
		this_user.train_data = dataset.train_data[this_user_train_index]
		this_user.train_label = dataset.train_label[this_user_train_index]
		
		# print(f"user {i} has classes:{this_user_class_count}")
		
		this_user.class_weight = this_user_class_count / class_size
		
		# this_user.class_weight = np.ones((len(np.unique(this_user.train_label)))) * training_set_size / (len(np.unique(this_user.train_label)) * (np.bincount(this_user.train_label)))
		# print("class weight:", this_user.class_weight)
		
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
		
		### we use testing data as 'eval_non_member' set, but we need to make sure that nonmembers are in the same class as the training data
		
		nonmember_instance_per_class = int(args.eval_data_size / num_classes_non_iid_per_user)
		this_user_nonmember_eval_index = []
		for this_class in assigned_classes:
			this_class_index = np.arange(len(this_user.test_label))[this_user.test_label == this_class]
			# print (f"this class nonmember size {len(this_class_index)}")
			selected_index = np.random.choice(this_class_index, nonmember_instance_per_class, replace=False)
			this_user_nonmember_eval_index.append(selected_index)
		
		# non_member_index = np.random.choice(len(this_user.test_label), args.eval_data_size, replace=False)
		non_member_index = np.array(this_user_nonmember_eval_index).flatten()
		evaluation_non_member = part_pytorch_dataset(copy.deepcopy(this_user.test_data[non_member_index]),
													 copy.deepcopy(this_user.test_label[non_member_index]), train=False,
													 transform=transform_test,
													 target_transform=target_transform)
		this_user.evaluation_non_member_dataset = evaluation_non_member
	
	### check remaining unassigned data
	dataset.remaining_index = index_left
	print(len(index_left))
	
	### we select some data as validation set
	validation_data_index = np.random.choice(index_left, len(index_left), replace=False)  ### this should be false, but just for the sake of # of user exp
	validation_data = dataset.train_data[validation_data_index]
	validation_label = dataset.train_label[validation_data_index]
	dataset.remaining_index = np.setdiff1d(index_left, validation_data_index)
	
	for user_idx in range(num_users):
		this_user = user_list[user_idx]
		assigned_classes = all_assigned_classes[user_idx]
		
		this_user_validation_index = np.concatenate([np.arange(len(validation_label))[validation_label == i] for i in assigned_classes]).flatten()
		# print (this_user_validation_index.shape)
		this_user_validation_index = this_user_validation_index.astype(np.int64)
		
		this_user.eval_validation_data = validation_data[this_user_validation_index]
		this_user.eval_validation_label = validation_label[this_user_validation_index]
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
		### create a validation-base-loader, to create a non member distribution, specifically for passive case
		validation_base_index = np.random.choice(len(this_user.eval_validation_label), min(1000, len(this_user.eval_validation_label)), replace=False)
		print(len(validation_base_index))
		args.validation_set_size = len(validation_base_index)
		this_user.validation_base_data = this_user.eval_validation_data[validation_base_index]
		this_user.validation_base_label = this_user.eval_validation_label[validation_base_index]
		this_user.validation_base_dataset = part_pytorch_dataset(this_user.validation_base_data, this_user.validation_base_label, train=False,
																 transform=transform_test,
																 target_transform=target_transform)
		this_user.validation_base_data_loader = torch.utils.data.DataLoader(this_user.validation_base_dataset,
																			batch_size=args.target_batch_size, shuffle=False,
																			num_workers=1)



def get_blackbox_auc_no_shadow(all_train_loss, all_test_loss, fpr_threshold=0.001):
	### randomly sample to get a balanced evaluation set
	min_len = min(len(all_test_loss), len(all_train_loss))
	test_index = np.random.choice(np.arange(len(all_test_loss)), min_len, replace=False)
	all_test_loss = all_test_loss[test_index]
	train_index = np.random.choice(np.arange(len(all_train_loss)), min_len, replace=False)
	all_train_loss = all_train_loss[train_index]
	
	## calcualte AUC and PLR
	label = np.concatenate((np.ones((len(all_train_loss))), np.zeros((len(all_test_loss)))))
	auc = roc_auc_score(label, np.concatenate((all_train_loss, all_test_loss), axis=0))
	print(f"AUC score {auc}")
	
	threshold = np.sort(all_test_loss)[int(len(all_test_loss) * fpr_threshold)]
	print(f"Loss threshold {threshold}")
	
	cnt = 0
	for i in range(len(all_train_loss)):
		if (all_train_loss[i] <= threshold):
			cnt += 1
	print(f"TPR {cnt / min_len}, FPR {fpr_threshold}, PLR {cnt / (min_len * fpr_threshold)}")
	
	return auc, 1.0 * cnt / (min_len * fpr_threshold), np.concatenate((all_train_loss, all_test_loss), axis=0), label


def proxy_metric_calculation(targeted_loss):
	targeted_loss = torch.log(torch.exp(-1 * targeted_loss) / (1 - torch.exp(-1 * targeted_loss) + 1e-8))
	targeted_loss, _ = torch.sort(targeted_loss)
	
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
	targeted_loss = np.reshape(targeted_loss,(-1,1))
	from sklearn.mixture import GaussianMixture

	means_init = np.reshape(np.array([avg1,avg2]),(-1,1))
	gm = GaussianMixture(n_components=2, random_state=0,covariance_type='spherical',init_params='random',means_init=means_init)
	gm.fit(targeted_loss)
	means = gm.means_
	covariances = gm.covariances_
	metric = np.abs(means[0][0]-means[1][0])/(covariances[0]+covariances[1])
	print (f"means {means},cov {covariances}")
	'''
	return metric


def ga_active_attacker_param_search(user_list, target_model, ori_target_model_state_dict, gradient_ascent_weight, global_weights):
	l = 0
	r = 100
	target_model.load_state_dict(global_weights)
	### valid acc here is the acc of model from last epoch, not current epoch
	_, _, valid_acc_metric, train_loss_metric, _, validation_loss_metric = get_train_test_acc(user_list, target_model, return_validation_result=True, print_option=False)
	best_alpha = 0
	logs = []
	min_val_loss = 1e10
	'''
	cos = calculate_param_cosine_similarity(gradient_ascent_weight, global_weights, ori_target_model_state_dict)
	normal_norm = calculate_param_change_magnitude(ori_target_model_state_dict,global_weights)
	active_norm = calculate_param_change_magnitude(ori_target_model_state_dict,gradient_ascent_weight)
	alpha = -1 * cos * normal_norm / active_norm
	return alpha, logs
	'''
	
	while (l < r):
		alpha = (l + r) / 2
		param_search_weight_dict = {}
		for (key1, val1), (key2, val2), (key3, val3) in zip(ori_target_model_state_dict.items(), gradient_ascent_weight.items(), global_weights.items()):
			param_search_weight_dict[key2] = val3 + (val2 - val1) * alpha
		target_model.load_state_dict(param_search_weight_dict)
		train_acc, test_acc, valid_acc, _, _, valid_loss = get_train_test_acc(user_list, target_model, return_validation_result=True, print_option=False)
		
		## we want to find the maximum alpha so the validation loss is not greater than validation loss last epoch for single user case
		'''
		if (valid_loss > validation_loss_metric):
			r = alpha - 0.05
		else:
			l = alpha + 0.05
			best_alpha = alpha
		#print (f"alpha{alpha}, val loss {valid_loss}, constraint {validation_loss_metric}")
		min_val_loss = min(min_val_loss,valid_loss)
		'''
		## for multi-user case, we need to keep validation acc not decreasing
		if (valid_acc < valid_acc_metric):
			r = alpha - 0.05
		else:
			l = alpha + 0.05
			best_alpha = alpha
	print(f"best alpha {best_alpha * 100}")
	return max(0, best_alpha), logs


def gd_active_attacker_param_search(user_list, target_model, ori_target_model_state_dict, gradient_ascent_weight, global_weights, scheduling=1, upper_bound=0):
	l = 0
	r = 10 * pow(10, scheduling)
	residual = 0.1
	if (args.model_name == 'densenet_cifar'):
		residual = 0.1
		r = 10 * pow(10, scheduling)
	
	best_alpha = 0
	target_model.load_state_dict(global_weights)
	targeted_loss = get_active_loss(user_list, target_model)
	loss_metric = torch.mean(targeted_loss).cpu().item()
	_, _, valid_acc_metric, _, _, validation_loss_metric = get_train_test_acc(user_list, target_model, return_validation_result=True, print_option=False)
	# dis_metric = proxy_metric_calculation(targeted_loss)
	logs = []
	best_alpha_validation_acc = valid_acc_metric
	# for utility-preserving attacker, we need to keep the validation loss not increasing and find the min loss metric
	# for utility-ignoring attacker, we just need to find the min loss metric
	# this min loss metric can be replaced with better metric.
	# print (f"initial validation loss {validation_loss_metric}, initial loss metric {loss_metric}, valid acc {valid_acc_metric}")
	logs.append((0, loss_metric, validation_loss_metric, 0, 0, valid_acc_metric))
	while (l < r):
		alpha = (l + r) / 2
		# for alpha in alpha_list:
		param_search_weight_dict = {}
		for (key1, val1), (key2, val2), (key3, val3) in zip(ori_target_model_state_dict.items(), gradient_ascent_weight.items(), global_weights.items()):
			param_search_weight_dict[key1] = val3 + (val2 - val1) * alpha
		target_model.load_state_dict(param_search_weight_dict)
		train_acc, test_acc, valid_acc, _, _, valid_loss = get_train_test_acc(user_list, target_model, return_validation_result=True, print_option=False)
		targeted_loss = get_active_loss(user_list, target_model)
		avg_targeted_loss = torch.mean(targeted_loss).cpu().item()
		# metric = proxy_metric_calculation(targeted_loss)
		# print(f"alpha {alpha},targeted avg loss {avg_targeted_loss}, validation acc {valid_acc}, train_acc {train_acc}, test_acc {test_acc}")#simple division metric {metric}")
		logs.append((alpha, avg_targeted_loss, valid_loss, train_acc, test_acc, valid_acc))  # metric,targeted_loss.cpu().numpy()))
		
		# this is utility preserving implementation.
		# if (args.random_seed == 10): #### args.num_user = 10
		# if (valid_loss > validation_loss_metric) or (avg_targeted_loss>loss_metric): ### loss constraint
		# print ("constraint 0.97")
		# if (valid_acc < (valid_acc_metric*0.99) or avg_targeted_loss > loss_metric): ### acc constraint
		if (valid_acc < (valid_acc_metric * 0.97) or avg_targeted_loss > loss_metric):  ### acc constraint
			# if (avg_targeted_loss > loss_metric):  ### no constraint
			r = alpha - residual
		else:
			l = alpha + residual
			best_alpha = alpha
			loss_metric = avg_targeted_loss
			best_alpha_validation_acc = valid_acc
	# this is utility ignoring implementation.
	# if (args.random_seed <10):
	#	if ((avg_targeted_loss>loss_metric)):
	#		r = alpha - 0.01
	#	else:
	#		l = alpha + 0.01
	#		best_alpha = alpha
	#		loss_metric = avg_targeted_loss
	
	# if (avg_targeted_loss<loss_metric):
	#	best_alpha = alpha
	#	loss_metric = avg_targeted_loss
	
	# if ((metric>dis_metric)):
	# r = alpha - 0.05
	#	best_alpha = alpha
	#	dis_metric = metric
	# else:
	# l = alpha + 0.05
	print(f"alpha {best_alpha}, loss metric {loss_metric}, validation acc before attack {valid_acc_metric},validation acc after attack {best_alpha_validation_acc}")
	# print (f"alpha {best_alpha}, validation loss metric {validation_loss_metric}")
	
	return best_alpha, logs


def logit_function(l, eps=1e-6):
	### we need to setup some bounds for probability (or loss) for numerical stability
	# eps = 1e-15
	l = np.clip(l, a_min=eps, a_max=1 - eps)
	return np.log(l / (1 - l))


def normpdf(x, mean, sd):
	var = float(sd) ** 2
	denom = (2 * math.pi * var) ** .5
	num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
	return num / denom

def get_blackbox_auc_per_class_nn(data,label,fpr_threshold):
	
	### choose half as train, the other half as test
	train_data_index = np.random.choice(len(label),int(0.5*len(label)),replace=False)
	test_data_index = np.setdiff1d(np.arange(len(label)),train_data_index)
	
	train_data = data[train_data_index]
	train_label = label[train_data_index]
	test_data = data[test_data_index]
	test_label = label[test_data_index]
	
	#print (train_data.shape,train_label.shape,test_data.shape,test_label.shape)
	
	from model.attack_model import simple_mlp_attacknet
	model = simple_mlp_attacknet(train_data.shape[1],100,50,2)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss().to(device)
	model.train().to(device)
	epochs = 100
	optimizer =  torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
	train_dataset = part_pytorch_dataset(train_data, train_label, train=False, transform=None,
										 target_transform=None)
	test_dataset = part_pytorch_dataset(test_data, test_label, train=False, transform=None,
										 target_transform=None)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,shuffle=False, num_workers=1)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100,shuffle=False, num_workers=1)
	
	for _ in range(epochs):
		for _, (images, labels, _) in enumerate(train_loader):
			optimizer.zero_grad()
			images, labels = images.to(device), labels.to(device)
			log_probs = model(images)
			loss = criterion(log_probs, labels)
			loss.backward()
			optimizer.step()
			
	## train acc
	correct = 0.0
	total = 0.0
	with torch.no_grad():
		model.eval()
		for images, labels, _ in train_loader:
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
	#print(f"attack model train acc {acc}")

	## train acc
	correct = 0.0
	total = 0.0
	with torch.no_grad():
		model.eval()
		for images, labels, _ in test_loader:
			model.zero_grad()
			images = images.to(device)
			outputs = model(images)
			labels = labels.to(device)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			#print (images.shape,labels.shape)
			correct += (predicted == labels).sum()
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	#print(f"attack model test acc {acc}")
	
	## evaluate auc and tpr
	membership_prob = []
	membership_label = test_label
	for images,labels,_ in test_loader:
		images, labels = images.to(device), labels.to(device)
		log_probs = model(images)
		log_probs = F.softmax(log_probs,dim=1)
		membership_prob.append(log_probs[:,1].detach().cpu().numpy())
		
	membership_prob = np.concatenate(membership_prob,axis=0)
	#print (membership_prob.shape,membership_label.shape)
	
	# auc
	auc_score = roc_auc_score(membership_label,membership_prob)
	# tpr at 0.001 fpr
	#fpr_threshold = 0.001
	negative_index = np.arange(len(test_label))[test_label == 0]
	threshold_index = int(len(negative_index) * fpr_threshold)
	threshold = np.sort(membership_prob[negative_index])[::-1][threshold_index]
	cnt = 0
	for i in range(len(membership_prob)):
		if (membership_prob[i] > threshold and test_label[i] == 1):
			cnt += 1
	print(cnt, len(negative_index), fpr_threshold, threshold_index)
	print(f"TPR {cnt / len(negative_index)}, FPR {fpr_threshold}, PLR {cnt / (threshold_index+1)}")
	return auc_score,cnt / len(negative_index)

def get_blackbox_auc_class_nn(all_prob,all_training_partition,all_validation_partition,all_class_label, fpr_threshold=1e-3):
	all_tpr = []
	all_auc = []
	all_class = np.unique(all_class_label)
	### flatten all inputs
	model_num = len(all_prob)
	for c in all_class:
		this_class_member_pred = []
		this_class_nonmember_pred = []
		for m in range(model_num):
			# this model & this class member / nonmember set
			this_pred = all_prob[m]
			this_class_label = all_class_label[m]
			this_class_index = np.arange(len(this_class_label))[this_class_label == c]
			this_class_train_index = [x for x in this_class_index if (x<len(all_training_partition[0]))]
			this_class_valid_index = [x for x in this_class_index if (x>=len(all_training_partition[0]))]
			this_class_this_model_member_pred = this_pred[this_class_train_index]
			this_class_member_pred.append(this_class_this_model_member_pred)
			this_class_this_model_nonmember_pred = this_pred[this_class_valid_index]
			this_class_nonmember_pred.append(this_class_this_model_nonmember_pred)
		
		### prepare a balanced set
		this_class_nonmember_pred = np.concatenate(this_class_nonmember_pred,axis=0)
		this_class_member_pred = np.concatenate(this_class_member_pred,axis=0)
		#print (this_class_member_pred.shape,this_class_nonmember_pred.shape)
		min_len = min(len(this_class_nonmember_pred),len(this_class_member_pred))
		
		data = np.concatenate((this_class_member_pred[:min_len],this_class_nonmember_pred[:min_len]),axis=0)
		label = np.concatenate((np.ones((min_len)),np.zeros((min_len))),axis=0)
		
		auc,tpr = get_blackbox_auc_per_class_nn(data,label,fpr_threshold)
		if (len(label>2000)):
			all_auc.append(auc)
			all_tpr.append(tpr)
		
	avg_auc = np.average(np.array(all_auc))
	avg_tpr = np.average(np.array(all_tpr))
	
	print(f"classwise NN attack: auc {avg_auc}, tpr {avg_tpr} at fpr = {fpr_threshold}, plr {avg_tpr/fpr_threshold}")
	return avg_auc,avg_tpr


def get_blackbox_auc_lira(all_pred, all_train_index, all_valid_index, all_class_label, fpr_threshold_list=[1e-2,1e-3]):
	## prepare membership label and all probablities
	num_instance = len(all_train_index[0]) + len(all_valid_index[0])
	
	all_metric_result = []
	all_labels = []
	
	for i in range(num_instance):
		# prepare this instances label
		# prepare this instances pred
		# print (f"instance index {i}")
		this_instance_pred = []
		this_instance_label = []
		this_instance_index = []
		this_instance_prob = []
		for j in range(len(all_train_index)):
			if (i in all_train_index[j]):
				this_instance_label.append(1)
				this_instance_index.append(all_train_index[j].tolist().index(i))
				this_instance_pred.append(all_pred[j][this_instance_index[-1]])
				this_pred = this_instance_pred[-1]
				this_label = all_class_label[j][this_instance_index[-1]]
				this_instance_prob.append(this_pred[this_label])
			else:
				this_instance_label.append(0)
				this_instance_index.append(all_valid_index[j].tolist().index(i) + len(all_train_index[j]))
				this_instance_pred.append(all_pred[j][this_instance_index[-1]])
				this_pred = this_instance_pred[-1]
				this_label = all_class_label[j][this_instance_index[-1]]
				this_instance_prob.append(this_pred[this_label])
		
		this_instance_label = np.array(this_instance_label)
		this_instance_index = np.array(this_instance_index)
		this_instance_pred = np.array(this_instance_pred)
		
		# print (i,len(all_train_index),len(this_instance_label),len(this_instance_prob),len(this_instance_index))
		# print (this_instance_prob)
		# this_instance_prob = logit_function(np.array(this_instance_prob).flatten())
		this_instance_prob = np.array(this_instance_prob).flatten()
		# print (this_instance_prob)
		# select half of member and half of nonmember as training samples
		# remaining half of members and half of nonmembers are testing sample
		half = int(0.5 * len(this_instance_label))
		# print (half)
		training_index = np.random.choice(len(this_instance_label), half, replace=False)
		testing_index = np.setdiff1d(np.arange(len(this_instance_label)), training_index)
		
		training_member_index = []
		training_nonmember_index = []
		for this_idx in training_index:
			if (this_instance_label[this_idx] == 1):
				training_member_index.append(this_idx)
			else:
				training_nonmember_index.append(this_idx)
		training_nonmember_index = np.array(training_nonmember_index).astype(np.int64)
		training_member_index = np.array(training_member_index).astype(np.int64)
		training_member_prob = this_instance_prob[training_member_index]
		training_nonmember_prob = this_instance_prob[training_nonmember_index]
		in_mu, in_std = np.average(training_member_prob), np.std(training_member_prob)
		# calculate out distribution
		out_mu, out_std = np.average(training_nonmember_prob), np.std(training_nonmember_prob)
		# calculate metric
		# print (in_mu,in_std,out_mu,out_std)
		this_instance_metric = []
		this_instance_metric_label = []
		eps = 1e-25
		for idx in testing_index:
			# print (testing_index[idx])
			# print (len(this_instance_prob))
			# print (len(this_instance_label))
			this_prob = this_instance_prob[idx]
			this_label = this_instance_label[idx]
			in_prob = normpdf(this_prob, in_mu, in_std + eps)
			out_prob = normpdf(this_prob, out_mu, out_std + eps)
			# print (in_prob,out_prob)
			this_metric = in_prob / (out_prob + eps)
			this_instance_metric.append(this_metric)
			this_instance_metric_label.append(this_label)
		# if (this_label == 1 and this_instance_metric == 0.0):
		#	print (in_mu,in_std,out_mu,out_std,this_prob)
		## only sample fixed in and fixed out for eval
		this_instance_metric_label = np.array(this_instance_metric_label)
		this_instance_metric = np.array(this_instance_metric)
		# fixed_num = 5
		# print (np.bincount(this_instance_metric_label))
		# if (np.any(np.bincount(this_instance_metric_label)<fixed_num) or len(np.bincount(this_instance_metric_label))<2):
		#	continue
		# all_in_index = np.arange(len(this_instance_metric_label))[this_instance_metric_label == 1]
		# in_index = np.random.choice(all_in_index,fixed_num,replace=False)
		# all_out_index = np.arange(len(this_instance_metric_label))[this_instance_metric_label == 0]
		# out_index = np.random.choice(all_out_index,fixed_num,replace=False)
		# all_index = np.concatenate((in_index,out_index))
		# this_instance_metric = this_instance_metric[all_index]
		# this_instance_metric_label = this_instance_metric_label[all_index]
		all_metric_result.append(this_instance_metric)
		all_labels.append(this_instance_metric_label)
	
	all_metric_result = np.array(all_metric_result).flatten()
	all_metric_result = np.nan_to_num(all_metric_result)
	all_labels = np.array(all_labels).flatten()
	
	pos_index = np.arange(len(all_labels))[all_labels == 1]
	neg_index = np.arange(len(all_labels))[all_labels == 0]
	min_len = min(len(pos_index), len(neg_index))
	sampled_pos_index = np.random.choice(pos_index, min_len, replace=False)
	sampled_neg_index = np.random.choice(neg_index, min_len, replace=False)
	sampled_index = np.concatenate((sampled_neg_index, sampled_pos_index), axis=0)
	all_metric_result = all_metric_result[sampled_index]
	all_labels = all_labels[sampled_index]
	
	print (min_len,len(pos_index),len(neg_index),all_metric_result.shape,all_labels.shape)
	# print (all_metric_result[:100],all_labels[:100])
	## calcualte AUC and PLR
	auc = roc_auc_score(all_labels, all_metric_result)
	print(f"AUC score {auc}")
	for fpr_threshold in fpr_threshold_list:
		negative_index = np.arange(len(all_labels))[all_labels == 0]
		threshold_index = int(len(negative_index) * fpr_threshold)
		threshold = np.sort(all_metric_result[negative_index])[::-1][threshold_index]
		print(f"Likelihood threshold {threshold}")
		cnt = 0
		for i in range(len(all_metric_result)):
			if (all_metric_result[i] >= threshold and all_labels[i] == 1):
				cnt += 1
				
		print (cnt, len(negative_index), fpr_threshold, threshold_index)
		print(f"TPR {cnt / len(negative_index)}, FPR {fpr_threshold}, PLR {cnt / threshold_index}")
	fpr, tpr, _ = metrics.roc_curve(all_labels, all_metric_result, pos_label=1)
	return all_metric_result, all_labels, auc, 1.0 * cnt / threshold_index, fpr, tpr


def get_blackbox_auc_lira(all_pred, all_train_index, all_valid_index, all_class_label, fpr_threshold_list=[1e-2, 1e-3]):
	## prepare membership label and all probablities
	num_instance = len(all_train_index[0]) + len(all_valid_index[0])
	
	all_metric_result = []
	all_labels = []
	
	for i in range(num_instance):
		# prepare this instances label
		# prepare this instances pred
		# print (f"instance index {i}")
		this_instance_pred = []
		this_instance_label = []
		this_instance_index = []
		this_instance_prob = []
		for j in range(len(all_train_index)):
			if (i in all_train_index[j]):
				this_instance_label.append(1)
				this_instance_index.append(all_train_index[j].tolist().index(i))
				this_instance_pred.append(all_pred[j][this_instance_index[-1]])
				this_pred = this_instance_pred[-1]
				this_label = all_class_label[j][this_instance_index[-1]]
				this_instance_prob.append(this_pred[this_label])
			else:
				this_instance_label.append(0)
				this_instance_index.append(all_valid_index[j].tolist().index(i) + len(all_train_index[j]))
				this_instance_pred.append(all_pred[j][this_instance_index[-1]])
				this_pred = this_instance_pred[-1]
				this_label = all_class_label[j][this_instance_index[-1]]
				this_instance_prob.append(this_pred[this_label])
		
		this_instance_label = np.array(this_instance_label)
		this_instance_index = np.array(this_instance_index)
		this_instance_pred = np.array(this_instance_pred)
		
		# print (i,len(all_train_index),len(this_instance_label),len(this_instance_prob),len(this_instance_index))
		# print (this_instance_prob)
		# this_instance_prob = logit_function(np.array(this_instance_prob).flatten())
		this_instance_prob = np.array(this_instance_prob).flatten()
		# print (this_instance_prob)
		# select half of member and half of nonmember as training samples
		# remaining half of members and half of nonmembers are testing sample
		half = int(0.5 * len(this_instance_label))
		# print (half)
		training_index = np.random.choice(len(this_instance_label), half, replace=False)
		testing_index = np.setdiff1d(np.arange(len(this_instance_label)), training_index)
		
		training_member_index = []
		training_nonmember_index = []
		for this_idx in training_index:
			if (this_instance_label[this_idx] == 1):
				training_member_index.append(this_idx)
			else:
				training_nonmember_index.append(this_idx)
		training_nonmember_index = np.array(training_nonmember_index).astype(np.int64)
		training_member_index = np.array(training_member_index).astype(np.int64)
		training_member_prob = this_instance_prob[training_member_index]
		training_nonmember_prob = this_instance_prob[training_nonmember_index]
		in_mu, in_std = np.average(training_member_prob), np.std(training_member_prob)
		# calculate out distribution
		out_mu, out_std = np.average(training_nonmember_prob), np.std(training_nonmember_prob)
		# calculate metric
		# print (in_mu,in_std,out_mu,out_std)
		this_instance_metric = []
		this_instance_metric_label = []
		eps = 1e-25
		for idx in testing_index:
			# print (testing_index[idx])
			# print (len(this_instance_prob))
			# print (len(this_instance_label))
			this_prob = this_instance_prob[idx]
			this_label = this_instance_label[idx]
			in_prob = normpdf(this_prob, in_mu, in_std + eps)
			out_prob = normpdf(this_prob, out_mu, out_std + eps)
			# print (in_prob,out_prob)
			this_metric = in_prob / (out_prob + eps)
			this_instance_metric.append(this_metric)
			this_instance_metric_label.append(this_label)
		# if (this_label == 1 and this_instance_metric == 0.0):
		#	print (in_mu,in_std,out_mu,out_std,this_prob)
		## only sample fixed in and fixed out for eval
		this_instance_metric_label = np.array(this_instance_metric_label)
		this_instance_metric = np.array(this_instance_metric)
		# fixed_num = 5
		# print (np.bincount(this_instance_metric_label))
		# if (np.any(np.bincount(this_instance_metric_label)<fixed_num) or len(np.bincount(this_instance_metric_label))<2):
		#	continue
		# all_in_index = np.arange(len(this_instance_metric_label))[this_instance_metric_label == 1]
		# in_index = np.random.choice(all_in_index,fixed_num,replace=False)
		# all_out_index = np.arange(len(this_instance_metric_label))[this_instance_metric_label == 0]
		# out_index = np.random.choice(all_out_index,fixed_num,replace=False)
		# all_index = np.concatenate((in_index,out_index))
		# this_instance_metric = this_instance_metric[all_index]
		# this_instance_metric_label = this_instance_metric_label[all_index]
		all_metric_result.append(this_instance_metric)
		all_labels.append(this_instance_metric_label)
	
	all_metric_result = np.array(all_metric_result).flatten()
	all_metric_result = np.nan_to_num(all_metric_result)
	all_labels = np.array(all_labels).flatten()
	
	pos_index = np.arange(len(all_labels))[all_labels == 1]
	neg_index = np.arange(len(all_labels))[all_labels == 0]
	min_len = min(len(pos_index), len(neg_index))
	sampled_pos_index = np.random.choice(pos_index, min_len, replace=False)
	sampled_neg_index = np.random.choice(neg_index, min_len, replace=False)
	sampled_index = np.concatenate((sampled_neg_index, sampled_pos_index), axis=0)
	all_metric_result = all_metric_result[sampled_index]
	all_labels = all_labels[sampled_index]
	
	print(min_len, len(pos_index), len(neg_index), all_metric_result.shape, all_labels.shape)
	# print (all_metric_result[:100],all_labels[:100])
	## calcualte AUC and PLR
	auc = roc_auc_score(all_labels, all_metric_result)
	print(f"AUC score {auc}")
	for fpr_threshold in fpr_threshold_list:
		negative_index = np.arange(len(all_labels))[all_labels == 0]
		threshold_index = int(len(negative_index) * fpr_threshold)
		threshold = np.sort(all_metric_result[negative_index])[::-1][threshold_index]
		print(f"Likelihood threshold {threshold}")
		cnt = 0
		for i in range(len(all_metric_result)):
			if (all_metric_result[i] >= threshold and all_labels[i] == 1):
				cnt += 1
		
		print(cnt, len(negative_index), fpr_threshold, threshold_index)
		print(f"TPR {cnt / len(negative_index)}, FPR {fpr_threshold}, PLR {cnt / threshold_index}")
	fpr, tpr, _ = metrics.roc_curve(all_labels, all_metric_result, pos_label=1)
	return all_metric_result, all_labels, auc, 1.0 * cnt / threshold_index, fpr, tpr


def log_roc(metric, label):
	fpr, tpr, thresholds = roc_curve(label, metric, pos_label=1)
	fpr = fpr[::-1]
	tpr = tpr[::-1]
	d = {}
	for x, y in zip(fpr, tpr):
		if (x in d):
			continue
		d[x] = y
	new_fpr = list(d.keys())
	new_tpr = list(d.values())
	fpr = np.array(new_fpr)[::-1]
	tpr = np.array(new_tpr)[::-1]
	
	uniform_fpr = [1e-4] + [1e-4 * i * 2 for i in range(1, 6)] + [1e-3 * i / 2 for i in range(2, 20)]
	uniform_fpr = uniform_fpr + [1e-2 * i / 2 for i in range(2, 200)]  # + [1e-1*i for i in range(1,11)]
	uniform_tpr = []
	for _, this_fpr in enumerate(uniform_fpr):
		for j in range(len(fpr)):
			if (fpr[j] >= this_fpr):
				this_tpr = tpr[j]
				break
		uniform_tpr.append(this_tpr)
	
	uniform_fpr = np.array(uniform_fpr)
	uniform_tpr = np.array(uniform_tpr)
	
	return uniform_fpr, uniform_tpr
