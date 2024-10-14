import torch


def merlin_attack_count(dataset, data, label, model, noise_magnitude=0.01):
	### all the hyperparameters are the same as the original paper
	### for normalized images, the noise magnitude is 0.01
	### for unnormalized images, we set the noise magnitude to 0.01*255
	### for purchase, we use 0.01
	repeat_times = 100
	counts = []
	if (dataset == 'cifar100' or dataset == 'cifar10' or dataset == 'purchase' or dataset == 'texas'):
		noise_magnitude = 0.01
	else:
		noise_magnitude = 0.01 * 255 / 2
	
	pred = F.softmax(model(data), dim=1)
	probs = np.array([pred[i, label[i]].detach().item() for i in range(len(label))])
	
	### for each instance in data, we randomly generate noise for 100 rounds and count the number of times that new loss is larger
	for img_idx in range(len(data)):
		this_img = data[img_idx]
		this_label = label[img_idx]
		### repeat this img for 100 times
		stacked_img = torch.stack([this_img for _ in range(repeat_times)])
		stacked_std = torch.ones_like(stacked_img) * noise_magnitude
		random_noise = torch.normal(mean=0, std=stacked_std)
		noisy_img = stacked_img + random_noise
		# print (stacked_img.size(),stacked_std.size(),random_noise.size(),noisy_img.size())
		noisy_pred = F.softmax(model(noisy_img), dim=1)
		noisy_probs = np.array([noisy_pred[i, this_label].detach().item() for i in range(repeat_times)])
		### larger loss means smaller probs, so we count the number of times for smaller probs.
		this_count = len(np.arange(len(noisy_probs))[noisy_probs < probs[img_idx]])
		# print (noisy_probs)
		# print (probs[img_idx])
		# print (this_count)
		counts.append(this_count)
	return counts


def modified_entropy(x, y):
	## shape of x: [# of data, # of classes]. x is a batch of prediction
	modified_entropy = []
	for i in range(len(y)):
		this_pred = x[i]
		this_label = y[i]
		this_modified_entropy = 0
		for j in range(len(this_pred)):
			if (j == this_label):
				this_modified_entropy = this_modified_entropy - (1 - this_pred[this_label]) * torch.log(this_pred[this_label])
			else:
				this_modified_entropy = this_modified_entropy - (this_pred[j]) * torch.log(1 - this_pred[j])
		modified_entropy.append(this_modified_entropy.detach().item())
	
	return np.nan_to_num(np.array(modified_entropy) * -1, posinf=1e9)


def get_blackbox_auc(user_list, target_model, dataset):
	yeom_auc = []
	yeom_tpr = []
	## yeom's attack
	for idx in range(len(user_list)):
		member_probs = []
		nonmember_probs = []
		for (images, labels, _) in user_list[idx].train_eval_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			preds = F.softmax(target_model(images), dim=1)
			# probs = preds[:,labels]
			probs = np.array([preds[i, labels[i]].detach().item() for i in range(len(labels))])
			# print (idx,probs.size())
			member_probs.append(probs)
		
		for (images, labels, _) in user_list[idx].test_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			preds = F.softmax(target_model(images), dim=1)
			# print (preds[0],torch.sum(preds[0]))
			# probs = preds[:, labels]
			probs = np.array([preds[i, labels[i]].detach().item() for i in range(len(labels))])
			# print (preds.size(),labels.size(), probs.shape)
			nonmember_probs.append(probs)
		
		member_probs = np.concatenate(member_probs).flatten()
		nonmember_probs = np.concatenate(nonmember_probs).flatten()
		min_len = min(len(member_probs), len(nonmember_probs))
		min_len = min(min_len, args.eval_data_size)
		# print (f"min len {min_len},member len{len(member_probs)}, nonmember len {len(nonmember_probs)}")
		member_index = np.random.choice(len(member_probs), min_len, replace=False)
		nonmember_index = np.random.choice(len(nonmember_probs), min_len, replace=False)
		# print (len(member_index),len(nonmember_index))
		probs = np.concatenate((member_probs[member_index], nonmember_probs[nonmember_index]), axis=0).flatten()
		labels = np.concatenate((np.ones((min_len)), np.zeros((min_len))), axis=0).astype(np.int64).flatten()
		
		# print (probs.shape,labels.shape)
		
		from sklearn.metrics import roc_auc_score
		auc_score = roc_auc_score(labels, probs)
		# print (f"BLACKBOX LOSS AUC {auc_score}")
		
		from sklearn.metrics import roc_auc_score, roc_curve
		fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
		
		return_tpr = get_tpr(pred=probs, label=labels)
		# print (f"FPR {10/min_len}, TPR {return_tpr}")
		
		yeom_auc.append(auc_score)
		yeom_tpr.append(return_tpr)
	
	print(f"yeom attack: avg auc {np.average(np.array(yeom_auc))}, avg tpr {np.average(np.array(yeom_tpr))} at fpr {10 / min_len}")
	print(f"auc std : {np.std(np.array(yeom_auc))}, tpr std :{np.std(np.array(yeom_tpr))}")
	
	#### here is the merlin attack
	merlin_auc = []
	merlin_tpr = []
	for idx in range(len(user_list)):
		member_counts = []
		nonmember_counts = []
		for (images, labels, _) in user_list[idx].train_eval_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			counts = merlin_attack_count(images, labels, target_model, noise_magnitude=0.01)
			member_counts.append(counts)
		
		for (images, labels, _) in user_list[idx].test_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			counts = merlin_attack_count(dataset, images, labels, target_model, noise_magnitude=0.01)
			nonmember_counts.append(counts)
		
		member_counts = np.concatenate(member_counts).flatten()
		nonmember_counts = np.concatenate(nonmember_counts).flatten()
		# print (member_counts.shape)
		min_len = min(len(member_counts), len(nonmember_counts))
		min_len = min(min_len, args.eval_data_size)
		# print(f"min len {min_len},member len{len(member_counts)}, nonmember len {len(nonmember_counts)}")
		member_index = np.random.choice(len(member_counts), min_len, replace=False)
		nonmember_index = np.random.choice(len(nonmember_counts), min_len, replace=False)
		# print (len(member_index),len(nonmember_index))
		counts = np.concatenate((member_counts[member_index], nonmember_counts[nonmember_index]), axis=0).flatten()
		labels = np.concatenate((np.ones((min_len)), np.zeros((min_len))), axis=0).astype(np.int64).flatten()
		
		from sklearn.metrics import roc_auc_score
		auc_score = roc_auc_score(labels, counts)
		from sklearn.metrics import roc_auc_score, roc_curve
		fpr, tpr, thresholds = roc_curve(labels, counts, pos_label=1)
		return_tpr = get_tpr(pred=counts, label=labels)
		# print(f"FPR {10 / min_len}, TPR {return_tpr}")
		merlin_auc.append(auc_score)
		merlin_tpr.append(return_tpr)
		
		# print (member_counts)
		# print (nonmember_counts)
		if (idx == 0):
			print(np.bincount(member_counts))
			print(np.bincount(nonmember_counts))
	
	print(merlin_auc)
	print(merlin_tpr)
	
	print(f"merlin attack: avg auc {np.average(np.array(merlin_auc))}, avg tpr {np.average(np.array(merlin_tpr))} at fpr {10 / min_len}")
	print(f"auc std : {np.std(np.array(merlin_auc))}, tpr std :{np.std(np.array(merlin_tpr))}")
	
	### here is the modified entropy attack
	song_auc = []
	song_tpr = []
	## song's attack
	for idx in range(len(user_list)):
		member_probs = []
		nonmember_probs = []
		for (images, labels, _) in user_list[idx].train_eval_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			preds = F.softmax(target_model(images), dim=1)
			# probs = preds[:,labels]
			probs = modified_entropy(preds, labels)
			# print (idx,probs.size())
			member_probs.append(probs)
		
		for (images, labels, _) in user_list[idx].test_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			preds = F.softmax(target_model(images), dim=1)
			# print (preds[0],torch.sum(preds[0]))
			# probs = preds[:, labels]
			probs = modified_entropy(preds, labels)
			# print (preds.size(),labels.size(), probs.shape)
			nonmember_probs.append(probs)
		
		member_probs = np.concatenate(member_probs).flatten()
		nonmember_probs = np.concatenate(nonmember_probs).flatten()
		min_len = min(len(member_probs), len(nonmember_probs))
		min_len = min(min_len, args.eval_data_size)
		# print(f"min len {min_len},member len{len(member_probs)}, nonmember len {len(nonmember_probs)}")
		member_index = np.random.choice(len(member_probs), min_len, replace=False)
		nonmember_index = np.random.choice(len(nonmember_probs), min_len, replace=False)
		# print (len(member_index),len(nonmember_index))
		probs = np.concatenate((member_probs[member_index], nonmember_probs[nonmember_index]), axis=0).flatten()
		labels = np.concatenate((np.ones((min_len)), np.zeros((min_len))), axis=0).astype(np.int64).flatten()
		
		# print (probs.shape,labels.shape)
		
		from sklearn.metrics import roc_auc_score
		auc_score = roc_auc_score(labels, probs)
		# print (f"BLACKBOX LOSS AUC {auc_score}")
		
		from sklearn.metrics import roc_auc_score, roc_curve
		fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
		
		return_tpr = get_tpr(pred=probs, label=labels)
		# print (f"FPR {10/min_len}, TPR {return_tpr}")
		
		song_auc.append(auc_score)
		song_tpr.append(return_tpr)
	
	print(f"modified entropy attack: avg auc {np.average(np.array(song_auc))}, avg tpr {np.average(np.array(song_tpr))} at fpr {10 / min_len}")
	print(f"auc std : {np.std(np.array(song_auc))}, tpr std :{np.std(np.array(song_tpr))}")
