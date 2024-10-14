import numpy as np
import torch
import torch.nn as nn
from utils import *
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from whitebox_attack import *
from user import *
from fed_attack_epoch import *
import torch.nn.functional as F

def get_all_info_per_instance(instance_grad, user_grad, batch_size=100,user_total_instance=4000,best_layer=None,test_rank=False,whole_nn=False):
	
	## instance_grad shape: # layers,  batch size, gradient of this layer
	## user_grad shape: # layers, gradient of this layer.
	## we need to extend user_grad to the same shape..
	## user total instance here means the total number of instance that used to produce this user_grad, for epoch case, this is args.target_data_size
	### for batch case, this should be just args.target_batch_size
	
	#if (best_layer!=None):
		# just calculate the statistics for the best layer to reduce computational cost
	#	user_grad = [user_grad[best_layer]]
	#	instance_grad = [instance_grad[best_layer]]
	
	#device = torch.device("cuda",1) ### this is for multi-gpu implementation
	#instance_grad = instance_grad.to(device)
	#user_grad = user_grad.to(device)
	
	###
	#print (f"batch-size{batch_size}")
	#print (len(instance_grad))
	#for p in instance_grad:
	#	print (p[0].size())
	#for p in instance_grad[0]:
	#	print (p.size())
	#print (len(user_grad))
	#for p in user_grad:
	#	print (p.size())
	
	
	device = torch.device("cuda", 0)
	norm_choice=1
	
	if (test_rank == 0 and whole_nn == 0): ### this is always true, except for the case of test on overparameterized model
	
		extended_user_grad = []
		for param in user_grad:
			new_param = torch.stack([param.to(device) for _ in range(batch_size)])
			#print(param.size(),new_param.size())
			extended_user_grad.append(new_param)
	
		this_instance_info = []
		for param1, param2 in zip(instance_grad, extended_user_grad):
			### need to remove bias from calculation to make the result file smaller..
			if (len(param1.size()) == 2): ### avoid bias for all
				continue
			#print (param2.shape)
			# cosine
			param1 = torch.stack([torch.flatten(param1[i]) for i in range(len(param1))])
			param2 = torch.stack([torch.flatten(param2[i]) for i in range(len(param2))])
			cos1 =  F.cosine_similarity(param1,param2,dim=1, eps=1e-10).detach().cpu().numpy()
			torch.cuda.empty_cache()
			# grad diff and grad norm
			#l2dist1 = torch.norm(param2 * user_total_instance,dim=1).detach()  # |B|
			#l2dist2 = torch.norm(param2 * user_total_instance - param1,dim=1).detach()# |(n*B-a)/(n-1)| if member this should be smaller
			l1dist1 = torch.norm(param2 * user_total_instance, p=1,dim=1).detach().cpu().numpy()  # |B|
			torch.cuda.empty_cache()
			l1dist2 = torch.norm(param2 * user_total_instance - param1, p=1,dim=1).detach().cpu().numpy() # |(n*B-a)/(n-1)|
			torch.cuda.empty_cache()
			l1norm = torch.norm(param1,p=1,dim=1).detach().cpu().numpy() # l1 norm of this instance
			torch.cuda.empty_cache()
			l2dist1 = torch.norm(param2 * user_total_instance, p=2,dim=1).detach().cpu().numpy()  # |B|
			torch.cuda.empty_cache()
			l2dist2 = torch.norm(param2 * user_total_instance - param1, p=2,dim=1).detach().cpu().numpy() # |(n*B-a)/(n-1)|
			torch.cuda.empty_cache()
			l2norm = torch.norm(param1,p=2,dim=1).detach().cpu().numpy() # l1 norm of this instance
			torch.cuda.empty_cache()
			'''
			counting_sign = torch.sign(param1 - param3) * torch.sign(param2 - param3)
			instance_pos_count = torch.sum(torch.maximum(torch.sign(param1 - param3), torch.zeros_like(param1))).cpu().numpy()
			instance_neg_count = torch.sum(torch.minimum(torch.sign(param1 - param3), torch.zeros_like(param1))).cpu().numpy()
			batch_pos_count = torch.sum(torch.maximum(torch.sign(param2 - param3), torch.zeros_like(param1))).cpu().numpy()
			batch_neg_count = torch.sum(torch.minimum(torch.sign(param2 - param3), torch.zeros_like(param1))).cpu().numpy()
			# batch_count = torch.sum(torch.sign(param2 - param3)).cpu().numpy()
			this_pos_count = torch.sum(torch.maximum(counting_sign, torch.zeros_like(param1))).cpu().numpy()
			this_neg_count = torch.sum(torch.minimum(counting_sign, torch.zeros_like(param1))).cpu().numpy()
			this_count = torch.sum(counting_sign).detach().cpu().numpy()
			
			this_layer = [cos1.item(), l1dist1.item(), l1dist2.item(),
			            torch.norm(torch.flatten(param1), p=1).item(),
			            l2dist1.item(), l2dist2.item(),
			            torch.norm(torch.flatten(param1)).item(),
			            instance_pos_count.item(), instance_neg_count.item(),
			            batch_pos_count.item(), batch_neg_count.item(),
			            this_pos_count.item(), this_neg_count.item(), this_count.item()]
			'''
			this_layer = [cos1, l1dist1, l1dist2, l1norm,l2dist1,l2dist2,l2norm]
			this_layer = np.stack(this_layer)
			#padding = np.stack([np.zeros_like(this_layer[0]) for _ in range(10)])
			#this_layer = np.concatenate((this_layer,padding),axis=0)
			this_instance_info.append(np.array(this_layer))
		
		this_instance_info = np.array(this_instance_info)
		this_instance_info = np.stack([this_instance_info[:,:,i] for i in range(batch_size)])
	
		return this_instance_info
	
	elif (whole_nn):
		### this is for the whole-NN cos / grad-diff implementation
		## instance_grad shape: # layers,  batch size, gradient of this layer
		## user_grad shape: # layers, gradient of this layer.
		#print ("WHOLE NN SELECTED")

		### we need to reshape extended user grad and instance grad to [#instance,param of whole NN(flatten)]
		flattened_user_grad = torch.cat([torch.flatten(user_grad[i]) for i in range(len(user_grad))])
		#print (flattened_user_grad.size())
		extended_user_grad = torch.stack([flattened_user_grad for _ in range(batch_size)])
		#print (extended_user_grad.size())
		### reshape instance_grad
		flattened_instance_grad = []
		for i in range(batch_size):
			this_instance_grad = torch.cat( [torch.flatten(instance_grad[j][i]) for j in range(len(instance_grad))] )
			flattened_instance_grad.append(this_instance_grad)
		flattened_instance_grad = torch.stack(flattened_instance_grad)
		#print (flattened_instance_grad.size())
		### calculate attack info
		this_instance_info = []
		cos1 =  F.cosine_similarity(flattened_instance_grad,extended_user_grad,dim=1, eps=1e-10).detach().cpu().numpy()
		torch.cuda.empty_cache()
		# grad diff and grad norm
		#l2dist1 = torch.norm(param2 * user_total_instance,dim=1).detach()  # |B|
		#l2dist2 = torch.norm(param2 * user_total_instance - param1,dim=1).detach()# |(n*B-a)/(n-1)| if member this should be smaller
		l1dist1 = torch.norm(extended_user_grad * user_total_instance, p=1,dim=1).detach().cpu().numpy()  # |B|
		torch.cuda.empty_cache()
		l1dist2 = torch.norm(extended_user_grad * user_total_instance - flattened_instance_grad, p=1,dim=1).detach().cpu().numpy() # |(n*B-a)/(n-1)|
		torch.cuda.empty_cache()
		l1norm = torch.norm(flattened_instance_grad,p=1,dim=1).detach().cpu().numpy() # l1 norm of this instance
		torch.cuda.empty_cache()
		l2dist1 = torch.norm(extended_user_grad * user_total_instance, p=2,dim=1).detach().cpu().numpy()  # |B|
		torch.cuda.empty_cache()
		l2dist2 = torch.norm(extended_user_grad * user_total_instance - flattened_instance_grad, p=2,dim=1).detach().cpu().numpy() # |(n*B-a)/(n-1)|
		torch.cuda.empty_cache()
		l2norm = torch.norm(flattened_instance_grad,p=2,dim=1).detach().cpu().numpy() # l1 norm of this instance
		torch.cuda.empty_cache()
		this_layer = [cos1, l1dist1, l1dist2, l1norm,l2dist1,l2dist2,l2norm]
		this_layer = np.stack(this_layer)
		#padding = np.stack([np.zeros_like(this_layer[0]) for _ in range(10)])
		#this_layer = np.concatenate((this_layer, padding), axis=0)
		this_instance_info.append(np.array(this_layer))
		this_instance_info = np.array(this_instance_info)
		this_instance_info = np.stack([this_instance_info[:,:,i] for i in range(batch_size)])
		#print (this_instance_info.shape)
		return this_instance_info
	
	else:
		### this is for the skin-special model cos / grad-diff implementation
		## instance_grad shape: # layers, batch size, gradient of this layer
		## user_grad shape: # layers, gradient of this layer.
		## we only use the gradient for the last two fc layers and concatenate them to be one vector.
		
		### we only need the parameters of the two fc layers, the shape is [393216, test_rank] and [test_rank, 4096]
		#print (f"rank {test_rank}")
		flattened_user_grad = []
		for p in user_grad:
			#print (p.shape)
			#print (list(p.shape))
			#print (list(p.shape) == [test_rank,9216])
			if (list(p.shape) == [1024,test_rank] or list(p.shape) == [test_rank,10]):
				flattened_user_grad.append(torch.flatten(p))
		flattened_user_grad = torch.cat(flattened_user_grad)
		extended_user_grad = torch.stack([flattened_user_grad for _ in range(batch_size)])
		
		#print (extended_user_grad.shape)
		
		flattened_instance_grad = []
		for this_layer_grad in instance_grad:
				#print (this_layer_grad.shape,list(this_layer_grad.shape),list(this_layer_grad[0].shape) == [test_rank,9216])
				if (list(this_layer_grad[0].shape) == [1024,test_rank] or list(this_layer_grad[0].shape) == [test_rank,10]):
					flattened_instance_grad.append(torch.stack([torch.flatten(p) for p in this_layer_grad]))

		flattened_instance_grad = torch.cat(flattened_instance_grad,dim=1)
		
		#print (flattened_instance_grad.shape)
		'''
		### we need to reshape extended user grad and instance grad to [#instance,param of whole NN(flatten)]
		flattened_user_grad = [torch.flatten(user_grad[i]).to(device) for i in range(len(user_grad))]
		#print (len(flattened_user_grad))
		flattened_user_grad = torch.cat(flattened_user_grad)
		# print (flattened_user_grad.size())
		extended_user_grad = torch.stack([flattened_user_grad for _ in range(batch_size)])
		# print (extended_user_grad.size())
		### reshape instance_grad
		flattened_instance_grad = []
		for i in range(batch_size):
			this_instance_grad = torch.cat([torch.flatten(instance_grad[j][i]) for j in range(len(instance_grad))])
			flattened_instance_grad.append(this_instance_grad)
		flattened_instance_grad = torch.stack(flattened_instance_grad)
		# print (flattened_instance_grad.size())
		
		### we know that, the param size of fc1 layer is 256*test_rank + test_rank / param size of fc2 layer is test_rank*23 + 23
		#total_param_number = 256*test_rank + test_rank + test_rank*23 + 23
		### take the param of the last two fc layers.
		#flattened_instance_grad = flattened_instance_grad[:,-total_param_number:]
		#extended_user_grad = extended_user_grad[:,-total_param_number:]
		'''
		
		#print (total_param_number,flattened_instance_grad.shape,extended_user_grad.shape)
		
		### calculate attack info
		this_instance_info = []
		cos1 = F.cosine_similarity(flattened_instance_grad, extended_user_grad, dim=1, eps=1e-10).detach().cpu().numpy()
		torch.cuda.empty_cache()
		# grad diff and grad norm
		# l2dist1 = torch.norm(param2 * user_total_instance,dim=1).detach()  # |B|
		# l2dist2 = torch.norm(param2 * user_total_instance - param1,dim=1).detach()# |(n*B-a)/(n-1)| if member this should be smaller
		l1dist1 = torch.norm(extended_user_grad * user_total_instance, p=1, dim=1).detach().cpu().numpy()  # |B|
		torch.cuda.empty_cache()
		l1dist2 = torch.norm(extended_user_grad * user_total_instance - flattened_instance_grad, p=1, dim=1).detach().cpu().numpy()  # |(n*B-a)/(n-1)|
		torch.cuda.empty_cache()
		l1norm = torch.norm(flattened_instance_grad, p=1, dim=1).detach().cpu().numpy()  # l1 norm of this instance
		torch.cuda.empty_cache()
		l2dist1 = torch.norm(extended_user_grad * user_total_instance, p=2,dim=1).detach().cpu().numpy()  # |B|
		torch.cuda.empty_cache()
		l2dist2 = torch.norm(extended_user_grad * user_total_instance - flattened_instance_grad, p=2,dim=1).detach().cpu().numpy() # |(n*B-a)/(n-1)|
		torch.cuda.empty_cache()
		l2norm = torch.norm(flattened_instance_grad,p=2,dim=1).detach().cpu().numpy() # l1 norm of this instance
		torch.cuda.empty_cache()
		this_layer = [cos1, l1dist1, l1dist2, l1norm,l2dist1,l2dist2,l2norm]
		this_layer = np.stack(this_layer)
		# padding = np.stack([np.zeros_like(this_layer[0]) for _ in range(10)])
		# this_layer = np.concatenate((this_layer, padding), axis=0)
		this_instance_info.append(np.array(this_layer))
		this_instance_info = np.array(this_instance_info)
		this_instance_info = np.stack([this_instance_info[:, :, i] for i in range(batch_size)])
		
		#print (this_instance_info.shape)
		
		return this_instance_info
	


def multi_party_member_attack(user_list, model, batch_size, get_gradient_func=None, user_update_list=None, thres=0.05,
                              attack_loader_list=None,user_total_instance=4000,max_instance_per_batch=50,best_layer=None,test_rank=False,whole_nn=False): ### using gpu 1
	### for this attack, user_update_list contains the update from each user after local training for 1(or more) epochs
	### the update from each user serves as the gradient
	### now we calculate the gradient of each instance to do grad_diff, cosine, counting and norm attack
	
	all_info = []
	all_label = []
	num_instance_per_user = 0
	for target_idx in range(len(user_list)):
		this_user_data_info = []
		
		#one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset, batch_size=1,shuffle=False)
		batch_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset, batch_size=max_instance_per_batch,shuffle=False)
		
		# if (len(attack_loader_list) != 0):  ### for batch level attack
		#  batch_instance_loader = attack_loader_list[target_idx]
		# print ("batch level atatck activated!")
		
		instance_count = 0
		#for image, label, _ in one_instance_loader:
		for image, label, instance_idx in batch_instance_loader:
			this_instance_info = []
			this_instance_grad = get_gradient_func(model, image, label)
			
			for eval_idx in range(len(user_update_list)):
				info = get_all_info_per_instance(this_instance_grad, user_update_list[eval_idx], batch_size=max_instance_per_batch,
				                                 user_total_instance=user_total_instance,best_layer=best_layer,test_rank=test_rank,whole_nn=whole_nn)
				this_instance_info.append(info)
				#print (f"eval_idx {eval_idx}, instance_idx {instance_idx},target_idx {target_idx}")
			
			model.zero_grad()
			torch.cuda.empty_cache()
			del this_instance_grad
			
			instance_count += max_instance_per_batch
			this_instance_info = np.array(this_instance_info)
			this_instance_info = np.stack([this_instance_info[:,i,:,:] for i in range(max_instance_per_batch)])
			#print (f"this instance info shape{this_instance_info.shape}")
			this_user_data_info.append(this_instance_info)
			all_label.append([target_idx for _ in range(max_instance_per_batch)])
		
		#print (f"total member instance {instance_count}")
		
		this_user_data_info = np.vstack([this_user_data_info[i] for i in range(len(this_user_data_info))])
		#print (f"this_user data info shape{this_user_data_info.shape}")
		all_info.append(this_user_data_info)
		num_instance_per_user = instance_count
	
	#print (f"num instance per user {num_instance_per_user}")
	
	### we also need to add non-member set
	
	for target_idx in range(len(user_list)):
		# specifically for ss case
		if (len(user_list)==1):
			break
			
		this_user_data_info = []
		#one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset,batch_size=1,shuffle=False)
		batch_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset, batch_size=max_instance_per_batch, shuffle=False)
		
		#if (len(attack_loader_list) != 0):  ### for batch level attack
		#    batch_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].test_dataset, batch_size=max_instance_per_batch,shuffle=False)
		#    #print ("batch level attack activated!")
		
		instance_count = 0
		for image, label, instance_idx in batch_instance_loader:
			this_instance_info = []
			this_instance_grad = get_gradient_func(model, image, label)
			
			for eval_idx in range(len(user_update_list)):
				info = get_all_info_per_instance(this_instance_grad, user_update_list[eval_idx], batch_size=max_instance_per_batch,
				                                 user_total_instance=user_total_instance,best_layer=best_layer,test_rank=test_rank,whole_nn=whole_nn)
				this_instance_info.append(info)
				#print (f"eval_idx {eval_idx}, instance_idx {instance_idx},target_idx {target_idx}")
			
			torch.cuda.empty_cache()
			model.zero_grad()
			
			instance_count += max_instance_per_batch
			this_instance_info = np.array(this_instance_info)
			this_instance_info = np.stack([this_instance_info[:,i,:,:] for i in range(max_instance_per_batch)])
			this_user_data_info.append(this_instance_info)
			all_label.append([len(user_list) for _ in range(max_instance_per_batch)])
			
			if (instance_count >= num_instance_per_user):
				break
		
		this_user_data_info = np.vstack([this_user_data_info[i] for i in range(len(this_user_data_info))])
		#print (f"this_user data info shape{this_user_data_info.shape}")
		all_info.append(this_user_data_info)
	
	#break ### just calculate the non-member statistics for one set of non-member.
	
	# print (f"total non-member instance {instance_count}")
	
	# for data in all_info:
	#    print (np.array(data).shape)
	all_label = np.array(all_label).flatten()
	all_info = np.squeeze(np.array(all_info))
	#print (all_info.shape)
	all_info = np.vstack([all_info[i] for i in range(len(all_info))])
	#print (all_info.shape,all_label.shape)
	return np.array(all_info), np.array(all_label)


def multi_party_member_attack_valid(user_list, model, batch_size, get_gradient_func=None, user_update_list=None, thres=0.05,
                              attack_loader_list=None, user_total_instance=4000, max_instance_per_batch=50, best_layer=None, test_rank=False,whole_nn=False):  ### using gpu 1
	### for this attack, user_update_list contains the update from each user after local training for 1(or more) epochs
	### the update from each user serves as the gradient
	### now we calculate the gradient of each instance to do grad_diff, cosine, counting and norm attack
	
	all_info = []
	all_label = []
	for target_idx in range(len(user_list)):
		if(target_idx>0):
			break
		this_user_data_info = []
		
		batch_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].validation_base_dataset, batch_size=max_instance_per_batch, shuffle=False)
		
		instance_count = 0
		for image, label, instance_idx in batch_instance_loader:
			this_instance_info = []
			this_instance_grad = get_gradient_func(model, image, label)
			
			for eval_idx in range(len(user_update_list)):
				info = get_all_info_per_instance(this_instance_grad, user_update_list[eval_idx], batch_size=max_instance_per_batch,
				                                 user_total_instance=user_total_instance, best_layer=best_layer, test_rank=test_rank,whole_nn=whole_nn)
				this_instance_info.append(info)
			
			model.zero_grad()
			torch.cuda.empty_cache()
			del this_instance_grad
			
			instance_count += max_instance_per_batch
			this_instance_info = np.array(this_instance_info)
			this_instance_info = np.stack([this_instance_info[:, i, :, :] for i in range(max_instance_per_batch)])
			this_user_data_info.append(this_instance_info)
			all_label.append([target_idx for _ in range(max_instance_per_batch)])
		
		
		this_user_data_info = np.vstack([this_user_data_info[i] for i in range(len(this_user_data_info))])
		all_info.append(this_user_data_info)
	
	all_label = np.array(all_label).flatten()
	all_info = np.squeeze(np.array(all_info))
	#all_info = np.vstack([all_info[i] for i in range(len(all_info))])
	#print(all_info.shape, all_label.shape)
	return np.array(all_info), np.array(all_label)


def multi_party_member_loss_attack(user_list, model, batch_size, get_gradient_func=None, user_update_list=None,
                                   user_model_list=None, thres=0.05, attack_loader_list=[],max_instance_per_batch=50,model_name='inception'): ### using gpu 1
	### for this attack, user_update_list contains the update from each user after local training for 1(or more) epochs
	### the update from each user serves as the gradient
	### now we calculate the gradient of each instance to do grad_diff, cosine, counting and norm attack
	
	# print(f"user_model_list_len {len(user_model_list)}")
	
	#for name, param in model.named_parameters():
	#    print(f"param dtype {param.dtype}")
	#    break
	
	# ori_state_dict = copy.deepcopy(model.state_dict())
	device = torch.device("cuda",0)
	criterion = nn.CrossEntropyLoss(reduction='none').to(device)
	model.to(device)
	
	all_info = []
	all_label = []
	num_instance_per_user = 0
	for target_idx in range(len(user_list)):
		this_user_data_info = []
		
		#one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset, batch_size=1,shuffle=False)
		batch_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset, batch_size=max_instance_per_batch,shuffle=False)
		
		#if (len(attack_loader_list) != 0):  ### for batch level attack
		#    batch_instance_loader = attack_loader_list[target_idx]
		#    # print ("batch level atatck activated!")
		
		instance_count = 0
		with torch.no_grad():
			for image, label, _ in batch_instance_loader:
				image = image.to(device)
				label = label.to(device)
				
				#print (f"data dtype {image.dtype}, label dtype {label.dtype}")
				
				this_instance_loss = []
				for eval_idx in range(len(user_model_list)):
					model.load_state_dict(user_model_list[eval_idx])
					if (model_name == 'inception'):
						log_probs,_ = model(image)
					else:
						log_probs = model(image)
					loss = criterion(log_probs, label)
					this_instance_loss.append(loss.detach().cpu().numpy())
					
					model.zero_grad()
				
				instance_count += max_instance_per_batch
				this_user_data_info.append(this_instance_loss)
				all_label.append([target_idx for _ in range(max_instance_per_batch)])
		
		# print (f"total member instance {instance_count}")
		num_instance_per_user = instance_count
		
		this_user_data_info = np.array(this_user_data_info)
		#print(this_user_data_info.shape)
		this_user_data_info = np.array([[this_user_data_info[i,:,j] for j in range(max_instance_per_batch)] for i in range(len(this_user_data_info))])
		this_user_data_info = np.vstack([this_user_data_info[i] for i in range(len(this_user_data_info))])
		#print(this_user_data_info.shape)
		
		all_info.append(this_user_data_info)
	
	### we also need to add non-member set
	
	for target_idx in range(len(user_list)):
		if (len(user_list) == 1):
			break
		this_user_data_info = []
		#one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset,batch_size=1,shuffle=False)
		batch_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset,batch_size=max_instance_per_batch,shuffle=False)
		
		#if (len(attack_loader_list) != 0):  ### for batch level attack
		#    batch_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].test_dataset, batch_size=max_instance_per_batch,shuffle=False)
		
		instance_count = 0
		
		with torch.no_grad():
			for image, label, _ in batch_instance_loader:
				image = image.to(device)
				label = label.to(device)
				this_instance_loss = []
				for eval_idx in range(len(user_model_list)):
					model.load_state_dict(user_model_list[eval_idx])
					if (model_name == 'inception'):
						log_probs,_ = model(image)
					else:
						log_probs = model(image)
					loss = criterion(log_probs, label)
					this_instance_loss.append(loss.detach().cpu().numpy())
					model.zero_grad()
				
				instance_count += max_instance_per_batch
				this_user_data_info.append(this_instance_loss)
				all_label.append([len(user_list) for _ in range(max_instance_per_batch)])
				if (instance_count >= num_instance_per_user):
					break
		
		this_user_data_info = np.array(this_user_data_info)
		#print(this_user_data_info.shape)
		this_user_data_info = np.array([[this_user_data_info[i, :, j] for j in range(max_instance_per_batch)] for i in range(len(this_user_data_info))])
		this_user_data_info = np.vstack([this_user_data_info[i] for i in range(len(this_user_data_info))])
		#print(this_user_data_info.shape)
		
		all_info.append(this_user_data_info)
	
	all_info = np.array(all_info)
	all_info = np.vstack([all_info[i] for i in range(len(all_info))])
	all_label = np.array(all_label).flatten()
	#print (np.array(all_info).shape,np.array(all_label).shape)
	return np.array(all_info), np.array(all_label)


def multi_party_member_loss_attack_valid(user_list, model, batch_size, get_gradient_func=None, user_update_list=None,
                                         user_model_list=None, thres=0.05, attack_loader_list=[], max_instance_per_batch=50,model_name='inception'):
	
	device = torch.device("cuda", 0)
	criterion = nn.CrossEntropyLoss(reduction='none').to(device)
	model.to(device)
	
	all_info = []
	all_label = []
	for target_idx in range(len(user_list)):
		if(target_idx>0):
			break
		this_user_data_info = []
		
		batch_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].validation_base_dataset, batch_size=max_instance_per_batch, shuffle=False)
		
		instance_count = 0
		with torch.no_grad():
			for image, label, _ in batch_instance_loader:
				image = image.to(device)
				label = label.to(device)
				this_instance_loss = []
				for eval_idx in range(len(user_model_list)):
					model.load_state_dict(user_model_list[eval_idx])
					if (model_name == 'inception'):
						log_probs,_ = model(image)
					else:
						log_probs = model(image)
					loss = criterion(log_probs, label)
					this_instance_loss.append(loss.detach().cpu().numpy())
					model.zero_grad()
				instance_count += max_instance_per_batch
				this_user_data_info.append(this_instance_loss)
				all_label.append([target_idx for _ in range(max_instance_per_batch)])
		
		this_user_data_info = np.array(this_user_data_info)
		this_user_data_info = np.array([[this_user_data_info[i, :, j] for j in range(max_instance_per_batch)] for i in range(len(this_user_data_info))])
		this_user_data_info = np.vstack([this_user_data_info[i] for i in range(len(this_user_data_info))])
		all_info.append(this_user_data_info)
	
	all_info = np.array(all_info)
	all_info = np.vstack([all_info[i] for i in range(len(all_info))])
	all_label = np.array(all_label).flatten()
	#print(np.array(all_info).shape, np.array(all_label).shape)
	return np.array(all_info), np.array(all_label)


def multi_party_member_prediction_attack(user_list, model, batch_size, get_gradient_func=None, user_update_list=None,
                                         user_model_list=None, thres=0.05):
    ### for this attack, user_update_list contains the update from each user after local training for 1(or more) epochs
    ### the update from each user serves as the gradient
    ### now we calculate the gradient of each instance to do grad_diff, cosine, counting and norm attack

    # print(f"user_model_list_len {len(user_model_list)}")

    # ori_state_dict = copy.deepcopy(model.state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)
    all_info = []
    all_label = []
    for target_idx in range(len(user_list)):
        this_user_data_info = []
        one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset, batch_size=1,
                                                          shuffle=False)
        instance_count = 0
        for image, label, _ in one_instance_loader:
            image = image.to(device)
            label = label.to(device)
            this_instance_loss = []
            for eval_idx in range(len(user_model_list)):
                model.load_state_dict(user_model_list[eval_idx])
                log_probs = model(image)
                loss = criterion(log_probs, label)
                this_instance_loss.append(log_probs.detach().cpu().numpy())

                model.zero_grad()

            instance_count += 1
            this_user_data_info.append(this_instance_loss)
            all_label.append(label.detach().cpu().numpy())

        # print (f"total member instance {instance_count}")

        all_info.append(this_user_data_info)

    ### we also need to add non-member set

    for target_idx in range(len(user_list)):
        this_user_data_info = []
        one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset,
                                                          batch_size=1,
                                                          shuffle=False)
        instance_count = 0

        for image, label, _ in one_instance_loader:
            image = image.to(device)
            label = label.to(device)
            this_instance_loss = []
            for eval_idx in range(len(user_model_list)):
                model.load_state_dict(user_model_list[eval_idx])
                log_probs = model(image)
                loss = criterion(log_probs, label)
                this_instance_loss.append(log_probs.detach().cpu().numpy())

                model.zero_grad()

            instance_count += 1
            this_user_data_info.append(this_instance_loss)
            all_label.append(label.detach().cpu().numpy())

        all_info.append(this_user_data_info)
        # print (f"total non-member instance {instance_count}")

    # print (np.array(all_info).shape,np.array(all_label).shape)

    # model.load_state_dict(ori_state_dict)

    return np.array(all_info), np.array(all_label)


def instance_cos(user_list, model, batch_size, get_gradient_func=None, user_update_list=None, user_model_list=None,
                 thres=0.05):
    ### for each user, we only select 500 to calculate this

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    all_cos = []
    count_limit = 500
    for user_i in range(len(user_list)):
        for user_j in range(len(user_list)):
            user_i_instance_loader = torch.utils.data.DataLoader(user_list[user_i].evaluation_member_dataset,
                                                                 batch_size=1,
                                                                 shuffle=False)
            user_j_instance_loader = torch.utils.data.DataLoader(user_list[user_j].evaluation_member_dataset,
                                                                 batch_size=1,
                                                                 shuffle=False)
            count_i = 0
            for image_i, label_i, _ in user_i_instance_loader:
                count_j = 0
                this_i_cos = []
                for image_j, label_j, _ in user_j_instance_loader:
                    this_j_cos = []
                    i_grad = get_gradient_func(model, image_i, label_i)
                    j_grad = get_gradient_func(model, image_j, label_j)
                    for param1, param2 in zip(i_grad, j_grad):
                        this_j_cos.append(cos(torch.flatten(param1), torch.flatten(param2)).detach().cpu().item())

                    count_j += 1
                    this_i_cos.append(this_j_cos)
                    if (count_j >= count_limit):
                        break

                count_i += 1
                if (count_i >= count_limit):
                    break

                all_cos.append(np.array(this_i_cos))

    print (np.array(all_cos).shape)

    return np.array(all_cos)
