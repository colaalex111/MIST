import torch
import torchvision
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from PIL import Image
#from memory_profiler import profile
#from utils import Cutout

class part_pytorch_dataset(data.Dataset):
	def __init__(self,x,y,train=True,transform=None, target_transform=None,float_target=False):
		self.transform = transform
		self.target_transform = target_transform
		self.train = train
		if (train):
			self.train_data = x
			self.train_labels = y
		else:
			self.test_data = x
			self.test_labels = y
		self.float_target =  float_target
	
	def __getitem__(self, index):
		if self.train:
			img, target = self.train_data[index], self.train_labels[index]
		else:
			img, target = self.test_data[index], self.test_labels[index]
		
		if self.transform is not None:
			new_img = self.transform(img)
		else:
			new_img = torch.from_numpy(np.array(img)).type(torch.float32) ## type(torch.float64)
		#print (target.shape)
		#print (self.float_target)
		if (not self.float_target):
			new_target = torch.from_numpy(np.array(target)).type(torch.LongTensor)
		else:
			new_target = torch.from_numpy(np.array(target)).type(torch.FloatTensor)
		
		#return new_img.type(torch.float32), new_target, index
		return new_img, new_target, index
	
	def __len__(self):
		if self.train:
			return len(self.train_data)
		else:
			return len(self.test_data)


class dataset():
	def __init__(self,dataset_name,membership_attack_number=0,cutout=False,gpu=1,n_holes=1,length=16):
		self.dataset_name = dataset_name
		self.gpu = gpu
		self.membership_attack_number = membership_attack_number
		self.cutout = cutout
		self.n_holes=n_holes
		self.length=length
		
		if (gpu==1):
			if (dataset_name == 'location'):
				self.train_data = np.load('/home/lijiacheng/dataset/location_train_data.npy')
				self.train_label = np.load('/home/lijiacheng/dataset/location_train_label.npy')
				self.test_data = np.load('/home/lijiacheng/dataset/location_test_data.npy')
				self.test_label = np.load('/home/lijiacheng/dataset/location_test_label.npy')
				self.data = np.concatenate((self.train_data,self.test_data),axis=0)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0)
				
				self.train_data = self.data[:4500]
				self.train_label = self.label[:4500]
				self.test_data = self.data[4500:]
				self.test_label = self.label[4500:]
				
				#print (np.unique(self.train_label))
				#print (np.unique(self.test_label))
				#print (np.unique(self.label))
			if (dataset_name == 'kidney'):
				self.train_data = np.load('/home/lijiacheng/dataset/kidney_train_data.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/kidney_train_label.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/kidney_test_data.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/kidney_test_label.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data,self.test_data),axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0)
			
			if (dataset_name == 'fashion_mnist'):
				self.train_data = np.load('/home/lijiacheng/dataset/fashion_mnist_train_images.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/fashion_mnist_train_labels.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/fashion_mnist_test_images.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/fashion_mnist_test_labels.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data, self.test_data), axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label, self.test_label), axis=0)
			
			if (dataset_name == 'kidney299'):
				self.train_data = np.load('/home/lijiacheng/dataset/kidney_train_data_299.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/kidney_train_label_299.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/kidney_test_data_299.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/kidney_test_label_299.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data,self.test_data),axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0)
				#print (self.train_label.shape,self.test_label.shape)
			
			if (dataset_name == 'skin'):
				self.train_data = np.load('/home/lijiacheng/dataset/skin_train_data.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/skin_train_label.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/skin_test_data.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/skin_test_label.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data,self.test_data),axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0)

			if (dataset_name == 'skin299'):
				self.train_data = np.load('/home/lijiacheng/dataset/skin_train_data_299.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/skin_train_label_299.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/skin_test_data_299.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/skin_test_label_299.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data, self.test_data), axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label, self.test_label), axis=0)
				
			if (dataset_name == 'chest'):
				self.train_data = np.load('/home/lijiacheng/dataset/chest_train_data.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/chest_train_label.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/chest_test_data.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/chest_test_label.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data,self.test_data),axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0)
				print (self.train_label.shape,self.test_label.shape)
				

			if (dataset_name == 'chest299'):
				self.train_data = np.load('/home/lijiacheng/dataset/chest_train_data_299.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/chest_train_label_299.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/chest_test_data_299.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/chest_test_label_299.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data,self.test_data),axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0)
				print (self.train_label.shape,self.test_label.shape)
			
			if (dataset_name == 'cifar100'):
				self.train_data = np.load('/home/lijiacheng/dataset/cifar100_train_data.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/cifar100_train_label.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/cifar100_test_data.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/cifar100_test_label.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data,self.test_data),axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0)
			
			if (dataset_name == 'covid'):
				self.train_data = np.load('/home/lijiacheng/dataset/covid_train_data.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/covid_train_label.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/covid_test_data.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/covid_test_label.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data,self.test_data),axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0).astype(np.int64)
				

			if (dataset_name == 'covid299'):
				self.train_data = np.load('/home/lijiacheng/dataset/covid_train_data_299.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/covid_train_label_299.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/covid_test_data_299.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/covid_test_label_299.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data,self.test_data),axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0).astype(np.int64)
			
			if (dataset_name == 'retina'):
				self.train_data = np.load('/home/lijiacheng/dataset/retina_train_data.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/retina_train_label.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/retina_test_data.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/retina_test_label.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data,self.test_data),axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0).astype(np.int64)

				train_index = np.random.choice(len(self.data), 32000, replace=False)
				test_index = np.setdiff1d(np.arange(len(self.data)), train_index)
				self.train_data = self.data[train_index]
				self.train_label = self.label[train_index]
				self.test_data = self.data[test_index]
				self.test_label = self.label[test_index]
				
				print (self.data.shape,self.label.shape)
			
			if (dataset_name == 'retina299'):
				self.train_data = np.load('/home/lijiacheng/dataset/retina_train_data_299.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/retina_train_label_299.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/retina_test_data_299.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/retina_test_label_299.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data, self.test_data), axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label, self.test_label), axis=0).astype(np.int64)
				
				train_index = np.random.choice(len(self.data), 32000, replace=False)
				test_index = np.setdiff1d(np.arange(len(self.data)), train_index)
				self.train_data = self.data[train_index]
				self.train_label = self.label[train_index]
				self.test_data = self.data[test_index]
				self.test_label = self.label[test_index]
				
				print(self.data.shape, self.label.shape)
			
			if (dataset_name == 'cifar10'):
				self.train_data = np.load('/home/lijiacheng/dataset/cifar10_train_data.npy').astype(np.uint8)
				self.train_label = np.load('/home/lijiacheng/dataset/cifar10_train_label.npy').astype(np.int64)
				self.test_data = np.load('/home/lijiacheng/dataset/cifar10_test_data.npy').astype(np.uint8)
				self.test_label = np.load('/home/lijiacheng/dataset/cifar10_test_label.npy').astype(np.int64)
				self.data = np.concatenate((self.train_data,self.test_data),axis=0).astype(np.uint8)
				self.label = np.concatenate((self.train_label,self.test_label),axis=0).astype(np.int64)
				
				#train_index = np.random.choice(np.arange(len(self.label)), 50000, replace=False)
				#test_index = np.setdiff1d(np.arange(len(self.label)), train_index)
				
				#self.train_data = self.data[train_index]
				#self.train_label = self.label[train_index]
				#self.test_data = self.data[test_index]
				#self.test_label = self.label[test_index]
				#print (self.train_data.shape,self.train_label.shape,self.test_data.shape,self.test_label.shape)
			
			if (dataset_name == 'texas'):
				self.train_data = np.load('/home/lijiacheng/dataset/texas_train_data.npy')
				self.train_label = np.load('/home/lijiacheng/dataset/texas_train_label.npy')-1
				self.test_data = np.load('/home/lijiacheng/dataset/texas_test_data.npy')
				self.test_label = np.load('/home/lijiacheng/dataset/texas_test_label.npy')-1
				self.data = np.concatenate((self.train_data, self.test_data), axis=0)
				self.label = np.concatenate((self.train_label, self.test_label), axis=0).astype(np.int64)
				self.label = np.squeeze(self.label)
				self.train_label = np.squeeze(self.train_label)
				self.test_label = np.squeeze(self.test_label)
				
				#print (len(self.label))
				### move 10000 testing into validation
				train_index = np.random.choice(np.arange(len(self.label)),50000,replace=False)
				test_index = np.setdiff1d(np.arange(len(self.label)),train_index)
				
				self.train_data = self.data[train_index]
				self.train_label = self.label[train_index]
				self.test_data = self.data[test_index]
				self.test_label = self.label[test_index]
				#print (self.train_data.shape,self.train_label.shape,self.test_data.shape,self.test_label.shape)
				#print (np.bincount(self.train_label),np.bincount(self.test_label))
				#print (np.amin(np.bincount(self.train_label)),np.amin(np.bincount(self.test_label)))
				#print (np.amin(np.bincount(self.label)),np.bincount(self.label))
				'''
				### do resampling here. for train each class 200, for test each class 36
				sampled_train_data = []
				sampled_test_data = []
				sampled_train_label = []
				sampled_test_label = []
				for i in range(100):
					# find all this class idx
					this_class_idx = np.arange(len(self.label))[self.label == i]
					#print (len(this_class_idx))
					train_idx = np.random.choice(this_class_idx,205,replace=False)
					#print (len(train_idx))
					this_class_idx_left = np.setdiff1d(this_class_idx,train_idx)
					#print (len(this_class_idx_left))
					test_idx = np.random.choice(this_class_idx_left,30,replace=False)
					#print (len(test_idx))
					
					sampled_train_data.append(self.data[train_idx])
					sampled_train_label.append(self.label[train_idx])
					sampled_test_data.append(self.data[test_idx])
					sampled_test_label.append(self.label[test_idx])
					
				sampled_train_data = np.vstack(sampled_train_data)
				sampled_train_label = np.array(sampled_train_label).flatten()
				sampled_test_data = np.vstack(sampled_test_data)
				sampled_test_label = np.array(sampled_test_label).flatten()
				#print (sampled_train_data.shape)
				#print (sampled_test_data.shape)
				#print (sampled_train_label.shape)
				#print (sampled_test_label.shape)
				
				self.train_data = sampled_train_data
				self.train_label = sampled_train_label
				self.test_data = sampled_test_data
				self.test_label = sampled_test_label
				self.data = np.concatenate((self.train_data, self.test_data), axis=0)
				self.label = np.concatenate((self.train_label, self.test_label), axis=0).astype(np.int64)
				self.label = np.squeeze(self.label)
				'''

				#print (self.train_data.shape,self.train_label.shape,self.test_data.shape,self.test_label.shape)
			
			
			
			if (dataset_name == 'purchase'):
				self.train_data = np.load('/home/lijiacheng/dataset/purchase_train_data.npy')
				self.train_label = np.load('/home/lijiacheng/dataset/purchase_train_label.npy')-1
				self.test_data = np.load('/home/lijiacheng/dataset/purchase_test_data.npy')
				self.test_label = np.load('/home/lijiacheng/dataset/purchase_test_label.npy')-1
				self.data = np.concatenate((self.train_data, self.test_data), axis=0)
				self.label = np.concatenate((self.train_label, self.test_label), axis=0).astype(np.int64)
				
				### move 10000 testing into validation
				train_index = np.random.choice(np.arange(len(self.label)),80000,replace=False)
				test_index = np.setdiff1d(np.arange(len(self.label)),train_index)
				
				self.train_data = self.data[train_index]
				self.train_label = self.label[train_index]
				self.test_data = self.data[test_index]
				self.test_label = self.label[test_index]
				
				print (self.train_data.shape,self.train_label.shape,self.test_data.shape,self.test_label.shape)
				print (np.bincount(self.train_label),np.bincount(self.test_label))
		#if (set_float):
		#	self.train_data = self.train_data.astype(np.float32)
			#self.train_label = self.train_label.astype(np.float32)
		#	self.test_data = self.test_data.astype(np.float32)
			#self.train_data = self.train_data.astype(np.float32)
		#	self.data = self.data.astype(np.float32)
			
	
	#@profile
	def select_part(self,data_number,membership_attack_number,reference_number=0,shadow_model_label=1):
		self.data_number = data_number
		#print ("data number = %d " % data_number)
		
		if (membership_attack_number==0):
			return self.old_select_part(data_number)
		
		else:
			self.in_train_eval_partition = np.random.choice(len(self.eval_attack_parition),int(len(self.eval_attack_parition)/2),replace=False)
			self.out_train_eval_partition = np.setdiff1d(np.arange(len(self.eval_attack_parition)),self.in_train_eval_partition)
			
			#print (self.eval_attack_parition)
			#print (self.in_train_eval_partition,self.out_train_eval_partition)
			
			if (shadow_model_label==1):
				self.train_partition_first = self.eval_attack_parition[self.in_train_eval_partition]
				self.train_partition_second = self.train_rest_partition[np.random.choice(len(self.train_rest_partition),data_number - int((membership_attack_number/2)),replace=False)]
				self.validation_partition = np.setdiff1d(self.train_rest_partition,self.train_partition_second)
				self.train_partition = np.concatenate((self.train_partition_first,self.train_partition_second))
				self.part_train_data = np.copy(self.train_data[self.train_partition])
				self.part_train_label = np.copy(self.train_label[self.train_partition])
				
				self.part_test_data = self.test_data
				self.part_test_label = self.test_label
				self.part_eval_data = np.copy(self.train_data[self.eval_attack_parition])
				self.part_eval_label = np.copy(self.train_label[self.eval_attack_parition])
				self.part_validation_data = np.copy(self.train_data[self.validation_partition])
				self.part_validation_label = np.copy(self.train_label[self.validation_partition])
				
				#### sort train/valid for mmd loss to compute on each class
				self.part_train_data = self.part_train_data[np.argsort(self.part_train_label)]
				self.part_train_label = self.part_train_label[np.argsort(self.part_train_label)]
				self.part_validation_data = self.part_validation_data[np.argsort(self.part_validation_label)]
				self.part_validation_label = self.part_validation_label[np.argsort(self.part_validation_label)]
				
				#### sanity check
				#print ((np.sort(self.eval_attack_parition[self.out_train_eval_partition]))[:20])
				#print (self.validation_partition[:20])
				#print (np.sort(self.train_partition)[:20])
			
			else:
				self.train_partition_first = self.eval_attack_parition[self.in_train_eval_partition]
				self.train_partition_second = np.random.choice(len(self.test_label),data_number - int((membership_attack_number/2)),replace=False)
				#self.validation_partition = np.setdiff1d(self.train_rest_partition,self.train_partition_second)
				self.validation_partition = self.train_rest_partition
				#self.train_partition = np.concatenate((self.train_partition_first,self.train_partition_second))
				self.part_train_data = np.copy(np.concatenate((self.train_data[self.train_partition_first],self.test_data[self.train_partition_second])))
				self.part_train_label = np.copy(np.concatenate((self.train_label[self.train_partition_first],self.test_label[self.train_partition_second])))
				
				self.part_test_partition_first = np.setdiff1d(np.arange(len(self.test_label)),self.train_partition_second)
				self.part_test_partition_second = self.eval_attack_parition[self.out_train_eval_partition]
				self.part_test_data = np.copy(np.concatenate((self.test_data[self.part_test_partition_first],self.train_data[self.part_test_partition_second])))
				self.part_test_label = np.copy(np.concatenate((self.test_label[self.part_test_partition_first],self.train_label[self.part_test_partition_second])))
				
				self.part_eval_data = np.copy(self.train_data[self.eval_attack_parition])
				self.part_eval_label = np.copy(self.train_label[self.eval_attack_parition])
				self.part_validation_data = np.copy(self.train_data[self.validation_partition])
				self.part_validation_label = np.copy(self.train_label[self.validation_partition])
				
				#self.part_test_data = np.concatenate((self.part_test_data,self.part_validation_data))
				#self.part_test_label = np.concatenate((self.part_test_label,self.part_validation_label))
				
				#### sort train/valid for mmd loss to compute on each class
				self.part_train_data = self.part_train_data[np.argsort(self.part_train_label)]
				self.part_train_label = self.part_train_label[np.argsort(self.part_train_label)]
				self.part_validation_data = self.part_validation_data[np.argsort(self.part_validation_label)]
				self.part_validation_label = self.part_validation_label[np.argsort(self.part_validation_label)]
		
		
		
		### sort the validation data according to the class index
		sorted_index = np.argsort(self.part_validation_label)
		self.part_validation_data = self.part_validation_data[sorted_index]
		self.part_validation_label = self.part_validation_label[sorted_index]
		
		### create a index list for starting index of each class
		self.starting_index = []
		#print ("starting index",self.starting_index)
		for i in np.unique(self.part_validation_label):
			for j in range(len(self.part_validation_label)):
				if (self.part_validation_label[j] == i):
					#print ("class %d index %d "%(i,j))
					self.starting_index.append(j)
					break
		
		### check length of train,valid,test
		print ("training data num",self.part_train_data.shape[0])
		print ("validation data num",self.part_validation_data.shape[0])
		print ("testing data num",self.part_test_data.shape[0])
		
		import gc
		collected = gc.collect()
		#print("Garbage collector: collected",
		#      "%d objects." % collected)
		
		
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
		
		target_transform = transforms.Compose([
			transforms.ToTensor()
		])
		
		if (self.cutout==1):
			transform_train.transforms.append(Cutout(n_holes=self.n_holes, length=self.length))
			print ("CUTOUT ACTIVATED!")
		
		if (self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100'):
			
			#print (self.train_data[0])
			train = part_pytorch_dataset(self.part_train_data,self.part_train_label,train=True,transform=transform_train,target_transform=target_transform)
			test = part_pytorch_dataset(self.part_test_data,self.part_test_label,train=False,transform=transform_test,target_transform=target_transform)
			eval_attack = part_pytorch_dataset(self.part_eval_data,self.part_eval_label,train=False,transform=transform_test,target_transform=target_transform)
			validation = part_pytorch_dataset(self.part_validation_data,self.part_validation_label,train=False,transform=transform_test,target_transform=target_transform)
			train_eval = part_pytorch_dataset(self.part_train_data,self.part_train_label,train=False,transform=transform_test,target_transform=target_transform)
			return train,test,eval_attack,validation,self.eval_attack_parition,self.in_train_eval_partition,self.out_train_eval_partition,self.starting_index,train_eval
		
		if ( self.dataset_name == 'mnist' or self.dataset_name =='fashion_mnist'):
			train = part_pytorch_dataset(self.part_train_data,self.part_train_label,train=True,transform=transforms.ToTensor(),target_transform=target_transform)
			test = part_pytorch_dataset(self.part_test_data,self.part_test_label,train=False,transform=transforms.ToTensor(),target_transform=target_transform)
			eval_attack = part_pytorch_dataset(self.part_eval_data,self.part_eval_label,train=False,transform=transforms.ToTensor(),target_transform=target_transform)
			validation = part_pytorch_dataset(self.part_validation_data,self.part_validation_label,train=False,transform=transforms.ToTensor(),target_transform=target_transform)
			train_eval = part_pytorch_dataset(self.part_train_data, self.part_train_label, train=False,
			                                  transform=transforms.ToTensor(), target_transform=target_transform)
			
			return train,test,eval_attack,validation,self.eval_attack_parition,self.in_train_eval_partition,self.out_train_eval_partition,self.starting_index,train_eval
		
		
		if (self.dataset_name == 'adult' or self.dataset_name == 'texas' or self.dataset_name == 'titanic' or self.dataset_name =='purchase'):
			train = part_pytorch_dataset(self.part_train_data,self.part_train_label,train=True,transform=None,target_transform=None)
			test = part_pytorch_dataset(self.part_test_data,self.part_test_label,train=False,transform=None,target_transform=None)
			eval_attack = part_pytorch_dataset(self.part_eval_data,self.part_eval_label,train=False,transform=None,target_transform=None)
			validation = part_pytorch_dataset(self.part_validation_data, self.part_validation_label,train=False, transform=None,target_transform=None)
			train_eval = part_pytorch_dataset(self.part_train_data, self.part_train_label, train=False,
			                                  transform=None, target_transform=None)
			
			return train,test,eval_attack,validation,self.eval_attack_parition,self.in_train_eval_partition,self.out_train_eval_partition,self.starting_index,train_eval
	
	
	def apply_ban_rules(self,ban_list):
		
		ban_index = np.where(ban_list == 1)[0]
		ban_index = np.array(ban_index)
		valid_index = np.setdiff1d(np.arange(len(self.eval_attack_parition)),ban_index)
		
		self.eval_attack_parition = self.eval_attack_parition[valid_index]
		
		print ("new length = %d" %(len(self.eval_attack_parition)))
	
	
	def old_select_part(self,data_number):
		self.data_number = data_number
		#print ("data number = %d " % data_number)
		self.train_partition = np.random.choice(len(self.label),data_number,replace=False)
		self.test_parition = np.setdiff1d(np.arange(len(self.label)),self.train_partition)
		self.part_train_data = self.data[self.train_partition]
		self.part_train_label = self.label[self.train_partition]
		self.part_test_data = self.data[self.test_parition]
		self.part_test_label = self.label[self.test_parition]
		
		# train + test
		
		#print (self.part_train_data.shape)
		#print (self.part_train_label.shape)
		#print (self.part_test_data.shape)
		#print (self.part_test_label.shape)
		
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		if (self.cutout==1):
			transform_train.transforms.append(Cutout(n_holes=self.n_holes, length=self.length))
			print ("CUTOUT ACTIVATED!")
		
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		
		if (self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100'):
			train = part_pytorch_dataset(self.part_train_data,self.part_train_label,train=True,transform=transform_train)
			test = part_pytorch_dataset(self.part_test_data,self.part_test_label,train=False,transform=transform_test)
			return train,test,self.train_partition,self.test_parition
		
		if (self.dataset_name == 'mnist'):
			train = part_pytorch_dataset(self.part_train_data,self.part_train_label,train=True,transform=transforms.ToTensor())
			test = part_pytorch_dataset(self.part_test_data,self.part_test_label,train=False,transform=transforms.ToTensor())
			return train,test,self.train_partition,self.test_parition
		
		if (self.dataset_name == 'adult' or self.dataset_name == 'texas' or self.dataset_name =='purchase' or self.dataset_name == 'titanic'):
			train = part_pytorch_dataset(self.part_train_data,self.part_train_label,train=True,transform=None)
			test = part_pytorch_dataset(self.part_test_data,self.part_test_label,train=False,transform=None)
			return train,test,self.train_partition,self.test_parition



            
            