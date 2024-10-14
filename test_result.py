
## alexnet cifar100
import numpy as np
from model_utils import get_blackbox_auc_lira
def test_one_case(data_name,dis_name,label_name,eps1=1e-6,eps2=1e-20):
	
	data_name = './expdata/' + data_name
	data = np.load(data_name)
	all_prob = data['arr_0']
	all_training_partition = data['arr_1']
	all_validation_partition = data['arr_2']
	all_class_label = data['arr_3']
	all_loss = data['arr_4']
	all_label = data['arr_5']
	all_prob_dis,all_label,auc,plr = get_blackbox_auc_lira(all_prob,all_training_partition,all_validation_partition,all_class_label=all_class_label,eps1=eps1,eps2=eps2)
	dis_name = f'./expdata/{eps1}_{eps2}_' + dis_name
	np.save(dis_name,all_prob_dis)
	label_name = f'./expdata/{eps1}_{eps2}_' + label_name
	np.save(label_name,all_label)
	print (f"data name {data_name}, LIRA attack, auc {auc}, plr {plr}")
	
data_name_list = ['cifar100_alexnet_10_20_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100_all_info.npz']

dis_name_list = ['cifar100_alexnet_10_20_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100_metric_distribution.npy']

label_name_list = ['cifar100_alexnet_10_20_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100_metric_label.npy']

eps1 = 1e-10
eps2= 1e-25
for data_name,dis_name,label_name in zip(data_name_list,dis_name_list,label_name_list):
	test_one_case(data_name,dis_name,label_name,eps1=eps1,eps2=eps2)
	
	