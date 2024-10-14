import numpy as np
import torch
import torch.nn as nn
from utils import *
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from whitebox_attack import *
from user import *

def get_all_user_update_info(user_update_list):

    ### now we consider cos(b_i,b_j) for all user pairs for all layers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    all_cos_matrix = []

    for param_idx in range(len(user_update_list[0])):

        cos_matrix = np.zeros((len(user_update_list), len(user_update_list)))
        for user_idx1 in range(len(user_update_list)):
            for user_idx2 in range(len(user_update_list)):
                #print (user_update_list[user_idx1][param_idx].size())
                this_cos = cos(torch.flatten(user_update_list[user_idx1][param_idx]),torch.flatten(user_update_list[user_idx2][param_idx]))
                cos_matrix[user_idx1,user_idx2] = this_cos.item()

        all_cos_matrix.append(cos_matrix)

    all_cos_matrix = np.array(all_cos_matrix)

    #print (all_cos_matrix.shape)

    return all_cos_matrix

def get_background_grad(user,model,get_gradient_func=None,non_member_flag=False,half_flag=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (non_member_flag):
        background_loader = user.test_data_loader
    else:
        background_loader = user.train_eval_data_loader

    background_grad = []
    total_background_batch = 0
    for idx,(background_data,background_label,_) in enumerate(background_loader):
        this_background_batch_gradient =  get_gradient_func(model, background_data.to(device), background_label.to(device))
        ## this should be the avg of this batch
        #print (len(this_background_batch_gradient))
        if (idx == 0):
            for layer_idx in range(len(this_background_batch_gradient)):
                background_grad.append(torch.zeros_like(this_background_batch_gradient[layer_idx]))
                #print (layer_idx," ",this_background_batch_gradient[layer_idx].size())

        for layer_idx in range(len(this_background_batch_gradient)):

            ### half b process
            #if (idx%2 == 0):
            #    background_grad[layer_idx]+=this_background_batch_gradient[layer_idx]

            background_grad[layer_idx]+=this_background_batch_gradient[layer_idx]

        total_background_batch = idx
        #total_background_batch = int(idx/2) ### for half b exp

    for layer_idx in range(len(background_grad)):
        background_grad[layer_idx] = background_grad[layer_idx]/total_background_batch

    total_background_batch+=1

    print (f"total background batch num:{total_background_batch}")

    return background_grad

def background_sign_sanity_check(user_update_list):

    for i in range(len(user_update_list)):
        for j in range(len(user_update_list)):
            if (i == j):
                continue
            print (f"cur user idx {i} and {j}")

            for idx,(param1,param2) in enumerate(zip(user_update_list[i],user_update_list[j])):
                pos_sign = torch.sum(torch.maximum((torch.sign(torch.flatten(param1) - torch.flatten(param2))),
                                           torch.zeros_like(torch.flatten(param1))))
                neg_sign = torch.sum(torch.minimum((torch.sign(torch.flatten(param1) - torch.flatten(param2))),
                                           torch.zeros_like(torch.flatten(param1))))

                print (f" param idx {idx}, pos sign {pos_sign}, neg sign {neg_sign}")


def get_all_info_non_member_epoch(user_list,model,batch_size,get_gradient_func=None,user_update_list=None,thres=0.05):

    ### for this attack, user_update_list contains the update from each user after local training for 1(or more) epochs
    ### the update from each user serves as the gradient
    ### now we calculate the gradient of each instance to do grad_diff, cosine, counting and norm attack

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    #background_sign_sanity_check(user_update_list)

    all_info = []
    #all_grad_distribution = [] ### this should be in shape [# of instance, # of conv or fc layers, 100]
    #all_grad_class = [] ### [# of instance]
    #all_grad_membership_label = [] ### [# of instance]

    for target_idx in range(len(user_list)):
        ### use training data from next user as the member-background
        background_grad = user_update_list[(target_idx+1)%len(user_list)]

        ## for now, we use half of b to verify the process
        #background_grad = get_background_grad(user_list[(target_idx) % len(user_list)], model, get_gradient_func,
        #                                      non_member_flag=False, half_flag=True)

        background_grad_sparse = get_sparse_gradient(background_grad, thres=thres)

        this_epoch_info = []

        user_update_sparse = get_sparse_gradient(user_update_list[target_idx],thres=thres)

        one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset,batch_size=1,shuffle=False)

        instance_count = 0
        for image,label,_ in one_instance_loader:

            this_instance_grad = get_gradient_func(model,image,label)
            this_instance_grad_sparse = get_sparse_gradient(this_instance_grad,thres=thres)

            this_instance_info = get_all_info_per_instance(this_instance_grad,user_update_list[target_idx],background_grad,batch_size=batch_size)
            this_instance_info_sparse = get_all_info_per_instance(this_instance_grad_sparse,user_update_sparse,background_grad_sparse,batch_size=batch_size,sparse_flag=True)
            #print (np.array(this_instance_info).shape,np.array(this_instance_info_sparse).shape)
            this_instance_all_info = np.concatenate((np.array(this_instance_info),np.array(this_instance_info_sparse)),axis=1)
            #print (f"this instance all info shape {this_instance_all_info.shape}")
            this_epoch_info.append(this_instance_all_info)
            model.zero_grad()
            instance_count+=1

            ### grad distribution process
            #this_instance_grad_dis = []
            #for param in this_instance_grad:
            #    if (len(torch.flatten(param))>100):
            #        this_instance_grad_dis.append(torch.flatten(param.detach()).cpu().numpy()[:10])

            #=print (np.array(this_instance_grad_dis).shape)

            #all_grad_distribution.append(np.array(this_instance_grad_dis))
            #all_grad_class.append(label.detach().item())
            #all_grad_membership_label.append(1)

        print (f"total member instance {instance_count}")

        #one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset,batch_size=1,shuffle=False)
        ### what if the non-members are training data from 3rd user?
        one_instance_loader = torch.utils.data.DataLoader(user_list[(target_idx+2)%len(user_list)].evaluation_member_dataset,
                                                          batch_size=1, shuffle=False)

        instance_count = 0
        for image,label,_ in one_instance_loader:
            this_instance_grad = get_gradient_func(model,image,label)
            this_instance_grad_sparse = get_sparse_gradient(this_instance_grad,thres=thres)

            this_instance_info = get_all_info_per_instance(this_instance_grad,user_update_list[target_idx],background_grad,batch_size=batch_size)
            this_instance_info_sparse = get_all_info_per_instance(this_instance_grad_sparse,user_update_sparse,background_grad_sparse,batch_size=batch_size,sparse_flag=True)
            this_instance_all_info = np.concatenate((np.array(this_instance_info),np.array(this_instance_info_sparse)),axis=1)
            #print (f"this instance all info shape {this_instance_all_info.shape}")
            this_epoch_info.append(this_instance_all_info)
            model.zero_grad()
            instance_count+=1

            ### grad distribution process
            #this_instance_grad_dis = []
            #for param in this_instance_grad:
            #    if (len(torch.flatten(param))>100):
            #        this_instance_grad_dis.append(torch.flatten(param.detach()).cpu().numpy()[:10])
            #all_grad_distribution.append(np.array(this_instance_grad_dis))
            #all_grad_class.append(label.detach().item())
            #all_grad_membership_label.append(0)

        print (f"total non member instance {instance_count}")

        all_info.append(this_epoch_info)

    print (np.array(all_info).shape)
    #print (np.array(all_grad_distribution).shape)
    #print (np.array(all_grad_class).shape)
    #print (np.array(all_grad_membership_label).shape)

    return np.array(all_info)

    #return np.array(all_info),np.array(all_grad_distribution),np.array(all_grad_class),np.array(all_grad_membership_label)


def record_gradient_dis(user_list,model,batch_size,get_gradient_func=None,preset_count=10):

    ### record the gradient distribution for 10 instances


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    all_info = []
    for target_idx in range(len(user_list)):

        one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset,
                                                          batch_size=1,shuffle=False)

        instance_count = 0
        for image,label,_ in one_instance_loader:
            this_instance_grad = get_gradient_func(model,image,label)
            grad_list = [param.detach().cpu().numpy() for param in this_instance_grad]
            all_info.append(grad_list)
            model.zero_grad()
            instance_count+=1

            if (instance_count>preset_count):
                break

        print (f"total member instance {instance_count}")

        one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset,
                                                          batch_size=1, shuffle=False)

        instance_count = 0
        for image,label,_ in one_instance_loader:
            this_instance_grad = get_gradient_func(model,image,label)
            grad_list = [param.detach().cpu().numpy() for param in this_instance_grad]
            all_info.append(grad_list)
            model.zero_grad()
            instance_count+=1

            ### grad distribution process
            #this_instance_grad_dis = []
            #for param in this_instance_grad:
            #    if (len(torch.flatten(param))>100):
            #        this_instance_grad_dis.append(torch.flatten(param.detach()).cpu().numpy()[:10])
            #all_grad_distribution.append(np.array(this_instance_grad_dis))
            #all_grad_class.append(label.detach().item())
            #all_grad_membership_label.append(0)
            if (instance_count>preset_count):
                break

        print (f"total non member instance {instance_count}")


    print (np.array(all_info).shape)

    return np.array(all_info)



'''
def run_active_multi_party_attacks(user_list,target_model,epoch,user_update_list,user_model_list,ori_model_weight_dict):

    naming_str =  get_naming_mid_str()  + str(epoch + 1) + '_' + str(
        args.dataset) + '_' + str(args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(
        args.model_name) + '.npy'

    print (naming_str)

    all_info,all_label = multi_party_member_attack(user_list,target_model,batch_size=args.target_batch_size,user_update_list=user_update_list,get_gradient_func=get_gradient)
    np.save('./expdata/all_info_multi_party_member_attack_' + naming_str, all_info)
    np.save('./expdata/all_label_multi_party_member_attack_' + naming_str, all_label)

    loss_info,loss_label = multi_party_member_loss_attack(user_list,target_model,batch_size=args.target_batch_size,user_update_list=user_update_list,get_gradient_func=get_gradient,user_model_list=user_model_list)
    np.save('./expdata/loss_info_multi_party_member_attack_' + naming_str, loss_info)
    np.save('./expdata/loss_label_multi_party_member_attack_' + naming_str, loss_label)

    target_loss_info,target_loss_label = multi_party_member_loss_attack(user_list,target_model,batch_size=args.target_batch_size,user_update_list=user_update_list,get_gradient_func=get_gradient,user_model_list=[ori_model_weight_dict])
    np.save('./expdata/target_model_before_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
    np.save('./expdata/target_model_before_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)

    weights_after_avg = average_weights(user_model_list)
    target_loss_info,target_loss_label = multi_party_member_loss_attack(user_list,target_model,batch_size=args.target_batch_size,user_update_list=user_update_list,get_gradient_func=get_gradient,user_model_list=[weights_after_avg])
    np.save('./expdata/target_model_after_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
    np.save('./expdata/target_model_after_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)

    print ("server adversary - active attack finished")
    print (naming_str)

def run_multi_party_attacks(user_list,target_model,epoch,user_update_list,user_model_list,ori_model_weight_dict):

    naming_str = get_naming_mid_str() + str(epoch + 1) + '_' + str(
        args.dataset) + '_' + str(args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(
        args.model_name) +'.npy'

    print (naming_str)
    # notice that, for loss attack, there is no model for non-member update,
    # but for our attack, we can use validation set to compute a non-member update
    #  [maybe we could use the testing set itself], but this will make the attack easier

    all_info,all_label = multi_party_member_attack(user_list,target_model,batch_size=args.target_batch_size,user_update_list=user_update_list,get_gradient_func=get_gradient)
    np.save('./expdata/all_info_multi_party_member_attack_' + naming_str, all_info)
    np.save('./expdata/all_label_multi_party_member_attack_' + naming_str, all_label)

    loss_info,loss_label = multi_party_member_loss_attack(user_list,target_model,batch_size=args.target_batch_size,user_update_list=user_update_list,get_gradient_func=get_gradient,user_model_list=user_model_list)
    np.save('./expdata/loss_info_multi_party_member_attack_' + naming_str, loss_info)
    np.save('./expdata/loss_label_multi_party_member_attack_' + naming_str, loss_label)

    target_loss_info,target_loss_label = multi_party_member_loss_attack(user_list,target_model,batch_size=args.target_batch_size,user_update_list=user_update_list,get_gradient_func=get_gradient,user_model_list=[ori_model_weight_dict])
    np.save('./expdata/target_model_before_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
    np.save('./expdata/target_model_before_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)

    weights_after_avg = average_weights(user_model_list)
    target_loss_info,target_loss_label = multi_party_member_loss_attack(user_list,target_model,batch_size=args.target_batch_size,user_update_list=user_update_list,get_gradient_func=get_gradient,user_model_list=[weights_after_avg])
    np.save('./expdata/target_model_after_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
    np.save('./expdata/target_model_after_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)

    #instance_cos_result = instance_cos(user_list,copy.deepcopy(target_model),batch_size=args.target_batch_size,user_update_list=user_update_list,get_gradient_func=get_gradient,user_model_list=[weights_after_avg])
    #np.save('./expdata/instance_cos_multi_party_member_attack_' + naming_str, instance_cos_result)

    print ("server adversary - multi party attack finished")

def run_active_multi_party_attacks_client(user_list,target_model,epoch,user_update_list,user_model_list,ori_model_weight_dict):

    ### if the client is the adversary, we need to change the user_update_list to make it the sum of updates from all other parties
    ### here we assume that the last party is the adversary

    sum_user_update_list = []
    for param in user_update_list[0]:
        sum_user_update_list.append(torch.zeros_like(param))

    for user_idx in range(len(user_list)-1):
        for idx,param in enumerate(user_update_list[user_idx]):
            sum_user_update_list[idx] = sum_user_update_list[idx] + param

    for param in sum_user_update_list:
        param = param / (len(user_list) - 1)

    ### for the loss attack, the available model is the model updated with sum of updates from all other parties
    temp_sum_weights = average_weights(user_model_list[:-1])

    ###

    naming_str =  get_naming_mid_str()  + str(epoch + 1) + '_' + str(
        args.dataset) + '_' + str(args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(
        args.model_name) + '.npy'

    print (naming_str)

    all_info,all_label = multi_party_member_attack(user_list[:-1],target_model,batch_size=args.target_batch_size,user_update_list=[sum_user_update_list],get_gradient_func=get_gradient)
    np.save('./expdata/all_info_multi_party_member_attack_' + naming_str, all_info)
    np.save('./expdata/all_label_multi_party_member_attack_' + naming_str, all_label)

    loss_info,loss_label = multi_party_member_loss_attack(user_list[:-1],target_model,batch_size=args.target_batch_size,user_update_list=[sum_user_update_list],get_gradient_func=get_gradient,user_model_list=[temp_sum_weights])
    np.save('./expdata/loss_info_multi_party_member_attack_' + naming_str, loss_info)
    np.save('./expdata/loss_label_multi_party_member_attack_' + naming_str, loss_label)

    target_loss_info,target_loss_label = multi_party_member_loss_attack(user_list[:-1],target_model,batch_size=args.target_batch_size,user_update_list=[sum_user_update_list],get_gradient_func=get_gradient,user_model_list=[ori_model_weight_dict])
    np.save('./expdata/target_model_before_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
    np.save('./expdata/target_model_before_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)

    weights_after_avg = average_weights(user_model_list)
    target_loss_info,target_loss_label = multi_party_member_loss_attack(user_list[:-1],target_model,batch_size=args.target_batch_size,user_update_list=[sum_user_update_list],get_gradient_func=get_gradient,user_model_list=[weights_after_avg])
    np.save('./expdata/target_model_after_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
    np.save('./expdata/target_model_after_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)

    print ("client adversary - active attack finished")

def run_multi_party_attacks_client(user_list,target_model,epoch,user_update_list,user_model_list,ori_model_weight_dict):

    naming_str = get_naming_mid_str() + str(epoch + 1) + '_' + str(
        args.dataset) + '_' + str(args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(
        args.model_name) + '.npy'

    print (naming_str)
    ### if the client is the adversary, we need to change the user_update_list to make it the sum of updates from all other parties
    ### here we assume that the last party is the adversary

    sum_user_update_list = []
    for param in user_update_list[0]:
        sum_user_update_list.append(torch.zeros_like(param))

    for user_idx in range(len(user_list)-1):
        for idx,param in enumerate(user_update_list[user_idx]):
            sum_user_update_list[idx] = sum_user_update_list[idx] + param

    for param in sum_user_update_list:
        param = param / (len(user_list) - 1)

    ### for the loss attack, the available model is the model updated with sum of updates from all other parties
    temp_sum_weights = average_weights(user_model_list[:-1])

    ###
    all_info,all_label = multi_party_member_attack(user_list[:-1],target_model,batch_size=args.target_batch_size,user_update_list=[sum_user_update_list],get_gradient_func=get_gradient)
    np.save('./expdata/all_info_multi_party_member_attack_' + naming_str, all_info)
    np.save('./expdata/all_label_multi_party_member_attack_' + naming_str, all_label)

    loss_info,loss_label = multi_party_member_loss_attack(user_list[:-1],target_model,batch_size=args.target_batch_size,user_update_list=[sum_user_update_list],get_gradient_func=get_gradient,user_model_list=[temp_sum_weights])
    np.save('./expdata/loss_info_multi_party_member_attack_' + naming_str, loss_info)
    np.save('./expdata/loss_label_multi_party_member_attack_' + naming_str, loss_label)

    target_loss_info,target_loss_label = multi_party_member_loss_attack(user_list[:-1],target_model,batch_size=args.target_batch_size,user_update_list=[sum_user_update_list],get_gradient_func=get_gradient,user_model_list=[ori_model_weight_dict])
    np.save('./expdata/target_model_before_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
    np.save('./expdata/target_model_before_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)

    weights_after_avg = average_weights(user_model_list)
    target_loss_info,target_loss_label = multi_party_member_loss_attack(user_list[:-1],target_model,batch_size=args.target_batch_size,user_update_list=[sum_user_update_list],get_gradient_func=get_gradient,user_model_list=[weights_after_avg])
    np.save('./expdata/target_model_after_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
    np.save('./expdata/target_model_after_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)

    #instance_cos_result = instance_cos(user_list,copy.deepcopy(target_model),batch_size=args.target_batch_size,user_update_list=user_update_list,get_gradient_func=get_gradient,user_model_list=[weights_after_avg])
    #np.save('./expdata/instance_cos_multi_party_member_attack_' + naming_str, instance_cos_result)
    print ("client adversary - multi party attack finished")


#def run_batch_attacks(user_list,target_model,epoch):
    #naming_str = str(args.dpsgd) + '_' + str(args.noise_scale) + '_' + str(args.mmd) + '_' + str(
    #    args.mmd_loss_lambda) + '_' + str(args.mixup) + '_' + str(epoch + 1) + '_' + str(
    #    args.dataset) + '_' + str(args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(
    #    args.model_name) + '.npy'
    ### for the following two attacks, we need to use jupyter notebook to get the AUC score and so on, we can add this function later, like eval_attack(m_pred,m_true)

    ### GET info for cosine attack and grad-minus attack for single batch case
    ### for this attack, we mainly want to show the AUC score for each batch
    ### we can compare this with the diff MI attack, since that attack is also batch based

    #all_info = get_all_info_non_member_singlebatch(user_list, target_model, batch_size=args.target_batch_size,
    #                                               get_gradient_func=get_gradient)
    #np.save('./expdata/all_info_non_member_singlebatch_' + naming_str, all_info)
    #print ("single batch attack finished")

    ### GET info for cosine attack and grad-minus attack for multi batch case
    ### for this attack, we assume that the center has many batches (and their gradients), and the center wants to figure out
    ### whether one instance is used in any of all those batches or not.
    #all_info, last_layer_gradient = get_all_info_non_member_multibatch(user_list, target_model,
    #                                                                             batch_size=args.target_batch_size,
    #                                                                             get_gradient_func=get_gradient)
    #np.save('./expdata/all_info_non_member_multibatch_' + naming_str, all_info)
    #np.save('./expdata/all_info_non_member_lastlayergradient_' + naming_str, np.array(last_layer_gradient))
    #print ("multibatch attack finished")

    ### sparse vector counting attack
    #all_info = get_all_info_non_member_singlebatch_sparse_vector_counting(user_list, model = target_model,
    #                                                                      batch_size=args.target_batch_size,
    #                                                                      get_gradient_func=get_gradient,
    #                                                                      thres=0.05)
    #np.save('./expdata/all_info_non_member_singlebatch_sparsecounting_' + naming_str, all_info)

    #print("single batch sparse counting finished")

    ### diff MI attack
    ### for this attack, we have no AUC score to report, only accuracy or F1-score. (all_info should contain m_pred and m_true)

    #m_true,m_pred = diffmi_attack(user_list,target_model,batch_size=20)

    #np.save('./expdata/diffmi_attack'+naming_str+'_m_true_20.npy',m_true)
    #np.save('./expdata/diffmi_attack'+naming_str+'_m_pred_20.npy',m_pred)

    #print ("diffmi attack batch_size 20 finished")

    #m_true,m_pred = diffmi_attack(user_list,target_model,batch_size=2*args.target_batch_size)

    #np.save('./expdata/diffmi_attack'+naming_str+'_m_true_200.npy',m_true)
    #np.save('./expdata/diffmi_attack'+naming_str+'_m_pred_200.npy',m_pred)

    #print ("diffmi attack batch_size 200 finished")

    ### nasr fed attack
    ### for this attack, we can report AUC score since the output is a probability.

    #_,_ = nasr_fed_attack(user_list,target_model,args.dataset)

    #print ("nasr attack finished")

    #exit(0)

    #np.save('./expdata/nasr_fed_attack_'+naming_str,all_info)
    # data is already saved by modifying the ml_privacy library
     
'''