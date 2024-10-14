import numpy as np
import torch
import torch.nn as nn
from utils import *
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from whitebox_attack import *
from user import *

def cos_l1(param1,param2):
    norm1 = torch.norm(torch.flatten(param1), p=1)
    norm2 = torch.norm(torch.flatten(param2), p=1)
    return torch.dot(torch.flatten(param1),torch.flatten(param2))/(norm1*norm2)


def get_all_info_non_member_singlebatch_sparse_vector_counting(user_list,model,batch_size,get_gradient_func=None,layer_index=45,thres=0.15):

    ## for this attack, we need a huge background set to calculate the avg
    ## we can also use zero as avg to test how the background can improve the performance
    ## for each batch, we calculate the gradient of that batch,
    ## for each instance, we calcualte the gradient of that instance,
    ## for every param, we consider the gradient wrt this param, x_grad[i] and batch_grad[i]
    ## the intuition is, if x_grad[i] is member, then sign(grad_x[i]) = sign(batch_grad[i] - background_grad[i])
    ## otherwise it should be non-member
    ## we can use the counting as the indicator

    print ("SPARSE VECTOR COUNTING")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_info = []
    ### we need to calculate the background_info, here we use the test set to do so

    background_loader = user_list[0].test_data_loader

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
            background_grad[layer_idx]+=this_background_batch_gradient[layer_idx]
        total_background_batch = idx

    for layer_idx in range(len(background_grad)):
        background_grad[layer_idx] = background_grad[layer_idx]/total_background_batch

    total_background_batch+=1

    print (f"total background batch num:{total_background_batch}")

    background_grad = get_sparse_gradient(background_grad,thres=thres)

    ### now we calculate batch_grad and x_grad
    for target_idx in range(len(user_list)):

        target_info = []

        train_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset, batch_size=batch_size,
                                                   shuffle=True)
        ### non-member set
        test_loader = torch.utils.data.DataLoader(user_list[target_idx].test_dataset, batch_size=batch_size,
                                                       shuffle=True)

        for (train_image, train_label,_), (test_image, test_label,_) in zip(
                train_loader, test_loader):

            this_batch_info = []

            this_train_batch_gradient = get_sparse_gradient(get_gradient_func(model,train_image,train_label),thres=thres)

            ### get the gradient for each instance in the training batch
            for i in range(batch_size):
                this_instance_info = []
                this_instance = torch.unsqueeze(train_image[i].to(device), dim=0)
                this_label = torch.unsqueeze(train_label[i].to(device), dim=0)
                this_instance_grad = get_sparse_gradient(get_gradient_func(model,this_instance,this_label),thres=thres)


                for param1, param2, param3 in zip(this_instance_grad, this_train_batch_gradient,background_grad):

                    ## both param1 and param2 are filtered vector, containing only top thres% values
                    background_sign = torch.sign(param2 - param3)
                    counting_sign = torch.sign(param1)*background_sign
                    this_count = torch.sum(counting_sign).cpu().numpy()
                    this_layer = [this_count]

                    this_instance_info.append(np.array(this_layer))

                this_batch_info.append(this_instance_info)
                model.zero_grad()


            for i in range(batch_size):
                this_instance_info = []
                this_instance = torch.unsqueeze(test_image[i].to(device), dim=0)
                this_label = torch.unsqueeze(test_label[i].to(device), dim=0)
                this_instance_grad = get_sparse_gradient(get_gradient_func(model,this_instance,this_label),thres=thres)

                for param1, param2,param3 in zip(this_instance_grad, this_train_batch_gradient,background_grad):
                    background_sign = torch.sign(param2 - param3)
                    counting_sign = torch.sign(param1) * background_sign
                    this_count = torch.sum(counting_sign).cpu().numpy()
                    this_layer = [this_count]

                    this_instance_info.append(np.array(this_layer))
                this_batch_info.append(this_instance_info)
                model.zero_grad()

            target_info.append(this_batch_info)
            ### for each batch, we have member's data + nonmember's data put together

        all_info.append(target_info)

    print (np.array(all_info).shape)

    return np.array(all_info)


def active_attacker_get_all_info_non_member_batch(attacker,user_list,model,attacker_set,attacker_selected_index=None,target_percentage=0.05,batch_size=100,get_gradient_func=None):

    ### we have target_num target instances blended into one batch
    ### we try to calculate the gradient of each instance and the gradient of this batch
    ### the goal is to tell if any target instance is included in this batch, and if yes, which one is included

    target_num = int(target_percentage*batch_size)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    #print (target_num)

    all_info = []
    all_label = []

    for target_idx in range(len(user_list)):

        target_info = []
        target_label = []
        batch_idx = 0

        target_loader = torch.utils.data.DataLoader(user_list[target_idx].target_train_dataset, batch_size=int(target_percentage * batch_size),
                                                    shuffle=False)
        one_instance_target_loader = torch.utils.data.DataLoader(user_list[target_idx].target_train_dataset, batch_size=1, shuffle=False)

        non_target_loader = torch.utils.data.DataLoader(user_list[target_idx].non_target_train_dataset,batch_size=batch_size-target_num,shuffle=True)

        #print (user_list[target_idx].target_train_dataset.__len__())


        for (non_target_train_images,non_target_train_labels),(target_train_images,target_train_labels) in (zip(non_target_loader,target_loader)):
            ### combine two set of images together
            combined_images = torch.cat((non_target_train_images,target_train_images),0)
            combined_labels = torch.cat((non_target_train_labels,target_train_labels),0)
            #print (combined_images.shape)

            this_train_batch_gradient = get_gradient_func(model,combined_images,combined_labels)
            
            ### for current batch, the ground truth is that only the target train images are included in this batch,
            ### any other target instance is not included.
            ### we should be able to tell this, for example the stats for included instances are the top target_num ones.

            ### create ground truth label
            ground_truth = np.zeros(user_list[target_idx].target_train_dataset.__len__())
            ground_truth[batch_idx*target_num:(batch_idx+1)*target_num]=1

            for this_instance_image,this_instance_label in one_instance_target_loader:
                this_info = []
                this_instance_grad = get_gradient_func(model,this_instance_image,this_instance_label)

                for param1, param2 in zip(this_instance_grad, this_train_batch_gradient):

                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))
                    l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                    l2dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)))
                    # |(n*B-a)/(n-1)| if member this should be smaller
                    l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                    l1dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)), p=1)

                    # |(n*B-a)/(n-1)|
                    this_layer = [cos1.item(),l1dist1.item(), l1dist2.item(),
                                          torch.norm(torch.flatten(param1), p=1).item(),
                                          l2dist1.item(), l2dist2.item(),
                                          torch.norm(torch.flatten(param1)).item()]

                    this_info.append(np.array(this_layer))

                target_info.append(np.array(this_info))
                model.zero_grad()

            target_label.append(np.array(ground_truth))

        all_info.append(np.array(target_info))
        target_label = np.array(target_label).flatten()
        all_label.append(np.array(target_label))
        batch_idx+=1

    print ("all info processed.")

    print (np.array(all_info).shape)
    print (np.array(all_label).shape)

    return np.array(all_info),np.array(all_label)


def active_attacker_get_all_info_non_member_batch_oakland(user_list,model,attacker_set,target_percentage=0.01,batch_size=100,get_gradient_func=None):

    target_num = int(target_percentage*batch_size)
    one_instance_loader = torch.utils.data.DataLoader(attacker_set,batch_size=int(target_percentage*batch_size),shuffle=False)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    all_info = []

    for target_idx in range(len(user_list)):

        target_info = []

        train_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset, batch_size=batch_size,
                                                   shuffle=False)
        test_eval_loader = torch.utils.data.DataLoader(user_list[target_idx].test_eval_dataset, batch_size=batch_size,
                                                       shuffle=False)

        #### here is the implementation of the oakland paper.. we do gradient ascent for the attacker set

        for (train_image, train_label) in train_loader:
            this_train_batch_gradient = get_gradient_func(model,train_image,train_label)

            for this_instance_image,this_instance_label in one_instance_loader:
                this_info = []
                this_instance_grad = get_gradient_func(model,this_instance_image,this_instance_label)

                for param1, param2 in zip(this_instance_grad, this_train_batch_gradient):

                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))
                    l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                    l2dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)))
                    # |(n*B-a)/(n-1)| if member this should be smaller
                    l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                    l1dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)), p=1)

                    # |(n*B-a)/(n-1)|
                    this_layer = [cos1.item(),l1dist1.item(), l1dist2.item(),
                                          torch.norm(torch.flatten(param1), p=1).item(),
                                          l2dist1.item(), l2dist2.item(),
                                          torch.norm(torch.flatten(param1)).item()]

                    this_info.append(np.array(this_layer))

                target_info.append(np.array(this_info))

                model.zero_grad()


        all_info.append(np.array(target_info))

    print ("all info processed.")

    print (np.array(all_info).shape)

    return np.array(all_info)

def get_all_info_member_multibatch(attacker,user_list, model,batch_size,get_gradient_func=None,attacker_evaluation_data_index=None):

    ### for this case, the members are the training samples of this user and non-members are the training samples from other users
    ### here we use the evaluation set from the attacker
    ### \frac{1}{# of users} of the evaluation set will be members and the remaining part will be non-member
    ### here we use ROC-AUC / PR-AUC and balanced accuracy to evaluate the attack methods
    print ("ALL INFO MEMBER BATCH:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    all_info = []
    all_label = []

    for target_idx in range(len(user_list)):

        target_info = []
        target_label = np.zeros(attacker.evaluation_dataset.__len__())
        length_per_user = int(len(target_label)/len(user_list))
        target_label[length_per_user*target_idx:length_per_user*(target_idx+1)] = 1

        train_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset, batch_size=batch_size,
                                                   shuffle=True)

        evaluation_loader = torch.utils.data.DataLoader(attacker.evaluation_dataset,batch_size=1,shuffle=False)
        ### the order of the evaluation set is determined by the user index
        ### for the first user, the true label is 1 for index 0-\frac{1}{# of users}, 0 for the rest

        for (train_image, train_label,_) in train_loader:

            this_train_batch_gradient = get_gradient_func(model,train_image.to(device),train_label.to(device))

            this_batch_info = []

            for (this_image,this_label,_) in evaluation_loader:

                this_instance_info = []

                this_instance_grad = get_gradient_func(model,this_image.to(device),this_label.to(device))

                for param1, param2 in zip(this_instance_grad, this_train_batch_gradient):

                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))

                    l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                    l2dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)))
                    # |(n*B-a)/(n-1)| if member this should be smaller
                    l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                    l1dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)), p=1)

                    # |(n*B-a)/(n-1)|
                    this_layer = [cos1.item(),l1dist1.item(), l1dist2.item(),
                                          torch.norm(torch.flatten(param1), p=1).item(),
                                          l2dist1.item(), l2dist2.item(),
                                          torch.norm(torch.flatten(param1)).item()]

                    this_instance_info.append(np.array(this_layer))

                this_batch_info.append(this_instance_info)

                model.zero_grad()

            target_info.append(this_batch_info)

        all_info.append(target_info)
        all_label.append(target_label)

    print ("all info processed.")
    print (np.array(all_info).shape)
    print (np.array(all_label).shape)

    return np.array(all_info),np.array(all_label)


def get_all_info_non_member_multibatch_single_layer(user_list, model,batch_size,get_gradient_func=None,layer_index=45):

    ### for this case, the members are the training samples of this user and non-members are testing samples
    ### in this case, the evaluation set is formed by 2 parts:
    ### 1. randomly sampled ones from the training set of the target user [the evaluation set of this user]
    ### 2. randomly samples ones from the testing set [randomly sampled batches from testing set]
    ### we can have balanced evaluation set here

    ### for resnet20, layer index should be 45

    ### here we use ROC-AUC / PR-AUC and balanced accuracy to evaluate the attack methods
    print ("ALL INFO NON MEMBER BATCH + AUX:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    all_info = []
    all_label = []
    all_aux_info = []

    ### we only use a specific layer to make the calculation faster
    for target_idx in range(len(user_list)):
        target_info = []
        target_label = []
        target_aux_info = []

        ### calculate and store batch gradient of that layer

        train_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset, batch_size=batch_size,
                                                   shuffle=True)
        member_evaluation_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset, batch_size=1, shuffle=False)
        non_member_evaluation_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset,batch_size=1,shuffle=False)

        train_set_gradient = []
        for (train_image, train_label, train_index) in train_loader:
            this_train_batch_gradient = get_gradient_func(model,train_image.to(device),train_label.to(device))
            train_set_gradient.append(this_train_batch_gradient[layer_index])

        for (this_image,this_label,_) in member_evaluation_loader:
            this_instance_gradient = get_gradient_func(model,this_image.to(device),this_label.to(device))
            this_layer_gradient = this_instance_gradient[layer_index]
            this_instance_info = []
            for batch_layer_gradient in train_set_gradient:
                param1 = this_layer_gradient
                param2 = batch_layer_gradient

                cos1 = cos(torch.flatten(param1), torch.flatten(param2))

                l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                l2dist2 = torch.norm(
                    (torch.flatten(param2) * batch_size - torch.flatten(
                        param1)))
                # |(n*B-a)/(n-1)| if member this should be smaller
                l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                l1dist2 = torch.norm(
                    (torch.flatten(param2) * batch_size - torch.flatten(
                        param1)), p=1)

                # |(n*B-a)/(n-1)|
                this_layer = [cos1.item(), l1dist1.item(), l1dist2.item(),
                              torch.norm(torch.flatten(param1), p=1).item(),
                              l2dist1.item(), l2dist2.item(),
                              torch.norm(torch.flatten(param1)).item()]

                this_instance_info.append(np.array(this_layer))

            target_info.append(this_instance_info)
            target_label.append(1)


        for (this_image,this_label,_) in non_member_evaluation_loader:
            this_instance_gradient = get_gradient_func(model,this_image.to(device),this_label.to(device))
            this_layer_gradient = this_instance_gradient[layer_index]
            this_instance_info = []
            for batch_layer_gradient in train_set_gradient:
                param1 = this_layer_gradient
                param2 = batch_layer_gradient

                cos1 = cos(torch.flatten(param1), torch.flatten(param2))

                l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                l2dist2 = torch.norm(
                    (torch.flatten(param2) * batch_size - torch.flatten(
                        param1)))
                # |(n*B-a)/(n-1)| if member this should be smaller
                l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                l1dist2 = torch.norm(
                    (torch.flatten(param2) * batch_size - torch.flatten(
                        param1)), p=1)

                # |(n*B-a)/(n-1)|
                this_layer = [cos1.item(), l1dist1.item(), l1dist2.item(),
                              torch.norm(torch.flatten(param1), p=1).item(),
                              l2dist1.item(), l2dist2.item(),
                              torch.norm(torch.flatten(param1)).item()]

                this_instance_info.append(np.array(this_layer))

            target_info.append(this_instance_info)
            target_label.append(0)

        all_info.append(target_info)
        all_label.append(target_label)

    all_info = np.array(all_info)
    print (all_info.shape)
    all_label = np.array(all_label)
    print (all_label.shape)

    return np.array(all_info), np.array(all_label)


def get_all_info_non_member_multibatch_sparse_vector(attacker,user_list, model,batch_size,get_gradient_func=None,attacker_evaluation_data_index=None,layer_index=45):

    ### for this case, the members are the training samples of this user and non-members are testing samples
    ### in this case, the evaluation set is formed by 2 parts:
    ### 1. randomly sampled ones from the training set of the target user [the evaluation set of this user]
    ### 2. randomly samples ones from the testing set [randomly sampled batches from testing set]
    ### we can have balanced evaluation set here

    ### here we use ROC-AUC / PR-AUC and balanced accuracy to evaluate the attack methods
    print ("ALL INFO NON MEMBER BATCH + AUX:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    all_info = []
    all_label = []
    all_aux_info = []

    ### we only use a specific layer to make the calculation faster
    for target_idx in range(len(user_list)):
        target_info = []
        target_label = []
        target_aux_info = []

        ### calculate and store batch gradient of that layer

        train_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset, batch_size=batch_size,
                                                   shuffle=True)
        member_evaluation_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset,
                                                               batch_size=1, shuffle=False)
        non_member_evaluation_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset,
                                                                   batch_size=1, shuffle=False)
        aux_loader = torch.utils.data.DataLoader(attacker.aux_dataset, batch_size=1, shuffle=False)

        train_set_gradient = []
        for (train_image, train_label, train_index) in train_loader:
            this_train_batch_gradient = get_sparse_gradient(get_gradient_func(model, train_image.to(device), train_label.to(device)))
            train_set_gradient.append(this_train_batch_gradient)

        for (this_image, this_label, _) in member_evaluation_loader:
            this_instance_gradient = get_sparse_gradient(get_gradient_func(model, this_image.to(device), this_label.to(device)))
            this_instance_info = []
            for this_train_set_gradient in train_set_gradient:
                this_batch_info = []
                for param1,param2 in zip(this_instance_gradient,this_train_set_gradient):
                    param1 = param1.to_dense()
                    param2 = param2.to_dense()
                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))
                    l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                    l2dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)))
                # |(n*B-a)/(n-1)| if member this should be smaller
                    l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                    l1dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)), p=1)
                # |(n*B-a)/(n-1)|
                    this_layer = [cos1.item(), l1dist1.item(), l1dist2.item(),
                                torch.norm(torch.flatten(param1), p=1).item(),
                                l2dist1.item(), l2dist2.item(),
                                torch.norm(torch.flatten(param1)).item()]

                    this_batch_info.append(np.array(this_layer))

                this_instance_info.append(this_batch_info)

            target_info.append(this_instance_info)
            target_label.append(1)

        for (this_image, this_label, _) in non_member_evaluation_loader:
            this_instance_gradient = get_sparse_gradient(get_gradient_func(model, this_image.to(device), this_label.to(device)))
            this_instance_info = []
            for this_train_set_gradient in train_set_gradient:
                this_batch_info = []
                for param1, param2 in zip(this_instance_gradient, this_train_set_gradient):
                    param1 = param1.to_dense()
                    param2 = param2.to_dense()
                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))
                    l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                    l2dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)))
                    # |(n*B-a)/(n-1)| if member this should be smaller
                    l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                    l1dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)), p=1)
                    # |(n*B-a)/(n-1)|
                    this_layer = [cos1.item(), l1dist1.item(), l1dist2.item(),
                                  torch.norm(torch.flatten(param1), p=1).item(),
                                  l2dist1.item(), l2dist2.item(),
                                  torch.norm(torch.flatten(param1)).item()]

                    this_batch_info.append(np.array(this_layer))

                this_instance_info.append(this_batch_info)

            target_info.append(this_instance_info)
            target_label.append(0)

        for (this_image, this_label, _) in aux_loader:
            this_instance_gradient = get_sparse_gradient(get_gradient_func(model, this_image.to(device), this_label.to(device)))
            this_instance_info = []
            for this_train_set_gradient in train_set_gradient:
                this_batch_info = []
                for param1, param2 in zip(this_instance_gradient, this_train_set_gradient):
                    param1 = param1.to_dense()
                    param2 = param2.to_dense()
                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))
                    l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                    l2dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)))
                    # |(n*B-a)/(n-1)| if member this should be smaller
                    l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                    l1dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)), p=1)
                    # |(n*B-a)/(n-1)|
                    this_layer = [cos1.item(), l1dist1.item(), l1dist2.item(),
                                  torch.norm(torch.flatten(param1), p=1).item(),
                                  l2dist1.item(), l2dist2.item(),
                                  torch.norm(torch.flatten(param1)).item()]

                    this_batch_info.append(np.array(this_layer))

                this_instance_info.append(this_batch_info)

            target_aux_info.append(this_instance_info)

        all_info.append(target_info)
        all_label.append(target_label)
        all_aux_info.append(target_aux_info)

    all_info = np.array(all_info)
    print (all_info.shape)
    all_label = np.array(all_label)
    print (all_label.shape)
    all_aux_info = np.array(all_aux_info)
    print (all_aux_info.shape)

    return np.array(all_info), np.array(all_label), np.array(all_aux_info)

def cosine_similarity_check(attacker,user_list, model,batch_size,get_gradient_func=None,attacker_evaluation_data_index=None,layer_index=45):
    print ("cosine_similarity_check")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    #cos = cos_l1

    member_cosine_info = []
    member_label = []
    non_member_cosine_info = []
    non_member_label = []
    cross_cosine_info = []
    cross_label = []

    ### we only use a specific layer to make the calculation faster
    for target_idx in range(len(user_list)):

        ### calculate and store batch gradient of that layer

        member_evaluation_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset,
                                                               batch_size=1, shuffle=False)
        non_member_evaluation_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset,
                                                                   batch_size=1, shuffle=False)


        for (first_image,first_label,_) in member_evaluation_loader:
            member_label.append(first_label.item())
            this_instance_info = []
            for (second_image,second_label,_) in member_evaluation_loader:
                first_instance_gradient = get_gradient_func(model, first_image.to(device), first_label.to(device))
                second_instance_gradient = get_gradient_func(model, second_image.to(device), second_label.to(device))

                for param1,param2 in zip(first_instance_gradient,second_instance_gradient):
                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))
                    this_instance_info.append(cos1.item())

            member_cosine_info.append(this_instance_info)


        for (first_image,first_label,_) in non_member_evaluation_loader:
            non_member_label.append(first_label.item())
            this_instance_info = []
            for (second_image,second_label,_) in non_member_evaluation_loader:
                first_instance_gradient = get_gradient_func(model, first_image.to(device), first_label.to(device))
                second_instance_gradient = get_gradient_func(model, second_image.to(device), second_label.to(device))

                for param1,param2 in zip(first_instance_gradient,second_instance_gradient):
                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))
                    this_instance_info.append(cos1.item())

            non_member_cosine_info.append(this_instance_info)


        for (first_image,first_label,_) in member_evaluation_loader:
            this_instance_info = []
            for (second_image,second_label,_) in non_member_evaluation_loader:
                cross_label.append([first_label.item(),second_label.item()])
                first_instance_gradient = get_gradient_func(model, first_image.to(device), first_label.to(device))
                second_instance_gradient = get_gradient_func(model, second_image.to(device), second_label.to(device))

                for param1,param2 in zip(first_instance_gradient,second_instance_gradient):
                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))
                    this_instance_info.append(cos1.item())

            cross_cosine_info.append(this_instance_info)

    return np.array(member_cosine_info),np.array(member_label),\
           np.array(non_member_cosine_info),np.array(non_member_label),np.array(cross_cosine_info),np.array(cross_label)


def get_all_info_non_member_singlebatch(user_list, model,batch_size,get_gradient_func=None):

    print ("get all info single batch exp")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from scipy.stats import entropy

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    all_info = []

    for target_idx in range(len(user_list)):

        target_info = []

        next_user = (target_idx + 1) % (len(user_list))
        train_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset, batch_size=batch_size,
                                                   shuffle=True)
        ### training background set
        test_loader = torch.utils.data.DataLoader(user_list[target_idx].test_dataset, batch_size=batch_size,
                                                       shuffle=True)

        for (train_image, train_label,_), (test_image, test_label,_) in zip(
                train_loader, test_loader):

            this_batch_info = []

            this_train_batch_gradient = get_gradient_func(model,train_image,train_label)

            ### get the gradient for each instance in the training batch
            for i in range(batch_size):
                this_instance_info = []
                this_instance = torch.unsqueeze(train_image[i].to(device), dim=0)
                this_label = torch.unsqueeze(train_label[i].to(device), dim=0)
                this_instance_grad = get_gradient_func(model,this_instance,this_label)

                for param1, param2 in zip(this_instance_grad, this_train_batch_gradient):
                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))
                    l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                    l2dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)))
                    # |(n*B-a)/(n-1)| if member this should be smaller
                    l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                    l1dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)), p=1)
                    # |(n*B-a)/(n-1)|
                    this_layer = [cos1.item(), l1dist1.item(), l1dist2.item(),
                                  torch.norm(torch.flatten(param1), p=1).item(),
                                  l2dist1.item(), l2dist2.item(),
                                  torch.norm(torch.flatten(param1)).item()]

                    this_instance_info.append(np.array(this_layer))

                this_batch_info.append(this_instance_info)
                model.zero_grad()


            for i in range(batch_size):
                this_instance_info = []
                this_instance = torch.unsqueeze(test_image[i].to(device), dim=0)
                this_label = torch.unsqueeze(test_label[i].to(device), dim=0)
                this_instance_grad = get_gradient_func(model,this_instance,this_label)

                for param1, param2 in zip(this_instance_grad, this_train_batch_gradient):
                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))
                    l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                    l2dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)))
                    # |(n*B-a)/(n-1)| if member this should be smaller
                    l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                    l1dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)), p=1)
                    # |(n*B-a)/(n-1)|
                    this_layer = [cos1.item(), l1dist1.item(), l1dist2.item(),
                                  torch.norm(torch.flatten(param1), p=1).item(),
                                  l2dist1.item(), l2dist2.item(),
                                  torch.norm(torch.flatten(param1)).item()]

                    this_instance_info.append(np.array(this_layer))

                this_batch_info.append(this_instance_info)
                model.zero_grad()

            target_info.append(this_batch_info)

        all_info.append(target_info)

    print (np.array(all_info).shape)

    return np.array(all_info)


def get_all_info_non_member_multibatch(user_list, model,batch_size,get_gradient_func=None):

    ### for this case, the members are the training samples of this user and non-members are testing samples
    ### in this case, the evaluation set is formed by 2 parts:
    ### 1. randomly sampled ones from the training set of the target user [the evaluation set of this user]
    ### 2. randomly samples ones from the testing set [randomly sampled batches from testing set]
    ### we can have balanced evaluation set here

    ### here we use ROC-AUC / PR-AUC and balanced accuracy to evaluate the attack methods
    print ("ALL INFO NON MEMBER BATCH:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    #cos = cos_l1

    all_info = []
    all_label = []
    all_last_layer_gradient = []
    #batch_norm_info = []

    for target_idx in range(len(user_list)):

        target_info = []
        target_label = []
        target_aux_info = []
        this_last_layer_gradient = []

        train_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset, batch_size=batch_size,
                                                   shuffle=True)
        member_evaluation_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_member_dataset, batch_size=1, shuffle=False)

        non_member_evaluation_loader = torch.utils.data.DataLoader(user_list[target_idx].evaluation_non_member_dataset,batch_size=1,shuffle=False)

        for (train_image, train_label, train_index) in train_loader:

            this_train_batch_gradient = get_gradient_func(model,train_image.to(device),train_label.to(device))

            this_batch_info = []

            for index,(this_image, this_label,_) in enumerate(member_evaluation_loader):

                this_instance_info = []

                this_instance_grad = get_gradient_func(model,this_image.to(device),this_label.to(device))

                for param1, param2 in zip(this_instance_grad, this_train_batch_gradient):
                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))

                    l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                    l2dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)))
                    # |(n*B-a)/(n-1)| if member this should be smaller
                    l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                    l1dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)), p=1)
                    # |(n*B-a)/(n-1)|

                    this_layer = [cos1.item(), l1dist1.item(), l1dist2.item(),
                                  torch.norm(torch.flatten(param1), p=1).item(),
                                  l2dist1.item(), l2dist2.item(),
                                  torch.norm(torch.flatten(param1)).item()]

                    special_num = torch.dot(torch.flatten(param1),torch.flatten(param1))/(batch_size*torch.norm(torch.flatten(param1),p=2))
                    this_layer.append(special_num.item())

                    this_instance_info.append(np.array(this_layer))

                this_batch_info.append(this_instance_info)

                model.zero_grad()

            for index,(this_image, this_label,_) in enumerate(non_member_evaluation_loader):

                #if (index == 0):
                #    print ("loader unshuffle check:")
                #    print (this_label)

                this_instance_info = []

                this_instance_grad = get_gradient_func(model,this_image.to(device),this_label.to(device))

                for param1, param2 in zip(this_instance_grad, this_train_batch_gradient):
                    cos1 = cos(torch.flatten(param1), torch.flatten(param2))

                    l2dist1 = torch.norm(torch.flatten(param2) * batch_size)  # |B|
                    l2dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)))
                    # |(n*B-a)/(n-1)| if member this should be smaller
                    l1dist1 = torch.norm(torch.flatten(param2) * batch_size, p=1)  # |B|
                    l1dist2 = torch.norm(
                        (torch.flatten(param2) * batch_size - torch.flatten(
                            param1)), p=1)

                    # |(n*B-a)/(n-1)|
                    this_layer = [cos1.item(), l1dist1.item(), l1dist2.item(),
                                  torch.norm(torch.flatten(param1), p=1).item(),
                                  l2dist1.item(), l2dist2.item(),
                                  torch.norm(torch.flatten(param1)).item()]

                    special_num = torch.dot(torch.flatten(param1),torch.flatten(param1))/(batch_size*torch.norm(torch.flatten(param1),p=2))
                    this_layer.append(special_num.item())

                    this_instance_info.append(np.array(this_layer))

                this_batch_info.append(this_instance_info)

                model.zero_grad()

            target_info.append(this_batch_info)
            #target_label.append(this_batch_label)

        all_info.append(target_info)
        all_label.append(target_label)
        ### get the last layer gradient

        for (this_image, this_label, _) in member_evaluation_loader:
            this_instance_grad = get_gradient_func(model, this_image.to(device), this_label.to(device))
            this_last_layer_gradient.append(this_instance_grad[-2].cpu().numpy()) ###  -1 is the gradient wrt bias. -2 is the gradient wrt weight of fc layer
            #print (this_last_layer_gradient[-1].shape)

        for (this_image, this_label, _) in non_member_evaluation_loader:
            this_instance_grad = get_gradient_func(model, this_image.to(device), this_label.to(device))
            this_last_layer_gradient.append(this_instance_grad[-2].cpu().numpy()) ###  -1 is the gradient wrt bias. -2 is the gradient wrt weight of fc layer
            #print (this_last_layer_gradient[-1].shape)

        all_last_layer_gradient.append(this_last_layer_gradient)

    print ("all info processed.")
    print (np.array(all_info).shape)
    #print (np.array(all_label).shape)
    print (np.array(all_last_layer_gradient).shape)

    return np.array(all_info),np.array(all_last_layer_gradient)



def fed_gradient_kl_attack_per_batch_non_member(user_list, model,batch_size,total_size,get_gradient_func=None):

    print ("fed grad cosine attack:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from scipy.stats import entropy
    all_cosine = []

    for target_idx in range(len(user_list)):

        cosine_list = []

        next_user = (target_idx + 1) % (len(user_list))

        train_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset, batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(user_list[next_user].train_eval_dataset, batch_size=batch_size,
                                                  shuffle=True)
        ### training background set
        test_eval_loader = torch.utils.data.DataLoader(user_list[target_idx].test_eval_dataset, batch_size=batch_size,
                                                       shuffle=True)
        #test_eval_loader = torch.utils.data.DataLoader(user_list[target_idx].test_dataset, batch_size=batch_size,
        #                                               shuffle=True)
        ### should I use test or test eval here? idk... let's try both.. using test set will make the attack easier.


        all_count = 0
        for (train_image, train_label), (test_image, test_label), (test_eval_image, test_eval_label) in zip(
                train_loader, test_loader, test_eval_loader):
            ### get the gradient for one batch of training and one batch of testing
            #
            # train_image = train_image.to(device)
            # train_label = train_label.to(device)
            # output = model(train_image)
            # criterion = nn.CrossEntropyLoss().to(device)
            # loss = criterion(output, train_label)
            # loss.backward()
            #
            # this_train_batch_gradient = copy.deepcopy([param.grad for param in model.parameters()])
            #
            # model.zero_grad()
            #
            # test_image = test_image.to(device)
            # test_label = test_label.to(device)
            # output = model(test_image)
            # criterion = nn.CrossEntropyLoss().to(device)
            # loss = criterion(output, test_label)
            # loss.backward()
            #
            # this_test_batch_gradient = copy.deepcopy([param.grad for param in model.parameters()])
            #
            # model.zero_grad()

            this_train_batch_gradient = get_gradient_func(model,train_image,train_label)
            this_test_batch_gradient = get_gradient_func(model,test_image,test_label)

            ### get the gradient for each instance in the training batch
            for i in range(batch_size):

                this_cos1 = []
                this_cos2 = []

                this_instance = torch.unsqueeze(train_image[i].to(device), dim=0)
                this_label = torch.unsqueeze(train_label[i].to(device), dim=0)
                # output = model(this_instance)
                # criterion = nn.CrossEntropyLoss().to(device)
                # loss = criterion(output, this_label)
                # loss.backward()
                #this_instance_grad = [param.grad for param in model.parameters()]
                this_instance_grad = get_gradient_func(model,this_instance,this_label)


                for param1, param2, param3 in zip(this_instance_grad, this_train_batch_gradient,
                                                  this_test_batch_gradient):
                    cos1 = entropy(torch.flatten(param1).cpu().numpy(), torch.flatten(param2).cpu().numpy())
                    cos2 = entropy(torch.flatten(param1).cpu().numpy(), torch.flatten(param3).cpu().numpy())

                    this_cos1.append(cos1.item())
                    this_cos2.append(cos2.item())

                #cosine_list.append(np.array(this_cos1)-np.array(this_cos2))
                cosine_list.append(np.array(this_cos1))

                model.zero_grad()

            all_count += batch_size

            for i in range(batch_size):

                this_cos1 = []
                this_cos2 = []

                this_instance = torch.unsqueeze(test_eval_image[i].to(device), dim=0)
                this_label = torch.unsqueeze(test_eval_label[i].to(device), dim=0)
                # output = model(this_instance)
                # criterion = nn.CrossEntropyLoss().to(device)
                # loss = criterion(output, this_label)
                # loss.backward()
                # this_instance_grad = [param.grad for param in model.parameters()]
                this_instance_grad = get_gradient_func(model,this_instance,this_label)

                for param1, param2, param3 in zip(this_instance_grad, this_train_batch_gradient,
                                                  this_test_batch_gradient):
                    cos1 = entropy(torch.flatten(param1).cpu().numpy(), torch.flatten(param2).cpu().numpy())
                    cos2 = entropy(torch.flatten(param1).cpu().numpy(), torch.flatten(param3).cpu().numpy())

                    this_cos1.append(cos1.item())
                    this_cos2.append(cos2.item())

                #cosine_list.append(np.array(this_cos1)-np.array(this_cos2))
                cosine_list.append(np.array(this_cos1))

                model.zero_grad()

        cosine_list = np.array(cosine_list)
        #print (cosine_list.shape)

        num_layers = cosine_list.shape[1]

        for layer_idx in range(num_layers):
            member_data = []
            non_member_data = []
            y_true = []
            this_layer_data = cosine_list[:, layer_idx]
            y_pred = this_layer_data
            for batch_idx in range(int(total_size / batch_size)):
                member_data.append(
                    this_layer_data[2 * batch_idx * batch_size:2 * batch_idx * batch_size + batch_size])
                non_member_data.append(this_layer_data[
                                       2 * batch_idx * batch_size + batch_size:2 * batch_idx * batch_size + batch_size * 2])
                y_true.append(np.concatenate((np.ones(batch_size), np.zeros(batch_size))))

            from sklearn.metrics import roc_auc_score
            y_true = np.array(y_true).flatten()

            if (layer_idx == num_layers-1):
                print ("AUC score", roc_auc_score(y_true, y_pred))
                print (y_true.shape)
                print (y_pred.shape)

        all_cosine.append(cosine_list)
    print ("fed grad cosine attack end")

    return np.array(all_cosine)




def fed_gradient_in_out_attack_per_batch_non_member(user_list, model, batch_size=100, total_size=500,get_gradient_func=None):
    # batch_size = 100
    # total_size = 500
    print ("fed grad in out attack:")

    all_l1 = []
    all_l2 = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for target_idx in range(len(user_list)):

        next_user_1 = (target_idx + 1) % (len(user_list))
        next_user_2 = (target_idx + 2) % (len(user_list))

        train_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset, batch_size=batch_size,
                                                   shuffle=True)
        next_user_1_loader = torch.utils.data.DataLoader(user_list[next_user_1].train_eval_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True)
        next_user_2_loader = torch.utils.data.DataLoader(user_list[target_idx].test_eval_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True)

        l1diff = []
        l2diff = []

        for (target_image, target_label), (next_user_1_image, next_user_1_label), (
                next_user_2_image, next_user_2_label) in zip(train_loader, next_user_1_loader, next_user_2_loader):
            # ### get the gradient for the training batch
            # target_image = target_image.to(device)
            # target_label = target_label.to(device)
            # output = model(target_image)
            # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
            # loss = criterion(output, target_label)
            # loss.backward()
            #
            # train_batch_grad = copy.deepcopy([param.grad for param in model.parameters()])
            #
            # model.zero_grad()
            #
            # ### get the gradient for the next_user_1 batch
            #
            # next_user_1_image = next_user_1_image.to(device)
            # next_user_1_label = next_user_1_label.to(device)
            # output = model(next_user_1_image)
            # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
            # loss = criterion(output, next_user_1_label)
            # loss.backward()
            #
            # next_user_1_batch_grad = copy.deepcopy([param.grad for param in model.parameters()])
            #
            # model.zero_grad()

            train_batch_grad = get_gradient_func(model,target_image,target_label)
            next_user_1_batch_grad = get_gradient_func(model,next_user_1_image,next_user_1_label)

            ### for every instance in train_batch

            for i in range(batch_size):
                this_target_image = torch.unsqueeze(target_image[i], dim=0).to(device)
                this_target_label = torch.unsqueeze(target_label[i], dim=0).to(device)
                # output = model(this_target_image)
                # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
                # loss = criterion(output, this_target_label)
                # loss.backward()
                #
                # this_instance_grad = copy.deepcopy([param.grad for param in model.parameters()])
                this_instance_grad = get_gradient_func(model,this_target_image,this_target_label)

                this_l1_dist = []
                this_l2_dist = []

                for param1, param2, param3 in zip(this_instance_grad, train_batch_grad,
                                                  next_user_1_batch_grad):
                    l2dist1 = torch.norm(torch.flatten(param2) - torch.flatten(param3))  # |B-C|
                    l2dist2 = torch.norm(
                        torch.flatten(param2) - torch.flatten(param3) * (batch_size) / (batch_size+1) - torch.flatten(
                            param1) / (batch_size+1))
                    # |B- C(n-1)/n + a/n|

                    l1dist1 = torch.norm(torch.flatten(param2) - torch.flatten(param3), p=1)  # |B-C|
                    l1dist2 = torch.norm(
                        torch.flatten(param2) - torch.flatten(param3) * (batch_size) / (batch_size+1) - torch.flatten(
                            param1) / (batch_size+1), p=1)
                    # |B- C(n-1)/n + a/n|

                    this_l1_dist.append(l1dist1.item() - l1dist2.item()) ### smaller means member
                    this_l2_dist.append(l2dist1.item() - l2dist2.item())


                l1diff.append(np.array(this_l1_dist))
                l2diff.append(np.array(this_l2_dist))

                model.zero_grad()

            ### for a batch of non-members

            # sprint ('end')

            for i in range(batch_size):
                this_next_user_2_image = torch.unsqueeze(next_user_2_image[i], dim=0).to(device)
                this_next_user_2_label = torch.unsqueeze(next_user_2_label[i], dim=0).to(device)
                # output = model(this_next_user_2_image)
                # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
                # loss = criterion(output, this_next_user_2_label)
                # loss.backward()
                #
                # this_instance_grad = copy.deepcopy([param.grad for param in model.parameters()])

                this_instance_grad = get_gradient_func(model,this_next_user_2_image,this_next_user_2_label)

                this_l1_dist = []
                this_l2_dist = []


                for param1, param2, param3 in zip(this_instance_grad, train_batch_grad,
                                                  next_user_1_batch_grad):
                    l2dist1 = torch.norm(torch.flatten(param2) - torch.flatten(param3))  # |B-C|
                    l2dist2 = torch.norm(
                        torch.flatten(param2) - torch.flatten(param3) * (batch_size) / (batch_size+1) - torch.flatten(
                            param1) / (batch_size+1))
                    # |B- C(n-1)/n + a/n|

                    l1dist1 = torch.norm(torch.flatten(param2) - torch.flatten(param3), p=1)  # |B-C|
                    l1dist2 = torch.norm(
                        torch.flatten(param2) - torch.flatten(param3) * (batch_size) / (batch_size+1) - torch.flatten(
                            param1) / (batch_size+1), p=1)
                    # |B- C(n-1)/n + a/n|

                    this_l1_dist.append(l1dist1.item() - l1dist2.item()) ### smaller means member
                    this_l2_dist.append(l2dist1.item() - l2dist2.item())

                l1diff.append(np.array(this_l1_dist))
                l2diff.append(np.array(this_l2_dist))

                model.zero_grad()

        l1diff = np.array(l1diff)
        l2diff = np.array(l2diff)

        #print (l1diff.shape)

        num_layers = l1diff.shape[1]

        for layer_idx in range(num_layers):
            member_data = []
            non_member_data = []
            y_true = []
            this_layer_data = l1diff[:,layer_idx]
            y_pred = this_layer_data
            for batch_idx in range(int(total_size/batch_size)):
                member_data.append(this_layer_data[2*batch_idx*batch_size:2*batch_idx*batch_size+batch_size])
                non_member_data.append(this_layer_data[2*batch_idx*batch_size+batch_size:2*batch_idx*batch_size+batch_size*2])
                y_true.append(np.concatenate((np.ones(batch_size), np.zeros(batch_size))))

            from sklearn.metrics import roc_auc_score
            y_true = np.array(y_true).flatten()
            if (layer_idx == num_layers-1):
                print ("l1 AUC score", roc_auc_score(y_true, y_pred))


        for layer_idx in range(num_layers):
            member_data = []
            non_member_data = []
            y_true = []
            this_layer_data = l2diff[:,layer_idx]
            y_pred = this_layer_data
            for batch_idx in range(int(total_size/batch_size)):
                member_data.append(this_layer_data[2*batch_idx*batch_size:2*batch_idx*batch_size+batch_size])
                non_member_data.append(this_layer_data[2*batch_idx*batch_size+batch_size:2*batch_idx*batch_size+batch_size*2])
                y_true.append(np.concatenate((np.ones(batch_size), np.zeros(batch_size))))

            from sklearn.metrics import roc_auc_score
            y_true = np.array(y_true).flatten()
            if (layer_idx == num_layers-1):
                print ("l2 AUC score", roc_auc_score(y_true, y_pred))


        all_l1.append(l1diff)
        all_l2.append(l2diff)

    print ("fed grad in out attack end")

    return np.array(all_l1), np.array(all_l2)


def fed_gradient_summing_up_attack_per_batch(user_list, model, batch_size=100, total_size=500,get_gradient_func=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_plus_count = []
    all_label = []

    for target_idx in range(len(user_list)):

        next_user_1 = (target_idx + 1) % (len(user_list))
        next_user_2 = (target_idx + 2) % (len(user_list))

        member_corr_count = 0
        non_member_corr_count = 0

        train_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset, batch_size=batch_size,
                                                   shuffle=True)
        next_user_1_loader = torch.utils.data.DataLoader(user_list[next_user_1].train_eval_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True)
        next_user_2_loader = torch.utils.data.DataLoader(user_list[next_user_2].train_eval_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True)

        for (target_image, target_label), (next_user_1_image, next_user_1_label), (
                next_user_2_image, next_user_2_label) in zip(train_loader, next_user_1_loader, next_user_2_loader):
            ### get the gradient for the training batch
            # target_image = target_image.to(device)
            # target_label = target_label.to(device)
            # output = model(target_image)
            # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
            # loss = criterion(output, target_label)
            # loss.backward()
            #
            # train_batch_grad = copy.deepcopy([param.grad for param in model.parameters()])
            #
            # model.zero_grad()
            #
            # ### get the gradient for the next_user_1 batch
            #
            # next_user_1_image = next_user_1_image.to(device)
            # next_user_1_label = next_user_1_label.to(device)
            # output = model(next_user_1_image)
            # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
            # loss = criterion(output, next_user_1_label)
            # loss.backward()
            #
            # next_user_1_batch_grad = copy.deepcopy([param.grad for param in model.parameters()])
            #
            # model.zero_grad()

            train_batch_grad = get_gradient_func(model,target_image,target_label)
            next_user_1_batch_grad = get_gradient_func(model,next_user_1_image,next_user_1_label)


            ### for every instance in train_batch

            for i in range(batch_size):
                this_target_image = torch.unsqueeze(target_image[i], dim=0).to(device)
                this_target_label = torch.unsqueeze(target_label[i], dim=0).to(device)
                # output = model(this_target_image)
                # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
                # loss = criterion(output, this_target_label)
                # loss.backward()
                #
                # this_instance_grad = copy.deepcopy([param.grad for param in model.parameters()])
                this_instance_grad = get_gradient_func(model,this_target_image,this_target_label)

                this_instance_count = []

                plus_count = 0

                for param1, param2, param3 in zip(this_instance_grad, train_batch_grad,
                                                  next_user_1_batch_grad):
                    estimation_in = torch.flatten((param2*batch_size-param1)/(batch_size-1)-param3)
                    estimation_out = torch.flatten((param2-param3))
                    ### if this instance is member, then estimation in should be smaller in absolute fashion

                    ####
                    # signs = torch.sign((torch.abs(estimation_out) - torch.abs(estimation_in)))

                    ####
                    # signs = torch.abs(torch.sign(param1)+torch.sign(param2-param3))-1

                    ####
                    this_a = torch.abs(torch.sign(param1 - param3) + torch.sign(param1 - param2))
                    this_b = torch.abs(torch.sign(param3 - param1) + torch.sign(param3 - param2))
                    signs = (this_b * (this_a + this_b) - 4) / 4
                    plus_count += torch.sum(signs)

                    this_instance_count.append(torch.sum(signs).item())

                if (plus_count>0):
                    member_corr_count+=1


                all_plus_count.append(np.array(this_instance_count))
                all_label.append(1)

                #print (i,plus_count)

                model.zero_grad()

            ### for a batch of non-members

            for i in range(batch_size):
                this_next_user_2_image = torch.unsqueeze(next_user_2_image[i], dim=0).to(device)
                this_next_user_2_label = torch.unsqueeze(next_user_2_label[i], dim=0).to(device)
                # output = model(this_next_user_2_image)
                # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
                # loss = criterion(output, this_next_user_2_label)
                # loss.backward()

                #this_instance_grad = copy.deepcopy([param.grad for param in model.parameters()])
                this_instance_grad = get_gradient_func(model,this_next_user_2_image,this_next_user_2_label)

                this_instance_count = []

                plus_count=0

                for param1, param2, param3 in zip(this_instance_grad, train_batch_grad,
                                                  next_user_1_batch_grad):
                    estimation_in = torch.flatten((param2*batch_size-param1)/(batch_size-1)-param3)
                    estimation_out = torch.flatten((param2-param3))
                    ### if this instance is member, then estimation in should be smaller in absolute fashion

                    ####
                    # signs = torch.sign((torch.abs(estimation_out) - torch.abs(estimation_in)))

                    ####
                    # signs = torch.abs(torch.sign(param1)+torch.sign(param2-param3))-1

                    ####
                    this_a = torch.abs(torch.sign(param1 - param3) + torch.sign(param1 - param2))
                    this_b = torch.abs(torch.sign(param3 - param1) + torch.sign(param3 - param2))
                    signs = (this_b * (this_a + this_b) - 4) / 4
                    plus_count += torch.sum(signs)

                    this_instance_count.append(torch.sum(signs).item())

                if (plus_count<0):
                    non_member_corr_count+=1


                all_plus_count.append(np.array(this_instance_count))
                all_label.append(0)


                model.zero_grad()

                #print (i,plus_count)

        #print (member_corr_count,non_member_corr_count,total_size,member_corr_count/total_size,non_member_corr_count/total_size)


    all_plus_count = np.array(all_plus_count)
    from sklearn.metrics import roc_auc_score
    #for i in range(all_plus_count.shape[1]):
    #    print ("auc score", roc_auc_score(np.array(all_label), np.array(all_plus_count[:,i])))

    print ("auc score", roc_auc_score(np.array(all_label), np.sum(all_plus_count,axis=1)))

    return np.array(all_plus_count)