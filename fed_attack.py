import numpy as np
import torch
import torch.nn as nn
from utils import *
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from whitebox_attack import *
from user import *
from fed_attack_batch import *
from fed_attack_epoch import *

def fed_gradient_norm_baseline_attack(user_list,model,get_gradient_func=None):

    print ("fed grad norm baseline attack:")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for target_idx in range(len(user_list)):

        all_member_norm = []

        one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset,batch_size=1,shuffle=True)

        all_count = 0
        for image,label in one_instance_loader:

            this_instance_grad = get_gradient_func(model,image,label)

            ### compute the norm diff
            norm1 = 0
            norm2 = 0
            norm3 = 0
            dist1 = 0
            for param1,param2 in zip(this_instance_grad,user_list[target_idx].train_gradient):
                #print (param1.shape)
                dist1+=torch.norm(torch.flatten(param1 - param2))
                norm1+=torch.norm(torch.flatten(param1))
                norm2+=torch.norm(torch.flatten(param2))
            dist2 = 0
            for param1,param2 in zip(this_instance_grad,user_list[target_idx].test_gradient):
                dist2+=torch.norm(torch.flatten(param1 - param2))
                norm3+=torch.norm(torch.flatten(param2))

            all_count+=1
            all_member_norm.append(norm1.item())
            model.zero_grad()

        all_nonmember_norm = []
        ### non member data come from the unassigned part of the original training set
        one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].test_eval_dataset, batch_size=1,
                                                          shuffle=True)
        for index,(image, label) in enumerate(one_instance_loader):
            ### get the gradient for this specific instance
            if (index>=all_count):
                break
            ### make sure that # of testing-member and # of testing-non-member are the same
            this_instance_grad = get_gradient_func(model,image,label)

            norm1 = 0
            norm2 = 0
            norm3 = 0
            dist1 = 0
            for param1,param2 in zip(this_instance_grad,user_list[target_idx].train_gradient):
                #print (param1.shape)
                dist1+=torch.norm(torch.flatten(param1 - param2))
                norm1+=torch.norm(torch.flatten(param1))
                norm2+=torch.norm(torch.flatten(param2))

                #if (index == 0):
                #    print (param1.shape)

            dist2 = 0
            for param1,param2 in zip(this_instance_grad,user_list[target_idx].test_gradient):
                dist2+=torch.norm(torch.flatten(param1 - param2))
                norm3+=torch.norm(torch.flatten(param2))

            all_nonmember_norm.append(norm1.item())

            model.zero_grad()

        ##### launch the gard norm attack from oakland paper
        length = len(all_member_norm)
        #print (length,len(all_nonmember_norm))
        all_member_norm = np.array(all_member_norm)
        all_nonmember_norm = np.array(all_nonmember_norm)
        member_train_index = np.random.choice(length, int(1 / 2 * length), replace=False)
        member_test_index = np.setdiff1d(np.arange(length), member_train_index)
        nonmember_train_index = np.random.choice(length, int(1 / 2 * length), replace=False)
        nonmember_test_index = np.setdiff1d(np.arange(length), nonmember_train_index)

        train_data = np.concatenate((all_member_norm[member_train_index], all_nonmember_norm[nonmember_train_index]))
        train_label = np.concatenate((np.ones(len(member_train_index)), np.zeros(len(nonmember_train_index))))
        test_data = np.concatenate((all_member_norm[member_test_index], all_nonmember_norm[nonmember_test_index]))
        test_label = np.concatenate((np.ones(len(member_test_index)), np.zeros(len(nonmember_test_index))))

        train_data = np.reshape(train_data, (-1, 1))
        test_data = np.reshape(test_data, (-1, 1))

        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(random_state=0, solver='lbfgs')
        clf.fit(train_data, train_label)
        print ("lr gradient norm attack accuracy %.2f" % (clf.score(test_data, test_label) * 100))

        ### get the auc score
        y_true = np.concatenate((np.ones(len(all_member_norm)),np.zeros(len(all_nonmember_norm))))
        y_pred = np.concatenate((all_member_norm,all_nonmember_norm))

        from sklearn.metrics import roc_auc_score
        print ("auc score", roc_auc_score(y_true, y_pred))

    print ("fed grad norm baseline attack end")


def fed_gradient_norm_baseline_all_training_attack(user_list,model,get_gradient_func=None):

    print ("fed grad norm baseline all training attack:")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for target_idx in range(len(user_list)):

        all_member_norm = []
        next_user = (1+target_idx)%len(user_list)

        one_instance_loader = torch.utils.data.DataLoader(user_list[target_idx].train_eval_dataset,batch_size=1,shuffle=True)

        member_corr_count = 0
        all_count = 0
        for image,label in one_instance_loader:
            this_instance_grad = get_gradient_func(model,image,label)

            ### compute the norm diff
            norm1 = 0
            norm2 = 0
            norm3 = 0
            dist1 = 0
            for param1,param2 in zip(this_instance_grad,user_list[target_idx].train_gradient):
                #print (param1.shape)
                dist1+=torch.norm(torch.flatten(param1 - param2))
                norm1+=torch.norm(torch.flatten(param1))
                norm2+=torch.norm(torch.flatten(param2))
            dist2 = 0
            #for param1,param2 in zip(this_instance_grad,user_list[target_idx].test_gradient):
            #    dist2+=torch.norm(torch.flatten(param1 - param2))
            #    norm3+=torch.norm(torch.flatten(param2))

            all_count+=1
            all_member_norm.append(norm1.item())
            model.zero_grad()

        all_nonmember_norm = []
        ### non member data come from the unassigned part of the original training set
        one_instance_loader = torch.utils.data.DataLoader(user_list[next_user].train_eval_dataset, batch_size=1,
                                                          shuffle=True)
        for index,(image, label) in enumerate(one_instance_loader):
            this_instance_grad = get_gradient_func(model,image,label)
            norm1 = 0
            norm2 = 0
            norm3 = 0
            dist1 = 0
            for param1,param2 in zip(this_instance_grad,user_list[target_idx].train_gradient):
                #print (param1.shape)
                dist1+=torch.norm(torch.flatten(param1 - param2))
                norm1+=torch.norm(torch.flatten(param1))
                norm2+=torch.norm(torch.flatten(param2))

                #if (index == 0):
                #    print (param1.shape)

            #dist2 = 0
            #for param1,param2 in zip(this_instance_grad,user_list[target_idx].test_gradient):
            #    dist2+=torch.norm(torch.flatten(param1 - param2))
            #    norm3+=torch.norm(torch.flatten(param2))

            all_nonmember_norm.append(norm1.item())

            model.zero_grad()

        ##### launch the gard norm attack from oakland paper
        length = len(all_member_norm)
        #print (length)
        all_member_norm = np.array(all_member_norm)
        all_nonmember_norm = np.array(all_nonmember_norm)
        member_train_index = np.random.choice(length, int(1 / 2 * length), replace=False)
        member_test_index = np.setdiff1d(np.arange(length), member_train_index)
        nonmember_train_index = np.random.choice(length, int(1 / 2 * length), replace=False)
        nonmember_test_index = np.setdiff1d(np.arange(length), nonmember_train_index)

        train_data = np.concatenate((all_member_norm[member_train_index], all_nonmember_norm[nonmember_train_index]))
        train_label = np.concatenate((np.ones(len(member_train_index)), np.zeros(len(nonmember_train_index))))
        test_data = np.concatenate((all_member_norm[member_test_index], all_nonmember_norm[nonmember_test_index]))
        test_label = np.concatenate((np.ones(len(member_test_index)), np.zeros(len(nonmember_test_index))))

        train_data = np.reshape(train_data, (-1, 1))
        test_data = np.reshape(test_data, (-1, 1))

        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(random_state=0, solver='lbfgs')
        clf.fit(train_data, train_label)
        print ("lr gradient norm attack accuracy %.2f" % (clf.score(test_data, test_label) * 100))

        y_true = np.concatenate((np.ones(len(all_member_norm)),np.zeros(len(all_nonmember_norm))))
        y_pred = np.concatenate((all_member_norm,all_nonmember_norm))

        from sklearn.metrics import roc_auc_score
        print ("auc score", roc_auc_score(y_true, y_pred))

    print ("fed grad norm baseline all training attack end")


def loss_profile_attack(user_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(len(user_list)):
        this_user = user_list[i]
        train_profile = np.array(this_user.train_loss_profile)
        test_profile = np.array(this_user.test_loss_profile)

        print (train_profile.shape)

        #### assume we know half of the true label, we use a logistic regression (or a small NN?) to attack

        length = len(train_profile)
        member_train_index = np.random.choice(length, int(1 / 2 * length), replace=False)
        member_test_index = np.setdiff1d(np.arange(length), member_train_index)
        nonmember_train_index = np.random.choice(length, int(1 / 2 * length), replace=False)
        nonmember_test_index = np.setdiff1d(np.arange(length), nonmember_train_index)

        train_data = np.concatenate((train_profile[member_train_index], test_profile[nonmember_train_index]))
        train_label = np.concatenate((np.ones(len(member_train_index)), np.zeros(len(nonmember_train_index))))
        test_data = np.concatenate((train_profile[member_test_index], test_profile[nonmember_test_index]))
        test_label = np.concatenate((np.ones(len(member_test_index)), np.zeros(len(nonmember_test_index))))

        #### logistic regression attack

        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(random_state=0, solver='lbfgs')
        clf.fit(train_data, train_label)

        print ("lr gradient norm attack accuracy %.2f" % (clf.score(train_data, train_label) * 100))
        print ("lr gradient norm attack accuracy %.2f" % (clf.score(test_data, test_label) * 100))

        #### small NN attack. epoch = 100, adam or adamax, lr=0.001, decay=1e-5
        attack_net = onelayer_AttackNet(train_data.shape[1]).to(device)
        attack_net.train()
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(attack_net.parameters(), lr=0.001, weight_decay=1e-5)
        #### create dataset and dataset loader
        train_set = part_pytorch_dataset(train_data,train_label,train=False,transform=None,target_transform=None)
        test_set = part_pytorch_dataset(test_data,test_label,train=False,transform=None,target_transform=None)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=1)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=1)
        for i in range(100):### epoch = 100
            for train_image,train_label in train_loader:
                attack_net.zero_grad()
                train_image = train_image.to(device)
                train_label = train_label.to(device)
                output = attack_net(train_image)
                loss = criterion(output,train_label)
                loss.backward()
                optimizer.step()

        ### after training, get train acc and test acc

        correct = 0.0
        total = 0.0
        attack_net.eval()
        for images, labels in train_loader:
            images = images.to(device)
            outputs = attack_net(images)
            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc = correct.item()
        acc = acc / total
        acc = acc * 100.0

        print ("attack net train acc %.2f" % (acc))

        correct = 0.0
        total = 0.0
        attack_net.eval()
        for images, labels in test_loader:
            images = images.to(device)
            outputs = attack_net(images)
            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc = correct.item()
        acc = acc / total
        acc = acc * 100.0

        print ("attack net test acc %.2f" % (acc))










