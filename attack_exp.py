from whitebox_attack import *
from blackbox_attack import *
import argparse
from data import dataset
from model import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def eval_model(target_model, train, test, eval_attack,validation, learning_rate, decay, epochs,
               batch_size, starting_index):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.dataset == 'cifar10'):
        num_classes = 10
    elif (args.dataset == 'cifar100'):
        num_classes = 100
    elif (args.dataset == 'purchase'):
        num_classes = 100
    elif (args.dataset == 'texas'):
        num_classes = 100
    elif (args.dataset == 'mnist'):
        num_classes = 10

    label_type = torch.LongTensor
    criterion = nn.CrossEntropyLoss()
    if (args.gpu):
        label_type = torch.cuda.LongTensor
        criterion = nn.CrossEntropyLoss().cuda()
    target_model.type(torch.FloatTensor)
    target_model.to(device)
    #target_model.train()
    optimizer = torch.optim.SGD(target_model.parameters(), lr=learning_rate, weight_decay=decay, momentum=0.9)
    if (args.dataset=='purchase' or args.dataset=='texas'):
        optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=decay)
        print ("learning rate:",learning_rate)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=1)
    validation_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=False,
                                                    num_workers=1)
    train_loader_in_order = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=1)


    ## some info we need to record for validation MI attacks
    validation_confidence_in_training = []
    validation_label_in_training = []
    avg_loss = 0

    for epoch in range(epochs):
        avg_loss = 0
        #start_time = time.time()
        target_model.train()

        if (epoch in args.schedule):
            learning_rate = learning_rate / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print ("new learning rate = %f" % (learning_rate))

        for index,(images, labels) in enumerate(train_loader):

            if (args.mixup):
                inputs, targets_a, targets_b, lam = mixup_data(images, labels, args.alpha)
                optimizer.zero_grad()
                inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)
                outputs = target_model(inputs)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(criterion, outputs)
                avg_loss+=loss.item()
                loss.backward()
                optimizer.step()

            elif (args.label_smoothing!=0):
                images = images.to(device)
                optimizer.zero_grad()
                outputs = target_model(images)
                outputs = F.log_softmax(outputs,dim=-1)
                labels = torch.reshape(labels,(-1,1))
                label_onehot = torch.FloatTensor(batch_size, num_classes)
                label_onehot.zero_()
                label_onehot.scatter_(1,labels,1)
                this_alpha = args.label_smoothing
                this_const = (1-this_alpha)/(num_classes-1)
                mod_labels = torch.mul(label_onehot,this_alpha-this_const)
                mod_labels = torch.add(mod_labels,this_const)
                mod_labels = mod_labels.to(device)
                criterion = nn.KLDivLoss()
                loss = criterion(outputs,mod_labels)
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()

            else:
                # normal training
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = target_model(images)
                loss = criterion(outputs, labels)
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()

        avg_loss = avg_loss/(index+1)

        ### test train/valid acc after every epoch
        correct = 0.0
        total = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            outputs = target_model(images)
            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc = correct.item()
        acc = acc / total
        acc = acc * 100.0
        this_training_acc = acc

        print ('Training Accuracy %f' %(acc))

        correct = 0.0
        total = 0.0
        target_model.eval()
        for images, labels in validation_loader:
            images = images.to(device)
            outputs = target_model(images)
            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc = correct.item()
        acc = acc / total
        acc = acc * 100.0

        print('Validation Accuracy %f ' % (acc))

        this_validation_accuracy = acc

        max_distribution_lambda = args.maxprob_loss_lambda
        corr_distribution_lambda = args.corr_loss_lambda
        mmd_loss_lambda = args.mmd_loss_lambda

        ## if the gap is less than 3%, then don't do MMD regularization. This threshold can be tuned.
        if (abs(this_validation_accuracy - this_training_acc) < 3 or args.mmd_loss_lambda<1e-5):
            continue
        else:
            pass

        validation_label_in_training = []
        validation_confidence_in_training = []
        for loss_index,(train_images,train_labels) in enumerate(train_loader_in_order):

            batch_num = train_labels.size()[0]
            optimizer.zero_grad()

            ### get the same number of images for each class
            valid_images = torch.zeros_like(train_images).type(torch.FloatTensor).to(device)
            valid_labels = torch.zeros_like(train_labels).type(torch.LongTensor).to(device)
            valid_index = 0
            for label_index,i in enumerate(torch.unique(train_labels)):
                this_frequency = torch.bincount(train_labels)[i].to(device)
                this_class_start = starting_index[i]
                ## i is the current class label

                ## try random?
                if (i<num_classes-1):
                    this_class_end = starting_index[i+1]-1
                else:
                    this_class_end = validation.__len__()-1

                #print ("class %d, start %d, end %d" %(i,this_class_start,this_class_end))

                for j in range(this_frequency): ### this part can be rewritten to be more efficient

                    random_index = np.random.randint(this_class_start,this_class_end)
                    #random_index = this_class_start+j
                    #print ("this random index",random_index,this_class_start+random_index)
                    new_images,new_labels =((validation).__getitem__(random_index)) ### replace j with random_index
                    valid_images[valid_index] = new_images.to(device)
                    #print ("new label",new_labels)
                    valid_labels[valid_index] = (torch.ones(1)*new_labels).type(torch.LongTensor).to(device)
                    valid_index+=1

            train_images = train_images.to(device)
            train_labels = train_labels.to(device)
            outputs = target_model(train_images)
            all_train_outputs = F.softmax(outputs,dim=1)
            all_train_outputs = all_train_outputs.view(-1,num_classes)
            train_labels = train_labels.view(batch_num,1)

            valid_images = valid_images.to(device)
            valid_labels = valid_labels.to(device)
            outputs = target_model(valid_images)
            all_valid_outputs = F.softmax(outputs,dim=1)
            all_valid_outputs = (all_valid_outputs).detach_()
            valid_labels = valid_labels.view(batch_num,1)

            validation_label_in_training.append(valid_labels.cpu().data.numpy())
            validation_confidence_in_training.append(all_valid_outputs.cpu().data.numpy())

            ### max_distribution_loss
            if (max_distribution_lambda>0):
                train_max_prob,_ = torch.max(all_train_outputs,dim=1)
                valid_max_prob,_ = torch.max(all_valid_outputs,dim=1)
                train_max_prob,_ = torch.sort(train_max_prob)
                valid_max_prob,_ = torch.sort(valid_max_prob)
                max_distribution_loss = (((train_max_prob-valid_max_prob)**2))*max_distribution_lambda
                #print ("max distribution loss",max_distribution_loss)
                max_distribution_loss = max_distribution_loss.mean()
                #print ("max distribution loss",max_distribution_loss)
                #if (loss_index == 0):
                    #print ("max prob loss used!")
                max_distribution_loss.backward()

            ### max distribution kuiper loss
            if (args.kuiper_loss_lambda>0):
                train_max_prob,_ = torch.max(all_train_outputs,dim=1)
                valid_max_prob,_ = torch.max(all_valid_outputs,dim=1)
                train_max_prob,_ = torch.sort(train_max_prob)
                valid_max_prob,_ = torch.sort(valid_max_prob)

                ### round into integers, 0.01 as an interval
                train_max_prob = torch.round(train_max_prob*100)
                valid_max_prob = torch.round(valid_max_prob*100)

                ### calculate bincount
                train_bincount = bincount(torch.arange(1,100).type(label_type),train_max_prob,max_item=100)
                valid_bincount = bincount(torch.arange(1,100).type(label_type),valid_max_prob,max_item=100)

                ## calculate CDF
                train_cdf = torch.cumsum(train_bincount,dim=0)
                valid_cdf = torch.cumsum(valid_bincount,dim=0)

                kuiper_loss=args.kuiper_loss_lambda*distroLoss(train_cdf,(torch.ones(1)*batch_num).type(torch.cuda.FloatTensor),valid_cdf,(torch.ones(1)*batch_num).type(torch.cuda.FloatTensor))
                #print ("kuiper loss = ",kuiper_loss)
                kuiper_loss.backward()

            ### class_distribution_loss
            if (corr_distribution_lambda>0):
                train_corr_prob = []
                valid_corr_prob = []
                for image_index in range(batch_num):
                    train_corr_prob.append(all_train_outputs[image_index][train_labels[image_index]])
                    valid_corr_prob.append(all_valid_outputs[image_index][valid_labels[image_index]])

                train_corr_prob = torch.stack(train_corr_prob)
                valid_corr_prob = torch.stack(valid_corr_prob)

                train_corr_prob,_ = torch.sort(train_corr_prob)
                valid_corr_prob,_ = torch.sort(valid_corr_prob)
                corr_distribution_loss = (((train_corr_prob-valid_corr_prob)**2))*corr_distribution_lambda
                corr_distribution_loss = corr_distribution_loss.mean()
                #if (loss_index == 0):
                #    print ("corr prob loss used!")
                corr_distribution_loss.backward()

            ### corr prob MMD loss
            #sfrom mmd_loss import mix_rbf_mmd2
            if (args.corr_mmd_loss_lambda>0):
                train_corr_prob = []
                valid_corr_prob = []
                for image_index in range(batch_num):
                    train_corr_prob.append(all_train_outputs[image_index][train_labels[image_index]])
                    valid_corr_prob.append(all_valid_outputs[image_index][valid_labels[image_index]])

                train_corr_prob = torch.stack(train_corr_prob)
                valid_corr_prob = torch.stack(valid_corr_prob)

                train_corr_prob,_ = torch.sort(train_corr_prob)
                valid_corr_prob,_ = torch.sort(valid_corr_prob)
                corr_prob_mmd_loss = mix_rbf_mmd2(train_corr_prob,valid_corr_prob,sigma_list=[1])*args.corr_mmd_loss_lambda
                #if (loss_index == 0):
                #    print ("correct label probability MMD loss",corr_prob_mmd_loss)
                corr_prob_mmd_loss.backward()

            ### compare two MMD loss
            #from mmd_loss import mix_rbf_mmd2
            #first_mmd_loss = mix_rbf_mmd2(train_corr_prob,valid_corr_prob,sigma_list=[1])
            #second_mmd_loss = distroLoss,torch.ones(1,dtype=torch.float32).cuda()*batch_num,all_valid_outputs,torch.ones(1,dtype=torch.float32).cuda()*batch_num,lossName='mmd')
            #print ("first mmd",torch.log(first_mmd_loss))
            #print ("second mmd",corr_prob_mmd_loss/args.corr_mmd_loss_lambda)

            ### MMD loss
            #from mmd_loss import mix_rbf_mmd2
            if (mmd_loss_lambda>0):
                mmd_loss = mix_rbf_mmd2(all_train_outputs,all_valid_outputs,sigma_list=[1])*mmd_loss_lambda
                mmd_loss.backward()

            ### kuiper loss
            #kuiper_loss = 0.01*distroLoss(all_train_outputs,torch.ones(1,dtype=torch.float32).cuda()*batch_num,all_valid_outputs,torch.ones(1,dtype=torch.float32).cuda()*batch_num,lossName='kuiper')
            #print ("this kuiper loss ",kuiper_loss)
            #if (abs(kuiper_loss)>0.001):
            #    kuiper_loss.backward(retain_graph=True)

            ### update the weights by max_distribution loss, corr distribution loss and mmd loss
            if (epoch != epochs-1):
                optimizer.step()

    print ("TRAINING FINISHED")

    ### train/validation accuracy after training
    correct = 0.0
    total = 0.0
    target_model.eval()

    for images, labels in train_loader:
        images = images.to(device)
        outputs = target_model(images)
        labels = labels.to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    acc = correct.item()
    acc = acc / total
    acc = acc * 100.0
    print("Train Accuracy %f " % (acc))
    training_acc = acc

    correct = 0.0
    total = 0.0

    for images, labels in test_loader:
        images = images.to(device)
        outputs = target_model(images)
        labels = labels.to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    acc = correct.item()
    acc = acc / total
    acc = acc * 100.0
    print("Test Accuracy %f " % (acc))
    testing_acc = acc

    ### if temperature scaling is used here to do calibration
    if (args.temperature_scaling):
        ### apply temperature scaling
        target_model = ModelWithTemperature(target_model)
        target_model.set_temperature(validation_loader)
        correct = 0.0
        total = 0.0
        target_model.eval()
        for images, labels in train_loader:
            images = images.to(device)
            outputs = target_model(images)
            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc = correct.item()
        acc = acc / total
        acc = acc * 100.0
        print("Train Accuracy %f " % (acc))
        training_acc = acc

        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = images.to(device)
            outputs = target_model(images)
            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc = correct.item()
        acc = acc / total
        acc = acc * 100.0
        print("Test Accuracy %f " % (acc))
        testing_acc = acc

    ### generate predictions for membership evaluation set

    eval_loader = torch.utils.data.DataLoader(eval_attack, batch_size=100, shuffle=False,num_workers=1)
    confidence = []
    for i, (images, labels) in enumerate(eval_loader):
        images = images.to(device)
        labels = labels.to(device)
        this_confidence = target_model(images)
        for i in range(images.size()[0]):
            confidence.append(F.softmax(this_confidence[i],dim=0).cpu().detach().data.numpy())
    ### another idea is to try to generate adversarial perturbation to test membership, see CISPA's blackbox paper in 2020
    confidence = np.array(confidence)
    confidence = np.reshape(confidence, (args.membership_attack_number, num_classes))

    torch.cuda.empty_cache()

    return target_model,confidence,training_acc,testing_acc,validation_confidence_in_training,validation_label_in_training,avg_loss


    ###
    #middle_output_train = []
    #middle_output_valid = []
    #if (args.test==1 and args.middle_output==1):
        #debug
        #middle_output_train,middle_output_valid = get_middle_output(target_model,train_loader_in_order,validation,starting_index)

    #middle_grad_valid = []
    #middle_grad_train = []#

    #if (args.test==1 and args.middle_gradient==1):
    #    middle_grad_train,middle_grad_valid = get_middle_gradient(target_model,train_loader_in_order,validation,starting_index)

    #if (args.mmd_loss_lambda>0):
    #    validation_confidence_in_training = np.reshape(np.array(validation_confidence_in_training),(-1,num_classes))
    #    validation_label_in_training = np.reshape(np.array(validation_label_in_training),(-1))


def attack_experiment():
    ### some settings
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(seed=12345)
    torch.set_printoptions(threshold=5000, edgeitems=20)

    ### dataset && membership inference data
    membership_attack_number = args.membership_attack_number
    target_dataset = dataset(dataset_name=args.dataset, gpu=args.gpu,
                             membership_attack_number=membership_attack_number,
                             cutout=args.cutout, n_holes=args.n_holes, length=args.length)
    num_classes = len(np.unique(target_dataset.label))

    total_classes = np.zeros((args.model_number, membership_attack_number))
    total_confidence = np.zeros((args.model_number, membership_attack_number,
                                 len(np.unique(target_dataset.train_label))))
    total_label = np.zeros((args.model_number, membership_attack_number))
    total_avg_loss = np.zeros((args.model_number))
    total_valid_confidence = []
    total_valid_label = []
    total_training_acc = 0
    total_testing_acc = 0
    total_training_acc_list = []
    total_testing_acc_list = []
    total_middle_output_train = []
    total_middle_output_test = []
    total_middle_output_train_label = []
    total_middle_output_test_label = []
    total_middle_gradient_train = []
    total_middle_gradient_test = []

    ## get corresponding target model
    for i in range(args.model_number):
        print ("model number %d" %(i))
        if (target_dataset.dataset_name == 'cifar100'):

            if (args.model_name == 'resnet20'):
                target_model = resnet(depth=20,num_classes=100)
            if (args.model_name =='resnet110'):
                target_model = resnet(depth=110,num_classes=100)
            if (args.model_name == 'densenet_cifar'):
                target_model = densenet(num_classes=100)
            if (args.model_name=='resnet101'):
                model = models.resnet101(pretrained=args.pretrained)
                number_features = model.fc.in_features
                model.fc = nn.Linear(number_features, 100)
                model.avgpool = nn.AvgPool2d(1, 1)
                target_model = model
            if (args.model_name=='resnet18'):
                model = models.resnet18(pretrained=args.pretrained)
                number_features = model.fc.in_features
                model.fc = nn.Linear(number_features, 100)
                model.avgpool = nn.AvgPool2d(1, 1)
                target_model = model
            if (args.model_name =='vgg16'):
                target_model = models.vgg16(pretrained=args.pretrained)
                target_model.avgpool = nn.Sequential(nn.Flatten())
                target_model.classifier = nn.Sequential(nn.Linear(512, 100))
                #target_model.classifier = nn.Sequential(nn.Conv2d(512,100,kernel_size=1),
                #                                        nn.AvgPool2d(kernel_size=1))
                #print (target_model)

            if (args.model_name =='densenet121'):
                target_model = models.densenet121(pretrained=args.pretrained)
                target_model.classifier = nn.Linear(1024,100)

            if (args.model_name == 'alexnet'):
                target_model = alexnet(num_classes=100)

        elif (target_dataset.dataset_name == 'cifar10'):
            if (args.model_name == 'resnet20'):
                target_model = resnet(depth=20,num_classes=10)
            if (args.model_name =='resnet110'):
                target_model = resnet(depth=110,num_classes=10)
            if (args.model_name == 'densenet_cifar'):
                target_model = densenet(num_classes=10)
            if (args.model_name=='resnet101'):
                model = models.resnet101(pretrained=args.pretrained)
                number_features = model.fc.in_features
                model.fc = nn.Linear(number_features, 10)
                model.avgpool = nn.AvgPool2d(1, 1)
                target_model = model
            #resnet18
            if (args.model_name=='resnet18'):
                model = models.resnet18(pretrained=args.pretrained)
                number_features = model.fc.in_features
                model.fc = nn.Linear(number_features, 10)
                model.avgpool = nn.AvgPool2d(1, 1)
                target_model = model
                #print (target_model)
            if (args.model_name =='vgg16'):
                target_model = models.vgg16(pretrained=args.pretrained)
                print (target_model)
                target_model.avgpool = nn.Sequential(nn.Flatten())
                target_model.classifier = nn.Sequential(nn.Linear(512, 10))
                print (target_model)
            if (args.model_name =='densenet121'):
                target_model = models.densenet121(pretrained=args.pretrained)
                target_model.classifier = nn.Sequential(nn.Linear(1024,10))
            if (args.model_name == 'alexnet'):
                target_model = alexnet(num_classes=10)
        else:
            target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))

        ### generate training / testing / validation data
        target_data_number = args.target_data_size
        train, test, eval_attack,validation, eval_partition, in_train_partition, out_train_partition,starting_index,train_eval = target_dataset.select_part(
            target_data_number, membership_attack_number, args.reference_number)

        ### membership information for eval set
        for j in range(len(in_train_partition)):
            total_label[i, in_train_partition[j]] = 1
        for j in range(len(out_train_partition)):
            total_label[i, out_train_partition[j]] = 0

        ###train target model & get confidence for train/test data
        target_model,confidence,training_acc,testing_acc,valid_conf,valid_label,this_avg_loss = \
            eval_model(target_model, train, test, eval_attack,validation,
                                learning_rate=args.target_learning_rate,
                       decay=args.target_l2_ratio, epochs=args.target_epochs,
                                batch_size=args.target_batch_size,
                                starting_index=starting_index)

        ###gather all info
        total_classes[i] = np.copy(target_dataset.part_eval_label)
        total_confidence[i] = confidence
        total_training_acc+=training_acc
        total_testing_acc+=testing_acc
        total_avg_loss[i] = this_avg_loss
        total_valid_confidence.append(valid_conf)
        total_valid_label.append(valid_label)
        total_training_acc_list.append(training_acc)
        total_testing_acc_list.append(testing_acc)


        ### if we want to do white-box attack

        if (args.whitebox):
            print ("DOING whitebox data collection!")
            if (args.middle_output):
                this_middle_output_train,this_middle_train_label = get_middle_output(target_model,train_eval,training_size=args.target_data_size)
                this_middle_output_test,this_middle_test_label = get_middle_output(target_model,test,training_size=args.target_data_size)
                total_middle_output_test.append(this_middle_output_test)
                total_middle_output_train.append(this_middle_output_train)
                total_middle_output_test_label.append(this_middle_test_label)
                total_middle_output_train_label.append(this_middle_train_label)

                #print (this_middle_output_train[0][0][:500])
                #print (this_middle_output_test[0][0][:500])

                ### we can use validation data here
                # this_middle_output_valid = get_middle_output(target_model,validation,training_size=args,target_data_size)

            if (args.middle_gradient):
                #print ("fast gradient compuattaion")
                #this_middle_gradient_train = get_middle_gradient(target_model, train_eval,training_size=args.target_data_size,required_num=args.required_num)
                #this_middle_gradient_test = get_middle_gradient(target_model, test,training_size=args.target_data_size,required_num=args.required_num)

                this_middle_gradient_train = get_middle_gradient_one(target_model,train_eval,training_size=args.target_data_size,required_num=args.required_num)
                this_middle_gradient_test = get_middle_gradient_one(target_model,test,training_size=args.target_data_size,required_num=args.required_num)
                total_middle_gradient_test.append(this_middle_gradient_test)
                total_middle_gradient_train.append(this_middle_gradient_train)

                ### gradient check
                #new_middle_gradient_train = get_middle_gradient(target_model, train,training_size=args.target_data_size,required_num=args.required_num)
                #new_middle_gradient_test = get_middle_gradient(target_model, test,training_size=args.target_data_size,required_num=args.required_num)
                #print ("gradient calculation check")
                #for grad1,grad2 in zip(this_middle_gradient_train,new_middle_gradient_train):
                #    print (grad1.shape)
                #    print (grad2.shape)
                #    print (np.sum(grad1[0]))
                #    print (np.sum(grad2[0]))
                #    print (np.sum(grad1[0]-grad2[0]))
                #    print (grad1[0][:20],grad2[0][:20])
                #    print (np.sum(grad1-grad2))



        del target_model

    if (args.whitebox and args.save_exp_data):
        #print ("training accuracy %.2f" %(total_training_acc))
        #print ("testing accuracy %.2f" %(total_testing_acc))

        path = '/home/lijiacheng/neighbor/expdata/' + args.dataset + '_' + str(args.target_data_size) + '_' + str(
                args.model_number) + '_' + str(args.model_name) + '_' + str(args.mmd_loss_lambda) + '_' + str(
                args.mixup) + '_' + str(args.target_learning_rate) + '_'
        #np.save(path + 'confidence.npy', total_confidence)
        #np.save(path + 'label.npy', total_label)
        #np.save(path + 'classes.npy', total_classes)

        #print (len(total_middle_output_train))

        if (args.middle_output):
            path = '/home/lijiacheng/neighbor/expdata/' + args.dataset + '_' + str(args.target_data_size) + '_' + str(
                args.model_number) + '_' + str(args.model_name) + '_' + str(args.mmd_loss_lambda) + '_' + str(
                args.mixup) + '_' + str(args.target_learning_rate) + '_'

            for i in range(len(total_middle_output_test[0])):

                print (total_middle_output_test[0][i].shape)
                print (total_middle_output_train[0][i].shape)

                print (path+str(i+1)+'_train_activation.npy')

                arr1 = np.array(total_middle_output_train[0][i])
                arr2 = np.array(total_middle_output_test[0][i])

                np.save(path+str(i+1)+'_train_activation.npy',arr1)
                np.save(path+str(i+1)+'_valid_activation.npy',arr2)

                ### do whitebox middle output umap/norm attack
                print ("attacking layer %d using activation" %(i+1))
                attack = whitebox_attack(args.membership_attack_number,name='activation_umap',num_classes=num_classes)
                attack.attack(arr1,arr2,train_label=total_middle_output_train_label[0][i],valid_label=total_middle_output_test_label[0][i])
                attack = whitebox_attack(args.membership_attack_number,name='activation_norm',num_classes=num_classes)
                attack.attack(arr1, arr2, train_label=total_middle_output_train_label[0][i],valid_label=total_middle_output_test_label[0][i])

        if (args.middle_gradient):
            path = '/home/lijiacheng/neighbor/expdata/' + args.dataset + '_' + str(args.target_data_size) + '_' + str(
                args.model_number) + '_' + str(args.model_name) + '_' + str(args.mmd_loss_lambda) + '_' + str(
                args.mixup) + '_' + str(args.target_learning_rate) + '_'

            for i in range(len(total_middle_gradient_test[0])):

                print (path + str(i + 1) + '_train_grad.npy')
                arr1 = np.array(total_middle_gradient_train[0][i])
                arr2 = np.array(total_middle_gradient_test[0][i])
                #np.save(path + str(i + 1) + '_train_grad.npy', arr1)
                #np.save(path + str(i + 1) + '_valid_grad.npy', arr2)

                print ("attacking layer %d using activation" %(i+1))
                attack = whitebox_attack(args.membership_attack_number,name='gradient_umap',num_classes=num_classes)
                attack.attack(arr1,arr2,train_label=target_dataset.part_train_label,valid_label=target_dataset.part_test_label)
                attack = whitebox_attack(args.membership_attack_number,name='gradient_norm',num_classes=num_classes)
                attack.attack(arr1,arr2,train_label=target_dataset.part_train_label,valid_label=target_dataset.part_test_label)

        #np.save(path+'valid_label.npy',target_dataset.part_test_label)
        #np.save(path+'train_label.npy',target_dataset.part_train_label)

        return total_training_acc,total_testing_acc


    print ("avg training acc = %f"%(total_training_acc/args.model_number))
    print ("avg_testing_acc = %f"%(total_testing_acc/args.model_number))
    for i in range(len(total_training_acc_list)):
        print ("train acc %.4f, test acc %.4f" %(total_training_acc_list[i],total_testing_acc_list[i]))


    #### launch blackbox attacks
    total_confidence = np.nan_to_num(total_confidence)
    ## baseline attack
    attack = blackbox_attack(args.membership_attack_number,name='baseline',num_classes=num_classes)
    attack.attack(total_confidence,total_classes,total_label)
    ## avgloss attack
    attack = blackbox_attack(args.membership_attack_number,name='avg_loss',num_classes=num_classes)
    attack.attack(total_confidence,total_classes,total_label,avg_loss=np.average(total_avg_loss))
    ## top1 attack
    attack = blackbox_attack(args.membership_attack_number,name='top1',num_classes=num_classes)
    attack.attack(total_confidence,total_classes,total_label)
    ## top3 attack
    attack = blackbox_attack(args.membership_attack_number,name='top3',num_classes=num_classes)
    attack.attack(total_confidence,total_classes,total_label)
    ## global prob attack
    attack = blackbox_attack(args.membership_attack_number,name='global_prob',num_classes=num_classes)
    attack.attack(total_confidence,total_classes,total_label)
    ## per class attack
    attack = blackbox_attack(args.membership_attack_number,name='per_class',num_classes=num_classes)
    attack.attack(total_confidence,total_classes,total_label)
    ## instance distance attack
    attack = blackbox_attack(args.membership_attack_number,name='instance_distance',num_classes=num_classes)
    attack.attack(total_confidence,total_classes,total_label)
    ## instance prob/ratio attack
    attack = blackbox_attack(args.membership_attack_number,name='instance_prob',num_classes=num_classes)
    attack.attack(total_confidence,total_classes,total_label)

    if (args.validation_mi):
        total_valid_confidence = np.array(total_valid_confidence)
        total_valid_classes = np.array(total_valid_label)
        print ("total valid shape",total_valid_confidence.shape,total_valid_classes.shape)

        ## need to implement here


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_data_size', type=int, default=3000)
    parser.add_argument('--target_model', type=str, default='cnn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_learning_rate', type=float, default=0.001)
    parser.add_argument('--target_batch_size', type=int, default=100)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--target_epochs', type=int, default=20)
    parser.add_argument('--attack_epochs', type=int, default=500)
    parser.add_argument('--target_l2_ratio', type=float, default=5e-4)
    parser.add_argument('--shadow_data_size', type=int, default=30000)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model_number', type=int, default=10)
    #parser.add_argument('--attack_times', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--membership_attack_number', type=int, default=500)
    parser.add_argument('--reference_number', type=int, default=1)

    parser.add_argument('--schedule', type=int, nargs='+', default=[80,120])
    parser.add_argument('--model_name',type=str,default='alexnet')
    parser.add_argument('--pretrained',type=int,default=0)
    parser.add_argument('--temperature_scaling',type=int,default=0)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--alpha',type=float,default='1.0')
    parser.add_argument('--mixup',type=int,default=0)
    parser.add_argument('--label_smoothing',type=float,default=0)
    #parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--cutout',type=int,default=0)
    parser.add_argument('--n_holes',type=int,default=1)
    parser.add_argument('--length',type=int,default=16)
    parser.add_argument('--mmd_loss_lambda',type=float,default=0)
    parser.add_argument('--maxprob_loss_lambda',type=float,default=0)
    parser.add_argument('--corr_loss_lambda',type=float,default=0)
    parser.add_argument('--corr_mmd_loss_lambda',type=float,default=0)
    parser.add_argument('--kuiper_loss_lambda',type=float,default=0)


    parser.add_argument('--validation_mi',type=int,default=0)

    parser.add_argument('--whitebox',type=int,default=0)
    parser.add_argument('--middle_output',type=int,default=0)
    parser.add_argument('--middle_gradient',type=int,default=0)
    parser.add_argument('--save_exp_data',type=int,default=1)
    parser.add_argument('--test',type=int,default=0)
    parser.add_argument('--required_num',type=int,default=200)

    import torch
    torch.manual_seed(123)
    import numpy as np
    np.random.seed(123)

    args = parser.parse_args()
    print (vars(args))

    attack_experiment()

    print (vars(args))