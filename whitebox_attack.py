import numpy as np
import torch
import torch.nn as nn
from utils import *
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
class whitebox_attack:

    def __init__(self,membership_attack_number,name='activation_umap',num_classes=10,training_size=10000):

        np.random.seed(seed = 12345)
        np.set_printoptions(suppress=True)

        self.name = name
        self.num_classes = num_classes
        self.membership_attack_number = membership_attack_number

    def attack(self,train_activation,valid_activation,train_label,valid_label):

        if (self.name == 'gradient_norm'):
            self._gradient_norm_attack(train_activation,valid_activation,train_label,valid_label)
        if (self.name == 'gradient_umap'):
            self._gradient_umap_attack(train_activation,valid_activation,train_label,valid_label)
        if (self.name == 'activation_umap'):
            self._activation_umap_attack(train_activation,valid_activation,train_label,valid_label)
        if (self.name == 'activation_norm'):
            self._activation_norm_attack(train_activation,valid_activation,train_label,valid_label)

    ### validation activation here can also be test activation.
    def _gradient_norm_attack(self,train_activation,valid_activation,all_train_label=[],all_valid_label=[]):
        length = train_activation.shape[0]
        if (train_activation.shape[1] == 0):
            return

        random_index = np.arange(length)
        train_activation = np.squeeze(train_activation)[random_index]
        valid_activation = np.squeeze(valid_activation)[random_index]

        confidence = np.concatenate((train_activation, valid_activation), axis=0)

        embedding = [np.linalg.norm(confidence[i]) for i in range(len(confidence))]
        embedding = np.array(embedding)
        embedding = np.reshape((embedding),(-1,1))
        #print ("activation norm debug")
        #print (embedding.shape)

        ### doing svm by using 50% as training

        train_train_index = np.random.choice(length, int(0.5 * length), replace=False)
        valid_train_index = np.random.choice(length, int(0.5 * length), replace=False) + length

        train_test_index = np.setdiff1d(np.arange(length), train_train_index)
        valid_test_index = np.setdiff1d(np.arange(length), valid_train_index - length) + length

        train_feature = np.concatenate((embedding[train_train_index], embedding[valid_train_index]), axis=0)
        train_label = np.concatenate((np.ones((int(0.5 * length))), np.zeros((int(0.5 * length)))))

        test_feature = np.concatenate((embedding[train_test_index], embedding[valid_test_index]), axis=0)
        test_label = np.concatenate((np.ones((int(0.5 * length))), np.zeros((int(0.5 * length)))))

        model = LogisticRegression(random_state=0, solver='lbfgs')
        model.fit(train_feature, train_label)
        print ("lr gradient norm whitebox attack accuracy %.2f" % (model.score(test_feature, test_label)*100))

        #from sklearn.svm import SVC
        #clf = SVC(gamma='auto')
        #clf.fit(train_feature, train_label)
        #print ("svm gradient norm whitebox attack accuracy %.2f" % (clf.score(test_feature, test_label)*100))
        #print (classification_report(train_label, clf.predict(train_feature)))
        #print (np.bincount(np.array(clf.predict(train_feature)).astype(np.int64)))
        #print (classification_report(test_label, clf.predict(test_feature)))
        #print (np.bincount(np.array(clf.predict(test_feature)).astype(np.int64)))


    def _activation_norm_attack(self,train_activation,valid_activation,all_train_label=[],all_valid_label=[]):

        length = train_activation.shape[0]
        random_index = np.arange(length)
        train_activation = np.squeeze(train_activation)[random_index]
        valid_activation = np.squeeze(valid_activation)[random_index]
        print ("activation norm debug:")
        print (train_activation.shape)
        confidence = np.concatenate((train_activation, valid_activation), axis=0)

        embedding = [np.linalg.norm(confidence[i]) for i in range(len(confidence))]
        embedding = np.array(embedding)
        embedding = np.reshape((embedding),(-1,1))


        # import matplotlib.pyplot as plt
        # figure = plt.figure(figsize=(10, 10))
        # plt.scatter(embedding[random_index, 0], embedding[random_index, 1], color='red', s=2)
        # plt.scatter(embedding[random_index + length, 0], embedding[random_index + length, 1], color='blue', s=2)
        # plt.show()

        ### doing svm by using 50% as training

        train_train_index = np.random.choice(length, int(0.5 * length), replace=False)
        valid_train_index = np.random.choice(length, int(0.5 * length), replace=False) + length

        train_test_index = np.setdiff1d(np.arange(length), train_train_index)
        valid_test_index = np.setdiff1d(np.arange(length), valid_train_index - length) + length

        print ("class distribution check:")
        print (np.bincount(all_train_label[train_train_index]))
        print (np.bincount(all_train_label[train_test_index]))
        print (np.bincount(all_valid_label[valid_train_index-length]))
        print (np.bincount(all_valid_label[valid_test_index-length]))

        train_feature = np.concatenate((embedding[train_train_index], embedding[valid_train_index]), axis=0)
        train_label = np.concatenate((np.ones((int(0.5 * length))), np.zeros((int(0.5 * length)))))

        test_feature = np.concatenate((embedding[train_test_index], embedding[valid_test_index]), axis=0)
        test_label = np.concatenate((np.ones((int(0.5 * length))), np.zeros((int(0.5 * length)))))

        model = LogisticRegression(random_state=0, solver='lbfgs')
        model.fit(train_feature, train_label)
        print ("activation norm whitebox attack accuracy %.2f" % (model.score(test_feature, test_label)*100))
        print (classification_report(train_label, model.predict(train_feature)))
        print (classification_report(test_label, model.predict(test_feature)))

    def _activation_umap_attack(self,train_activation,valid_activation,all_train_label=[],all_valid_label=[]):

        num_classes = np.unique(all_train_label)

        length = train_activation.shape[0]
        if (train_activation.shape[1] == 0):
            return

        #random_index = np.arange(length)
        #train_activation = np.squeeze(train_activation)[random_index]
        #valid_activation = np.squeeze(valid_activation)[random_index]

        #confidence = np.concatenate((train_activation, valid_activation), axis=0)
        #random_index = np.random.choice(len(confidence), len(confidence), replace=False)
        #confidence = confidence[random_index]

        #train_index = []
        #valid_index = []
        #for i in range(len(random_index)):
        #    if (random_index[i] < length):
        #        train_index.append(i)
        #    else:
        #        valid_index.append(i)

        #import umap
        #reducer = umap.UMAP(n_neighbors=10, min_dist=0.1)
        #embedding = reducer.fit_transform(confidence)

        ## should we do this in a per class fashion?
        ### doing svm by using 50% as training
        #train_train_index = np.random.choice(train_index, int(0.5 * len(train_index)), replace=False)
        #valid_train_index = np.random.choice(valid_index, int(0.5 * len(valid_index)), replace=False)
        #train_test_index = np.setdiff1d(train_index, train_train_index)
        #valid_test_index = np.setdiff1d(valid_index, valid_train_index)

        #train_feature = np.concatenate((embedding[train_train_index], embedding[valid_train_index]), axis=0)
        #train_label = np.concatenate((np.ones(len(train_train_index)), np.zeros((len(valid_train_index)))))

        #test_feature = np.concatenate((embedding[train_test_index], embedding[valid_test_index]), axis=0)
        #test_label = np.concatenate((np.ones(len(train_test_index)), np.zeros((len(valid_test_index)))))

        #from sklearn.svm import SVC
        #clf = SVC(gamma='auto')
        #clf.fit(train_feature, train_label)
        #print ("activation umap whitebox attack accuracy %.2f" % (clf.score(test_feature, test_label)*100))
        #print (classification_report(train_label, clf.predict(train_feature)))
        #print (classification_report(test_label, clf.predict(test_feature)))
        #
        overall_acc = 0
        for this_class in num_classes:
            print (this_class)
            this_class_index_train = np.arange(len(all_train_label))[all_train_label == int(this_class)]
            this_class_index_valid = np.arange(len(all_valid_label))[all_valid_label == int(this_class)]
            length = min(len(this_class_index_train),len(this_class_index_valid))
            # print ("whitebox activation umap debug")
            random_index = np.arange(length)
            this_train_activation = np.copy(train_activation[this_class_index_train[random_index]])
            this_valid_activation = np.copy(valid_activation[this_class_index_valid[random_index]])

            print (length)
            print (this_train_activation.shape)
            print (this_valid_activation.shape)

            confidence = np.concatenate((this_train_activation, this_valid_activation), axis=0)
            import umap

            reducer = umap.UMAP(n_neighbors=10, min_dist=0.1)
            embedding = reducer.fit_transform(confidence)
            print (embedding.shape)

            train_train_index = np.random.choice(length, int(0.5 * length), replace=False)
            valid_train_index = np.random.choice(length, int(0.5 * length), replace=False) + length

            train_test_index = np.setdiff1d(np.arange(length), train_train_index)
            valid_test_index = np.setdiff1d(np.arange(length), valid_train_index - length) + length

            print (len(train_train_index),len(valid_train_index),len(train_test_index),len(valid_test_index))

            train_feature = np.concatenate((embedding[train_train_index], embedding[valid_train_index]), axis=0)
            this_train_label = np.concatenate((np.ones((len(train_train_index))), np.zeros((len(valid_train_index)))))

            print (len(this_train_label),len(train_feature))

            test_feature = np.concatenate((embedding[train_test_index], embedding[valid_test_index]), axis=0)
            this_test_label = np.concatenate((np.ones((len(train_test_index))), np.zeros((len(valid_test_index)))))

            print (len(this_test_label),len(test_feature))

            from sklearn.svm import SVC
            clf = SVC(gamma='auto',kernel='rbf')
            clf.fit(train_feature, this_train_label)
            print ("class:",this_class)
            print ("activation umap whitebox attack accuracy %.2f" % (clf.score(test_feature, this_test_label) * 100))
            overall_acc+=clf.score(test_feature,this_test_label)

            #print(train_label.shape)

        print ("overall attack accuracy %.2f" %(overall_acc*100/10))

    def _gradient_umap_attack(self,train_activation,valid_activation,all_train_label=[],all_valid_label=[]):

        length = train_activation.shape[0]
        if (train_activation.shape[1] == 0):
            return

        random_index = np.arange(length)
        train_activation = np.squeeze(train_activation)[random_index]
        valid_activation = np.squeeze(valid_activation)[random_index]

        confidence = np.concatenate((train_activation, valid_activation), axis=0)
        import umap
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1)
        embedding = reducer.fit_transform(confidence)

        ### doing svm by using 50% as training

        train_train_index = np.random.choice(length, int(0.5 * length), replace=False)
        valid_train_index = np.random.choice(length, int(0.5 * length), replace=False) + length

        train_test_index = np.setdiff1d(np.arange(length), train_train_index)
        valid_test_index = np.setdiff1d(np.arange(length), valid_train_index - length) + length


        train_feature = np.concatenate((embedding[train_train_index], embedding[valid_train_index]), axis=0)
        train_label = np.concatenate((np.ones((int(0.5 * length))), np.zeros((int(0.5 * length)))))

        test_feature = np.concatenate((embedding[train_test_index], embedding[valid_test_index]), axis=0)
        test_label = np.concatenate((np.ones((int(0.5 * length))), np.zeros((int(0.5 * length)))))

        from sklearn.svm import SVC
        clf = SVC(gamma='auto')
        clf.fit(train_feature, train_label)
        print ("svm gradient umap whitebox attack accuracy %.2f" % (clf.score(test_feature, test_label)*100))
        print (classification_report(train_label, clf.predict(train_feature)))
        print (classification_report(test_label, clf.predict(test_feature)))

        #print (np.bincount(np.array(clf.predict(train_feature)).astype(np.int64)))
        #print (classification_report(test_label, clf.predict(test_feature)))
        #print (np.bincount(np.array(clf.predict(test_feature)).astype(np.int64)))
        #prediction = np.array(clf.predict(train_feature)).astype(np.int64)

        #for i in np.unique(all_train_label):
        #    this_class_index = np.arange(len(all_train_label))[all_train_label == i]
        #    this_class_prediction = prediction[this_class_index]
        #    print ("class ",i,":prediction count:",np.bincount(this_class_prediction))

        #for i in np.unique(all_valid_label):
        #    this_class_index = np.arange(len(all_valid_label))[all_valid_label == i]
        #    this_class_prediction = prediction[this_class_index]
        #    print ("class ",i,":prediction count:",np.bincount(this_class_prediction))

        #from sklearn.svm import SVC
        #clf = LogisticRegression(random_state=0, solver='saga')
        #clf.fit(train_feature, train_label)
        #print ("lr gradient umap whitebox attack accuracy %.2f" % (clf.score(test_feature, test_label)*100))
        #print (classification_report(train_label, clf.predict(train_feature)))
        #print (np.bincount(np.array(clf.predict(train_feature)).astype(np.int64)))
        #print (classification_report(test_label, clf.predict(test_feature)))
        #print (np.bincount(np.array(clf.predict(test_feature)).astype(np.int64)))


       # for i in np.unique(all_train_label):
       #     this_class_index = np.arange(len(all_train_label))[all_train_label == i]
       #     this_class_prediction = prediction[this_class_index]
       #     print ("class ",i,":prediction count:",np.bincount(this_class_prediction))

       # for i in np.unique(all_valid_label):
       #     this_class_index = np.arange(len(all_valid_label))[all_valid_label == i]
       #     this_class_prediction = prediction[this_class_index]
       #     print ("class ",i,":prediction count:",np.bincount(this_class_prediction))



def get_middle_gradient(model,train,training_size=10000,starting_index=[],layer_index=0,required_num=500):

    _supported_layers = ['Linear', 'Conv2d']

    print ("calculating middle gradient")

    dtype = torch.cuda.FloatTensor
    label_type = torch.cuda.LongTensor

    layer_list = list(model.children())
    num_layers = len(layer_list)

    middle_gradient_train = [[] for _ in range(num_layers)]

    train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True, num_workers=0)

    autograd_hacks.add_hooks(model)
    loss_fn = nn.CrossEntropyLoss().cuda()

    all_label = []

    for loss_index, (train_images, train_labels) in enumerate(train_loader):
        #print ("loss index %d " %(loss_index))
        if (100*(loss_index)>=required_num):
            #print (train_labels)
            break
            #return []
        ### 100 is the batch size

        model.zero_grad()
        train_images = Variable(train_images).type(dtype)
        train_labels = Variable(train_labels).type(label_type)
        output = model(train_images)
        loss = loss_fn(output, train_labels)
        loss.backward()
        autograd_hacks.compute_grad1(model)

        all_label.append(train_labels.data.cpu().numpy())

        for index,layer in enumerate(model.children()):
            if (autograd_hacks._layer_type(layer) in _supported_layers):
                for sec_index,param in enumerate(layer.parameters()):
                #this_gradient = param.grad1.data.cpu().numpy()
                    middle_gradient_train[index].append(param.grad1.data.cpu().numpy())
                    #print (param.shape)
                    #print (torch.sum(param))
                    #print (np.sum(param.grad1.data.cpu().numpy()))
                    #print (np.sum(param.grad.data.cpu().numpy()))
                    break
                    ### bias is not included

        autograd_hacks.clear_activation(model)
        autograd_hacks.clear_backprops(model)
        # param.grad: gradient averaged over the batch
        # param.grad1[i]: gradient with respect to example i

        ### check if computation is correct
        #for param in model.parameters():
            #print (param.shape)
            #print (param.grad1.shape)
            #print (torch.sum(torch.abs(param.grad-param.grad1.mean(dim=0))))
            #assert (torch.allclose(param.grad1.mean(dim=0), param.grad))

        ### another implementation could be [this would only give you the accumulated gradient, but not gradient for each one].
        #print ("testing another implementation")
        #output = model(train_images)
        #loss_fn = nn.CrossEntropyLoss(reduction='none').cuda()
        #loss = loss_fn(output,train_labels)
        #loss.backward(gradient=torch.ones_like(loss))
        #for param in model.parameters():
        #    print (param.grad.shape)

        #model.zero_grad()
        #torch.cuda.empty_cache()

    for index,layer in enumerate(model.children()):
        #print ("this layer:",index)
        #print (layer)
        count = 0
        for sec_index in range(len(middle_gradient_train[index])):
            #print (sec_index,len(middle_gradient_train[index][sec_index]))
            count += len(middle_gradient_train[index][sec_index])
            middle_gradient_train[index][sec_index] = np.reshape(middle_gradient_train[index][sec_index],(100,-1))
            #print (middle_gradient_train[index][sec_index].shape)

        #print (count)
        middle_gradient_train[index] = np.reshape(np.array(middle_gradient_train[index]),(required_num,-1))
        #print (middle_gradient_train[index].shape)

    model.zero_grad()
    autograd_hacks.clear_activation(model)
    autograd_hacks.clear_backprops(model)
    autograd_hacks.remove_hooks(model)

    all_label = np.reshape((np.array(all_label)),(-1))

    #print (np.bincount(all_label))

    return middle_gradient_train


def get_middle_gradient_one(model,train,starting_index=[],layer_index=0,training_size=10000,required_num=500):

    dtype = torch.cuda.FloatTensor
    label_type = torch.cuda.LongTensor
    all_label = []
    ## problem of layer-based iteration: nn.sequential module and you need to iterate through every layer inside.
    ## For resnet and densenet, each block cannot be accessed by LAYER. the only way is by PARAM.
    #layer_list = list(model.children())
    #new_layer_list = []
    #for layer in layer_list:
    #    if (isinstance(layer,nn.Sequential)):
    #        for this_layer in layer.children():
    #            new_layer_list.append(this_layer)
    #            print (len(new_layer_list),layer)
    #    else:
    #        new_layer_list.append(layer)
    #        print (len(new_layer_list),layer)
    #num_layers = len(new_layer_list)
    #print ("new_layer_list:")
    #for index,layer in enumerate(new_layer_list):
    #    print (index,layer)

    #middle_gradient_train = [[] for _ in range(len(new_layer_list))]
    count=0
    for param in model.parameters():
        #print (param.shape)
        if (len(param.shape)==4 or len(param.shape)==2):
        #if (len(param.shape) == 2):
            ### only record gradient for conv layer
            #print ("param get!")
            count+=1
    middle_gradient_train = [[] for _ in range(count)]
    #print ("overall layer num:",count)

    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False, num_workers=0)

    loss_fn = nn.CrossEntropyLoss().cuda()

    for loss_index, (train_images, train_labels) in enumerate(train_loader):

        #print ("loss index %d " %(loss_index))

        if (1*(loss_index)>=required_num):
            break

        ### 1 is the batch size
        model.zero_grad()
        train_images = Variable(train_images).type(dtype)
        train_labels = Variable(train_labels).type(label_type)
        output = model(train_images)
        loss = loss_fn(output, train_labels)
        loss.backward()

        all_label.append(train_labels.data.cpu().numpy())

        ## get gradient by param
        this_count = 0
        for param in model.parameters():
            #print (param.shape,param.grad.shape)
            if (len(param.shape) == 4 or len(param.shape)==2):
            #if (len(param.shape) ==2):
                #print (this_count)
                middle_gradient_train[this_count].append(param.grad.data.cpu().numpy())

                #print (param.shape)
                #print (torch.sum(param))
                #print (np.sum(param.grad.data.cpu().numpy()))
                #break
                this_count+=1

            #if (len(param.shape) == 2):
            #    if (loss_index%50==0):
            #        print (sum(sum(middle_gradient_train[this_count-1][-1])))

        ## get gradient by layer
        #for index,layer in enumerate(new_layer_list):
        #    if (hasattr(layer, 'weight')):
        #        print (layer.weight.shape)
        #        print (layer.weight.grad.shape)
        #        middle_gradient_train[index].append(layer.weight.grad)
        #torch.cuda.empty_cache()
        #break

        #model.zero_grad()
        #torch.cuda.empty_cache()

    for i in range(len(middle_gradient_train)):
        middle_gradient_train[i] = np.reshape((np.array(middle_gradient_train[i])),(required_num,-1))
        #print (middle_gradient_train[i].shape)
    #middle_gradient_train = np.array(middle_gradient_train)
    #print (middle_gradient_train.shape)

    all_label = np.reshape((np.array(all_label)),(-1))
    #print (np.bincount(all_label))
    #print (all_label)
    return middle_gradient_train


def get_middle_output(model,train,starting_index=[],layer_index=0,training_size=10000):
    print ("calculating middle output")

    dtype = torch.cuda.FloatTensor
    label_type = torch.cuda.LongTensor

    middle_output = []
    train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True, num_workers=0)

    middle_label = []

    #print (model)

    layer_list = []

    for layer in model.children():
        layer_list.append(layer)
        #print (len(layer_list),layer)

    for i in range(len(layer_list)-1):

        this_middle_output = []
        this_middle_label = []
        new_modules = layer_list[:i+1]
        new_model = nn.Sequential(*new_modules)

        ### weight check for the new models
        #for old_param,new_param in zip(model.parameters(),new_model.parameters()):
        #    print (old_param.shape)
        #    print (new_param.shape)
        #    print (torch.sum(old_param-new_param)
        ## weights are the same.
        ### new model check
        #print(new_model)

        count = 0
        for loss_index, (train_images, train_labels) in enumerate(train_loader):

            model.zero_grad()
            train_images = Variable(train_images).type(dtype)
            train_labels = Variable(train_labels).type(label_type)
            output = new_model(train_images)
            this_middle_output.append(output.data.cpu().numpy())

            #torch.cuda.empty_cache()

            this_middle_label.append(train_labels.data.cpu().numpy())

            #if (count == 0):
            #    print (this_middle_output[-1].shape)

            count+=this_middle_output[-1].shape[0]
            if (count>=training_size):
                break

            #print (train_labels[0])

        middle_output.append(np.reshape(np.array(this_middle_output),(training_size,-1)))
        middle_label.append(np.reshape(np.array(this_middle_label),(-1)))


    ### last layer shape mismatch for some strange reasons.

    this_middle_output = []
    this_middle_label = []
    count = 0
    acc = 0
    for loss_index, (train_images, train_labels) in enumerate(train_loader):

        model.zero_grad()
        torch.cuda.empty_cache()

        train_images = Variable(train_images).type(dtype)
        train_labels = Variable(train_labels).type(label_type)
        output = model(train_images)
        this_middle_output.append(output.data.cpu().numpy())
        this_middle_label.append(train_labels.data.cpu().numpy())

        _, predicted = torch.max(output, 1)
        acc += ((predicted == train_labels).sum()).item()

        if (count == 0):
            print (this_middle_output[-1].shape)

        ### this is to match the size for training and testing
        count += this_middle_output[-1].shape[0]
        if (count >= training_size):
            break

    #print ("accuracy %.2f" %(acc*100/training_size))

    middle_output.append(np.reshape(np.array(this_middle_output), (training_size, -1)))
    middle_label.append(np.reshape(np.array(this_middle_label),(-1)))

    #for i in range(len(middle_output)):
    #    print ("layer ",i)
    #    print (layer_list[i])
    #    print (middle_output[i].shape)

    #all_label = np.reshape((np.array(all_label)), (-1))
    #print (np.bincount(all_label))

    return middle_output,middle_label




