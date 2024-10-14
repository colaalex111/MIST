import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
from data import part_pytorch_dataset
from model import *

class blackbox_attack:

    def __init__(self,membership_attack_number,name='baseline_attack',num_classes=10):

        np.set_printoptions(suppress=True)
        self.name = name
        self.num_classes = num_classes
        self.membership_attack_number = membership_attack_number


    def attack(self,total_confidences,total_classes,total_labels,output_file,avg_loss=0):

        if (self.name == 'baseline'):
            self._baseline_attack(total_confidences,total_classes,total_labels)
        if (self.name == 'avg_loss'):
            self.avg_loss = avg_loss
            self._avg_loss_attack(total_confidences,total_classes,total_labels)
        if (self.name == 'top1'):
            self._top1_attack(total_confidences,total_classes,total_labels)
        if (self.name == 'top3'):
            self._top3_attack(total_confidences,total_classes,total_labels)
        if (self.name == 'global_prob'):
            return self._global_prob_attack(total_confidences,total_classes,total_labels,output_file)
        if (self.name == 'per_class'):
            self._per_class_attack(total_confidences,total_classes,total_labels)
        if (self.name == 'instance_distance'):
            self._instance_distance_attack(total_confidences,total_classes,total_labels)
        if (self.name == 'instance_prob'):
            self._instance_prob_attack(total_confidences,total_classes,total_labels)

    def get_attack_input(self,model,user):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        train_iter = torch.utils.data.DataLoader(user.evaluation_member_dataset,batch_size=100,shuffle=True)
        test_iter = torch.utils.data.DataLoader(user.evaluation_non_member_dataset,batch_size=100,shuffle=True)

        total_confidences = []
        total_classes = []
        total_labels = []

        for images,labels,_ in train_iter:
            images = images.to(device)
            outputs = model(images)
            labels = labels.to(device)
            total_confidences.append(outputs.detach().cpu().numpy())
            total_classes.append(labels.detach().cpu().numpy())
            #print (labels.size())
            total_labels.append(np.ones((labels.size(0))))

        for images,labels,_ in test_iter:
            images = images.to(device)
            outputs = model(images)
            labels = labels.to(device)
            total_confidences.append(outputs.detach().cpu().numpy())
            total_classes.append(labels.detach().cpu().numpy())
            #print (labels.size())
            total_labels.append(np.zeros((labels.size(0))))

        total_confidences = np.array(total_confidences)
        total_confidences = np.reshape(total_confidences,(-1,total_confidences.shape[-1]))
        #print (total_confidences.size())
        total_classes = np.hstack(total_classes)
        #print (total_classes.size())
        total_labels = np.hstack(total_labels)
        #print (total_labels.size())

        #print (total_confidences.shape,total_classes.shape,total_labels.shape)
        #print (total_confidences[0],np.sum(total_confidences[0]),total_classes[0],total_labels[0])

        return total_confidences,total_classes,total_labels

    def _baseline_attack(self,total_confidences,total_classes,total_labels):

        reshaped_classes = total_classes
        reshaped_classes = np.reshape(reshaped_classes, (-1))
        features = np.reshape(total_confidences, (-1,self.num_classes))
        features = np.argmax(features, axis=1)
        corr = [features[i] == reshaped_classes[i] for i in range(len(reshaped_classes))]
        corr = np.array(corr)
        labels = np.reshape(total_labels,(-1))
        acc = 0
        for i in range(len(reshaped_classes)):
            if (corr[i] == 1 and labels[i] == 1):
                acc += 1
            if (corr[i] == 0 and labels[i] == 0):
                acc += 1

        acc = acc / len(reshaped_classes)
        print ("baseline attack acc = %.2f" % (acc*100))
        print (classification_report(labels, corr))

    def _avg_loss_attack(self,total_confidences,total_classes,total_labels):

        reshaped_classes = (total_classes.copy()).astype(np.int64)
        reshaped_classes = np.reshape(reshaped_classes, (-1))
        total_num = len(reshaped_classes)
        train_indices = np.random.choice(total_num,int(total_num / 2), replace=False)
        test_indices = np.setdiff1d(np.arange(total_num), train_indices)
        features = np.reshape(total_confidences, (total_num,self.num_classes))
        features = [features[i, reshaped_classes[i]] for i in range(features.shape[0])]
        features = np.array(features)
        features = np.log(features) * -1
        features = np.nan_to_num(features)
        labels = np.reshape(total_labels,(-1))
        # shadow_avg_loss = np.average(total_avg_loss)/(args.target_data_size*args.model_number/2)
        #shadow_avg_loss = np.average(total_avg_loss)
        shadow_avg_loss = self.avg_loss
        print ("shadow avg loss = %.2f " % (shadow_avg_loss))
        corr = 0
        predict = np.zeros((len(labels)))
        for i in range(len(labels)):
            predict[i] = (shadow_avg_loss > features[i])
            if (predict[i] == 1 and labels[i] == 1):
                corr += 1
            if (predict[i] == 0 and labels[i] == 0):
                corr += 1
        print ("avg loss acc %.2f" % (corr*100 / len(labels)))
        print (classification_report(labels, predict))

    def _top1_attack(self,total_confidences,total_classes,total_labels):

        reshaped_classes = (total_classes.copy()).astype(np.int64)
        reshaped_classes = np.reshape(reshaped_classes, (-1))
        total_num = len(reshaped_classes)
        train_indices = np.random.choice(total_num, int(total_num / 2), replace=False)
        test_indices = np.setdiff1d(np.arange(total_num), train_indices)
        features = np.reshape(total_confidences, (total_num, -1))

        train_features = np.amax(features[train_indices], axis=1)
        train_features = np.reshape(train_features, (-1, 1))
        test_features = np.amax(features[test_indices], axis=1)
        test_features = np.reshape(test_features, (-1, 1))
        train_features = np.nan_to_num(train_features)
        test_features = np.nan_to_num(test_features)

        labels = np.reshape(total_labels, (-1))
        labels = np.nan_to_num(labels)
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]

        model = LogisticRegression(random_state=0, solver='lbfgs')
        model.fit(train_features, train_labels)
        print ("lr on global highest accuracy = %f " % (model.score(test_features, test_labels)*100))
        #print (classification_report(test_labels, model.predict(test_features)))

    def _top3_attack(self,total_confidences,total_classes,total_labels):

        reshaped_classes = (total_classes.copy()).astype(np.int64)
        reshaped_classes = np.reshape(reshaped_classes, (-1))
        total_num = len(reshaped_classes)
        train_indices = np.random.choice(total_num,int(total_num / 2), replace=False)
        test_indices = np.setdiff1d(np.arange(total_num), train_indices)
        features = np.reshape(total_confidences, (total_num, -1))
        features = np.sort(features, axis=1)
        print ("top3 attack", features[0])
        features = features[:, -3:]

        #print (features[0])
        # print ("top3 attack feature shape",features.shape)
        # print ("top3 attack",features[0])
        #print (features.shape)

        train_features = features[train_indices]
        test_features = features[test_indices]

        labels = np.reshape(total_labels, (-1))

        train_labels = labels[train_indices]
        test_labels = labels[test_indices]

        model = LogisticRegression(random_state=0, solver='lbfgs')
        model.fit(train_features, train_labels)
        print ("lr on global top 3 accuracy = %f " % (model.score(test_features, test_labels)*100))
        y_pred = model.predict(test_features)
        #print (classification_report(test_labels, y_pred))

        ### NN top3 attack
        ### similar acc as LR top3 attack
        ### uncomment if you want to use

        nn_top3_acc = self._nn_attack(train_features, train_labels, test_features, test_labels)
        print ("NN on global top 3 accuracy = %f" % (nn_top3_acc*100))

    def _global_prob_attack(self,total_confidences,total_classes,total_labels,output_file):
        #### sometimes using Logistic regression solver cannot give the best threshold, remember to try manual threshold
        reshaped_classes = (total_classes.copy()).astype(np.int64)
        reshaped_classes = np.reshape(reshaped_classes, (-1))
        total_num = len(reshaped_classes)
        train_indices = np.random.choice(total_num,int(total_num / 2), replace=False)
        test_indices = np.setdiff1d(np.arange(total_num), train_indices)
        features = np.reshape(total_confidences, (total_num, -1))
        features = [features[i, reshaped_classes[i]] for i in range(features.shape[0])]
        features = np.array(features)
        features = np.nan_to_num(features)
        train_features = features[train_indices]
        train_features = np.reshape(train_features, (-1, 1))
        test_features = features[test_indices]
        test_features = np.reshape(test_features, (-1, 1))
        labels = np.reshape(total_labels, (-1))
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]

        model = LogisticRegression(random_state=0, solver='saga')

        ### train features might contain NAN, thus
        train_features = np.nan_to_num(train_features)
        model.fit(train_features, train_labels)

        if (output_file!=None):
            output_file.write("lr on global class label accuracy = %f \n " % (model.score(test_features, test_labels)*100))
        else:
            print("lr on global class label accuracy = %f \n " % (model.score(test_features, test_labels)*100))
        
        return model.score(test_features, test_labels)*100

    def _per_class_attack(self,total_confidences,total_classes,total_labels):

        per_class_acc = 0.0
        nn_per_class_acc = 0.0
        per_class_avg_precision = 0.0
        per_class_avg_recall = 0.0
        valid_per_class_acc = 0.0
        # per class test
        for i in range(self.num_classes):
            class_indices = np.arange(self.membership_attack_number)[total_classes[0, :] == i]
            this_class_confidence = total_confidences[:, class_indices]
            this_label = total_labels[:, class_indices]

            # print ("sanity check for per class")
            # print (this_class_confidence.shape)
            # print (this_classes.shape)
            # print (this_label.shape)

            this_class_number = len(class_indices)
            train_indices = np.random.choice(this_class_number, int(this_class_number / 2), replace=False)
            test_indices = np.setdiff1d(np.arange(this_class_number), train_indices)
            features = np.reshape(this_class_confidence, (-1,self.num_classes))
            this_label = this_label.flatten()
            train = features[train_indices]
            train_label = this_label[train_indices]
            test = features[test_indices]
            test_label = this_label[test_indices]

            ### LR per class test
            model = LogisticRegression(random_state=0, solver='lbfgs')
            model.fit(train, train_label)
            lr_perclass_confidence_acc = model.score(test, test_label)

            #### NEED to calculate avg precision and recall
            this_average_precision = precision_score(test_label, model.predict(test), average='macro')
            per_class_avg_precision += this_average_precision
            this_average_recall = recall_score(test_label, model.predict(test), average='macro')
            per_class_avg_recall += this_average_recall

            ### one layer NN per class test
            nn_perclass_confidence_acc = self._nn_attack(train, train_label, test, test_label)

            per_class_acc += lr_perclass_confidence_acc
            nn_per_class_acc += nn_perclass_confidence_acc

        print ("total per class lr acc = %f" % (per_class_acc * 100 / self.num_classes))
        print ("total per class nn acc = %f" % (nn_per_class_acc * 100 / self.num_classes))
        print ("total per class lr precision = %f" % (per_class_avg_precision * 100 / self.num_classes))
        print ("total per class lr recall = %f" % (per_class_avg_recall * 100 / self.num_classes))

    def _instance_distance_attack(self,total_confidences,total_classes,total_labels):


        total_distance_acc = 0.0
        total_dist_precision = 0.0
        total_dist_recall = 0.0

        for i in range(self.membership_attack_number):
            this_confidence = total_confidences[:, i]
            this_label = total_labels[:, i]
            this_class = total_classes[:, i]

            #in_count = np.count_nonzero(this_label)
            # print (this_confidence.shape)
            # print ("for instance %d: we have %d in training, %d not in training" %(i,in_count,args.model_number - in_count))

            distance_acc,dist_precision,dist_recall = self._instance_attack_membership('distance',this_confidence, this_label, this_class)
            total_distance_acc += distance_acc
            total_dist_precision += dist_precision
            total_dist_recall += dist_recall

        print ('total distance confidence = %f' % (total_distance_acc * 100 / self.membership_attack_number))
        print ("total distance precision = %f" % (total_dist_precision * 100 / self.membership_attack_number))
        print ("total distance recall = %f" % (total_dist_recall * 100 / self.membership_attack_number))

    def _instance_prob_attack(self,total_confidences,total_classes,total_labels):

        total_ratio_acc = 0.0
        total_ratio_precision = 0.0
        total_ratio_recall = 0.0

        for i in range(self.membership_attack_number):
            this_confidence = total_confidences[:, i]
            this_label = total_labels[:, i]
            this_class = total_classes[:, i]

            #in_count = np.count_nonzero(this_label)
            # print (this_confidence.shape)
            # print ("for instance %d: we have %d in training, %d not in training" %(i,in_count,args.model_number - in_count))

            ratio_acc,ratio_precision,ratio_recall = self._instance_attack_membership('prob',this_confidence,this_label,this_class)

            total_ratio_acc += ratio_acc
            total_ratio_precision += ratio_precision
            total_ratio_recall += ratio_recall

        print ("total ratio confidence = %f" % (total_ratio_acc * 100 / self.membership_attack_number))
        print ("total ratio precision = %f" % (total_ratio_precision * 100 / self.membership_attack_number))
        print ("total ratio recall = %f" % (total_ratio_recall * 100 / self.membership_attack_number))


    def _instance_attack_membership(self,attack_name,confidence,label,class_label):

        ###
        model_number = confidence.shape[0]

        ### show total in and out
        total_in = np.count_nonzero(label)
        total_out = len(label) - total_in
        if (total_in == 0 or total_out == 0):
            return [0.5] * 3

        #### generate train / test indices
        train_indices = np.random.choice(model_number, int(1 * model_number / 2), replace=False)
        test_indices = np.setdiff1d(np.arange(model_number), train_indices)
        train_label = label[train_indices]
        test_label = label[test_indices]
        train_1_count = np.count_nonzero(train_label)
        train_0_count = len(train_indices) - train_1_count

        test_1_count = np.count_nonzero(test_label)
        test_0_count = len(test_indices) - test_1_count

        cnt = 0

        while (train_0_count == 0 or train_1_count == 0 or test_0_count == 0 or test_1_count == 0):
            train_indices = np.random.choice(model_number, int(1 * model_number / 2), replace=False)
            test_indices = np.setdiff1d(np.arange(model_number), train_indices)
            train_label = label[train_indices]
            test_label = label[test_indices]
            train_1_count = np.count_nonzero(train_label)
            train_0_count = len(train_indices) - train_1_count

            test_1_count = np.count_nonzero(test_label)
            test_0_count = len(test_indices) - test_1_count

            cnt += 1
            if (cnt > 200):
                return [0.5] * 3


        if (attack_name == 'distance'):
            train_in_indices = []
            train_out_indices = []
            for i in range(len(train_indices)):
                if (label[train_indices[i]] == 1):
                    train_in_indices.append(train_indices[i])
                else:
                    train_out_indices.append(train_indices[i])
            train_indices = np.sort(train_indices)
            train_in_indices.sort()
            train_out_indices.sort()
            train_in_conf = confidence[train_in_indices]
            train_out_conf = confidence[train_out_indices]

            in_avg_conf = np.average(train_in_conf, axis=0)
            out_avg_conf = np.average(train_out_conf, axis=0)

            corr = 0
            import scipy
            from scipy.stats import entropy
            y_pred = np.zeros((len(test_indices)))
            for i in range(len(test_indices)):
                in_distance = scipy.stats.entropy(in_avg_conf, confidence[test_indices[i]])
                out_distance = scipy.stats.entropy(out_avg_conf, confidence[test_indices[i]])
                y_pred[i] = (in_distance < out_distance)
                if (in_distance < out_distance and label[test_indices[i]] == 1):
                    corr += 1
                if (out_distance < in_distance and label[test_indices[i]] == 0):
                    corr += 1
            distance_acc = corr * 1.0 / len(test_indices)
            dist_precision = precision_score(label[test_indices], y_pred, average='macro')
            dist_recall = recall_score(label[test_indices], y_pred, average='macro')

            return distance_acc,dist_precision,dist_recall

        if (attack_name == 'prob'):
            corr_class = int(class_label[0])
            features = confidence[:, corr_class]
            features = np.log(0.01 + features) - np.log(1.01 - features)
            features = np.reshape(features, (model_number, -1))
            train = features[train_indices]
            train_label = label[train_indices]
            test = features[test_indices]
            test_label = label[test_indices]
            model = LogisticRegression(random_state=0, solver='saga')
            model.fit(train, train_label)
            ratio_confidence_acc = model.score(test, test_label)
            precision = precision_score(test_label, model.predict(test), average='macro')
            recall = recall_score(test_label, model.predict(test), average='macro')

            return ratio_confidence_acc,precision,recall

        pass

    def _nn_attack(self,train, train_label, test, test_label):

        dem = train.shape[1]
        total_num = train.shape[0]
        test_total_num = test.shape[0]

        train = np.reshape(train, (total_num, 1, 1, dem))
        test = np.reshape(test, (test_total_num, 1, 1, dem))

        train = part_pytorch_dataset(train, train_label, train=True, transform=transforms.ToTensor())
        test = part_pytorch_dataset(test, test_label, train=False, transform=transforms.ToTensor())

        epochs = 100

        attack_model = onelayer_AttackNet(dem=dem) ## check the model details in model dir

        dtype = torch.FloatTensor
        label_type = torch.LongTensor
        criterion = nn.CrossEntropyLoss()

        attack_model.type(dtype)

        optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01, weight_decay=1e-7)

        train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)

        for epoch in range(epochs):
            for images, labels in train_loader:
                images = Variable(images).type(dtype)
                labels = Variable(labels).type(label_type)
                optimizer.zero_grad()
                outputs = attack_model(images)
                loss = criterion(outputs, labels)
                total_loss = loss
                total_loss.backward()
                optimizer.step()

        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images).type(dtype)
            outputs = attack_model(images)
            labels = labels.type(label_type)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        acc = correct.item() * 1.0
        acc = acc / total
        testing_acc = acc

        return testing_acc
