import torch 
import torch.nn as nn
import torch.nn.functional as F
class ShadowNet(nn.Module):

    def __init__(self,dataset,input_feature_number=0,output_feature_number=0):
        super(ShadowNet, self).__init__()
        self.dataset = dataset
        self.input_feature_number = input_feature_number
        self.output_feature_number = output_feature_number

        if (self.dataset == 'cifar100'):
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                #nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                #nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.fc1 = nn.Linear(8*8*64, 128)
            self.fc2 = nn.Linear(128,100)

        if (self.dataset == 'cifar10'):
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                #nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                #nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.fc1 = nn.Linear(8*8*64, 128)
            self.fc2 = nn.Linear(128,10)

        if (self.dataset == 'mnist'):
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                #nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                #nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.fc1 = nn.Linear(7*7*64, 128)
            self.fc2 = nn.Linear(128,10)

        if (self.dataset == 'adult'):
            self.fc1 = nn.Linear(input_feature_number,128)
            self.fc2 = nn.Linear(128,output_feature_number)
        
        if (self.dataset == 'texas'):
            self.fc1 = nn.Linear(input_feature_number,128)
            self.fc2 = nn.Linear(128,output_feature_number)

        if (self.dataset == 'titanic'):
            self.fc1 = nn.Linear(input_feature_number,128)
            self.fc2 = nn.Linear(128,output_feature_number)
        
    def forward(self, x):
        
        if (self.dataset == 'mnist' or self.dataset == 'cifar10' or self.dataset == 'cifar100'):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)
            out = F.log_softmax(out, dim=1)

        if (self.dataset == 'adult' or self.dataset == 'texas' or self.dataset == 'titanic'):
            out = x.view(x.size(0),-1)
            out = self.fc1(out)
            out = self.fc2(out)
            out = F.log_softmax(out,dim=1)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
