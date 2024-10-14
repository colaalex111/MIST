'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import torch

__all__ = ['alexnet']

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 =  nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.activation1 =  nn.ReLU(inplace=True)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.activation2 = nn.ReLU(inplace=True)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.activation3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.activation4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.activation5 = nn.ReLU(inplace=True)
        self.maxpooling5  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.maxpooling2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.conv4(x)
        x = self.activation4(x)
        x = self.conv5(x)
        x = self.activation5(x)
        x = self.maxpooling5(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return x


def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model
class alexnet_tinyimagenet(nn.Module):
	def __init__(self, n_class):
		super(alexnet_tinyimagenet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=2)
		self.act1 = nn.ReLU(inplace=True)
		self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
		self.act2 = nn.ReLU(inplace=True)
		self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
		self.act3 = nn.ReLU(inplace=True)
		self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
		self.act4 = nn.ReLU(inplace=True)
		self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.act5 = nn.ReLU(inplace=True)
		self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)

		self.drop1 = nn.Dropout()
		self.fc1 = nn.Linear(256 * 6 * 6, 4096)
		self.act6 = nn.ReLU(inplace=True)
		self.drop2 = nn.Dropout()
		self.fc2 = nn.Linear(4096, 4096)
		self.act7 = nn.ReLU(inplace=True)
		self.fc3 = nn.Linear(4096, n_class)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.act1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.act2(x)
		x = self.maxpool2(x)
		x = self.conv3(x)
		x = self.act3(x)
		x = self.conv4(x)
		x = self.act4(x)
		x = self.conv5(x)
		x = self.act5(x)
		x = self.maxpool5(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.drop1(x)
		x = self.fc1(x)
		x = self.act6(x)
		x = self.drop2(x)
		x = self.fc2(x)
		x = self.act7(x)
		x = self.fc3(x)
		return x