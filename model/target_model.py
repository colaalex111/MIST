import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AlexNet_large(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class mnist_special(nn.Module):
	def __init__(self,dataset):
		super(mnist_special,self).__init__()
		self.dataset = dataset
		self.flatten = nn.Flatten()
		self.linear1 = nn.Linear(28*28,10)
	
	def forward(self,x):
		x = self.flatten(x)
		x = self.linear1(x)
		return x


class retina_special(nn.Module):
	
	def __init__(self, test_rank=0):
		super(retina_special, self).__init__()
		
		self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
		self.activation1 = nn.ReLU(inplace=True)
		self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
		self.activation2 = nn.ReLU(inplace=True)
		self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
		self.activation3 = nn.ReLU(inplace=True)
		self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
		self.activation4 = nn.ReLU(inplace=True)
		self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.activation5 = nn.ReLU(inplace=True)
		self.maxpooling5 = nn.MaxPool2d(kernel_size=3, stride=2)
		
		self.linear1 = nn.Linear(256, test_rank)
		self.linear2 = nn.Linear(test_rank, 4)
	
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
		
		x = torch.flatten(x, 1)
		#print (x.size())
		
		x = self.linear1(x)
		out = self.linear2(x)
		return out

class TargetNet(nn.Module):
	
	def __init__(self,dataset,input_feature_number=0,output_feature_number=0):
		super(TargetNet, self).__init__()
		self.dataset = dataset
		self.input_feature_number = input_feature_number
		self.output_feature_number = output_feature_number
		
		if (self.dataset == 'location'):
			self.features = nn.Sequential(
				nn.Linear(446, 1024),
				nn.Tanh(),
				nn.Linear(1024, 512),
				nn.Tanh(),
				nn.Linear(512, 256),
				nn.Tanh(),
				nn.Linear(256, 30)
			)
		
		if (self.dataset == 'fashion_mnist'):
			self.conv1 = nn.Conv2d(1,32,kernel_size=5,padding=2)
			self.act1 = nn.ReLU()
			self.max1 =  nn.MaxPool2d(2)
			self.conv2 =  nn.Conv2d(32,64, kernel_size=5, padding=2)
			self.act2 = nn.ReLU()
			self.max2 =  nn.MaxPool2d(2)
			self.flatten = nn.Flatten()
			self.fc1 = nn.Linear(3136,100)
			self.fc2 = nn.Linear(100,10)
			
		
		if (self.dataset == 'medical_mnist'):
			self.conv1 =  nn.Conv2d(3, 32, kernel_size=5, padding=2)
			self.act1 = nn.ReLU()
			self.max1 =  nn.MaxPool2d(2)
			self.conv2 =  nn.Conv2d(32, 64, kernel_size=5, padding=2)
			self.act2 = nn.ReLU()
			self.max2 =  nn.MaxPool2d(2)
			self.flatten = nn.Flatten()
			self.fc1 = nn.Linear(8*8*64, 120)
			self.fc2 = nn.Linear(120,84)
			self.fc3 = nn.Linear(84,6)
		
		
		if (self.dataset == 'kidney'):
			self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
			self.activation1 = nn.ReLU(inplace=True)
			self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
			self.activation2 = nn.ReLU(inplace=True)
			self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
			self.activation3 = nn.ReLU(inplace=True)
			self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
			self.activation4 = nn.ReLU(inplace=True)
			self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
			self.activation5 = nn.ReLU(inplace=True)
			self.maxpooling5 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.linear1 = nn.Linear(256,4)
		
		if (self.dataset == 'skin'):
			self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
			self.activation1 = nn.ReLU(inplace=True)
			self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
			self.activation2 = nn.ReLU(inplace=True)
			self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
			self.activation3 = nn.ReLU(inplace=True)
			self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
			self.activation4 = nn.ReLU(inplace=True)
			self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
			self.activation5 = nn.ReLU(inplace=True)
			self.maxpooling5 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.linear1 = nn.Linear(256,23)
		
		if (self.dataset == 'chest' or self.dataset == 'covid'):
			self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
			self.activation1 = nn.ReLU(inplace=True)
			self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
			self.activation2 = nn.ReLU(inplace=True)
			self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
			self.activation3 = nn.ReLU(inplace=True)
			self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
			self.activation4 = nn.ReLU(inplace=True)
			self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
			self.activation5 = nn.ReLU(inplace=True)
			self.maxpooling5 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.linear1 = nn.Linear(256,2)
		
		if (self.dataset == 'retina'):
			self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
			self.activation1 = nn.ReLU(inplace=True)
			self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
			self.activation2 = nn.ReLU(inplace=True)
			self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
			self.activation3 = nn.ReLU(inplace=True)
			self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
			self.activation4 = nn.ReLU(inplace=True)
			self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
			self.activation5 = nn.ReLU(inplace=True)
			self.maxpooling5 = nn.MaxPool2d(kernel_size=3, stride=2)
			self.linear1 = nn.Linear(256,4)
		
		if (self.dataset == 'purchase' or self.dataset == 'texas'):
			self.fc1 = nn.Linear(input_feature_number,1024)
			self.fc2 = nn.Linear(1024,512)
			self.fc3 = nn.Linear(512,256)
			self.fc4 = nn.Linear(256,100)
		
	
	def forward(self, x):
			
		if (self.dataset == 'medical_mnist'):
			out = self.conv1(x)
			out = self.act1(out)
			out = self.max1(out)
			out = self.conv2(out)
			out = self.act2(out)
			out = self.max2(out)
			out = self.flatten(out)
			out = self.fc1(out)
			out = self.fc2(out)
			out = self.fc3(out)
			return out

		if (self.dataset == 'fashion_mnist'):
			out = self.conv1(x)
			out = self.act1(out)
			out = self.max1(out)
			out = self.conv2(out)
			out = self.act2(out)
			out = self.max2(out)
			out = self.flatten(out)
			#print (out.shape)
			out = self.fc1(out)
			#print (out.shape)
			out = self.fc2(out)
			return out
		
		if (self.dataset == 'purchase' or self.dataset == 'texas'):
			out = x.view(x.size(0),-1)
			out = self.fc1(out)
			out = F.tanh(out)
			out = self.fc2(out)
			out = F.tanh(out)
			out = self.fc3(out)
			out = F.tanh(out)
			out = self.fc4(out)
			out = F.tanh(out)
			return out
		
		if (self.dataset == 'retina' or self.dataset =='chest' or self.dataset =='skin' or self.dataset =='kidney' or self.dataset =='covid'):
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
			x = torch.flatten(x,1)
			x = self.linear1(x)
			return x
		
		if (self.dataset == 'location'):
			hidden_out = self.features(x)
			return hidden_out