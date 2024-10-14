import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class simple_mlp_attacknet(nn.Module):
	def __init__(self,input_shape,dem0,dem1,num_classes):
		super(simple_mlp_attacknet,self).__init__()
		self.num_classes = num_classes
		self.fc1 = nn.Linear(input_shape,dem0)
		self.fc2 = nn.Linear(dem0,dem1)
		self.fc3 = nn.Linear(dem1,num_classes)
	
	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		x = x.view(x.size(0), -1)
		#print(x.shape)
		x = self.fc3(x)
		if (self.num_classes == 1):
			x = F.sigmoid(x)
		return x

class simple_rnn_attacknet(nn.Module):
    def __init__(self,data_size,hidden_size,output_size):
        super(simple_rnn_attacknet,self).__init__()
        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data, last_hidden):
        ### last hidden needs to be initialized
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return hidden, output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class simple_cnn_attacknet(nn.Module):
    def __init__(self,in_channel,dem0,num_classes,kernel_size):
        super(simple_cnn_attacknet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=5,kernel_size=(1,kernel_size))
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=5,out_channels=10,kernel_size=(1,kernel_size))
        self.activation2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=10,out_channels=15,kernel_size=(1,kernel_size))
        self.activation3 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dem0,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = x.view(x.size(0),-1)
        #print (x.size())
        x = self.fc1(x)
        return x


class onelayer_AttackNet(nn.Module):

    def __init__(self, dem):
        super(onelayer_AttackNet, self).__init__()
        self.fc1 = nn.Linear(dem, 64)
        self.fc2 = nn.Linear(64,2)

    def forward(self, x1):
        x1 = x1.view(x1.size(0), -1)
        #print (x1.size())
        out = self.fc1(x1)
        out = F.relu(out)
        #print (out.size())
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out


class oakland_whitebox_attacknet_fcgradient(nn.Module):

    def __init__(self,dim):
        dim1 = int(dim[0])
        dim2 = int(dim[1])
        super(oakland_whitebox_attacknet_fcgradient,self).__init__()
        self.drop1 = nn.Dropout(p=0.2,inplace=True)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=100,kernel_size=(1,dim2),stride=1)
        self.activation1 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p=0.2,inplace=True)
        self.fc1 = nn.Linear(1,2024)
        self.activation2 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(p=0.2,inplace=True)
        self.fc2 = nn.Linear(2024,512)
        self.activation3 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512,256)
        self.activation4 = nn.ReLU(inplace=True)

        #### layers above are to process the gradient of fc layer

        self.fc4 = nn.Linear(256,256)
        self.activation5 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(256,128)
        self.activation6 = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(128,64)
        self.activation7 = nn.ReLU(inplace=True)
        self.fc7 = nn.Linear(64,1)
        self.activation8 = nn.ReLU(inplace=True)

        #### layers above are the components of the encoder network

    def forward(self,x):
        x = self.drop1(x)
        x = self.conv1(x)
        x = self.activation1(x)
        x = x.view(x.size(0),-1)
        x = self.drop2(x)
        x = self.fc1(x)
        x = self.activation2(x)
        x = self.drop3(x)
        x = self.fc2(x)
        x = self.activation3(x)
        x = self.fc3(x)
        x = self.activation4(x)
        x = self.fc4(x)
        x = self.activation5(x)
        x = self.fc5(x)
        x = self.activation6(x)
        x = self.fc6(x)
        x = self.activation7(x)
        x = self.fc7(x)
        x = self.activation8(x)
        return x
    ### here we only use the gradient wrt the last layer as input and use a CNN to process the data
    ###

    class gan_AttackNet(nn.Module):

        def __init__(self, dem):
            super(gan_AttackNet, self).__init__()
            self.fc1 = nn.Linear(dem, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 64)

            self.fc4 = nn.Linear(dem, 512)
            self.fc5 = nn.Linear(512, 64)

            self.fc6 = nn.Linear(128, 64)
            self.fc7 = nn.Linear(64, 2)

        def forward(self, x1, x2):
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)

            out1 = self.fc1(x1)
            out1 = F.relu(out1)
            out1 = self.fc2(out1)
            out1 = F.relu(out1)
            out1 = self.fc3(out1)
            out1 = F.relu(out1)

            out2 = self.fc4(x2)
            out2 = F.relu(out2)
            out2 = self.fc5(out2)
            out2 = F.relu(out2)

            out = torch.cat((out1, out2), dim=1)

            out = self.fc6(out)
            out = F.relu(out)
            out = self.fc7(out)
            out = F.softmax(out, dim=1)

            return out
        ## TYPE ONE
