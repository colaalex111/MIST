
import torch.nn as nn
from data import *
import torch
import numpy as np
from utils import *
class test_net(nn.Module):

    def __init__(self):
        super(test_net, self).__init__()
        self.fc1 = nn.Linear(4,1)

    def forward(self,x):
        x = self.fc1(x)
        x = x.view(-1)
        return x


data = []
label = []
net = test_net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules




for i in range(1,9):
    data.append([i,i,i,i])
    label.append(i)

data = np.array(data)
label = np.array(label).astype(np.int64)
dataset = part_pytorch_dataset(data,label,train=False,transform=None,target_transform=None)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0, momentum=0)
loss_fc = nn.MSELoss(reduction='sum').to(device)

autograd_hacks.add_hooks(net)
net.to(device)

optimizer.zero_grad()

for index,(data,label) in enumerate(loader):
    print ("round %d" % index)
    data = data.to(device)
    label = label.to(device)
    output = net(data)
    #print (output.shape)
    #print (label.shape)
    loss = loss_fc(label,output)
    loss.backward()
    print ("after grad")
    for index, layer in enumerate(net.children()):
            for sec_index, param in enumerate(layer.parameters()):
                # this_gradient = param.grad1.data.cpu().numpy()
                #print ("weight",param.data)
                #print ("grad1",param.grad1)
                print ("true grad",param.grad)

net.zero_grad()

privacy_engine = PrivacyEngine(
        net,
        1,
        8, ### overall training set size
        alphas=[1 + x / 10.0 for x in range(1, 10)] + list(range(12, 64)), ### params for renyi dp
        noise_multiplier=1e-11, ### sigma
        max_grad_norm=1e10) ### this is from dp-sgd paper)
privacy_engine.attach(optimizer)


for index,(data,label) in enumerate(loader):
    print ("round %d" % index)
    data = data.to(device)
    label = label.to(device)
    output = net(data)
    #print (output.shape)
    #print (label.shape)
    loss = loss_fc(label,output)
    loss.backward()
    optimizer.privacy_engine.step()
    print("noise added grad")
    for index, layer in enumerate(net.children()):
            for sec_index, param in enumerate(layer.parameters()):
                # this_gradient = param.grad1.data.cpu().numpy()
                #print ("weight",param.data)
                #print ("grad1",param.grad1)
                print ("true grad",param.grad)

net.zero_grad()