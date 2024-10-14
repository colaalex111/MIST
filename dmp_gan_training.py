import model
from blackbox_attack import *
import argparse
from data import dataset
from model import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from user import *
from data import *
from tqdm import tqdm
import copy
from opacus import PrivacyEngine
from model_utils import *
from model_utils import _batchnorm_to_groupnorm_new
from model_utils import get_train_loss
from opacus.validators import ModuleValidator
from worst_case_metric import find_vulnerable_points
from sklearn.metrics import roc_auc_score
import math
from model_utils import _ECELoss
from decisionboundaryattack import DecisionBlackBoxAttack
from hsja import HSJAttack
import torchvision.datasets as datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nc=3
# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
#number of discriminator filters
ndf = 64

#target_dataset = dataset(dataset_name='cifar100')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
		
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output


import time
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

start_time = time.time()
elapsed_time = 0

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
#load weights to test the model
#netG.load_state_dict(torch.load(args.gan_path))

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
#load weights to test the model
#netD.load_state_dict(torch.load('weights/netD_epoch_24.pth'))
print(netD)

criterion = nn.BCELoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(128, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

niter = 100 #100
g_loss = []
d_loss = []
BATCH_SIZE = 128

import datetime

#loading the dataset
trainset = datasets.CIFAR100(root="./data", download=True,  train=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
testset = datasets.CIFAR100(root="./data", download=True,  train=False,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataset = torch.utils.data.ConcatDataset([trainset, testset])


private_data_set = torch.utils.data.Subset(dataset, np.arange(20000))

dataloader = torch.utils.data.DataLoader(private_data_set, batch_size=256,
                                            shuffle=False, num_workers=4)


start = datetime.datetime.now()

for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if(i==len(dataloader)-1):
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        '''
        #save the output
        if i % 100 == 0:
            print('saving the output')
            vutils.save_image(real_cpu,'output/cifar100_real_samples-%d.png'%data_len,normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),'output/cifar100_fake_samples_epoch_%03d-%d.png' % (epoch, data_len),normalize=True)
        '''
    print(epoch)
    '''
    # Check pointing for every epoch
    torch.save(netG.state_dict(), 'weights/cifar100_netG_epoch_%d-%d.pth' % (epoch, data_len))
    torch.save(netD.state_dict(), 'weights/cifar100_netD_epoch_%d-%d.pth' % (epoch, data_len))
    '''

end = datetime.datetime.now()
duration = (end - start).total_seconds()
mod_time = float(duration)
print()
print("\t\t===>time for training %.4f "%(mod_time ))


#### once we have this model, we generate 20000 data instances so we can use it for DMP

num_synthetic = 20000
size_transform=transforms.Compose([
   transforms.Resize(32),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
inv_normalize = transforms.Compose([
    transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5,), (1/0.5, 1/0.5, 1/0.5))
])

len_t = num_synthetic//BATCH_SIZE
if num_synthetic%BATCH_SIZE:
    len_t += 1

with torch.no_grad():
    first = True
    for ind in range(len_t):
        if(ind == len_t-1):
            noise = torch.randn(num_synthetic%BATCH_SIZE, nz, 1, 1, device=device)
        else:
            noise = torch.randn(BATCH_SIZE, nz, 1, 1, device=device)
        fake = netG(noise)
        if(first):
            # first de-normalize, then resize
            #print("org gan output", fake.cpu().numpy().max(), fake.cpu().numpy().min(), flush=True)
            fake = inv_normalize(fake)
            #print("inv norm gan output", fake.cpu().numpy().max(), fake.cpu().numpy().min(), flush=True)
            fake = size_transform(fake)
            #print("32*32 norm output", fake.cpu().numpy().max(), fake.cpu().numpy().min(), flush=True)
            synthetic_data = fake
            first=False
            #print(fake.size())
        else:
            fake = inv_normalize(fake)
            fake = size_transform(fake)
            synthetic_data = torch.cat( (synthetic_data, fake), 0 )

print( synthetic_data.cpu().numpy().max(), synthetic_data.cpu().numpy().min(), )
print("total num of synthetic data ", synthetic_data.size(), flush=True)

np.save('syn_data.npy',synthetic_data.cpu().numpy())
