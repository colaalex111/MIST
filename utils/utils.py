
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import numpy as np
import torch
import copy
def onehot_transform(label):
    class_number = len(np.unique(label))
    length = label.shape[0]
    onehot_label = []
    for i in range(length):
        new_label = np.zeros(class_number)
        new_label[label[i]] = 1
        onehot_label.append(new_label)
    return np.array(onehot_label)
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''


def self_swap(x,alpha=0.15):
	### take some features from other samples. do we need to have mixed label as well?
	should_swap = torch.bernoulli(alpha * torch.ones((x.shape)).to(x.device))
	corrupted_X = torch.where(should_swap == 1, x[torch.randperm(x.shape[0])], x)
	#mask = (corrupted_X != X).float()
	return corrupted_X


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
	
    lam = torch.tensor(lam).type(torch.float64)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def bincount(x, w=None,max_item=None):
    """
    Bincount for torch (allowing for gradients)
    :param x: A torch tensor with integers
    :param w: Optional weights
    :return: A tensor (result) of size max(x) + 1 with (weighted) counts of elements in x

    Example Usage:
    # N = 100000
    # x = torch.randint(100, (N, )).int()
    # b = bincount(x)
    """

    if w is None:
        w = torch.ones_like(x, dtype=torch.float32)

    if max_item is None:
        m = x.max().item() + 1
    else:
        m = max_item+1

    bincount = torch.zeros(m, dtype=torch.float32, device=x.device)
    #bincount.requires_grad_(True)

    for i, xi in enumerate(x):
        #bincount[torch.round(xi).type(torch.cuda.LongTensor)] = bincount[torch.round(xi).type(torch.cuda.LongTensor)] + w[i]
        bincount[xi] = bincount[xi] + w[i]

    return bincount


def distroLoss(distA, nA, distB, nB, lossName="kuiper"):
    """
    Find the distribution loss given two empirical distributions and sample sizes.
    Parameters
    ----------
    distA : (torch.Tensor) Empirical distribution A
    nA : Sample size of A
    distB : (torch.Tensor) Empirical distribution B
    nB : Sample size of B
    lossName : One from (ks, kuiper, kuiper_approx, kuiper_ub, mmd)

    Returns
    -------
    Log distribution loss

    """

    # Assume that distA and distB have the same support (t = 0 to T).
    #print (nA,nB)

    assert distA.shape[0] == distB.shape[0], "Distributions of different length"
    effectiveN = torch.sqrt((nA * nB) / (nA + nB))

    if lossName == "ks":
        # Find log KS loss between the two KM distributions
        D = torch.abs(torch.max(distA - distB))
        lam = (effectiveN + 0.12 + 0.11 / effectiveN) * D
        lambda_squared = lam ** 2

        if lam < 1e-4:
            # If lambda is too small, return logLoss = 0
            # The sum below would require more terms to converge.
            # As lambda -> 0, we would actually require j -> $\infty$
            logloss = 0

        else:
            kspValue = 0
            for j in range(1, 1000):
                val = (-1) ** (j - 1) * torch.exp(2 * (1 - j * j) * lambda_squared)
                kspValue = kspValue + val

            logKpValue = torch.log(kspValue * 2) - 2 * lambda_squared
            logloss = logKpValue

    if lossName == "mmd":
        gramMatrixSize = distA.shape[0]

        def gaussianKernel(t, t_, sigma=1.0):
            assert (
                len(t.shape) == 2 and t.transpose(0, 1).shape == t_.shape
            ), "Shapes not compatible"
            return torch.exp(-((t - t_) ** 2 / sigma ** 2))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t = torch.arange(gramMatrixSize).float().to(device)
        K = gaussianKernel(t.unsqueeze(0), t.unsqueeze(1))

        #print (K)
        #print (K.shape)

        # Find outer products
        aa = torch.ger(distA, distA)
        bb = torch.ger(distB, distB)
        ab = torch.ger(distA, distB)

        mmd = torch.sum(K * (aa + bb - 2 * ab))
        logloss = -torch.log(
            mmd
        )  # We need to maximize the divergence (hence, the negative)

    if lossName.startswith("kuiper"):
        Dplus = torch.max(distA - distB).clamp(min=0)
        Dminus = torch.max(distB - distA).clamp(min=0)
        V = Dplus + Dminus

        logloss = kuiperVariants(effectiveN, V, lossName)

    return logloss


def kuiperVariants(effectiveN, V, lossName="kuiper"):
    lam = (effectiveN + 0.155 + 0.24 / effectiveN) * V
    lambda_squared = lam ** 2

    if lossName == "kuiper_approx":
        # Heuristic approximation
        logloss = -lambda_squared

    elif lossName == "kuiper":
        # See numerical recipes book 14.3

        if lam < 1e-4:
            # If lambda is too small, return logLoss = 0
            # The sum below would require more terms to converge.
            # As lambda -> 0, we would actually require j -> $\infty$
            logloss = 0

        else:
            kpValue = 0
            for j in range(1, 1000):
                val = (4 * j * j * lambda_squared - 1) * torch.exp(
                    2 * (1 - j * j) * lambda_squared
                )
                kpValue = kpValue + val

            logKpValue = torch.log(kpValue * 2) - 2 * lambda_squared
            logloss = logKpValue

    elif lossName == "kuiper_ub":
        # See paper for the upper bound calculation
        sqrt2lam = 1 / (math.sqrt(2) * lam)
        r_lower = torch.floor(sqrt2lam)
        r_upper = torch.ceil(sqrt2lam)

        kuiperTerm1 = lambda r, lam_sq: 4 * r ** 2 * lam_sq - 1
        logKuiperTerm2 = lambda r, lam_sq: -2 * r ** 2 * lam_sq

        # Slight difference from the notations in the paper. Here, w and v take lambda_squared as input instead of lambda.
        w = lambda r, lam_sq: -r * torch.exp(logKuiperTerm2(r, lam_sq))
        v = lambda r, lam_sq: kuiperTerm1(r, lam_sq) * torch.exp(
            logKuiperTerm2(r, lam_sq)
        )

        logloss = 0
        if r_lower >= 1:
            logloss = (
                logloss
                + w(r_lower, lambda_squared)
                - w(1, lambda_squared)
                + v(r_lower, lambda_squared)
            )

            logloss = logloss + v(r_upper, lambda_squared) - w(r_upper, lambda_squared)
            logloss = torch.log(logloss)

        else:
            # Here, Lambda is large. The loss simply becomes 0 {because of exp(-2*lambda**2)} and logloss goes to -inf.
            # Hence, can't use this : logloss = logloss + torch.log(v(r_upper, lambda_squared) - w(r_upper, lambda_squared))
            # Use logKuiperTerms to find the logLoss directly.

            logloss = (
                logloss
                + torch.log(kuiperTerm1(r_upper, lambda_squared))
                + logKuiperTerm2(r_upper, lambda_squared)
            )

    return logloss


def boundsForKuiperVariants(N, lossList=None, show=False, save=False):
    """
    Compute and plot the values for various Kuiper variants along with an "almost" exact approximation of Kuiper.

    Parameters
    ----------
    N : Effective sample size of the distributions. Loss is based on a p-value, hence dependent on the sample size.
    lossList : List of losses to compare

    Returns
    -------
    A tensor of size(100, len(lossList)) with values of the losses over a range of values for V (loss statistic).

    """

    if lossList is None:
        lossList = ["kuiper", "kuiper_ub"]

    listV = torch.arange(0.01, 1, 0.01)
    logKp = torch.zeros(listV.shape[0], len(lossList))

    for j, mode in enumerate(lossList):
        for i, V in enumerate(listV):
            logKp[i][j] = kuiperVariants(N, V, mode)

        #plt.plot(listV.numpy(), logKp.numpy()[:, j], label=mode)

    #plt.legend()
    #if save:
    #    plt.savefig("plot_kuiperVariants_{N}.pdf".format(N=N), dpi=300)
    #if show:
    #    plt.show()
    #plt.close()

    return logKp


def find_best_thres(feature,label):

    sorted_index = np.argsort(feature)
    feature = feature[sorted_index]
    label = label[sorted_index]

    #for i in range(40):
    #    print (feature[i*100])

    #print (feature[:100],label[:100])
    #print (feature[-100:],label[-100:])
    max_acc = -1
    max_thres = -1
    for index in range(len(label)):
        this_thres = feature[index]
        this_count = 0

        for i in range(len(label)):
            if (feature[i]>=this_thres and label[i]==1):
                this_count+=1
            if (feature[i]<this_thres and label[i]==0):
                this_count+=1

        if (this_count>max_acc):
            max_thres = this_thres
            max_acc = this_count

            #print ("this threshold %.2f, this acc %.2f " %(this_thres,this_count/len(label)))

    #print ("max acc %.2f" %(max_acc/len(label)))

    return max_thres


def convert_batchnorm_modules(
    model,
    converter
):
    return replace_all_modules(model, nn.modules.batchnorm._BatchNorm, converter)

def _replace_child(
    root, child_name, converter
):
    #print ("coverting a child")
    parent = root
    nameList = child_name.split(".")
    #print ("namelist:",nameList)
    for name in nameList[:-1]:
        parent = parent._modules[name]
    # set to identity
    parent._modules[nameList[-1]] = converter(parent._modules[nameList[-1]])

def replace_all_modules(
    root,
    target_class,
    converter
):
    if isinstance(root, target_class):
        return converter(root)

    for name, obj in root.named_modules():
        #print (name,obj)
        if isinstance(obj, target_class):
            #print ("coverting")
            _replace_child(root, name, converter)
    return root

def _batchnorm_to_groupnorm_new(module):
    #print (module)
    #print (module.num_features)
    return nn.GroupNorm(num_groups=module.num_features, num_channels=module.num_features, affine=True)



def get_sparse_gradient(x, thres):
    ### the input shape of x should be [# of layers, shape of grad of this layer]
    ### the output shape of x is [# of layers, flattened shape of grad of this layer], type:list of tensors
    ### we assume that there are multiple layers

    #print (len(x))

    num_layers = len(x)

    output_x = []

    for layer_idx in range(num_layers):
        this_layer_grad = x[layer_idx]
        # now we need to generate a mask to hide those small values
        this_layer_grad_copy = torch.flatten(copy.deepcopy(this_layer_grad))
        # we need to sort the copy and get the argmax
        _, indices = torch.sort(this_layer_grad_copy, descending=True)
        opt_len = int(len(indices) * thres)
        #print (this_layer_grad_copy)
        #print (indices)
        mask = torch.zeros_like(this_layer_grad_copy)
        mask = mask.scatter_(0, indices[:opt_len], 1.)
        #print (mask)
        this_layer_grad_output = torch.flatten(this_layer_grad) * mask

        output_x.append(this_layer_grad_output)

    return output_x

'''
def get_sparse_gradient(gradient, threshold=0.3):
    ### assume the inputs coming in [layer, shape of gradient] shape

    layer_num = len(gradient)

    sparse_gradient = []
    for i in range(layer_num):
        this_layer_gradient = torch.flatten(gradient[i])
        #print (this_layer_gradient.shape[0])
        ### get the threshold value, here we assume that we only use top10% of absolute value
        # print (torch.sort(torch.abs(this_layer_gradient)))
        this_threshold, _ = torch.sort(torch.abs(this_layer_gradient))
        if (this_layer_gradient.shape[0]<10):
            this_threshold = 0
        else:
            this_threshold = this_threshold[int(this_layer_gradient.shape[0] * (1 - threshold))]

        this_layer_gradient = torch.where((torch.abs(this_layer_gradient)) > this_threshold, this_layer_gradient,
                                          torch.zeros_like(this_layer_gradient))
        ### construct sparse tensor
        # get the value and the index for non-zero values
        this_index = torch.squeeze(torch.nonzero(this_layer_gradient))
        #print (this_index)
        length = this_index.shape[0]
        this_index = this_index.reshape(1, length)
        this_value = torch.squeeze(this_layer_gradient[this_index])
        this_sparse_tensor = torch.sparse_coo_tensor(this_index, this_value, this_layer_gradient.shape)
        sparse_gradient.append(this_sparse_tensor)

    return sparse_gradient
'''
