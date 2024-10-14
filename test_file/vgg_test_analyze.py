import numpy as np

def analyze():
    train_ori = np.load('/home/lijiacheng/expdata/vgg_train_ori_confidence_cifar10.npy')
    train_perturb = np.load('/home/lijiacheng/expdata/vgg_train_perturb_confidence_cifar10.npy')

    test_ori = np.load('/home/lijiacheng/expdata/vgg_test_ori_confidence_cifar10.npy')
    test_perturb = np.load('/home/lijiacheng/expdata/vgg_train_test_confidence_cifar10.npy')



def main():
    analyze()


if __name__ == '__main__':
    main()