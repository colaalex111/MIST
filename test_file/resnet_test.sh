CUDA_VISIBLE_DEVICES=0 python3 resnet_test.py --mixup 0 --dataset cifar100 --target_learning_rate 0.1 --target_epochs 160 --schedule 80 120 --target_batch_size 128
