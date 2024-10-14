CUDA_VISIBLE_DEVICES=0 python3 densenet_test.py --mixup 1 --dataset cifar10 --target_learning_rate 0.1 --target_epochs 300 --schedule 150 225 --target_batch_size 128
