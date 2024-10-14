CUDA_VISIBLE_DEVICES=3 python3 squeezenet_test.py --dataset cifar10 --target_learning_rate 0.01 --target_epochs 250 --schedule 100 150 --target_batch_size 64
