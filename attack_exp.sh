CUDA_VISIBLE_DEVICES=1 python3 attack_exp.py --required_num 500 --whitebox 1 --save_exp_data 1 --middle_output 0 --middle_gradient 1 --mmd_loss_lambda 0 --mixup 0 --alpha 1 --model_name resnet20 --model_number 1 --target_data_size 10000 --membership_attack_number 5000  --dataset cifar10 --target_learning_rate 0.1 --target_l2_ratio 1e-5 --early_stopping 0 --target_epochs 160 --schedule 80 120 --target_batch_size 100 --kuiper_loss_lambda 0 --maxprob_loss_lambda 0 --corr_loss_lambda 0  --label_smoothing 0 --validation_mi 0 --pretrained 0  --gpu 1