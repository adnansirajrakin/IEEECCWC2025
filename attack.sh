python -u attack_targeted.py \
--gpus 0 \
--model vgg_small_1w1a \
--results_dir [DIR] \
--data_path [DATA_PATH] \
--dataset cifar10 \
--epochs 1000 \
--lr 0.1 \
-b 128 \
-bt 128 \
--Tmin 1e-2 \
--Tmax 1e1 \
--lr_type cos \
--warm_up \
--evaluate 1 \



