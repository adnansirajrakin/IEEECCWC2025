nohup python -u attack_un.py \
--gpus 1 \
--model resnet20_1w1a  \
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
>> res20_un.out &




