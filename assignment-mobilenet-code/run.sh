export CUBLAS_WORKSPACE_CONFIG=:16:8

python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 300 \
--lr 0.05 --wd 0.0005 \
--lr_scheduler \
--mixup \
--seed 0 \
--fig_name lr=0.05-lr_sche-wd=0.0005-mixup.png \
--test

# learning_rate: 0.5
# 15 epochs
python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 15 \
--lr 0.5 --wd 0 \
--seed 0 \
--fig_name lr=0.5-wd=0-eps=15.png \
--test

# learning_rate: 0.05
# 15 epochs
python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 15 \
--lr 0.05 --wd 0 \
--seed 0 \
--fig_name lr=0.05-wd=0-eps=15.png \
--test

# learning_rate: 0.01
# 15 epochs
python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 15 \
--lr 0.01 --wd 0 \
--seed 0 \
--fig_name lr=0.01-wd=0-eps=15.png \
--test

# learning_rate: 0.01
# no learning rate schedule
# 300 epochs
python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 300 \
--lr 0.01 --wd 0 \
--seed 0 \
--fig_name lr=0.01-wd=0-eps=300.png \
--test

# learning_rate: 0.01
# learning rate schedule
# 300 epochs
python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 300 \
--lr 0.01 --wd 0 \
--lr_scheduler \
--seed 0 \
--fig_name lr=0.01-lr_sche-wd=0-eps=300.png \
--test

# learning_rate: 0.01
# learning rate schedule
# weight_decay: 0.0005
# 300 epochs
python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 300 \
--lr 0.01 --wd 0.0005 \
--lr_scheduler \
--seed 0 \
--fig_name lr=0.01-lr_sche-wd=0.0005-eps=300.png \
--test

# learning_rate: 0.01
# learning rate schedule
# weight_decay: 0.0001
# 300 epochs
python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 300 \
--lr 0.01 --wd 0.0001 \
--lr_scheduler \
--seed 0 \
--fig_name lr=0.01-lr_sche-wd=0.0001-eps=300.png \
--test

# learning_rate: 0.01
# learning rate schedule
# weight_decay: 0.0005
# mixup
# 300 epochs
python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 300 \
--lr 0.01 --wd 0.0005 \
--lr_scheduler \
--mixup \
--seed 0 \
--fig_name lr=0.01-lr_sche-wd=0.0005-mixup-eps=300.png \
--test
