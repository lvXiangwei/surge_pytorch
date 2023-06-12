conda activate tf1.15

#查看修改之后的代码能否恢复之前性能
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/rec/SIGIR21-SURGE/my_data/ \
#                     --batch_size 500 --early_stop 2 --show_step 500 --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_origin.yaml \
#                     --name check_origin --dataset taobao_global --gpu_id 0 # ok 30min, /data/home/hejiansu/KT/xiangwei/surge/my_save/2023-06-05_16:17:51

# # 测试第一版修改后的assist09
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/rec/SIGIR21-SURGE/my_data/ \
#                     --name test_assist09_gpu --dataset assist09_small --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0 # 27min

# # 第一版assist09
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/rec/SIGIR21-SURGE/my_data/ \
#                     --name assist09_v1 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0 # 5min

# 查看验证auc 下降是否因 batch size 过大的原因, no
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/rec/SIGIR21-SURGE/my_data/ \
#                     --name assist09_v1_test_batch_256 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 256 # 5min

# 查看训练集auc 的变化过程, 0.85 - 0.87
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/rec/SIGIR21-SURGE/my_data/ \
#                     --name assist09_v1_check_train_info --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 256 # 5min

# # 修改train data, max_len 改为200
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v1_fix_data --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 256 # 10min

# 基于上调整学习率, 0.7244
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v1_change_bs_512_lr_1e-5 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 512 --learning_rate 0.00005 

# # 观察到过拟合，调整 drop out {'auc': 0.7152, 'logloss': 0.5873}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v1_dropout --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.0001 

# # 观察到过拟合，调整 drop out, {'auc': 0.7308, 'logloss': 0.5856}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v1_dropout_0.2_lr_5e-4_bs_1024 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.0005 

# 观察到过拟合，embeding 维度 item_embedding_dim 64 --cate_embedding_dim 64 {'auc': 0.7224, 'logloss': 0.6098}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v1_dropout_0.2_lr_5e-4_bs_1024_input_dim_64 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.0005 

# 同上, --item_embedding_dim 32 --cate_embedding_dim 32 {'auc': 0.7263, 'logloss': 0.6086}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v1_dropout_0.2_lr_5e-4_bs_1024_input_dim_32 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.0005 

# 同上, --item_embedding_dim 16 --cate_embedding_dim 16 {'auc': 0.7243, 'logloss': 0.5908}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v1_dropout_0.2_lr_5e-4_bs_1024_input_dim_16 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.0005 

## 同上, input_dim 40, 20 droupout [0.1, 0.1], embeddinng dropout 0 {'auc': 0.7313, 'logloss': 0.5762}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v1_dropout_0.1_lr_5e-5_bs_1024_input_dim_40_20 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00005

# # 同上, input_dim 32, 8 droupout [0.1, 0.1], {'auc': 0.7317, 'logloss': 0.5811}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v1_dropout_0.1_lr_5e-5_bs_1024_input_dim_32_8 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00005

# # 同上, 修改loss 为 bpr loss， 修改auc 评测方法, {'auc_bpr': 0.8652089027723545}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v2_bpr_test --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00005 --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml

# # 同上, auc 变化剧烈，减小lr 1e-5, 再跑一次，RANDOM_SEED = 42 修改auc 评测方法, {'auc': 0.7386958219445529}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v2_bpr_test_2 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00001 --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml

# # 同上, auc ，修正lr 5e-5，RANDOM_SEED = 42 修改auc 评测方法, {'auc': 0.8796303527267995}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v2_bpr_test_lr_1e-5_seed_42(_reproduce) --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00005 --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml

# 同上, 修改随机seed，查看是否稳定 修正lr 5e-5，RANDOM_SEED = 0 修改auc 评测方法, {'auc': 0.7400494598464141}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v2_bpr_test_lr_5e-5_seed_0 --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00005 \
#                     --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml --seed 0

# 同上, 放大lr，查看是否稳定 修正lr 1e-4，RANDOM_SEED = 0 修改auc 评测方法, {'auc': 0.7551997917480151}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v2_bpr_test_lr_1e-4_seed_0 --dataset assist09 --gpu_id 1 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.0001 \
#                     --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml --seed 0

# # 同上, 修正lr 1e-5，RANDOM_SEED = 0 修改auc 评测方法, {'auc': 0.5532213978914486}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v2_bpr_test_lr_1e-5_seed_0 --dataset assist09 --gpu_id 1 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00001 \
#                     --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml --seed 0
