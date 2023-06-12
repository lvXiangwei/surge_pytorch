conda activate tf1.15

# 同上, 修改loss 为 bpr loss, seed修改auc 评测方法, {'auc_bpr': } {'auc': 0.9672523753742028}, {'auc': 0.47046726539112327}, {'auc': 0.6230639073278668}, {'auc': 0.5850839515814135}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v2_test_repro_no_seed_5 --dataset assist09 --gpu_id 1 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00005 \
#                     --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml 

# # 同上, auc ，修正lr 5e-5，RANDOM_SEED = 42 修改auc 评测方法, {'auc': 0.8796303527267995}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v2_bpr_test_lr_1e-5_seed_42(_reproduce) --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00005 \
#                        --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml

# # 同上, 修改loss 为 bpr loss， 修改auc 评测方法, {'auc_bpr': 0.8652089027723545}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v2_bpr_test --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00005 --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml

# # 同上, auc ，修正lr 5e-5，RANDOM_SEED = 42 修改auc 评测方法, {'auc': 0.8796303527267995}
# python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
#                     --name assist09_v2_bpr_test_lr_1e-5_seed_42(_reproduce) --dataset assist09 --gpu_id 0 \
#                     --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00005 \
#                        --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml

python sequential.py --data_path /data/home/hejiansu/KT/xiangwei/surge/my_data \
                    --name assist09_v3 --dataset assist09 --gpu_id 1 \
                    --train_num_ngs 0 --val_num_ngs 0 --test_num_ngs 0  --batch_size 1024 --learning_rate 0.00005 \
                    --yaml_file /data/home/hejiansu/KT/xiangwei/surge/my_code/gcn_bpr.yaml 