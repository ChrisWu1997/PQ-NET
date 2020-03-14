#python train.py --proj_dir /mnt/disk4/wurundi/PartGenApp \
#                --module part_ae \
#                --data_root data \
#                --category Chair \
#                --resolution 16 \
#                --nr_epochs 30 \
#                --batch_size 40 \
#                --lr 5e-4 \
#                --lr_step_size 200 \
#                --save_frequency 50 \
#                -g 0,1,7,9 \
#                --vis
#python train.py --proj_dir /mnt/disk4/wurundi/PartGenApp \
#                --module part_ae \
#                --data_root data \
#                --category Chair \
#                --resolution 32 \
#                --nr_epochs 80 \
#                --batch_size 40 \
#                --lr 5e-4 \
#                --lr_step_size 200 \
#                --save_frequency 50 \
#                -g 0,1,7,9 \
#                --vis \
#                --continue
python train.py --proj_dir /mnt/disk4/wurundi/PartGenApp \
                --module part_ae \
                --data_root data \
                --category Chair \
                --resolution 64 \
                --nr_epochs 210 \
                --batch_size 40 \
                --lr 5e-4 \
                --lr_step_size 200 \
                --save_frequency 50 \
                -g 0,1 \
                --vis \
                --continue
