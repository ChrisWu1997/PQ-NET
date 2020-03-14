python train.py --proj_dir /mnt/disk4/wurundi/PartGenApp \
                --module seq2seq \
                --data_root data \
                --category Chair \
                --nr_epochs 2000 \
                --batch_size 64 \
                --lr 1e-3 \
                --save_frequency 500 \
                -g 1 \
                --vis