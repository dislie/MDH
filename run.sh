#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python main.py --dataset cub-2011 --root /data/yhx/data/CUB_200_2011 --max-epoch 30 --batch-size 16 --max-iter 40 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 25,35 --num-samples 2000 --info 'CUB' --momen 0.91