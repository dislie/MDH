#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python main.py --dataset food101 --root /data/yhx/data/food-101 --max-epoch 30 --batch-size 16 --max-iter 50 --code-length 48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 35,45 --num-samples 2000 --info 'Food101' --momen 0.91
