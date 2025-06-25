#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --dataset vegfru --root /data/yhx/data/vegfru/ --max-epoch 30 --batch-size 16 --max-iter 50 --code-length 48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 25,35 --num-samples 4000 --info 'VegFru' --momen 0.91