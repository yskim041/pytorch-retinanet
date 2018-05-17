#!/bin/bash

python3 train.py \
    --gpus "0" \
    --num_classes 49 \
    --train_batch_size 2 \
    --test_batch_size 1 \
    --img_dir "datasets/food_all_images" \
    --train_list "datasets/food_ann_train.txt" \
    --test_list "datasets/food_ann_test.txt" \
    --checkpoint "checkpoint/food_ckpt.pth"
