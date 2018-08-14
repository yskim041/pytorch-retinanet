#!/bin/bash

python3 test.py \
    --gpus "0" \
    --num_classes 20 \
    --img_path "image/voc_test/000001.jpg" \
    --checkpoint "checkpoint/retinanet_ckpt.pth"
