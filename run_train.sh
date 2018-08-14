#!/bin/bash

if [ ! -f ./pretrained/net.pth ]; then
    echo "Run 'init_retinanet.py' first by executing:"
    echo ""
    echo "    python init_retinanet.py"
    echo ""
    echo "in scripts/ and check if the net file is generated in pretrained/."
    exit
fi

python3 train.py \
    --gpus "0" \
    --num_classes 20 \
    --train_batch_size 2 \
    --test_batch_size 1 \
    --net "pretrained/retinanet_net.pth" \
    --img_dir "data/voc_all_images" \
    --train_list "data/voc_train.txt" \
    --test_list "data/voc_test.txt" \
    --checkpoint "checkpoint/retinanet_ckpt.pth"
