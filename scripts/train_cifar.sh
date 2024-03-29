#!/bin/bash
python3 train_cifar.py \
    --model_config="../configs/default.json" \
    --lr=8e-4 \
    --epochs=20 \
    --save_dir="../checkpoints/vit" \