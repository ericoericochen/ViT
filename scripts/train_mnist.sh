#!/bin/bash
python3 train_mnist.py \
    --model_config="../configs/default.json" \
    --lr=3e-4 \
    --epochs=5 \
    --checkpoint_epoch=1 \
    --save_dir="../checkpoints/vit_mnist" \