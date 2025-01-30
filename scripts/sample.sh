#!/bin/bash
python sample.py \
--ckpt sr4x-bicubic \
--n-gpu-per-node 1 \
--dataset-dir /storage/ImageNet_full \
--batch-size 1 \
--clip-denoise \
--nfe 1000 \
--step-size 1.0 \
--use-cddb \
--method cddb_reg