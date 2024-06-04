#!/bin/bash

MODEL=ApiQ/Llama-2-7b-hf-w4g64r64
CKPT_DIR=./exp_results/gsm8k/Llama-2-7b-hf-w4g64r64

python finetuning/test_gsm8k.py \
  --model_name_or_path $MODEL \
  --ckpt_dir $CKPT_DIR \
  --batch_size 16 