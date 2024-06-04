#!/bin/bash

MODEL=ApiQ/Llama-2-7b-hf-w4g64r64
OUTPUT_DIR=./exp_results/gsm8k/Llama-2-7b-hf-w4g64r64


torchrun --nproc_per_node=1 finetuning/train_gsm8k.py \
  --model_name_or_path $MODEL \
  --learning_rate 3e-4 \
  --seed 11 \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --do_train \
  --report_to none \
