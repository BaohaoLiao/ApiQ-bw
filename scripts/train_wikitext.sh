#!/bin/bash

MODEL=LoftQ/Llama-2-7b-hf-4bit-64rank
OUTPUT_DIR=./exp_results/wikitext-2


python finetuning/train_clm.py \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --learning_rate 3e-4  \
    --seed 11 \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "epoch" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_train --do_eval \
    --logging_steps 50 \
    --report_to none \
    --block_size 1024