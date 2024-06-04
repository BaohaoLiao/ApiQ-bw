#!/bin/bash

TASK=commonsense
MODEL=ApiQ/Llama-2-7b-hf-w4g64r64
DATA_DIR=./dataset
OUTPUT_DIR=./exp_results/commonsense/Llama-2-7b-hf-w3g64r64

mkdir -p $OUTPUT_DIR
torchrun --nproc_per_node=1 finetuning/train_multitask.py \
    --do_train \
    --do_eval \
    --model_name_or_path $MODEL \
    --task $TASK \
    --data_dir $DATA_DIR \
    --test_split test \
    --use_normalized_template \
    --max_length 512 \
    --seed 42 \
    --learning_rate 3e-3 \
    --max_grad_norm 0.3 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --warmup_ratio 0.1 \
    --greedy_decoding \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --disable_tqdm true \
    --report_to "none" \
    --remove_unused_columns false \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir 2>&1 | tee $OUTPUT_DIR/out