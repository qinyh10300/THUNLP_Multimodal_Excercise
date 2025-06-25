#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`

MODEL="/mnt/data/user/tc_agi/zhy3488/models/MiniCPM-V-2_6"
DATA_DIR="/data/zhanghaoye/checkpoints/MiniCPM-V-26-logps"
DATA="data/preference_train.json"
REF_NAME="minicpm-v-26_test_reconstruct0929"

MODEL_MAX_Length=2048

deepspeed --master_port 29600 --include localhost:0,1,2,3 mllm/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --data_dir $DATA_DIR \
    --ref_name $REF_NAME \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --tune_vision true \
    --tune_llm true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --max_steps 2000 \
    --output_dir output/mllm_preference_training \
    --logging_dir output/mllm_preference_training/log \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed mllm/ds_pref_config_zero2.json \
    --report_to "tensorboard" \
    --dataloader_num_workers 16 \
    --preference_use_average_logp False \
    --preference_beta 0.5 \
    --task Preference