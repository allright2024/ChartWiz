#!/bin/bash

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path maywell/Synatra-42dot-1.3B \
    --version synatra_mini \
    --data_path /home/work/ai-hub/data/train/json_data/shuffled_desc_summ_table_all.json \
    --image_folder /home/work/ai-hub/data/train/img_data \
    --vision_tower ybelkada/pix2struct-base \
    --pretrain_mm_mlp_adapter /home/work/ai-hub/Test_LLaVA/checkpoints/synatra_1.3b_pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava_v1.5_synatra_1.3b_3_types_RGB \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 49000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
