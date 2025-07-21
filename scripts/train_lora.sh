#!/bin/bash

# Example LoRA fine-tuning script for Parsit
# More memory efficient for smaller GPUs

MODEL_VERSION="parsit-qwen-1.5b-lora"
DATA_PATH="./data/document_data.json"
IMAGE_FOLDER="./data/document_images"
OUTPUT_DIR="./checkpoints/${MODEL_VERSION}"

BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=2e-4
EPOCHS=5

python scripts/train_parsit.py \
    --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --version parsit \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower "google/siglip-so400m-patch14-384" \
    --mm_projector_type "mlp2x_gelu" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 2 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5