#!/bin/bash

# Example training script for Parsit on document analysis tasks
# Adjust paths and parameters according to your setup

# Model and data configuration
MODEL_VERSION="parsit-qwen-1.5b"
DATA_PATH="./data/document_data.json"
IMAGE_FOLDER="./data/document_images"
OUTPUT_DIR="./checkpoints/${MODEL_VERSION}"

# Training hyperparameters
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5
EPOCHS=3
MAX_LENGTH=2048

# Vision configuration  
VISION_TOWER="google/siglip-so400m-patch14-384"
MM_PROJECTOR_TYPE="mlp2x_gelu"

# Launch training
python scripts/train_parsit.py \
    --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --version parsit \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --mm_projector_type ${MM_PROJECTOR_TYPE} \
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
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${MAX_LENGTH} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb