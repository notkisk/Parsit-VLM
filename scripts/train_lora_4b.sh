#!/bin/bash

# LoRA fine-tuning script for Parsit Qwen3-4B
# Memory efficient for smaller GPUs

export NUM_GPUS=2
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo

# Memory management for 4B model with LoRA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_VERSION="parsit-qwen3-4b-lora"
DATA_PATH="./data/document_data.json"
IMAGE_FOLDER="./data/document_images"
OUTPUT_DIR="./checkpoints/${MODEL_VERSION}"

BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
EPOCHS=5

# Get absolute paths
REPO_ROOT="$(pwd)"
TRAIN_SCRIPT="$REPO_ROOT/scripts/train_parsit.py"

python "$TRAIN_SCRIPT" \
    --model_name_or_path "Qwen/Qwen3-4B" \
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
    --mm_projector_lr 1e-5 \
    --report_to wandb

echo "LoRA training completed successfully for Qwen3-4B!"