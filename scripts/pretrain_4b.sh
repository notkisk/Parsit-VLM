#!/bin/bash

# Auto-detect GPU count or use environment variable
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    export NUM_GPUS=${NUM_GPUS:-$DETECTED_GPUS}
else
    export NUM_GPUS=${NUM_GPUS:-1}
fi
echo "Using $NUM_GPUS GPU(s)"
export OMP_NUM_THREADS=8

# NCCL settings for multi-GPU communication (only needed for >1 GPU)
if [ "$NUM_GPUS" -gt 1 ]; then
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_DEBUG=WARN
    export NCCL_SOCKET_IFNAME=lo
fi

# Memory management for CUDA OOM prevention
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Remove CUDA_LAUNCH_BLOCKING for better performance in multi-GPU
# export CUDA_LAUNCH_BLOCKING=1

LLM_VERSION="Qwen/Qwen3-4B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="parsit-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# Get absolute paths
REPO_ROOT="$(pwd)"
TRAIN_SCRIPT="$REPO_ROOT/parsit/train/train.py"
DEEPSPEED_CONFIG="$REPO_ROOT/scripts/zero3_4b.json"
DATA_PATH="$REPO_ROOT/mlp-projector-pretrain/blip_laion_cc_sbu_558k_subset_5k.json" 
IMAGE_FOLDER="$REPO_ROOT/mlp-projector-pretrain/images"

torchrun --standalone --nproc_per_node=$NUM_GPUS \
    "$TRAIN_SCRIPT" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "${LLM_VERSION}" \
    --version "${PROMPT_VERSION}" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "/checkpoints/projectors/${BASE_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 500 \
    --learning_rate 8e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "$BASE_RUN_NAME" \
    --attn_implementation sdpa

# Error handling for torchrun
if [ $? -ne 0 ]; then
  echo "Error: torchrun command failed. Please check the logs above for details."
  exit 2
fi

echo "Training completed successfully for Qwen3-4B!"