#!/bin/bash

export NUM_GPUS=2
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo

# Memory management for larger 4B model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LLM_VERSION="Qwen/Qwen3-4B" 
# for 4b model we recommend bs=1, accum=8, lower lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="parsit-google_siglip2-so400m-patch14-384-Qwen_Qwen3-4B-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="parsit-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-finetune_stage" 
PREV_STAGE_CHECKPOINT="/checkpoints/projectors/${BASE_RUN_NAME}" # replace it with your last checkpoint training from pretrain stage
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "FINETUNE_RUN_NAME: ${RUN_NAME}"

# Get absolute paths
REPO_ROOT="$(pwd)"
TRAIN_SCRIPT="$REPO_ROOT/parsit/train/train.py"
DEEPSPEED_CONFIG="$REPO_ROOT/scripts/zero3_4b.json"

ACCELERATE_CPU_AFFINITY=1 torchrun --standalone --nproc_per_node=$NUM_GPUS \
    "$TRAIN_SCRIPT" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /path/to/finetune_data.yaml \
    --image_folder /path/to/images \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=1e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /checkpoints/finetune/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --attn_implementation sdpa

# Error handling for torchrun
if [ $? -ne 0 ]; then
  echo "Error: torchrun command failed. Please check the logs above for details."
  exit 2
fi

echo "Finetuning completed successfully for Qwen3-4B!"