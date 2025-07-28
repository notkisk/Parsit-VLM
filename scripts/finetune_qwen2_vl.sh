#!/bin/bash

# Qwen2.5-VL Fine-tuning Script with Multi-GPU Support
# Supports flexible GPU scaling: 1, 2, 4, 8+ GPUs
# Usage: 
#   Single GPU:    ./finetune_qwen2_vl.sh
#   Multi-GPU:     NUM_GPUS=4 ./finetune_qwen2_vl.sh
#   With model:    MODEL_SIZE=7B NUM_GPUS=8 ./finetune_qwen2_vl.sh

set -e

# Auto-detect GPU count or use environment variable
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    export NUM_GPUS=${NUM_GPUS:-$DETECTED_GPUS}
else
    export NUM_GPUS=${NUM_GPUS:-1}
fi

echo "🚀 Starting Qwen2.5-VL fine-tuning with $NUM_GPUS GPU(s)"

# Model size selection: 3B or 7B
MODEL_SIZE="${MODEL_SIZE:-3B}"
echo "📦 Using Qwen2.5-VL-${MODEL_SIZE} model"

if [ "$MODEL_SIZE" == "7B" ]; then
    MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
    # 7B model requires more memory and different training parameters
    if [ "$NUM_GPUS" -eq 1 ]; then
        echo "⚠️  Warning: 7B model with 1 GPU requires significant VRAM (>40GB)"
        BATCH_SIZE=1
        GRAD_ACCUM=8
        LEARNING_RATE=5e-6
        DEEPSPEED_CONFIG="zero3_7b.json"
    else
        BATCH_SIZE=1
        GRAD_ACCUM=4
        LEARNING_RATE=1e-5
        DEEPSPEED_CONFIG="zero3_7b.json"
    fi
else
    MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
    # 3B model parameters
    if [ "$NUM_GPUS" -eq 1 ]; then
        BATCH_SIZE=2
        GRAD_ACCUM=4
        LEARNING_RATE=1e-5
        DEEPSPEED_CONFIG="zero2.json"
    else
        BATCH_SIZE=2
        GRAD_ACCUM=2
        LEARNING_RATE=2e-5
        DEEPSPEED_CONFIG="zero3.json"
    fi
fi

# Environment setup
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Multi-GPU communication settings (only needed for >1 GPU)
if [ "$NUM_GPUS" -gt 1 ]; then
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_DEBUG=WARN
    export NCCL_SOCKET_IFNAME=lo
    echo "🔗 Multi-GPU communication configured"
fi

# Paths configuration
REPO_ROOT="$(pwd)"
TRAIN_SCRIPT="$REPO_ROOT/parsit/train/train.py"
DEEPSPEED_CONFIG_PATH="$REPO_ROOT/scripts/$DEEPSPEED_CONFIG"

# Dataset configuration - customize these paths
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/finetune_data.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$REPO_ROOT/data/images}"

# Verify required files exist
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Training data not found at $DATA_PATH"
    echo "Please set DATA_PATH environment variable or create the default dataset"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ Error: Image folder not found at $IMAGE_FOLDER"
    echo "Please set IMAGE_FOLDER environment variable or create the default image directory"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG_PATH" ]; then
    echo "❌ Error: DeepSpeed config not found at $DEEPSPEED_CONFIG_PATH"
    exit 1
fi

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-/checkpoints/qwen2_vl_finetune}"
RUN_NAME="qwen2.5-vl-${MODEL_SIZE,,}-finetune-$(date +%Y%m%d_%H%M%S)"

echo "📊 Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  GPUs: $NUM_GPUS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Learning Rate: $LEARNING_RATE"
echo "  DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "  Output: $OUTPUT_DIR/$RUN_NAME"
echo "  Data: $DATA_PATH"
echo "  Images: $IMAGE_FOLDER"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build training arguments dynamically
TRAIN_ARGS="
        --deepspeed $DEEPSPEED_CONFIG_PATH
        --model_name_or_path $MODEL_NAME
        --model_class_name ParsitQwen2VLForConditionalGeneration
        --version plain
        --data_path $DATA_PATH
        --image_folder $IMAGE_FOLDER
        --tune_mm_mlp_adapter True
        --mm_projector_lr 2e-5
        --bf16 True
        --output_dir $OUTPUT_DIR/$RUN_NAME
        --num_train_epochs ${NUM_EPOCHS:-3}
        --per_device_train_batch_size $BATCH_SIZE
        --per_device_eval_batch_size 1
        --gradient_accumulation_steps $GRAD_ACCUM
        --save_strategy steps
        --save_steps 500
        --save_total_limit 3
        --learning_rate $LEARNING_RATE
        --weight_decay 0.01
        --warmup_ratio 0.03
        --lr_scheduler_type cosine
        --logging_steps 10
        --max_grad_norm 1.0
        --tf32 True
        --model_max_length 4096
        --gradient_checkpointing True
        --dataloader_num_workers 8
        --lazy_preprocess True
        --report_to wandb
        --run_name $RUN_NAME
        --remove_unused_columns False
        --attn_implementation sdpa"

# Add max_steps if specified
if [ -n "${MAX_STEPS:-}" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --max_steps $MAX_STEPS"
fi

# Training command based on GPU count
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "🎯 Starting single GPU training..."
    python "$TRAIN_SCRIPT" $TRAIN_ARGS
else
    # Multi-GPU training with torchrun
    echo "🔥 Starting multi-GPU training with torchrun..."
    torchrun --standalone --nproc_per_node=$NUM_GPUS "$TRAIN_SCRIPT" $TRAIN_ARGS
fi

# Error handling
if [ $? -eq 0 ]; then
    echo "✅ Qwen2.5-VL fine-tuning completed successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR/$RUN_NAME"
    echo "🎉 Ready for inference and evaluation"
else
    echo "❌ Fine-tuning failed. Check logs above for details."
    exit 1
fi

echo "📈 Training Summary:"
echo "  Final model: $OUTPUT_DIR/$RUN_NAME"
echo "  Configuration: ${MODEL_SIZE} model on ${NUM_GPUS} GPU(s)"
echo "  Next steps: Run inference or continue training with different parameters"