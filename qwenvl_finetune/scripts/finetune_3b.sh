#!/bin/bash

# Qwen2.5-VL 3B Model Fine-tuning Script
# This script provides automated fine-tuning with GPU detection and optimal settings

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/configs/training/finetune_3b.yaml"

# Default parameters (can be overridden by environment variables)
MODEL_SIZE="${MODEL_SIZE:-3B}"
DATA_PATH="${DATA_PATH:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen25vl_3b_finetune}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EPOCHS="${EPOCHS:-3}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed/zero2_3b.json}"

# Logging and monitoring
WANDB_PROJECT="${WANDB_PROJECT:-qwen25vl-finetune}"
WANDB_NAME="${WANDB_NAME:-qwen25vl-3b-$(date +%Y%m%d_%H%M%S)}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "=========================================="
echo "Qwen2.5-VL 3B Fine-tuning Script"
echo "=========================================="

# Auto-detect GPU count and configure NCCL
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    export NUM_GPUS=${NUM_GPUS:-$DETECTED_GPUS}
    
    echo "Detected GPUs: $NUM_GPUS"
    
    # Configure NCCL for multi-GPU training
    if [ "$NUM_GPUS" -gt 1 ]; then
        export NCCL_IB_DISABLE=1
        export NCCL_P2P_DISABLE=0
        export NCCL_DEBUG=WARN
        export NCCL_SOCKET_IFNAME=lo
        echo "Multi-GPU NCCL configuration applied"
    fi
else
    export NUM_GPUS=1
    echo "Warning: nvidia-smi not found, assuming 1 GPU"
fi

# Memory optimization settings
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Validate required parameters
if [ -z "$DATA_PATH" ]; then
    echo "Error: DATA_PATH environment variable is required"
    echo "Usage: DATA_PATH=/path/to/data.json ./scripts/finetune_3b.sh"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

if [ -n "$IMAGE_FOLDER" ] && [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Error: Image folder not found: $IMAGE_FOLDER"
    exit 1
fi

# Validate configuration files
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Error: DeepSpeed configuration file not found: $DEEPSPEED_CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")/logs"

# Log system information
echo ""
echo "System Information:"
echo "  Project Root: $PROJECT_ROOT"
echo "  Data Path: $DATA_PATH"
echo "  Image Folder: ${IMAGE_FOLDER:-Not specified}"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  DeepSpeed Config: $DEEPSPEED_CONFIG"
echo ""

# Check memory requirements
if [ "$NUM_GPUS" -eq 1 ]; then
    REQUIRED_MEMORY=16  # GB
    if command -v nvidia-smi >/dev/null 2>&1; then
        AVAILABLE_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        AVAILABLE_MEMORY_GB=$((AVAILABLE_MEMORY / 1024))
        
        echo "GPU Memory Check:"
        echo "  Available: ${AVAILABLE_MEMORY_GB}GB"
        echo "  Recommended: ${REQUIRED_MEMORY}GB"
        
        if [ "$AVAILABLE_MEMORY_GB" -lt "$REQUIRED_MEMORY" ]; then
            echo "Warning: Available GPU memory may be insufficient"
            echo "Consider:"
            echo "  - Reducing batch size (current: $BATCH_SIZE)"
            echo "  - Using gradient checkpointing (enabled by default)"
            echo "  - Using DeepSpeed ZeRO-3 with CPU offloading"
        fi
    fi
fi

# Set up Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Create training script
TRAIN_SCRIPT="${PROJECT_ROOT}/train_model.py"

# Generate training script on-the-fly if it doesn't exist
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Creating training script..."
    cat > "$TRAIN_SCRIPT" << 'EOF'
#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qwen25vl import load_qwen25vl_model, Qwen25VLTrainer, DataProcessor, TrainingArguments
from qwen25vl.utils import setup_logging
from qwen25vl.training import ConversationDataset, DataConfig

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Fine-tuning")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--data_path", required=True, help="Path to training data")
    parser.add_argument("--image_folder", help="Path to image folder")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--deepspeed", help="DeepSpeed configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.deepspeed:
        config['deepspeed']['config_file'] = args.deepspeed
        
    # Setup logging
    setup_logging(
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        log_dir=config['training']['logging_dir']
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Qwen2.5-VL fine-tuning")
    
    # Load model
    logger.info(f"Loading {config['model']['model_size']} model...")
    model = load_qwen25vl_model(
        model_size=config['model']['model_size'],
        torch_dtype=config['model']['torch_dtype'],
        use_lora=False
    )
    
    # Setup data processor
    data_config = DataConfig(
        max_length=config['data']['max_length'],
        max_image_tokens=config['data']['max_image_tokens'],
        image_aspect_ratio=config['data']['image_aspect_ratio'],
        conversation_template=config['data']['conversation_template'],
        system_message=config['data']['system_message'],
        image_folder=args.image_folder,
        validate_images=config['data']['validate_images'],
        validate_conversations=config['data']['validate_conversations'],
        skip_invalid_samples=config['data']['skip_invalid_samples']
    )
    
    data_processor = DataProcessor(
        tokenizer=model.tokenizer,
        processor=model.processor,
        config=data_config
    )
    
    # Create datasets
    logger.info("Loading training data...")
    train_dataset = data_processor.create_dataset(args.data_path, "train")
    
    # Analyze dataset
    stats = data_processor.analyze_dataset(train_dataset)
    logger.info(f"Dataset loaded: {stats['total_samples']} samples")
    
    # Setup training arguments
    training_args = TrainingArguments(
        # Use config values
        **{k: v for k, v in config['training'].items() if k not in ['logging_dir']},
        deepspeed=config['deepspeed']['config_file'] if config['deepspeed']['enabled'] else None
    )
    
    # Create trainer
    trainer = Qwen25VLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=model.tokenizer
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
EOF
    chmod +x "$TRAIN_SCRIPT"
fi

# Prepare training command
TRAIN_CMD="python $TRAIN_SCRIPT"
TRAIN_CMD="$TRAIN_CMD --config $CONFIG_FILE"
TRAIN_CMD="$TRAIN_CMD --data_path $DATA_PATH"
TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --deepspeed $DEEPSPEED_CONFIG"

if [ -n "$IMAGE_FOLDER" ]; then
    TRAIN_CMD="$TRAIN_CMD --image_folder $IMAGE_FOLDER"
fi

# Set environment variables for the training process
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_NAME="$WANDB_NAME"

echo "Starting training with command:"
echo "$TRAIN_CMD"
echo ""

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training
    echo "Starting multi-GPU training on $NUM_GPUS GPUs..."
    torchrun --standalone --nproc_per_node=$NUM_GPUS $TRAIN_CMD
else
    # Single GPU training
    echo "Starting single-GPU training..."
    $TRAIN_CMD
fi

TRAIN_EXIT_CODE=$?

# Check training result
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo "Output directory: $OUTPUT_DIR"
    echo "Logs directory: $(dirname "$OUTPUT_DIR")/logs"
    
    # Display final model info
    if [ -f "$OUTPUT_DIR/config.json" ]; then
        echo ""
        echo "Model saved successfully:"
        ls -la "$OUTPUT_DIR"
    fi
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "=========================================="
    echo "Check the logs for detailed error information:"
    echo "  Logs: $(dirname "$OUTPUT_DIR")/logs"
    exit $TRAIN_EXIT_CODE
fi