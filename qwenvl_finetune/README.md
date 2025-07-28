# Qwen2.5-VL Fine-tuning System

A comprehensive fine-tuning framework for Qwen2.5-VL vision-language models with support for:
- Full fine-tuning and LoRA/QLoRA parameter-efficient training
- Multi-GPU training with DeepSpeed ZeRO optimization
- 4-bit/8-bit quantization for memory efficiency
- Dynamic resolution image/video processing
- Comprehensive training monitoring and logging

## 🚀 Quick Start

### Installation
```bash
# Install the package
pip install -e .

# Verify installation
python simple_test.py
```

### Basic Training Commands

#### 1. Memory-Efficient LoRA Training (Recommended)
```bash
# Using provided synthetic test data
export DATA_PATH="./test_data/datasets/synthetic_test.json"
export IMAGE_FOLDER="./test_data/images"
./scripts/finetune_lora_3b.sh
```

#### 2. Full Fine-tuning (High Memory)
```bash
# Using provided synthetic test data
export DATA_PATH="./test_data/datasets/synthetic_test.json"
export IMAGE_FOLDER="./test_data/images"
./scripts/finetune_3b.sh
```

## 📊 Training Configurations

### Available Configurations
- **Full Fine-tuning**: `configs/training/finetune_3b.yaml` / `finetune_7b.yaml`
- **LoRA Training**: `configs/lora/lora_3b.yaml` / `lora_7b.yaml`
- **DeepSpeed**: `configs/deepspeed/zero2_3b.json` / `zero3_7b.json`

### Model Sizes
- **3B Model**: `Qwen/Qwen2.5-VL-3B-Instruct` (~4GB VRAM with LoRA+4bit)
- **7B Model**: `Qwen/Qwen2.5-VL-7B-Instruct` (~8GB VRAM with LoRA+4bit)

## 🎛️ Environment Variables

Customize training with these environment variables:

```bash
# Data Configuration
export DATA_PATH="/path/to/your/dataset.json"
export IMAGE_FOLDER="/path/to/your/images/"

# Training Parameters
export BATCH_SIZE=1                    # Per-device batch size
export GRADIENT_ACCUMULATION=8        # Gradient accumulation steps
export LEARNING_RATE=2e-5             # Learning rate
export EPOCHS=3                       # Number of epochs

# Output Configuration
export OUTPUT_DIR="./checkpoints/my_model"

# Monitoring
export WANDB_PROJECT="my-qwen-finetune"
export LOG_LEVEL="INFO"

# Memory Optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 📝 Data Format

Your training data should follow this JSON format:

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "What do you see in this image?"
      },
      {
        "from": "gpt", 
        "value": "I can see a document with tables and text content..."
      }
    ],
    "image": "image_filename.jpg",
    "source": "dataset_name",
    "sample_id": 0
  }
]
```

### Image Processing
- **Supported formats**: JPG, PNG, WebP
- **Resolution**: Dynamic resolution (up to 4096 tokens)
- **Aspect ratio**: Preserved with intelligent padding

## ⚙️ Advanced Usage

### Multi-GPU Training
```bash
# Automatically detects available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
./scripts/finetune_3b.sh  # Uses DeepSpeed for multi-GPU
```

### Custom Training Script
```bash
python train_model.py \
  --config configs/lora/lora_3b.yaml \
  --data_path ./your_data.json \
  --image_folder ./your_images/ \
  --output_dir ./checkpoints/custom_model
```

### Memory Optimization Strategies

#### For 16-24GB GPUs:
1. **Use LoRA + 4-bit quantization** (recommended)
2. **Batch size**: 1-2
3. **Gradient accumulation**: 8-16

#### For 8-16GB GPUs:
1. **Use LoRA + 4-bit quantization**
2. **Batch size**: 1
3. **Gradient accumulation**: 16+
4. **Enable CPU offloading** in DeepSpeed config

#### For <8GB GPUs:
1. **Use LoRA + 8-bit quantization**
2. **Batch size**: 1
3. **Use DeepSpeed ZeRO-3 with CPU offloading**

## 📈 Monitoring Training

### Real-time Monitoring
```bash
# TensorBoard
tensorboard --logdir ./checkpoints/qwen25vl_3b_finetune/runs

# Log files
tail -f ./logs/qwen25vl_3b_finetune/training_*.log
```

### System Health Check
```bash
# Test system components
python simple_test.py

# Check GPU usage
nvidia-smi
watch -n 1 nvidia-smi
```

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory (OOM)
1. **Reduce batch size** to 1
2. **Increase gradient accumulation** steps
3. **Enable 4-bit quantization**
4. **Use LoRA instead of full fine-tuning**
5. **Enable CPU offloading** in DeepSpeed

#### Slow Training
1. **Increase batch size** if memory allows
2. **Use multiple GPUs** with DeepSpeed
3. **Disable gradient checkpointing** if memory permits
4. **Use flash attention** (if available)

#### Data Loading Issues
1. **Check image paths** are correct
2. **Verify JSON format** matches expected structure
3. **Test with synthetic data** first: `python simple_test.py`

#### Import Errors
```bash
# Reinstall package
pip install -e .

# Check dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
qwen25vl_finetune/
├── configs/                    # Training configurations
│   ├── training/              # Full fine-tuning configs
│   ├── lora/                  # LoRA training configs
│   └── deepspeed/             # DeepSpeed configurations
├── qwen25vl/                  # Main package
│   ├── models/                # Model implementations
│   ├── training/              # Training components
│   └── utils/                 # Utilities
├── scripts/                   # Training scripts
├── test_data/                 # Synthetic test data
└── checkpoints/               # Training outputs
```

## 🎯 Performance Benchmarks

### Training Speed (steps/second)
- **3B LoRA + 4-bit**: ~0.5-1.0 steps/s (19GB GPU)
- **3B Full**: ~0.2-0.4 steps/s (24GB+ GPU)
- **7B LoRA + 4-bit**: ~0.3-0.6 steps/s (24GB+ GPU)

### Memory Usage
- **3B LoRA + 4-bit**: ~4-6GB VRAM
- **3B Full**: ~16-20GB VRAM
- **7B LoRA + 4-bit**: ~8-12GB VRAM

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature-name`
3. **Make changes and test**: `python simple_test.py`
4. **Submit pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the excellent Qwen2.5-VL models
- **HuggingFace** for the transformers library
- **Microsoft DeepSpeed** for efficient training
- **PEFT Library** for parameter-efficient fine-tuning