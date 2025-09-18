# Parsit Changelog

All notable changes to the Parsit project are documented in this file.

## [Unreleased] - 2025-07-26

### 🔥 Critical Fixes

#### **Latest Critical Fixes (2025-07-26)**
- **FIXED: Single GPU training setup**: Resolved multi-GPU configuration errors on single GPU systems with auto-detection
- **FIXED: Model identifier compatibility**: Updated to use correct Qwen3 model paths (Qwen/Qwen3-4B vs non-existent -Instruct variants)
- **FIXED: NCCL configuration issues**: Improved distributed training initialization for various hardware setups

#### **Previous Critical Fixes (2025-07-24)**
- **FIXED: Checkpoint system improvements**: Enhanced checkpoint saving and loading reliability for consistent training resumption
- **FIXED: Training state persistence**: Improved model state management to prevent checkpoint corruption during multi-GPU training
- **FIXED: DeepSpeed checkpoint compatibility**: Resolved checkpoint format issues that caused training interruptions

#### **Previous Critical Fixes (2025-07-23)**
- **FIXED: Multi-GPU DeepSpeed initialization**: Resolved "IndexError: list index out of range" in DeepSpeed ZeRO-3 stage3.py when accessing empty optimizer parameter groups
- **FIXED: Parameter unfreezing timing**: Moved parameter unfreezing logic before ParsitTrainer initialization to ensure parameters are available for optimizer creation
- **FIXED: Single GPU usage on multi-GPU setup**: Fixed NCCL configuration and distributed training initialization to properly utilize all available GPUs
- **FIXED: Empty optimizer parameter groups**: Added comprehensive parameter validation in create_optimizer() method to prevent DeepSpeed failures
- **FIXED: Distributed training communication**: Optimized NCCL settings for local multi-GPU training with P2P communication

#### **Previous Critical Fixes (2025-07-22)**
- **FIXED: Checkpoint save crash**: Resolved `TypeError: Trainer._save_checkpoint() takes 3 positional arguments but 4 were given` in `parsit_trainer.py:394`
- **FIXED: Parameter counting bug**: Resolved massive parameter count errors (trillion-scale false counts → accurate 6.56M parameters)
- **FIXED: DeepSpeed ZeRO-3 parameter visibility**: Implemented proper parameter shape calculation that works with parameter partitioning
- **FIXED: Gradient explosion**: Added LayerNorm stabilization to MLP projector preventing 10M+ gradient spikes
- **FIXED: WandB parameter logging**: Corrected parameter statistics display (was showing 0, now shows real counts)
- **FIXED: Memory management**: Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to prevent OOM crashes

#### **Previous Fixes (2025-01-22)**
- **Fixed pad token issue**: Restored proper tokenization by changing `pad_token=""` to `pad_token="<|endoftext|>"` in `train.py:334`
- **Restored dataloader worker configuration**: Re-added `worker_init_fn=seed_worker` and `prefetch_factor` settings for reproducibility and performance
- **Fixed loss/grad_norm calculation**: Implemented DeepSpeed-aware parameter unfreezing that resolves training metrics not being calculated

### ✨ Major Features

#### **Recent Major Improvements (2025-07-26)**

#### **Qwen3-4B Model Support**
- **Added Qwen3-4B language model support**: Extended architecture to support 4B parameter models alongside existing 1.7B support
- **Model-specific training scripts**: Created optimized training configurations for 4B models with appropriate hyperparameters
- **Memory optimization for larger models**: Implemented CPU offloading and enhanced DeepSpeed configurations for 4B training
- **Automatic model size detection**: Added intelligent model size selection in training scripts with `MODEL_SIZE` environment variable

#### **Training Infrastructure Enhancements**
- **Auto GPU detection**: Enhanced scripts with automatic GPU count detection and single-GPU fallback capabilities
- **Optimized DeepSpeed configurations**: Created model-specific ZeRO-3 configurations with CPU offloading for memory efficiency
- **Training parameter optimization**: Adjusted batch sizes, gradient accumulation, and learning rates for optimal 4B model training
- **Hardware compatibility improvements**: Better support for various GPU configurations and memory constraints

#### **Script and Configuration Updates**
- **New training scripts**: Added `scripts/pretrain_4b.sh`, `scripts/finetune_4b.sh`, and `scripts/train_lora_4b.sh` for 4B model training
- **Enhanced existing scripts**: Updated `scripts/pretrain.sh` with model size selection capabilities
- **Improved error handling**: Better validation and fallback mechanisms for different hardware setups
- **Documentation updates**: Updated README and documentation to reflect 4B model support and usage examples

#### **Previous Major Improvements (2025-07-23)**

#### **Multi-GPU Training Infrastructure**
- **Enhanced distributed training support**: Comprehensive multi-GPU setup with proper parameter synchronization across ranks
- **DeepSpeed ZeRO-3 optimization**: Improved parameter partitioning and optimizer initialization for large-scale training
- **NCCL communication optimization**: Fine-tuned settings for local multi-GPU environments with P2P GPU communication
- **Real-time parameter validation**: Added pre-trainer validation to catch parameter unfreezing issues early
- **Dataset download automation**: Created `scripts/download_pretrain_data.sh` for automatic git-lfs setup and dataset downloading

#### **Previous Major Improvements (2025-07-22)**

#### **Enhanced Parameter Management & Debugging**
- **Robust parameter counting**: Added fallback logic for DeepSpeed ZeRO-3 parameter partitioning scenarios
- **Improved parameter logging**: Enhanced visibility with shape information and proper size calculation
- **WandB integration**: Automatic parameter count updates for monitoring and debugging
- **Real-time parameter validation**: Comprehensive error detection and reporting during training initialization

#### **Training Stability & Performance**
- **LayerNorm gradient stabilization**: Added LayerNorm layers between MLP components to prevent gradient explosion
- **Enhanced checkpointing**: Reduced save intervals from 50,000 → 500 steps for better recovery capabilities  
- **Memory optimization**: Improved CUDA memory allocation with expandable segments configuration
- **Training configuration tuning**: Optimized gradient accumulation and dataloader worker settings

#### **Architecture Improvements**
- **MLP projector enhancement**: Added LayerNorm stabilization to `mlp2x_gelu` architecture preventing training instability
- **Better error handling**: Comprehensive exception handling for DeepSpeed parameter operations
- **Monitoring improvements**: Enhanced gradient norm tracking and parameter status reporting

#### DeepSpeed-Aware Parameter Management (Previous)
- **Added sophisticated parameter unfreezing logic** (`train.py:520-557`)
  - Uses `deepspeed.zero.GatheredParameters` context manager for ZeRO-3 compatibility
  - Handles parameter availability across distributed ranks properly
  - Prevents parameter access issues that caused loss/grad_norm calculation failures
  - Supports granular control via `mm_tunable_parts` configuration

#### Enhanced Training Infrastructure
- **Added comprehensive parameter logging** (`train.py:563-583`)
  - Real-time parameter status reporting during training initialization
  - Clear visibility into trainable vs frozen parameters
  - Early detection of parameter freezing issues
  - Detailed parameter count reporting

#### Tensor Parallelism Support
- **Added tensor parallelism detection** (`parsit_trainer.py:194`)
  - `self.is_tp_enabled = getattr(self.accelerator.state, "tp_plugin", None) is not None`
  - Better multi-GPU scalability for large models
  - Enhanced distributed training capabilities

### 🔧 Improvements

#### Training Configuration
- **Multimodal arguments bridge** (`train.py:397-398`)
  - Fixed argument passing between `ModelArguments` and `DataArguments` classes
  - Ensures `mm_use_im_start_end` and `mm_use_im_patch_token` are properly propagated
  - Maintains functionality without major architectural refactoring

#### Data Processing Enhancements
- **Added robust dataset validation** (`train.py:621-624, 667-672`)
  - Malformed data detection and automatic skipping
  - Prevents training crashes from corrupted dataset entries
  - Graceful error handling for missing 'conversations' keys

#### Conversation Format Optimization
- **Updated conversation separator logic** (`train.py:816`)
  - Changed from `conv.sep + conv.roles[1]` to `conv.roles[1] + "\n"`
  - Optimized for CHATML conversation format used by Parsit
  - Maintains compatibility with `{"from": "human"/"gpt", "value": "..."}` dataset format

#### Accelerator Configuration
- **Streamlined Accelerator initialization** (`parsit_trainer.py:186`)
  - Removed deprecated `dispatch_batches` and `split_batches` parameters
  - Cleaner initialization for newer transformers versions
  - Maintained core distributed training functionality

### 🛠️ Technical Details

#### Architecture Compatibility
- **Dataset format support**: Native compatibility with `{"from": "human"/"gpt", "value": "..."}` format
- **Conversation templates**: Uses CHATML format with proper role mapping
- **Vision-language integration**: Maintains LLaVA-NeXT multimodal processing pipeline

#### DeepSpeed Integration
The most critical improvement addresses the loss/grad_norm calculation issue through proper DeepSpeed ZeRO-3 parameter management:

```python
# Old approach (caused issues)
for p in model.get_model().mm_projector.parameters():
    p.requires_grad = True

# New DeepSpeed-aware approach (fixes issues)
with deepspeed.zero.GatheredParameters(params_to_unfreeze, modifier_rank=0):
    if training_args.local_rank == 0:
        for p_to_unfreeze in params_to_unfreeze:
            if hasattr(p_to_unfreeze, 'ds_status') and p_to_unfreeze.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                continue
            p_to_unfreeze.requires_grad = True
```

### 🐛 Bug Fixes
- **Parameter availability handling**: Fixed issues where parameters were not available on all ranks in ZeRO-3
- **Training metrics calculation**: Resolved loss and gradient norm not being computed during training
- **Memory optimization**: Proper parameter gathering prevents OOM issues in distributed training
- **Reproducibility**: Restored deterministic training through proper worker seeding

### 📚 Documentation & Tooling

#### New Tools
- **Added Qwen3-4B test script** (`scripts/test_qwen3_4b.py`)
  - Validates 4B model compatibility with Parsit architecture
  - Tests model loading and configuration without requiring training data
  - Provides compatibility verification for both 1.7B and 4B models

- **Added dataset converter script** (`scripts/data_converter.py`)
  - Validates Parsit dataset format
  - Converts from LLaVA and ShareGPT formats
  - Supports both JSON and JSONL files
  - Auto-detection of dataset formats

#### New Configuration Files
- **Added 4B-specific DeepSpeed config** (`scripts/zero3_4b.json`)
  - Optimized ZeRO-3 configuration with CPU offloading for 4B models
  - Enhanced memory management for larger models
  - Reduced parameter thresholds for single-GPU training

#### Usage Examples
```bash
# Test 4B model compatibility
python scripts/test_qwen3_4b.py

# Train with 4B model using environment variable
MODEL_SIZE=4B bash scripts/pretrain.sh

# Train with 4B model using dedicated script
bash scripts/pretrain_4b.sh

# Validate dataset format
python scripts/data_converter.py --input data.json --validate-only

# Convert dataset format
python scripts/data_converter.py --input data.json --output converted.json --format auto
```

### 🔍 Analysis Summary

#### Root Cause of Loss/Grad_norm Issue
The original issue was caused by **improper parameter unfreezing in DeepSpeed ZeRO-3 environments**:

1. **Problem**: Direct parameter access (`p.requires_grad = True`) without ZeRO context manager
2. **Symptom**: Parameters appeared frozen, causing zero gradients and no loss calculation
3. **Solution**: Use `deepspeed.zero.GatheredParameters` context manager for proper parameter access
4. **Result**: Training metrics now calculate correctly with proper gradient flow

#### Validation of Changes
- ✅ **DeepSpeed integration**: Significantly improves ZeRO-3 compatibility
- ✅ **Parameter logging**: Essential for debugging training issues
- ✅ **Tensor parallelism**: Future-proofs for larger model training
- ✅ **Dataloader workers**: Critical for reproducibility and performance
- ✅ **Conversation format**: Properly handles expected dataset format
- ✅ **Dataset validation**: Prevents training failures from malformed data

### 🚀 Performance Impact
- **Training stability**: Eliminated parameter access errors in distributed settings
- **Debugging capability**: Real-time parameter status monitoring
- **Data loading**: Restored optimized prefetching and worker seeding
- **Memory efficiency**: Proper parameter gathering in ZeRO-3
- **Scalability**: Enhanced multi-GPU training support

### 📝 Notes
- All changes maintain backward compatibility with existing Parsit models
- Dataset format `{"from": "human"/"gpt", "value": "..."}` works natively without conversion
- Training scripts now provide detailed parameter status logging for transparency
- DeepSpeed ZeRO-3 optimization is significantly more robust and stable

---

## Previous Versions

### [Base] - LLaVA-NeXT Fork
Initial fork from LLaVA-NeXT with basic document analysis adaptations.

---

**Note**: This changelog documents the critical improvements made to resolve training issues and enhance the Parsit training infrastructure. All changes have been validated for compatibility with transformers 4.53.2 and maintain the core LLaVA-NeXT multimodal capabilities.