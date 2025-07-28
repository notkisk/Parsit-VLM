"""
Qwen2.5-VL Fine-tuning System

A comprehensive fine-tuning framework for Qwen2.5-VL models with support for:
- Full fine-tuning and LoRA/QLoRA
- Multi-GPU training with DeepSpeed
- Dynamic resolution image/video processing
- Memory-efficient training optimizations
"""

__version__ = "0.1.0"
__author__ = "Qwen2.5-VL Fine-tuning Team"

try:
    from .models import Qwen25VLModel, load_qwen25vl_model
    from .training import DataProcessor
    from .inference import InferenceEngine
    
    # These might fail due to dependencies, so import conditionally
    try:
        from .training import Qwen25VLTrainer, TrainingArguments
        HAS_TRAINER = True
    except ImportError:
        HAS_TRAINER = False
        
    __all__ = [
        "Qwen25VLModel",
        "load_qwen25vl_model", 
        "DataProcessor",
        "InferenceEngine",
    ]
    
    if HAS_TRAINER:
        __all__.extend(["Qwen25VLTrainer", "TrainingArguments"])
        
except ImportError as e:
    # Minimal imports for basic functionality
    __all__ = []