"""
Utility functions for model loading and configuration
"""

import os
import json
import torch
from typing import Dict, Any, Optional, List, Union
from transformers import Qwen2_5_VLConfig
from peft import LoraConfig, TaskType
import logging

from .qwen25vl_model import Qwen25VLModel

logger = logging.getLogger(__name__)


def get_model_config(model_size: str = "3B") -> Dict[str, Any]:
    """
    Get optimized configuration for different Qwen2.5-VL model sizes
    
    Args:
        model_size: Model size ("3B" or "7B")
        
    Returns:
        Dictionary with model configuration
    """
    
    base_configs = {
        "3B": {
            "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
            "max_sequence_length": 32768,
            "recommended_batch_size": 4,
            "recommended_gradient_accumulation": 4,
            "min_vram_gb": 12,
            "recommended_vram_gb": 16,
            "supports_quantization": True,
        },
        "7B": {
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct", 
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
            "max_sequence_length": 32768,
            "recommended_batch_size": 2,
            "recommended_gradient_accumulation": 8,
            "min_vram_gb": 24,
            "recommended_vram_gb": 32,
            "supports_quantization": True,
        }
    }
    
    if model_size not in base_configs:
        raise ValueError(f"Unsupported model size: {model_size}. Supported: {list(base_configs.keys())}")
        
    return base_configs[model_size]


def get_lora_config(
    model_size: str = "3B",
    task_complexity: str = "medium",
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get LoRA configuration optimized for different scenarios
    
    Args:
        model_size: Model size ("3B" or "7B")
        task_complexity: Task complexity ("light", "medium", "heavy")
        custom_config: Custom LoRA configuration to override defaults
        
    Returns:
        LoRA configuration dictionary
    """
    
    # Base LoRA configurations for different complexity levels
    complexity_configs = {
        "light": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
        },
        "medium": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none", 
            "task_type": TaskType.CAUSAL_LM,
        },
        "heavy": {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
        }
    }
    
    # Model-specific adjustments
    model_adjustments = {
        "3B": {
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        },
        "7B": {
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head"  # Include more modules for larger model
            ]
        }
    }
    
    if task_complexity not in complexity_configs:
        raise ValueError(f"Unsupported task complexity: {task_complexity}")
        
    if model_size not in model_adjustments:
        raise ValueError(f"Unsupported model size: {model_size}")
        
    # Combine configurations
    config = complexity_configs[task_complexity].copy()
    config.update(model_adjustments[model_size])
    
    # Apply custom overrides
    if custom_config:
        config.update(custom_config)
        
    return config


def apply_lora_config(model: Qwen25VLModel, lora_config: Dict[str, Any]) -> Qwen25VLModel:
    """
    Apply LoRA configuration to a model
    
    Args:
        model: Qwen25VLModel instance
        lora_config: LoRA configuration dictionary
        
    Returns:
        Model with LoRA applied
    """
    
    logger.info("Applying LoRA configuration to model")
    model.apply_lora(lora_config)
    
    return model


def load_qwen25vl_model(
    model_size: str = "3B",
    quantization: Optional[str] = None,
    use_lora: bool = False,
    lora_config: Optional[Dict[str, Any]] = None,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    custom_model_path: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    attn_implementation: str = "eager",
    **kwargs
) -> Qwen25VLModel:
    """
    Load Qwen2.5-VL model with specified configuration
    
    Args:
        model_size: Model size ("3B" or "7B")
        quantization: Quantization method ("4bit", "8bit", None)
        use_lora: Whether to apply LoRA
        lora_config: Custom LoRA configuration
        device_map: Device mapping strategy
        torch_dtype: Torch data type
        custom_model_path: Custom path to model (overrides model_size)
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Loaded Qwen25VLModel instance
    """
    
    # Get model configuration
    if custom_model_path:
        model_name_or_path = custom_model_path
        logger.info(f"Using custom model path: {custom_model_path}")
    else:
        model_config = get_model_config(model_size)
        model_name_or_path = model_config["model_name"]
        logger.info(f"Loading {model_size} model: {model_name_or_path}")
    
    # Set up quantization (prioritize direct parameters over quantization string)
    if quantization == "4bit":
        load_in_4bit = True
    elif quantization == "8bit":
        load_in_8bit = True
    
    if load_in_4bit:
        logger.info("Using 4-bit quantization")
    elif load_in_8bit:
        logger.info("Using 8-bit quantization")
        
    # Load model
    model = Qwen25VLModel(
        model_name_or_path=model_name_or_path,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        **kwargs
    )
    
    # Apply LoRA if requested
    if use_lora:
        if lora_config is None:
            lora_config = get_lora_config(model_size)
            logger.info("Using default LoRA configuration")
        else:
            logger.info("Using custom LoRA configuration")
            
        model = apply_lora_config(model, lora_config)
        
    # Print model information
    param_stats = model.get_parameter_count()
    memory_stats = model.get_memory_footprint()
    
    logger.info(f"Model loaded successfully:")
    logger.info(f"  Total parameters: {param_stats['total']:,}")
    logger.info(f"  Trainable parameters: {param_stats['trainable']:,} ({param_stats['trainable_percentage']:.2f}%)")
    logger.info(f"  Memory footprint: {memory_stats.get('model_memory_mb', 0):.1f} MB")
    
    return model


def estimate_memory_requirements(
    model_size: str,
    batch_size: int,
    sequence_length: int = 2048,
    use_quantization: bool = False,
    use_lora: bool = False,
    gradient_checkpointing: bool = True
) -> Dict[str, float]:
    """
    Estimate memory requirements for training
    
    Args:
        model_size: Model size ("3B" or "7B")
        batch_size: Training batch size  
        sequence_length: Maximum sequence length
        use_quantization: Whether quantization is used
        use_lora: Whether LoRA is used
        gradient_checkpointing: Whether gradient checkpointing is enabled
        
    Returns:
        Dictionary with memory estimates in GB
    """
    
    # Parameter counts (approximate)
    param_counts = {
        "3B": 3_000_000_000,
        "7B": 7_000_000_000
    }
    
    if model_size not in param_counts:
        raise ValueError(f"Unsupported model size: {model_size}")
        
    param_count = param_counts[model_size]
    
    # Base model memory (parameters + buffers)
    if use_quantization:
        model_memory = param_count * 1.5 / (1024**3)  # ~1.5 bytes per parameter with quantization
    else:
        model_memory = param_count * 4 / (1024**3)    # 4 bytes per parameter (float32/bfloat16)
        
    # Optimizer memory (AdamW: ~8 bytes per parameter)
    if use_lora:
        # LoRA reduces trainable parameters significantly
        trainable_params = param_count * 0.01  # ~1% of parameters
    else:
        trainable_params = param_count
        
    optimizer_memory = trainable_params * 8 / (1024**3)
    
    # Activation memory (depends on batch size and sequence length)
    activation_memory = batch_size * sequence_length * param_count * 4 / (1024**3) / 1000  # Rough estimate
    
    if gradient_checkpointing:
        activation_memory *= 0.3  # Gradient checkpointing reduces activation memory
        
    # Buffer and overhead
    overhead = 2.0  # GB
    
    total_memory = model_memory + optimizer_memory + activation_memory + overhead
    
    return {
        "model_memory_gb": model_memory,
        "optimizer_memory_gb": optimizer_memory, 
        "activation_memory_gb": activation_memory,
        "overhead_gb": overhead,
        "total_memory_gb": total_memory,
        "recommended_vram_gb": total_memory * 1.2  # Add 20% safety margin
    }


def validate_hardware_requirements(
    model_size: str,
    batch_size: int,
    use_quantization: bool = False,
    use_lora: bool = False
) -> Dict[str, Any]:
    """
    Validate hardware requirements and provide recommendations
    
    Args:
        model_size: Model size ("3B" or "7B") 
        batch_size: Training batch size
        use_quantization: Whether quantization is used
        use_lora: Whether LoRA is used
        
    Returns:
        Dictionary with validation results and recommendations
    """
    
    # Get memory estimates
    memory_est = estimate_memory_requirements(
        model_size=model_size,
        batch_size=batch_size,
        use_quantization=use_quantization,
        use_lora=use_lora
    )
    
    # Check available GPU memory
    gpu_available = 0
    gpu_count = 0
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_available = max(gpu_available, gpu_memory)
            
    results = {
        "memory_estimate": memory_est,
        "gpu_count": gpu_count,
        "max_gpu_memory_gb": gpu_available,
        "requirements_met": gpu_available >= memory_est["recommended_vram_gb"],
        "recommendations": []
    }
    
    # Generate recommendations
    if not results["requirements_met"]:
        if not use_quantization:
            results["recommendations"].append("Enable 4-bit quantization to reduce memory usage")
        if not use_lora:
            results["recommendations"].append("Use LoRA to significantly reduce memory requirements")
        if batch_size > 1:
            results["recommendations"].append(f"Reduce batch size from {batch_size} to 1")
        results["recommendations"].append("Enable gradient checkpointing")
        results["recommendations"].append("Consider using DeepSpeed ZeRO-3 with CPU offloading")
        
    return results