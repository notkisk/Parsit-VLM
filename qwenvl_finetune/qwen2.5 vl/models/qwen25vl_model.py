"""
Qwen2.5-VL Model wrapper with enhanced functionality for fine-tuning
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import logging

logger = logging.getLogger(__name__)


class Qwen25VLModel(nn.Module):
    """
    Enhanced Qwen2.5-VL model wrapper for fine-tuning with advanced features
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        config: Optional[Qwen2_5_VLConfig] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "eager",
        trust_remote_code: bool = True,
        device_map: str = "auto",
        **kwargs
    ):
        super().__init__()
        
        self.model_name_or_path = model_name_or_path
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        
        # Set up quantization config if needed
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
        # Set torch dtype
        dtype_mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_mapping.get(torch_dtype, torch.bfloat16)
        
        # Load model
        logger.info(f"Loading Qwen2.5-VL model: {model_name_or_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            config=config,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            **kwargs
        )
        
        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        
        # Set model to train mode
        self.model.train()
        
        # Store model info
        self.config = self.model.config
        self.is_peft_model = False
        self.peft_config = None
        
        logger.info(f"Model loaded successfully. Parameters: {self.get_parameter_count()}")
        
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for the model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0
        }
        
    def apply_lora(
        self,
        lora_config: Dict[str, Any],
        target_modules: Optional[List[str]] = None
    ) -> None:
        """Apply LoRA configuration to the model"""
        
        if target_modules is None:
            # Default target modules for Qwen2.5-VL
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head"
            ]
            
        # Remove conflicting keys from lora_config  
        clean_lora_config = {k: v for k, v in lora_config.items() 
                           if k not in ['enabled', 'task_type', 'target_modules']}
        
        lora_config_obj = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            **clean_lora_config
        )
        
        logger.info(f"Applying LoRA with config: {lora_config}")
        self.model = get_peft_model(self.model, lora_config_obj)
        self.is_peft_model = True
        self.peft_config = lora_config_obj
        
        # Print parameter statistics after LoRA
        param_stats = self.get_parameter_count()
        logger.info(f"LoRA applied. Trainable parameters: {param_stats['trainable']:,} "
                   f"({param_stats['trainable_percentage']:.2f}%)")
                   
    def merge_and_unload_lora(self):
        """Merge LoRA weights and unload PEFT model"""
        if self.is_peft_model:
            logger.info("Merging LoRA weights...")
            self.model = self.model.merge_and_unload()
            self.is_peft_model = False
            self.peft_config = None
            logger.info("LoRA weights merged successfully")
        else:
            logger.warning("Model is not a PEFT model, nothing to merge")
            
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model with proper handling of LoRA models"""
        logger.info(f"Saving model to: {save_directory}")
        
        if self.is_peft_model:
            # Save PEFT model
            self.model.save_pretrained(save_directory, **kwargs)
        else:
            # Save full model
            self.model.save_pretrained(save_directory, **kwargs)
            
        # Save processor and tokenizer
        self.processor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        logger.info("Model saved successfully")
        
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        peft_model_path: Optional[str] = None,
        **kwargs
    ):
        """Load model from pretrained checkpoint with optional LoRA weights"""
        
        # Create base model
        model = cls(model_name_or_path, **kwargs)
        
        # Load LoRA weights if specified
        if peft_model_path:
            logger.info(f"Loading LoRA weights from: {peft_model_path}")
            model.model = PeftModel.from_pretrained(model.model, peft_model_path)
            model.is_peft_model = True
            
        return model
        
    def forward(self, **kwargs):
        """Forward pass through the model"""
        return self.model(**kwargs)
        
    def generate(self, **kwargs):
        """Generate text using the model"""
        return self.model.generate(**kwargs)
        
    def prepare_inputs_for_training(
        self,
        images: Optional[List] = None,
        text: Optional[str] = None,
        conversations: Optional[List[Dict]] = None,
        **kwargs
    ):
        """Prepare inputs for training with proper tokenization and image processing"""
        
        if conversations:
            # Handle conversation format
            inputs = self.processor(
                text=conversations,
                images=images,
                return_tensors="pt",
                padding=True,
                **kwargs
            )
        elif text:
            # Handle simple text input
            inputs = self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                padding=True,
                **kwargs
            )
        else:
            raise ValueError("Either 'text' or 'conversations' must be provided")
            
        return inputs
        
    def get_memory_footprint(self) -> Dict[str, float]:
        """Get memory footprint of the model"""
        if hasattr(self.model, 'get_memory_footprint'):
            return {
                "model_memory_mb": self.model.get_memory_footprint() / (1024 ** 2),
                "allocated_memory_mb": torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0,
                "cached_memory_mb": torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0
            }
        else:
            return {
                "allocated_memory_mb": torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0,
                "cached_memory_mb": torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0
            }
            
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Gradient checkpointing not available for this model")
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing - compatibility method for HF Trainer"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            if gradient_checkpointing_kwargs:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            else:
                self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Gradient checkpointing not available for this model")
            
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled")
            
    def freeze_vision_tower(self):
        """Freeze vision encoder parameters"""
        if hasattr(self.model, 'vision_tower'):
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False
            logger.info("Vision tower frozen")
        else:
            logger.warning("Vision tower not found in model")
            
    def unfreeze_vision_tower(self):
        """Unfreeze vision encoder parameters"""
        if hasattr(self.model, 'vision_tower'):
            for param in self.model.vision_tower.parameters():
                param.requires_grad = True
            logger.info("Vision tower unfrozen")
        else:
            logger.warning("Vision tower not found in model")