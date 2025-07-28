"""
Enhanced trainer for Qwen2.5-VL with DeepSpeed and LoRA support
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from transformers import (
    Trainer,
    TrainingArguments as HFTrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
import deepspeed
from peft import PeftModel
import logging
import wandb
from accelerate import Accelerator

from ..models import Qwen25VLModel
from ..utils import TrainingLogger, get_memory_stats, clear_memory_cache, log_model_parameters

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(HFTrainingArguments):
    """Extended training arguments for Qwen2.5-VL fine-tuning"""
    
    # Model specific arguments
    model_size: str = field(default="3B", metadata={"help": "Model size: 3B or 7B"})
    use_lora: bool = field(default=False, metadata={"help": "Use LoRA for parameter-efficient training"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[str] = field(default=None, metadata={"help": "LoRA target modules (comma-separated)"})
    
    # Training optimization
    use_deepspeed: bool = field(default=True, metadata={"help": "Use DeepSpeed for training"})
    deepspeed_config: Optional[str] = field(default=None, metadata={"help": "Path to DeepSpeed config file"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    use_flash_attention: bool = field(default=True, metadata={"help": "Use flash attention"})
    
    # Memory optimization
    use_cpu_offload: bool = field(default=False, metadata={"help": "Offload optimizer states to CPU"})
    max_memory_gb: Optional[float] = field(default=None, metadata={"help": "Maximum GPU memory to use (GB)"})
    
    # Data arguments
    conversation_template: str = field(default="chatml", metadata={"help": "Conversation template to use"})
    max_image_tokens: int = field(default=4096, metadata={"help": "Maximum number of image tokens"})
    image_aspect_ratio: str = field(default="anyres", metadata={"help": "Image aspect ratio handling"})
    
    # Logging and monitoring
    log_memory_usage: bool = field(default=True, metadata={"help": "Log memory usage during training"})
    memory_logging_interval: int = field(default=100, metadata={"help": "Memory logging interval (steps)"})
    
    # Checkpoint management
    save_safetensors: bool = field(default=True, metadata={"help": "Save checkpoints in safetensors format"})
    checkpoint_keep_steps: Optional[int] = field(default=None, metadata={"help": "Keep checkpoints for last N steps"})


class Qwen25VLTrainer(Trainer):
    """Enhanced trainer for Qwen2.5-VL with advanced features"""
    
    def __init__(
        self,
        model: Union[Qwen25VLModel, nn.Module],
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        **kwargs
    ):
        # Initialize training logger
        self.training_logger = TrainingLogger("qwen25vl_trainer", args.logging_steps)
        
        # Store original arguments
        self.qwen_args = args
        
        # Set up memory monitoring
        self.log_memory_usage = args.log_memory_usage
        self.memory_logging_interval = args.memory_logging_interval
        
        # Initialize parent class
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs
        )
        
        # Apply training optimizations
        self._setup_training_optimizations()
        
        # Log model information
        if isinstance(model, Qwen25VLModel):
            log_model_parameters(model.model, logger)
        else:
            log_model_parameters(model, logger)
            
        logger.info("Qwen25VLTrainer initialized successfully")
        
    def _setup_training_optimizations(self):
        """Set up training optimizations"""
        
        # Enable gradient checkpointing
        if self.qwen_args.gradient_checkpointing:
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
            elif hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
            
        # Memory optimizations
        if torch.cuda.is_available():
            # Set memory fraction if specified
            if self.qwen_args.max_memory_gb:
                for i in range(torch.cuda.device_count()):
                    total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    memory_fraction = min(self.qwen_args.max_memory_gb / total_memory, 1.0)
                    torch.cuda.set_per_process_memory_fraction(memory_fraction, i)
                logger.info(f"Set GPU memory fraction to {memory_fraction:.2f}")
                
            # Clear cache
            clear_memory_cache()
            
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items: int = None) -> torch.Tensor:
        """Enhanced training step with memory monitoring"""
        
        # Log memory usage periodically
        if (self.log_memory_usage and 
            self.state.global_step % self.memory_logging_interval == 0):
            self.training_logger.log_memory_usage(f"Step {self.state.global_step} - Before forward: ")
            
        # Perform training step
        if num_items is not None:
            loss = super().training_step(model, inputs, num_items)
        else:
            loss = super().training_step(model, inputs)
        
        # Log additional metrics
        current_lr = self.get_lr()
        
        # Log step information
        self.training_logger.log_step(
            step=self.state.global_step,
            total_steps=self.state.max_steps if self.state.max_steps > 0 else self.state.num_train_epochs * len(self.get_train_dataloader()),
            loss=loss.item() if isinstance(loss, torch.Tensor) else loss,
            lr=current_lr,
            metrics={}
        )
        
        return loss
        
    def get_lr(self) -> float:
        """Get current learning rate"""
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_last_lr()[0]
        else:
            return self.args.learning_rate
            
    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with memory monitoring"""
        
        # Add memory usage to logs
        if self.log_memory_usage:
            memory_stats = get_memory_stats()
            if "gpu" in memory_stats and memory_stats["gpu"].get("available", True):
                for gpu_key, gpu_stats in memory_stats["gpu"].items():
                    if gpu_key.startswith("gpu_"):
                        device_id = gpu_key.split("_")[1]
                        logs[f"memory/gpu_{device_id}_allocated_gb"] = gpu_stats["allocated_gb"]
                        logs[f"memory/gpu_{device_id}_utilization_percent"] = gpu_stats["utilization_percent"]
                        
            logs["memory/system_used_gb"] = memory_stats["system"]["used_gb"]
            logs["memory/system_usage_percent"] = memory_stats["system"]["usage_percent"]
            
        # Add gradient norm if available
        if hasattr(self, '_grad_norm'):
            logs["train/grad_norm"] = self._grad_norm
            
        # Call parent logging
        super().log(logs)
        
    def _save_checkpoint(self, model, metrics=None):
        """Enhanced checkpoint saving with LoRA support"""
        
        logger.info(f"Saving checkpoint at step {self.state.global_step}")
        
        # Handle LoRA model saving
        actual_model = getattr(model, 'model', model)  # Unwrap if needed
        if isinstance(actual_model, PeftModel) or hasattr(actual_model, 'save_pretrained'):
            try:
                # Save LoRA adapter
                checkpoint_folder = f"checkpoint-{self.state.global_step}"
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                
                # Create directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Save PEFT model
                if hasattr(actual_model, 'save_pretrained'):
                    actual_model.save_pretrained(output_dir)
                
                # Save training state
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
                
                # Save training arguments
                torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                
                logger.info(f"LoRA checkpoint saved to {output_dir}")
                
            except Exception as e:
                logger.warning(f"LoRA checkpoint saving failed: {e}")
                logger.info("Attempting standard checkpoint saving...")
                try:
                    super()._save_checkpoint(model, metrics)
                except Exception as e2:
                    logger.error(f"Standard checkpoint saving also failed: {e2}")
                    logger.info("Training will continue without saving this checkpoint")
        else:
            # Standard checkpoint saving
            super()._save_checkpoint(model, metrics)
            
        # Clean up old checkpoints if specified
        if self.qwen_args.checkpoint_keep_steps:
            self._cleanup_old_checkpoints()
            
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to save disk space"""
        
        checkpoint_dirs = []
        for item in os.listdir(self.args.output_dir):
            if item.startswith("checkpoint-"):
                try:
                    step = int(item.split("-")[1])
                    checkpoint_dirs.append((step, item))
                except ValueError:
                    continue
                    
        # Sort by step number (descending)
        checkpoint_dirs.sort(reverse=True)
        
        # Keep only the most recent N checkpoints
        keep_count = self.qwen_args.checkpoint_keep_steps
        for step, dirname in checkpoint_dirs[keep_count:]:
            checkpoint_path = os.path.join(self.args.output_dir, dirname)
            try:
                import shutil
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
                
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with memory monitoring"""
        
        logger.info("Starting evaluation...")
        
        with torch.no_grad():
            # Clear cache before evaluation
            clear_memory_cache()
            
            # Run evaluation
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            
            # Log evaluation results
            eval_loss = metrics.get(f"{metric_key_prefix}_loss", 0.0)
            self.training_logger.log_epoch(
                epoch=int(self.state.epoch) if self.state.epoch else 0,
                train_loss=self.state.log_history[-1].get("train_loss", 0.0) if self.state.log_history else 0.0,
                eval_loss=eval_loss
            )
            
        return metrics
        
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Enhanced training with comprehensive monitoring"""
        
        logger.info("Starting Qwen2.5-VL training...")
        
        # Log training configuration
        config_dict = {
            "model_size": self.qwen_args.model_size,
            "use_lora": self.qwen_args.use_lora,
            "use_deepspeed": self.qwen_args.use_deepspeed,
            "gradient_checkpointing": self.qwen_args.gradient_checkpointing,
            "per_device_train_batch_size": self.args.per_device_train_batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "learning_rate": self.args.learning_rate,
            "num_train_epochs": self.args.num_train_epochs,
            "warmup_ratio": self.args.warmup_ratio,
            "weight_decay": self.args.weight_decay,
            "lr_scheduler_type": self.args.lr_scheduler_type,
        }
        
        from ..utils.logging_utils import log_training_config
        log_training_config(config_dict, logger)
        
        # Log initial memory usage
        if self.log_memory_usage:
            self.training_logger.log_memory_usage("Initial: ")
            
        # Start training
        try:
            result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
            
            logger.info("Training completed successfully!")
            
            # Log training summary
            summary = self.training_logger.get_metrics_summary()
            logger.info("Training Summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")
                
            return result
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
            
        finally:
            # Final memory cleanup
            clear_memory_cache()
            
    def create_optimizer(self):
        """Create optimizer with DeepSpeed compatibility"""
        
        if self.qwen_args.use_deepspeed and self.args.deepspeed:
            # DeepSpeed will handle optimizer creation
            return None
        else:
            # Use parent implementation
            return super().create_optimizer()
            
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Create learning rate scheduler"""
        
        if self.qwen_args.use_deepspeed and self.args.deepspeed:
            # DeepSpeed will handle scheduler creation
            return None
        else:
            # Use parent implementation
            return super().create_scheduler(num_training_steps, optimizer)
            
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with gradient tracking for monitoring"""
        
        # Compute loss
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False)
            outputs = None
            
        # Track gradient norm for monitoring
        if self.model.training:
            total_norm = 0
            param_count = 0
            
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                self._grad_norm = total_norm
                
        return (loss, outputs) if return_outputs else loss