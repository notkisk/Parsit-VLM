"""
Enhanced logging utilities for training monitoring
"""

import os
import sys
import logging
import time
from typing import Optional, Dict, Any
import torch
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    format_string: Optional[str] = None,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Set up comprehensive logging configuration
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        format_string: Custom format string
        log_dir: Directory to save logs (creates timestamped file if log_file not provided)
        
    Returns:
        Configured logger instance
    """
    
    # Configure log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    if format_string is None:
        format_string = (
            "[%(asctime)s] [%(levelname)8s] [%(name)s:%(lineno)d] %(message)s"
        )
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file or log_dir:
        if log_file is None and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"training_{timestamp}.log")
            
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_file}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


class TrainingLogger:
    """Enhanced logger for training metrics and progress"""
    
    def __init__(self, name: str = "training", log_interval: int = 100):
        self.logger = get_logger(name)
        self.log_interval = log_interval
        self.start_time = time.time()
        self.step_times = []
        self.metrics_history = []
        
    def log_step(
        self,
        step: int,
        total_steps: int,
        loss: float,
        lr: float,
        metrics: Optional[Dict[str, Any]] = None,
        force_log: bool = False
    ):
        """Log training step information"""
        
        current_time = time.time()
        step_time = current_time - (self.step_times[-1] if self.step_times else self.start_time)
        self.step_times.append(current_time)
        
        # Keep only recent step times for accurate rate calculation
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]
            
        if step % self.log_interval == 0 or force_log or step == total_steps:
            # Calculate rates
            if len(self.step_times) > 1:
                avg_step_time = sum(self.step_times[-10:]) / len(self.step_times[-10:]) if len(self.step_times) >= 10 else step_time
                steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
            else:
                steps_per_sec = 0
                
            # Calculate ETA
            remaining_steps = total_steps - step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600
            
            # Format progress
            progress_pct = (step / total_steps) * 100
            
            log_msg = (
                f"Step {step:6d}/{total_steps} ({progress_pct:5.1f}%) | "
                f"Loss: {loss:.6f} | LR: {lr:.2e} | "
                f"Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_hours:.1f}h"
            )
            
            # Add additional metrics
            if metrics:
                metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
                log_msg += " | " + " | ".join(metric_strs)
                
            self.logger.info(log_msg)
            
            # Store metrics for history
            self.metrics_history.append({
                "step": step,
                "loss": loss,
                "lr": lr,
                "steps_per_sec": steps_per_sec,
                "timestamp": current_time,
                **(metrics or {})
            })
            
    def log_epoch(self, epoch: int, train_loss: float, eval_loss: Optional[float] = None):
        """Log epoch summary"""
        
        elapsed_time = time.time() - self.start_time
        elapsed_hours = elapsed_time / 3600
        
        log_msg = f"Epoch {epoch} completed | Train Loss: {train_loss:.6f}"
        if eval_loss is not None:
            log_msg += f" | Eval Loss: {eval_loss:.6f}"
        log_msg += f" | Elapsed: {elapsed_hours:.1f}h"
        
        self.logger.info("=" * 80)
        self.logger.info(log_msg)
        self.logger.info("=" * 80)
        
    def log_memory_usage(self, prefix: str = ""):
        """Log current memory usage"""
        log_memory_usage(self.logger, prefix)
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics"""
        if not self.metrics_history:
            return {}
            
        recent_metrics = self.metrics_history[-10:]  # Last 10 steps
        
        return {
            "total_steps": len(self.metrics_history),
            "latest_loss": recent_metrics[-1]["loss"],
            "avg_recent_loss": sum(m["loss"] for m in recent_metrics) / len(recent_metrics),
            "avg_steps_per_sec": sum(m["steps_per_sec"] for m in recent_metrics) / len(recent_metrics),
            "total_training_time_hours": (time.time() - self.start_time) / 3600
        }


def log_memory_usage(logger: logging.Logger, prefix: str = ""):
    """Log current GPU memory usage"""
    
    if not torch.cuda.is_available():
        logger.info(f"{prefix}CUDA not available")
        return
        
    device_count = torch.cuda.device_count()
    
    for device_id in range(device_count):
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)   # GB
        max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024**3)  # GB
        
        device_props = torch.cuda.get_device_properties(device_id)
        total_memory = device_props.total_memory / (1024**3)  # GB
        
        logger.info(
            f"{prefix}GPU {device_id} ({device_props.name}): "
            f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | "
            f"Max Allocated: {max_allocated:.2f}GB | Total: {total_memory:.2f}GB | "
            f"Utilization: {(allocated/total_memory)*100:.1f}%"
        )


def log_model_parameters(model, logger: logging.Logger):
    """Log detailed model parameter information"""
    
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    param_by_layer = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        else:
            frozen_params += param_count
            
        # Group by layer type
        layer_type = name.split('.')[0] if '.' in name else name
        if layer_type not in param_by_layer:
            param_by_layer[layer_type] = {"total": 0, "trainable": 0}
        param_by_layer[layer_type]["total"] += param_count
        if param.requires_grad:
            param_by_layer[layer_type]["trainable"] += param_count
    
    logger.info("=" * 60)
    logger.info("MODEL PARAMETER SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
    logger.info(f"Frozen parameters: {frozen_params:,} ({(frozen_params/total_params)*100:.2f}%)")
    logger.info("")
    logger.info("Parameters by layer:")
    
    for layer_type, counts in sorted(param_by_layer.items()):
        trainable_pct = (counts["trainable"] / counts["total"]) * 100 if counts["total"] > 0 else 0
        logger.info(f"  {layer_type}: {counts['total']:,} total, {counts['trainable']:,} trainable ({trainable_pct:.1f}%)")
    
    logger.info("=" * 60)


def log_training_config(config: Dict[str, Any], logger: logging.Logger):
    """Log training configuration details"""
    
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    
    for key, value in sorted(config.items()):
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
            
    logger.info("=" * 60)