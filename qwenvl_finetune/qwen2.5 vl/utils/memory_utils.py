"""
Memory management and monitoring utilities
"""

import gc
import torch
import psutil
import threading
import time
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


def get_memory_stats() -> Dict[str, Any]:
    """
    Get comprehensive memory statistics
    
    Returns:
        Dictionary with memory usage information
    """
    stats = {}
    
    # System memory
    system_memory = psutil.virtual_memory()
    stats["system"] = {
        "total_gb": system_memory.total / (1024**3),
        "available_gb": system_memory.available / (1024**3),
        "used_gb": system_memory.used / (1024**3),
        "usage_percent": system_memory.percent
    }
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_stats = {}
        for device_id in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(device_id)
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            max_allocated = torch.cuda.max_memory_allocated(device_id)
            
            gpu_stats[f"gpu_{device_id}"] = {
                "name": device_props.name,
                "total_gb": device_props.total_memory / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "max_allocated_gb": max_allocated / (1024**3),
                "free_gb": (device_props.total_memory - reserved) / (1024**3),
                "utilization_percent": (allocated / device_props.total_memory) * 100
            }
        stats["gpu"] = gpu_stats
    else:
        stats["gpu"] = {"available": False}
    
    return stats


def clear_memory_cache():
    """Clear GPU and system memory caches"""
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    # Force garbage collection
    gc.collect()
    
    logger.info("Memory caches cleared")


@contextmanager
def memory_monitor(operation_name: str = "operation"):
    """
    Context manager to monitor memory usage during an operation
    
    Args:
        operation_name: Name of the operation being monitored
    """
    
    logger.info(f"Starting memory monitoring for: {operation_name}")
    
    # Get initial memory stats
    initial_stats = get_memory_stats()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    try:
        yield
    finally:
        # Get final memory stats
        final_stats = get_memory_stats()
        
        # Calculate memory changes
        system_change = (
            final_stats["system"]["used_gb"] - initial_stats["system"]["used_gb"]
        )
        
        logger.info(f"Memory usage for {operation_name}:")
        logger.info(f"  System memory change: {system_change:+.2f} GB")
        
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                gpu_key = f"gpu_{device_id}"
                if gpu_key in final_stats["gpu"]:
                    initial_gpu = initial_stats["gpu"][gpu_key]["allocated_gb"]
                    final_gpu = final_stats["gpu"][gpu_key]["allocated_gb"]
                    max_gpu = final_stats["gpu"][gpu_key]["max_allocated_gb"]
                    
                    logger.info(f"  GPU {device_id} memory change: {final_gpu - initial_gpu:+.2f} GB")
                    logger.info(f"  GPU {device_id} peak usage: {max_gpu:.2f} GB")


class MemoryMonitor:
    """Continuous memory monitoring in a separate thread"""
    
    def __init__(
        self,
        interval: float = 10.0,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.interval = interval
        self.callback = callback
        self.running = False
        self.thread = None
        self.stats_history = []
        
    def start(self):
        """Start memory monitoring"""
        if self.running:
            logger.warning("Memory monitor is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"Memory monitoring started (interval: {self.interval}s)")
        
    def stop(self):
        """Stop memory monitoring"""
        if not self.running:
            logger.warning("Memory monitor is not running")
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.interval + 1)
        logger.info("Memory monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                stats = get_memory_stats()
                stats["timestamp"] = time.time()
                
                # Store stats history (keep last 100 entries)
                self.stats_history.append(stats)
                if len(self.stats_history) > 100:
                    self.stats_history = self.stats_history[-100:]
                
                # Call callback if provided
                if self.callback:
                    self.callback(stats)
                    
                # Check for memory issues
                self._check_memory_issues(stats)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                
            time.sleep(self.interval)
            
    def _check_memory_issues(self, stats: Dict[str, Any]):
        """Check for potential memory issues and log warnings"""
        
        # System memory check
        if stats["system"]["usage_percent"] > 90:
            logger.warning(f"High system memory usage: {stats['system']['usage_percent']:.1f}%")
            
        # GPU memory check
        if "gpu" in stats and stats["gpu"].get("available", True):
            for gpu_key, gpu_stats in stats["gpu"].items():
                if gpu_key.startswith("gpu_") and gpu_stats["utilization_percent"] > 95:
                    logger.warning(
                        f"High GPU memory usage on {gpu_key}: {gpu_stats['utilization_percent']:.1f}%"
                    )
                    
    def get_peak_usage(self) -> Dict[str, Any]:
        """Get peak memory usage from monitoring history"""
        if not self.stats_history:
            return {}
            
        peak_stats = {
            "system": {"max_used_gb": 0, "max_usage_percent": 0},
            "gpu": {}
        }
        
        for stats in self.stats_history:
            # System peak
            system_used = stats["system"]["used_gb"]
            system_percent = stats["system"]["usage_percent"]
            
            peak_stats["system"]["max_used_gb"] = max(
                peak_stats["system"]["max_used_gb"], system_used
            )
            peak_stats["system"]["max_usage_percent"] = max(
                peak_stats["system"]["max_usage_percent"], system_percent
            )
            
            # GPU peaks
            if "gpu" in stats and stats["gpu"].get("available", True):
                for gpu_key, gpu_stats in stats["gpu"].items():
                    if gpu_key.startswith("gpu_"):
                        if gpu_key not in peak_stats["gpu"]:
                            peak_stats["gpu"][gpu_key] = {
                                "max_allocated_gb": 0,
                                "max_utilization_percent": 0
                            }
                            
                        peak_stats["gpu"][gpu_key]["max_allocated_gb"] = max(
                            peak_stats["gpu"][gpu_key]["max_allocated_gb"],
                            gpu_stats["allocated_gb"]
                        )
                        peak_stats["gpu"][gpu_key]["max_utilization_percent"] = max(
                            peak_stats["gpu"][gpu_key]["max_utilization_percent"],
                            gpu_stats["utilization_percent"]
                        )
                        
        return peak_stats


def estimate_model_memory(
    param_count: int,
    precision: str = "bfloat16",
    training: bool = True,
    optimizer: str = "adamw"
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model
    
    Args:
        param_count: Number of model parameters
        precision: Model precision ("float32", "bfloat16", "float16")
        training: Whether this is for training (includes optimizer states)
        optimizer: Optimizer type ("adamw", "sgd")
        
    Returns:
        Dictionary with memory estimates in GB
    """
    
    # Bytes per parameter based on precision
    precision_bytes = {
        "float32": 4,
        "bfloat16": 2,
        "float16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    param_bytes = precision_bytes.get(precision, 2)
    
    # Model parameters
    model_memory = param_count * param_bytes / (1024**3)
    
    estimates = {
        "model_params_gb": model_memory,
        "gradients_gb": model_memory if training else 0,  # Same size as model for gradients
        "optimizer_states_gb": 0
    }
    
    # Optimizer states
    if training:
        if optimizer.lower() == "adamw":
            # AdamW stores first and second moments (2x model size) + other states
            estimates["optimizer_states_gb"] = model_memory * 2.5
        elif optimizer.lower() == "sgd":
            # SGD with momentum stores momentum (1x model size)
            estimates["optimizer_states_gb"] = model_memory * 1.2
            
    # Total memory
    estimates["total_model_gb"] = sum(estimates.values())
    
    # Add overhead for activations, buffers, etc. (rough estimate)
    estimates["activation_overhead_gb"] = estimates["total_model_gb"] * 0.2
    estimates["total_estimated_gb"] = estimates["total_model_gb"] + estimates["activation_overhead_gb"]
    
    return estimates


def optimize_memory_for_training(
    model,
    enable_gradient_checkpointing: bool = True,
    clear_cache: bool = True
):
    """
    Apply memory optimizations for training
    
    Args:
        model: The model to optimize
        enable_gradient_checkpointing: Whether to enable gradient checkpointing
        clear_cache: Whether to clear memory cache
    """
    
    logger.info("Applying memory optimizations...")
    
    # Clear cache
    if clear_cache:
        clear_memory_cache()
        
    # Enable gradient checkpointing
    if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
        
    # Set model to use memory efficient attention if available
    if hasattr(model, 'config') and hasattr(model.config, 'use_memory_efficient_attention'):
        model.config.use_memory_efficient_attention = True
        logger.info("Memory efficient attention enabled")
        
    # Log memory usage after optimization
    stats = get_memory_stats()
    if "gpu" in stats and stats["gpu"].get("available", True):
        for gpu_key, gpu_stats in stats["gpu"].items():
            if gpu_key.startswith("gpu_"):
                logger.info(f"{gpu_key}: {gpu_stats['allocated_gb']:.2f}GB allocated, "
                           f"{gpu_stats['free_gb']:.2f}GB free")


@contextmanager
def temporary_memory_optimization():
    """
    Temporary context for memory optimization during specific operations
    """
    
    # Clear cache before operation
    clear_memory_cache()
    
    try:
        yield
    finally:
        # Clear cache after operation
        clear_memory_cache()