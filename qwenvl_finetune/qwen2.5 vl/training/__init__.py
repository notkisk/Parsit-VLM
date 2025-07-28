"""
Training components for Qwen2.5-VL fine-tuning
"""

from .data_processor import DataProcessor, ConversationDataset, DataConfig
from .conversation import ConversationHandler, ChatMLTemplate

try:
    from .trainer import Qwen25VLTrainer, TrainingArguments
    HAS_TRAINER = True
except ImportError:
    HAS_TRAINER = False

__all__ = [
    "DataProcessor",
    "ConversationDataset", 
    "DataConfig",
    "ConversationHandler",
    "ChatMLTemplate",
]

if HAS_TRAINER:
    __all__.extend(["Qwen25VLTrainer", "TrainingArguments"])