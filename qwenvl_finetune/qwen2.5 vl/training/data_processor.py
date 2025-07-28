"""
Data processing and dataset management for Qwen2.5-VL fine-tuning
"""

import os
import json
import yaml
import random
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from dataclasses import dataclass

from .conversation import ConversationHandler, ConversationConfig, ConversationTemplate
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing"""
    max_length: int = 2048
    max_image_tokens: int = 4096
    image_aspect_ratio: str = "anyres"
    conversation_template: str = "chatml"
    system_message: str = "You are a helpful AI assistant."
    image_folder: Optional[str] = None
    video_folder: Optional[str] = None
    
    # Data augmentation
    enable_image_augmentation: bool = False
    augmentation_prob: float = 0.3
    
    # Validation
    validate_images: bool = True
    validate_conversations: bool = True
    skip_invalid_samples: bool = True


class ConversationDataset(Dataset):
    """Dataset for conversation-based fine-tuning with multimodal support"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor,
        config: DataConfig,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.split = split
        
        # Initialize conversation handler
        conv_config = ConversationConfig(
            template=ConversationTemplate(config.conversation_template),
            system_message=config.system_message
        )
        self.conversation_handler = ConversationHandler(tokenizer, processor, conv_config)
        
        # Load data
        self.data = self._load_data(data_path)
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load and validate data from various formats"""
        
        data = []
        
        if data_path.endswith('.json'):
            data = self._load_json(data_path)
        elif data_path.endswith('.jsonl'):
            data = self._load_jsonl(data_path)
        elif data_path.endswith('.yaml') or data_path.endswith('.yml'):
            data = self._load_yaml_config(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
            
        # Validate and filter data
        if self.config.validate_conversations or self.config.validate_images:
            data = self._validate_data(data)
            
        return data
        
    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
        
    def _load_yaml_config(self, path: str) -> List[Dict[str, Any]]:
        """Load data from YAML configuration with multiple datasets"""
        
        with open(path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            
        datasets = yaml_config.get('datasets', [])
        combined_data = []
        
        for dataset_config in datasets:
            dataset_path = dataset_config.get('path')
            sampling_strategy = dataset_config.get('sampling', 'all')
            weight = dataset_config.get('weight', 1.0)
            
            if not dataset_path:
                logger.warning("Dataset path not specified, skipping")
                continue
                
            # Load individual dataset
            if dataset_path.endswith('.json'):
                dataset_data = self._load_json(dataset_path)
            elif dataset_path.endswith('.jsonl'):
                dataset_data = self._load_jsonl(dataset_path)
            else:
                logger.warning(f"Unsupported dataset format: {dataset_path}")
                continue
                
            # Apply sampling strategy
            dataset_data = self._apply_sampling_strategy(dataset_data, sampling_strategy)
            
            # Apply weight (duplicate samples)
            if weight > 1.0:
                dataset_data = dataset_data * int(weight)
                
            combined_data.extend(dataset_data)
            logger.info(f"Loaded {len(dataset_data)} samples from {dataset_path}")
            
        return combined_data
        
    def _apply_sampling_strategy(self, data: List[Dict], strategy: str) -> List[Dict]:
        """Apply sampling strategy to dataset"""
        
        if strategy == 'all':
            return data
            
        if ':' in strategy:
            method, amount = strategy.split(':', 1)
            
            if amount.endswith('%'):
                # Percentage sampling
                percentage = float(amount[:-1]) / 100
                target_size = int(len(data) * percentage)
            else:
                # Absolute number sampling
                target_size = int(amount)
                
            target_size = min(target_size, len(data))
            
            if method == 'random':
                return random.sample(data, target_size)
            elif method == 'first':
                return data[:target_size]
            elif method == 'last':
                return data[-target_size:]
            else:
                logger.warning(f"Unknown sampling method: {method}")
                return data
                
        return data
        
    def _validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate data samples and filter invalid ones"""
        
        valid_data = []
        invalid_count = 0
        
        for sample in data:
            is_valid = True
            
            # Validate conversation format
            if self.config.validate_conversations:
                conversations = sample.get('conversations', [])
                if not self.conversation_handler.validate_conversation(conversations):
                    is_valid = False
                    invalid_count += 1
                    if not self.config.skip_invalid_samples:
                        logger.warning(f"Invalid conversation format in sample: {sample}")
                        
            # Validate images
            if self.config.validate_images and 'image' in sample:
                image_path = self._get_image_path(sample['image'])
                if not self._validate_image(image_path):
                    is_valid = False
                    invalid_count += 1
                    if not self.config.skip_invalid_samples:
                        logger.warning(f"Invalid image in sample: {sample}")
                        
            if is_valid or not self.config.skip_invalid_samples:
                valid_data.append(sample)
                
        if invalid_count > 0:
            logger.info(f"Found {invalid_count} invalid samples out of {len(data)}")
            if self.config.skip_invalid_samples:
                logger.info(f"Skipped invalid samples, using {len(valid_data)} valid samples")
                
        return valid_data
        
    def _get_image_path(self, image_file: str) -> str:
        """Get full path to image file"""
        if os.path.isabs(image_file):
            return image_file
        elif self.config.image_folder:
            return os.path.join(self.config.image_folder, image_file)
        else:
            return image_file
            
    def _validate_image(self, image_path: str) -> bool:
        """Validate that image file exists and can be loaded"""
        try:
            if not os.path.exists(image_path):
                return False
            Image.open(image_path).verify()
            return True
        except Exception:
            return False
            
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and process image"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply augmentation if enabled
            if self.config.enable_image_augmentation and self.split == 'train':
                if random.random() < self.config.augmentation_prob:
                    image = self._augment_image(image)
                    
            return image
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
            
    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Apply random augmentation to image"""
        
        # Simple augmentations - can be expanded
        augmentations = []
        
        # Random brightness adjustment
        if random.random() < 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
            
        # Random contrast adjustment  
        if random.random() < 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
            
        return image
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        
        sample = self.data[idx]
        
        # Extract components
        conversations = sample.get('conversations', [])
        
        # Load images
        images = None
        if 'image' in sample:
            image_path = self._get_image_path(sample['image'])
            image = self._load_image(image_path)
            if image is not None:
                images = [image]
                
        # Load videos (if supported)
        videos = None
        if 'video' in sample:
            # Video loading would be implemented here
            pass
            
        # Prepare training sample
        try:
            processed_sample = self.conversation_handler.prepare_training_sample(
                conversations=conversations,
                images=images,
                videos=videos,
                max_length=self.config.max_length
            )
            
            # Remove tensor dimension if needed - ensure proper shape
            for key in processed_sample:
                if hasattr(processed_sample[key], 'squeeze'):
                    processed_sample[key] = processed_sample[key].squeeze(0)
            
            return processed_sample
            
        except Exception as e:
            logger.warning(f"Failed to process sample {idx}: {e}")
            # Return empty sample or skip
            return {
                'input_ids': torch.tensor([]),
                'labels': torch.tensor([]),
                'sample_id': idx,
                'source': 'failed'
            }
            
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a sample without processing"""
        sample = self.data[idx]
        
        info = {
            'conversations': sample.get('conversations', []),
            'has_image': 'image' in sample,
            'has_video': 'video' in sample,
            'source': sample.get('source', 'unknown')
        }
        
        if 'image' in sample:
            info['image_path'] = self._get_image_path(sample['image'])
            
        return info


class DataProcessor:
    """Main data processor for managing datasets and dataloaders"""
    
    def __init__(
        self,
        tokenizer,
        processor,
        config: DataConfig
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        logger.info("DataProcessor initialized")
        
    def create_dataset(
        self,
        data_path: str,
        split: str = "train"
    ) -> ConversationDataset:
        """Create a dataset from data path"""
        
        return ConversationDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config,
            split=split
        )
        
    def create_dataloader(
        self,
        dataset: ConversationDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None
    ) -> DataLoader:
        """Create a dataloader from dataset"""
        
        if collate_fn is None:
            collate_fn = self._default_collate_fn
            
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
    def _default_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default collation function for batching samples"""
        
        if not batch:
            return {}
            
        # Filter out failed samples
        valid_batch = [sample for sample in batch if sample.get('source') != 'failed']
        
        if not valid_batch:
            logger.warning("All samples in batch failed to process")
            return {
                'input_ids': torch.tensor([]),
                'labels': torch.tensor([]),
                'attention_mask': torch.tensor([])
            }
            
        # Get maximum sequence length in batch
        max_length = max(sample['input_ids'].shape[-1] for sample in valid_batch)
        
        # Pad sequences to same length
        padded_batch = {
            'input_ids': [],
            'labels': [],
            'attention_mask': []
        }
        
        for sample in valid_batch:
            input_ids = sample['input_ids'].squeeze(0) if sample['input_ids'].dim() > 1 else sample['input_ids']
            labels = sample['labels'].squeeze(0) if sample['labels'].dim() > 1 else sample['labels']
            
            # Pad sequences
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                input_ids = torch.cat([input_ids, torch.full((pad_length,), pad_token_id)])
                labels = torch.cat([labels, torch.full((pad_length,), -100)])
                attention_mask = torch.cat([torch.ones(len(sample['input_ids'])), torch.zeros(pad_length)])
            else:
                attention_mask = torch.ones(len(input_ids))
                
            padded_batch['input_ids'].append(input_ids)
            padded_batch['labels'].append(labels)
            padded_batch['attention_mask'].append(attention_mask)
            
        # Stack tensors
        for key in padded_batch:
            padded_batch[key] = torch.stack(padded_batch[key])
            
        # Add other keys from first sample
        if valid_batch:
            for key, value in valid_batch[0].items():
                if key not in padded_batch and not isinstance(value, torch.Tensor):
                    padded_batch[key] = value
                    
        return padded_batch
        
    def analyze_dataset(self, dataset: ConversationDataset) -> Dict[str, Any]:
        """Analyze dataset statistics"""
        
        logger.info("Analyzing dataset...")
        
        stats = {
            'total_samples': len(dataset),
            'conversation_lengths': [],
            'has_image_count': 0,
            'has_video_count': 0,
            'sources': {},
            'avg_turns_per_conversation': 0
        }
        
        total_turns = 0
        
        for i in range(min(len(dataset), 1000)):  # Sample first 1000 for analysis
            try:
                sample_info = dataset.get_sample_info(i)
                
                # Conversation statistics
                conversations = sample_info['conversations']
                stats['conversation_lengths'].append(len(conversations))
                total_turns += len(conversations)
                
                # Media statistics
                if sample_info['has_image']:
                    stats['has_image_count'] += 1
                if sample_info['has_video']:
                    stats['has_video_count'] += 1
                    
                # Source statistics
                source = sample_info['source']
                stats['sources'][source] = stats['sources'].get(source, 0) + 1
                
            except Exception as e:
                logger.warning(f"Failed to analyze sample {i}: {e}")
                
        # Calculate averages
        if stats['conversation_lengths']:
            stats['avg_conversation_length'] = sum(stats['conversation_lengths']) / len(stats['conversation_lengths'])
            stats['avg_turns_per_conversation'] = total_turns / len(stats['conversation_lengths'])
            
        stats['multimodal_percentage'] = (stats['has_image_count'] / stats['total_samples']) * 100
        
        logger.info(f"Dataset analysis complete:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Average conversation length: {stats.get('avg_conversation_length', 0):.1f}")
        logger.info(f"  Multimodal samples: {stats['has_image_count']} ({stats['multimodal_percentage']:.1f}%)")
        
        return stats
        
    def validate_data_format(self, data_path: str) -> Dict[str, Any]:
        """Validate data format and return summary"""
        
        logger.info(f"Validating data format: {data_path}")
        
        try:
            # Create temporary dataset for validation
            temp_dataset = self.create_dataset(data_path, "validation")
            
            # Run analysis
            stats = self.analyze_dataset(temp_dataset)
            
            # Add validation results
            stats['is_valid'] = True
            stats['validation_errors'] = []
            
            return stats
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {
                'is_valid': False,
                'validation_errors': [str(e)],
                'total_samples': 0
            }