#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qwen25vl import load_qwen25vl_model, Qwen25VLTrainer, DataProcessor, TrainingArguments
from qwen25vl.utils import setup_logging
from qwen25vl.training import ConversationDataset, DataConfig

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Fine-tuning")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--data_path", required=True, help="Path to training data")
    parser.add_argument("--image_folder", help="Path to image folder")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--deepspeed", help="DeepSpeed configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.deepspeed:
        config['deepspeed']['config_file'] = args.deepspeed
        
    # Setup logging
    setup_logging(
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        log_dir=config['training']['logging_dir']
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Qwen2.5-VL fine-tuning")
    
    # Load model
    logger.info(f"Loading {config['model']['model_size']} model...")
    
    # Check if LoRA is configured
    use_lora = 'lora' in config and config.get('lora', {}).get('enabled', True)
    lora_config = config.get('lora', {}) if use_lora else None
    
    model = load_qwen25vl_model(
        model_size=config['model']['model_size'],
        torch_dtype=config['model']['torch_dtype'],
        use_lora=use_lora,
        lora_config=lora_config,
        load_in_4bit=config['model'].get('load_in_4bit', False),
        load_in_8bit=config['model'].get('load_in_8bit', False),
        attn_implementation=config['model'].get('attn_implementation', 'eager')
    )
    
    # Setup data processor
    data_config = DataConfig(
        max_length=config['data']['max_length'],
        max_image_tokens=config['data']['max_image_tokens'],
        image_aspect_ratio=config['data']['image_aspect_ratio'],
        conversation_template=config['data']['conversation_template'],
        system_message=config['data']['system_message'],
        image_folder=args.image_folder,
        validate_images=config['data']['validate_images'],
        validate_conversations=config['data']['validate_conversations'],
        skip_invalid_samples=config['data']['skip_invalid_samples']
    )
    
    data_processor = DataProcessor(
        tokenizer=model.tokenizer,
        processor=model.processor,
        config=data_config
    )
    
    # Create datasets
    logger.info("Loading training data...")
    train_dataset = data_processor.create_dataset(args.data_path, "train")
    
    # Analyze dataset
    stats = data_processor.analyze_dataset(train_dataset)
    logger.info(f"Dataset loaded: {stats['total_samples']} samples")
    
    # Setup training arguments
    training_args = TrainingArguments(
        # Use config values
        **{k: v for k, v in config['training'].items() if k not in ['logging_dir']},
        deepspeed=config['deepspeed']['config_file'] if config['deepspeed']['enabled'] else None
    )
    
    # Create trainer
    trainer = Qwen25VLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=model.tokenizer
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
