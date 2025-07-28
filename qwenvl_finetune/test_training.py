#!/usr/bin/env python3

"""
Test script to validate our Qwen2.5-VL fine-tuning system
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qwen25vl.models import load_qwen25vl_model
from qwen25vl.training import DataProcessor, DataConfig
from qwen25vl.utils import setup_logging

def test_data_loading():
    """Test data loading and validation"""
    
    logger.info("Testing data loading...")
    
    # Use synthetic test data
    data_path = "./test_data/datasets/synthetic_test.json"
    image_folder = "./test_data/images"
    
    if not os.path.exists(data_path):
        logger.error(f"Test data not found: {data_path}")
        return False
        
    # Setup data configuration
    data_config = DataConfig(
        max_length=512,
        max_image_tokens=1024,
        image_aspect_ratio="anyres",
        conversation_template="chatml",
        system_message="You are a helpful AI assistant.",
        image_folder=image_folder,
        validate_images=True,
        validate_conversations=True,
        skip_invalid_samples=True
    )
    
    try:
        # Load a small model for testing (we'll use 3B with quantization)
        logger.info("Loading model for testing...")
        model = load_qwen25vl_model(
            model_size="3B",
            quantization="4bit",  # Use 4-bit quantization for testing
            use_lora=True,
            custom_model_path="Qwen/Qwen2.5-VL-3B-Instruct"
        )
        
        # Create data processor
        data_processor = DataProcessor(
            tokenizer=model.tokenizer,
            processor=model.processor,
            config=data_config
        )
        
        # Validate data format
        validation_result = data_processor.validate_data_format(data_path)
        logger.info(f"Data validation result: {validation_result}")
        
        if not validation_result.get('is_valid', False):
            logger.error("Data validation failed")
            return False
            
        # Create test dataset
        logger.info("Creating test dataset...")
        dataset = data_processor.create_dataset(data_path, "train")
        
        # Analyze dataset
        stats = data_processor.analyze_dataset(dataset)
        logger.info(f"Dataset stats: {stats}")
        
        # Test data loading
        logger.info("Testing data sample loading...")
        sample = dataset[0]
        logger.info(f"Sample keys: {list(sample.keys())}")
        logger.info(f"Input IDs shape: {sample['input_ids'].shape if 'input_ids' in sample else 'N/A'}")
        
        logger.info("Data loading test successful!")
        return True
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_training():
    """Test minimal training setup"""
    
    logger.info("Testing minimal training setup...")
    
    try:
        # Test training configuration loading
        config_file = "./configs/lora/lora_3b.yaml"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Training configuration loaded successfully")
        else:
            logger.warning(f"Config file not found: {config_file}")
            
        # Test DeepSpeed configuration
        deepspeed_config = "./configs/deepspeed/zero2_3b.json"
        if os.path.exists(deepspeed_config):
            logger.info("DeepSpeed configuration found")
        else:
            logger.warning(f"DeepSpeed config not found: {deepspeed_config}")
            
        logger.info("Minimal training setup test successful!")
        return True
        
    except Exception as e:
        logger.error(f"Minimal training test failed: {e}")
        return False

def main():
    """Main test function"""
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Starting Qwen2.5-VL Fine-tuning System Test")
    logger.info("="*60)
    
    # Test data loading
    data_test_passed = test_data_loading()
    
    # Test minimal training setup
    training_test_passed = test_minimal_training()
    
    # Summary
    logger.info("="*60)
    logger.info("Test Results Summary:")
    logger.info(f"  Data Loading Test: {'PASSED' if data_test_passed else 'FAILED'}")
    logger.info(f"  Training Setup Test: {'PASSED' if training_test_passed else 'FAILED'}")
    
    overall_result = data_test_passed and training_test_passed
    logger.info(f"  Overall Result: {'PASSED' if overall_result else 'FAILED'}")
    logger.info("="*60)
    
    if overall_result:
        logger.info("🎉 All tests passed! System is ready for training.")
        logger.info("\nNext steps:")
        logger.info("1. Run: DATA_PATH=./test_data/datasets/synthetic_test.json IMAGE_FOLDER=./test_data/images ./scripts/finetune_3b.sh")
        logger.info("2. Monitor training progress in logs")
        logger.info("3. Check checkpoints in ./checkpoints/")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        
    return overall_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)