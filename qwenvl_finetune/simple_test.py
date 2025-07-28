#!/usr/bin/env python3

"""
Simple test to verify the system components work
"""

import os
import json
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_synthetic_data():
    """Test that synthetic data exists and is valid"""
    
    data_path = "./test_data/datasets/synthetic_test.json"
    image_folder = "./test_data/images"
    
    logger.info("Testing synthetic data...")
    
    # Check files exist
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return False
        
    if not os.path.exists(image_folder):
        logger.error(f"Image folder not found: {image_folder}")
        return False
    
    # Load and validate JSON
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        
        # Check first sample
        if data:
            sample = data[0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            
            # Check conversations
            if 'conversations' in sample:
                convs = sample['conversations']
                logger.info(f"First sample has {len(convs)} conversation turns")
                
            # Check image file exists
            if 'image' in sample:
                image_path = os.path.join(image_folder, sample['image'])
                if os.path.exists(image_path):
                    logger.info(f"Image file exists: {sample['image']}")
                else:
                    logger.error(f"Image file missing: {image_path}")
                    return False
                    
        return True
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

def test_configs():
    """Test that configuration files exist"""
    
    logger.info("Testing configuration files...")
    
    configs_to_check = [
        "./configs/training/finetune_3b.yaml",
        "./configs/lora/lora_3b.yaml", 
        "./configs/deepspeed/zero2_3b.json",
        "./scripts/finetune_3b.sh"
    ]
    
    all_exist = True
    for config_file in configs_to_check:
        if os.path.exists(config_file):
            logger.info(f"✓ {config_file}")
        else:
            logger.error(f"✗ {config_file}")
            all_exist = False
            
    return all_exist

def test_package_imports():
    """Test that we can import our package components"""
    
    logger.info("Testing package imports...")
    
    try:
        # Test basic imports
        from qwen25vl.utils import setup_logging, get_memory_stats
        logger.info("✓ Utils imported successfully")
        
        from qwen25vl.training import DataConfig
        logger.info("✓ Training components imported successfully")
        
        # Test memory stats
        memory_stats = get_memory_stats()
        logger.info(f"✓ Memory stats: {memory_stats['system']['total_gb']:.1f}GB system RAM")
        
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    logger.info("="*50)
    logger.info("Qwen2.5-VL System Quick Test")
    logger.info("="*50)
    
    tests = [
        ("Synthetic Data", test_synthetic_data),
        ("Configuration Files", test_configs),
        ("Package Imports", test_package_imports)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"{test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Test Results:")
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! System is ready.")
        logger.info("\nTo start training:")
        logger.info("DATA_PATH=./test_data/datasets/synthetic_test.json IMAGE_FOLDER=./test_data/images ./scripts/finetune_3b.sh")
    else:
        logger.error("❌ Some tests failed.")
        
    return passed == len(results)

if __name__ == "__main__":
    main()