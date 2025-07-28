#!/usr/bin/env python3

"""
Quick script to download and prepare a small subset of DocLayNet-Instruct dataset for testing
"""

import os
import json
import logging
from datasets import load_dataset
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import hashlib
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test_setup(num_samples=5):
    """Quick setup for testing with minimal samples"""
    
    output_dir = Path("./test_data")
    images_dir = output_dir / "images"
    datasets_dir = output_dir / "datasets"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Setting up quick test with {num_samples} samples")
    
    try:
        # Load just a few samples from the dataset
        logger.info("Loading DocLayNet-Instruct dataset (validation split)...")
        dataset = load_dataset("nnul/DocLayNet-Instruct-v1-preprocessed", split=f"validation[:{num_samples}]")
        logger.info(f"Loaded {len(dataset)} samples")
        
        converted_samples = []
        
        for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            try:
                # Process conversations
                conversations = sample.get('conversations', [])
                if not conversations:
                    logger.warning(f"No conversations in sample {i}")
                    continue
                
                # Handle image
                image_filename = f"test_sample_{i:03d}.jpg"
                image_path = images_dir / image_filename
                
                # Try to get image from sample
                if 'image' in sample:
                    try:
                        if isinstance(sample['image'], dict) and 'bytes' in sample['image']:
                            # Image is embedded as bytes
                            image_bytes = sample['image']['bytes']
                            image = Image.open(BytesIO(image_bytes)).convert('RGB')
                        elif hasattr(sample['image'], 'save'):
                            # Image is PIL Image
                            image = sample['image'].convert('RGB')
                        else:
                            # Try to treat as URL or path
                            image_url = str(sample['image'])
                            response = requests.get(image_url, timeout=10)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content)).convert('RGB')
                        
                        # Save image
                        image.save(image_path, 'JPEG', quality=85)
                        logger.info(f"Saved image: {image_path}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process image for sample {i}: {e}")
                        # Create a dummy image for testing
                        dummy_image = Image.new('RGB', (224, 224), color='white')
                        dummy_image.save(image_path, 'JPEG', quality=85)
                        logger.info(f"Created dummy image: {image_path}")
                
                # Create training sample
                training_sample = {
                    "conversations": conversations,
                    "image": image_filename,
                    "source": "doclaynet_instruct_test",
                    "sample_id": i
                }
                
                converted_samples.append(training_sample)
                logger.info(f"Processed sample {i}: {len(conversations)} conversation turns")
                
            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                continue
        
        # Save converted dataset
        output_file = datasets_dir / "doclaynet_test.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(converted_samples)} samples to {output_file}")
        
        # Print sample information
        logger.info("\nSample Information:")
        for i, sample in enumerate(converted_samples[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Image: {sample['image']}")
            print(f"Conversations: {len(sample['conversations'])} turns")
            
            for j, conv in enumerate(sample['conversations'][:2]):  # Show first 2 conversations
                role = conv.get('from', 'unknown')
                value = conv.get('value', '')[:100] + ('...' if len(conv.get('value', '')) > 100 else '')
                print(f"  {j+1}. {role}: {value}")
        
        logger.info(f"\n{'='*50}")
        logger.info("Quick test setup completed!")
        logger.info(f"{'='*50}")
        logger.info(f"Dataset: {output_file}")
        logger.info(f"Images: {images_dir}")
        logger.info(f"Samples: {len(converted_samples)}")
        
        return output_file, images_dir, len(converted_samples)
        
    except Exception as e:
        logger.error(f"Quick test setup failed: {e}")
        raise

if __name__ == "__main__":
    quick_test_setup(num_samples=5)