#!/usr/bin/env python3

"""
Script to download and prepare DocLayNet-Instruct dataset for testing
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

class DocLayNetDownloader:
    def __init__(self, output_dir="./test_data"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.datasets_dir = self.output_dir / "datasets"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Datasets directory: {self.datasets_dir}")
    
    def download_dataset(self, split="validation", max_samples=None):
        """Download the DocLayNet-Instruct dataset"""
        
        logger.info(f"Loading DocLayNet-Instruct dataset (split: {split})")
        
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("nnul/DocLayNet-Instruct-v1-preprocessed", split=split)
            logger.info(f"Dataset loaded successfully: {len(dataset)} samples")
            
            # Limit samples for testing if specified
            if max_samples and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))
                logger.info(f"Limited to {max_samples} samples for testing")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def process_sample(self, sample, sample_id):
        """Process a single sample and download its image"""
        
        try:
            # Get image URL
            image_url = sample.get('image', {}).get('path') if isinstance(sample.get('image'), dict) else sample.get('image')
            
            if not image_url:
                logger.warning(f"No image URL found for sample {sample_id}")
                return None
            
            # Generate image filename
            image_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
            image_filename = f"sample_{sample_id:05d}_{image_hash}.jpg"
            image_path = self.images_dir / image_filename
            
            # Download image if not exists
            if not image_path.exists():
                try:
                    if isinstance(sample.get('image'), dict) and 'bytes' in sample['image']:
                        # Image is embedded as bytes
                        image_bytes = sample['image']['bytes']
                        image = Image.open(BytesIO(image_bytes)).convert('RGB')
                    else:
                        # Image is a URL
                        response = requests.get(image_url, timeout=30)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                    
                    # Save image
                    image.save(image_path, 'JPEG', quality=85)
                    
                except Exception as e:
                    logger.warning(f"Failed to download image for sample {sample_id}: {e}")
                    return None
            
            # Process conversations
            conversations = sample.get('conversations', [])
            if not conversations:
                logger.warning(f"No conversations found for sample {sample_id}")
                return None
            
            # Convert to our format
            processed_sample = {
                "conversations": conversations,
                "image": image_filename,
                "source": "doclaynet_instruct",
                "sample_id": sample_id
            }
            
            return processed_sample
            
        except Exception as e:
            logger.error(f"Failed to process sample {sample_id}: {e}")
            return None
    
    def convert_to_training_format(self, dataset, output_file="train_data.json"):
        """Convert dataset to our training format"""
        
        logger.info("Converting dataset to training format...")
        
        converted_samples = []
        failed_samples = 0
        
        for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            processed_sample = self.process_sample(sample, i)
            
            if processed_sample:
                converted_samples.append(processed_sample)
            else:
                failed_samples += 1
        
        logger.info(f"Successfully processed {len(converted_samples)} samples")
        logger.info(f"Failed to process {failed_samples} samples")
        
        # Save converted dataset
        output_path = self.datasets_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Converted dataset saved to: {output_path}")
        
        return output_path, len(converted_samples)
    
    def validate_data(self, json_file):
        """Validate the converted data"""
        
        logger.info("Validating converted data...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            valid_samples = 0
            invalid_samples = 0
            
            for i, sample in enumerate(data):
                is_valid = True
                
                # Check required fields
                if 'conversations' not in sample:
                    logger.warning(f"Sample {i}: Missing 'conversations' field")
                    is_valid = False
                
                if 'image' not in sample:
                    logger.warning(f"Sample {i}: Missing 'image' field")
                    is_valid = False
                
                # Check image file exists
                if 'image' in sample:
                    image_path = self.images_dir / sample['image']
                    if not image_path.exists():
                        logger.warning(f"Sample {i}: Image file not found: {image_path}")
                        is_valid = False
                
                # Check conversations format
                if 'conversations' in sample:
                    conversations = sample['conversations']
                    if not isinstance(conversations, list) or len(conversations) == 0:
                        logger.warning(f"Sample {i}: Invalid conversations format")
                        is_valid = False
                    else:
                        for j, conv in enumerate(conversations):
                            if not isinstance(conv, dict) or 'from' not in conv or 'value' not in conv:
                                logger.warning(f"Sample {i}, conversation {j}: Invalid format")
                                is_valid = False
                
                if is_valid:
                    valid_samples += 1
                else:
                    invalid_samples += 1
            
            logger.info(f"Validation complete:")
            logger.info(f"  Valid samples: {valid_samples}")
            logger.info(f"  Invalid samples: {invalid_samples}")
            logger.info(f"  Success rate: {valid_samples/(valid_samples+invalid_samples)*100:.1f}%")
            
            return valid_samples, invalid_samples
            
        except Exception as e:
            logger.error(f"Failed to validate data: {e}")
            return 0, 0
    
    def create_small_test_set(self, json_file, output_file="test_small.json", num_samples=10):
        """Create a small test set for initial testing"""
        
        logger.info(f"Creating small test set with {num_samples} samples...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Take first N valid samples
            test_samples = data[:num_samples]
            
            # Save small test set
            output_path = self.datasets_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_samples, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Small test set saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create small test set: {e}")
            return None
    
    def print_sample_info(self, json_file, num_samples=3):
        """Print information about sample conversations"""
        
        logger.info(f"Sample information from {json_file}:")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for i in range(min(num_samples, len(data))):
                sample = data[i]
                print(f"\n--- Sample {i+1} ---")
                print(f"Image: {sample.get('image', 'N/A')}")
                print(f"Source: {sample.get('source', 'N/A')}")
                
                conversations = sample.get('conversations', [])
                print(f"Conversations ({len(conversations)} turns):")
                
                for j, conv in enumerate(conversations):
                    role = conv.get('from', 'unknown')
                    value = conv.get('value', '')[:100] + ('...' if len(conv.get('value', '')) > 100 else '')
                    print(f"  {j+1}. {role}: {value}")
                
        except Exception as e:
            logger.error(f"Failed to print sample info: {e}")

def main():
    """Main function"""
    
    logger.info("Starting DocLayNet-Instruct dataset download and preparation")
    
    # Initialize downloader
    downloader = DocLayNetDownloader()
    
    try:
        # Download dataset (validation split)
        dataset = downloader.download_dataset(split="validation", max_samples=50)  # Limit for testing
        
        # Convert to training format
        json_file, num_samples = downloader.convert_to_training_format(dataset, "doclaynet_validation.json")
        
        # Validate data
        valid_samples, invalid_samples = downloader.validate_data(json_file)
        
        # Create small test set
        small_test_file = downloader.create_small_test_set(json_file, "doclaynet_test_small.json", 5)
        
        # Print sample information
        if small_test_file:
            downloader.print_sample_info(small_test_file)
        
        logger.info("\n" + "="*50)
        logger.info("Dataset preparation completed successfully!")
        logger.info("="*50)
        logger.info(f"Full dataset: {json_file} ({valid_samples} valid samples)")
        logger.info(f"Small test set: {small_test_file} (5 samples)")
        logger.info(f"Images directory: {downloader.images_dir}")
        logger.info("\nReady for training!")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()