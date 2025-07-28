#!/usr/bin/env python3
"""
Create a sample dataset for testing Qwen2.5-VL fine-tuning
This creates a minimal dataset for verification purposes
"""

import json
import os
from PIL import Image
import numpy as np

def create_sample_dataset():
    """Create a minimal sample dataset for testing"""
    
    # Create directories
    data_dir = "data"
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create sample images (colored rectangles for testing)
    sample_images = []
    for i in range(5):
        # Create a simple colored image
        color = (i * 50 % 255, (i * 80) % 255, (i * 120) % 255)
        img = Image.new('RGB', (224, 224), color)
        
        # Add some text to make it more interesting
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            # Try to use default font
            font = ImageFont.load_default()
        except:
            font = None
        
        text = f"Sample {i+1}"
        if font:
            draw.text((50, 100), text, fill=(255, 255, 255), font=font)
        else:
            draw.text((50, 100), text, fill=(255, 255, 255))
        
        # Save image
        img_path = os.path.join(images_dir, f"sample_{i+1}.jpg")
        img.save(img_path, "JPEG")
        sample_images.append(f"sample_{i+1}.jpg")
        print(f"Created {img_path}")
    
    # Create sample dataset JSON
    dataset = []
    
    # Sample conversations for fine-tuning
    sample_conversations = [
        {
            "image": "sample_1.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWhat do you see in this image?"
                },
                {
                    "from": "gpt",
                    "value": "I see a colored rectangle with the text 'Sample 1' on it. This appears to be a test image with a solid color background."
                }
            ]
        },
        {
            "image": "sample_2.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescribe the contents of this image."
                },
                {
                    "from": "gpt",
                    "value": "This is a test image showing a colored rectangle with 'Sample 2' text overlaid on it. The image has a uniform background color."
                }
            ]
        },
        {
            "image": "sample_3.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWhat text is visible in this image?"
                },
                {
                    "from": "gpt",
                    "value": "The text 'Sample 3' is visible in this image, displayed on a colored background."
                }
            ]
        },
        {
            "image": "sample_4.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nAnalyze this image."
                },
                {
                    "from": "gpt",
                    "value": "This is a simple test image consisting of a solid colored rectangle with the text 'Sample 4' written on it. It appears to be created for testing purposes."
                }
            ]
        },
        {
            "image": "sample_5.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWhat is the main feature of this image?"
                },
                {
                    "from": "gpt",
                    "value": "The main feature of this image is the text 'Sample 5' displayed on a colored background, indicating this is the fifth sample in a series of test images."
                }
            ]
        }
    ]
    
    # Save dataset JSON
    dataset_path = os.path.join(data_dir, "finetune_data.json")
    with open(dataset_path, 'w') as f:
        json.dump(sample_conversations, f, indent=2)
    
    print(f"\n✅ Sample dataset created successfully!")
    print(f"📁 Images: {len(sample_images)} files in {images_dir}/")
    print(f"📄 Dataset: {dataset_path}")
    print(f"💬 Conversations: {len(sample_conversations)} samples")
    
    # Verify all files exist
    print(f"\n🔍 Verification:")
    for item in sample_conversations:
        img_path = os.path.join(images_dir, item["image"]) 
        exists = os.path.exists(img_path)
        status = "✅" if exists else "❌"
        print(f"{status} {item['image']}")
    
    print(f"\n🚀 Ready for testing!")
    print(f"Run: ./scripts/finetune_qwen2_vl.sh")
    
    return dataset_path, images_dir

if __name__ == "__main__":
    create_sample_dataset()