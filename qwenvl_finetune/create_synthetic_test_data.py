#!/usr/bin/env python3

"""
Create synthetic test data for testing the Qwen2.5-VL fine-tuning system
"""

import os
import json
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_document_image(width=800, height=600, text_content="Sample Document"):
    """Create a synthetic document image with text"""
    
    # Create white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Add title
    draw.text((50, 50), "Document Analysis Test", fill='black', font=title_font)
    
    # Add some content
    content_lines = [
        "This is a synthetic document created for testing.",
        "It contains multiple lines of text to simulate",
        "a real document that might be processed by",
        "a vision-language model for analysis.",
        "",
        "Key Information:",
        f"• Content: {text_content}",
        "• Type: Test Document", 
        "• Purpose: Fine-tuning validation",
        "",
        "Additional details can be found below:",
        "- Item 1: Sample data point",
        "- Item 2: Another data element",
        "- Item 3: Final test item"
    ]
    
    y_pos = 120
    for line in content_lines:
        draw.text((50, y_pos), line, fill='black', font=font)
        y_pos += 35
    
    # Add a simple table
    table_y = y_pos + 20
    draw.rectangle([50, table_y, width-50, table_y+120], outline='black', width=2)
    draw.line([50, table_y+40, width-50, table_y+40], fill='black', width=1)
    draw.line([200, table_y, 200, table_y+120], fill='black', width=1)
    
    # Table headers
    draw.text((60, table_y+10), "Category", fill='black', font=font)
    draw.text((220, table_y+10), "Value", fill='black', font=font)
    
    # Table content
    draw.text((60, table_y+50), "Status", fill='black', font=font)
    draw.text((220, table_y+50), "Active", fill='black', font=font)
    draw.text((60, table_y+85), "Count", fill='black', font=font)
    draw.text((220, table_y+85), "42", fill='black', font=font)
    
    return img

def create_synthetic_dataset(num_samples=10):
    """Create synthetic dataset for testing"""
    
    output_dir = Path("./test_data")
    images_dir = output_dir / "images"
    datasets_dir = output_dir / "datasets"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating synthetic dataset with {num_samples} samples")
    
    # Sample questions and answers for document analysis
    qa_templates = [
        {
            "question": "What is the main title of this document?",
            "answer": "Document Analysis Test"
        },
        {
            "question": "What is the purpose of this document?",
            "answer": "Fine-tuning validation"
        },
        {
            "question": "What is the status shown in the table?",
            "answer": "Active"
        },
        {
            "question": "What is the count value in the table?",
            "answer": "42"
        },
        {
            "question": "How many items are listed in the additional details section?",
            "answer": "3"
        },
        {
            "question": "What type of document is this?",
            "answer": "Test Document"
        },
        {
            "question": "Describe the structure of this document.",
            "answer": "This document contains a title, multiple paragraphs of text, key information in bullet points, and a simple table with category-value pairs."
        },
        {
            "question": "What can you tell me about the content of this document?",
            "answer": "This is a synthetic document created for testing vision-language models. It contains structured information including titles, bullet points, and tabular data for comprehensive analysis."
        }
    ]
    
    converted_samples = []
    
    for i in range(num_samples):
        # Create image
        content_variation = f"Sample {i+1}"
        image = create_synthetic_document_image(text_content=content_variation)
        
        # Save image
        image_filename = f"synthetic_doc_{i:03d}.jpg"
        image_path = images_dir / image_filename
        image.save(image_path, 'JPEG', quality=85)
        
        # Select random Q&A pairs
        selected_qa = random.sample(qa_templates, random.randint(2, 4))
        
        # Create conversation
        conversations = []
        
        for qa in selected_qa:
            # Add human question
            conversations.append({
                "from": "human",
                "value": f"<image>\n{qa['question']}"
            })
            
            # Add assistant answer
            conversations.append({
                "from": "gpt", 
                "value": qa['answer']
            })
        
        # Create training sample
        training_sample = {
            "conversations": conversations,
            "image": image_filename,
            "source": "synthetic_test",
            "sample_id": i
        }
        
        converted_samples.append(training_sample)
        logger.info(f"Created sample {i+1}: {len(conversations)} conversation turns")
    
    # Save dataset
    output_file = datasets_dir / "synthetic_test.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(converted_samples)} samples to {output_file}")
    
    # Print sample information
    logger.info("\nSample Information:")
    for i, sample in enumerate(converted_samples[:2]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Image: {sample['image']}")
        print(f"Conversations: {len(sample['conversations'])} turns")
        
        for j, conv in enumerate(sample['conversations'][:4]):  # Show first 4 conversations
            role = conv.get('from', 'unknown')
            value = conv.get('value', '')[:150] + ('...' if len(conv.get('value', '')) > 150 else '')
            print(f"  {j+1}. {role}: {value}")
    
    logger.info(f"\n{'='*60}")
    logger.info("Synthetic test dataset created successfully!")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {output_file}")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Samples: {len(converted_samples)}")
    
    return output_file, images_dir, len(converted_samples)

if __name__ == "__main__":
    create_synthetic_dataset(num_samples=10)