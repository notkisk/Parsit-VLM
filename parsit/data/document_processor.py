import json
import os
import glob
from PIL import Image
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DocumentDataProcessor:
    """Process documents into training format for Parsit"""
    
    def __init__(self, image_folder: str, output_dir: str = "./processed_data"):
        self.image_folder = image_folder
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def process_pdf_qa_dataset(self, qa_pairs: List[Dict], output_file: str = "document_qa.json"):
        """
        Process PDF QA dataset into Parsit training format
        
        Args:
            qa_pairs: List of {"image_path": str, "question": str, "answer": str} dicts
            output_file: Output JSON file name
        """
        processed_data = []
        
        for item in qa_pairs:
            image_path = item["image_path"]
            question = item["question"]
            answer = item["answer"]
            
            # Verify image exists
            full_image_path = os.path.join(self.image_folder, image_path)
            if not os.path.exists(full_image_path):
                logger.warning(f"Image not found: {full_image_path}")
                continue
                
            # Create conversation format
            conversation_item = {
                "id": f"doc_qa_{len(processed_data)}",
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question}"
                    },
                    {
                        "from": "gpt", 
                        "value": answer
                    }
                ]
            }
            processed_data.append(conversation_item)
            
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Processed {len(processed_data)} QA pairs to {output_path}")
        return output_path
    
    def process_ocr_dataset(self, ocr_data: List[Dict], output_file: str = "document_ocr.json"):
        """
        Process OCR dataset for text extraction training
        
        Args:
            ocr_data: List of {"image_path": str, "text": str} dicts
            output_file: Output JSON file name
        """
        processed_data = []
        
        for item in ocr_data:
            image_path = item["image_path"]
            text = item["text"]
            
            full_image_path = os.path.join(self.image_folder, image_path)
            if not os.path.exists(full_image_path):
                logger.warning(f"Image not found: {full_image_path}")
                continue
                
            conversation_item = {
                "id": f"doc_ocr_{len(processed_data)}",
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nPlease extract all text from this document while maintaining the original formatting and structure."
                    },
                    {
                        "from": "gpt",
                        "value": text
                    }
                ]
            }
            processed_data.append(conversation_item)
            
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Processed {len(processed_data)} OCR samples to {output_path}")
        return output_path
    
    def process_document_understanding_dataset(self, understanding_data: List[Dict], output_file: str = "document_understanding.json"):
        """
        Process document understanding dataset
        
        Args:
            understanding_data: List of {"image_path": str, "description": str, "analysis": str} dicts
            output_file: Output JSON file name
        """
        processed_data = []
        
        for item in understanding_data:
            image_path = item["image_path"]
            description = item.get("description", "")
            analysis = item.get("analysis", "")
            
            full_image_path = os.path.join(self.image_folder, image_path)
            if not os.path.exists(full_image_path):
                logger.warning(f"Image not found: {full_image_path}")
                continue
                
            # Create comprehensive analysis prompt
            response = f"Document Analysis:\n{description}\n\nDetailed Analysis:\n{analysis}"
            
            conversation_item = {
                "id": f"doc_understand_{len(processed_data)}",
                "image": image_path,
                "conversations": [
                    {
                        "from": "human", 
                        "value": "<image>\nPlease analyze this document and provide a comprehensive description of its content, structure, layout, and key information."
                    },
                    {
                        "from": "gpt",
                        "value": response
                    }
                ]
            }
            processed_data.append(conversation_item)
            
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Processed {len(processed_data)} understanding samples to {output_path}")
        return output_path
    
    def combine_datasets(self, dataset_files: List[str], output_file: str = "combined_dataset.json"):
        """Combine multiple processed datasets"""
        combined_data = []
        
        for file_path in dataset_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    combined_data.extend(data)
                    logger.info(f"Added {len(data)} samples from {file_path}")
            else:
                logger.warning(f"File not found: {file_path}")
                
        # Shuffle and reassign IDs
        import random
        random.shuffle(combined_data)
        for i, item in enumerate(combined_data):
            item["id"] = f"combined_{i}"
            
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Combined dataset saved with {len(combined_data)} total samples to {output_path}")
        return output_path
    
    def validate_dataset(self, dataset_file: str):
        """Validate processed dataset"""
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        valid_count = 0
        errors = []
        
        for i, item in enumerate(data):
            # Check required fields
            if "image" not in item or "conversations" not in item:
                errors.append(f"Item {i}: Missing required fields")
                continue
                
            # Check image file exists
            image_path = os.path.join(self.image_folder, item["image"])
            if not os.path.exists(image_path):
                errors.append(f"Item {i}: Image not found - {image_path}")
                continue
                
            # Check conversation format
            conversations = item["conversations"]
            if len(conversations) < 2:
                errors.append(f"Item {i}: Insufficient conversation turns")
                continue
                
            if conversations[0]["from"] != "human" or conversations[1]["from"] != "gpt":
                errors.append(f"Item {i}: Invalid conversation format")
                continue
                
            valid_count += 1
            
        logger.info(f"Dataset validation: {valid_count}/{len(data)} valid samples")
        if errors:
            logger.warning(f"Found {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                logger.warning(f"  {error}")
                
        return valid_count, errors


def create_sample_dataset():
    """Create sample dataset for testing"""
    sample_data = [
        {
            "image_path": "sample_invoice.jpg",
            "question": "What is the total amount on this invoice?",
            "answer": "The total amount on this invoice is $1,250.00."
        },
        {
            "image_path": "sample_receipt.jpg", 
            "question": "What items were purchased?",
            "answer": "The purchased items are: Office supplies ($45.99), Printer paper ($12.50), and Pens ($8.99)."
        }
    ]
    
    processor = DocumentDataProcessor("./sample_images")
    return processor.process_pdf_qa_dataset(sample_data)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample dataset
    dataset_file = create_sample_dataset()
    print(f"Sample dataset created: {dataset_file}")