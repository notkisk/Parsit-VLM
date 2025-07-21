#!/usr/bin/env python3

"""
Basic usage examples for Parsit VLM

This script demonstrates how to use Parsit for various document analysis tasks.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parsit.inference import ParsitInference

def main():
    # Initialize Parsit model
    # Replace with your actual model path
    model_path = "path/to/your/parsit/model"
    
    try:
        print("Loading Parsit model...")
        model = ParsitInference(model_path)
        print("Model loaded successfully!")
        
        # Example 1: Document Analysis
        print("\n" + "="*50)
        print("Example 1: Document Analysis")
        print("="*50)
        
        document_path = "sample_document.jpg"  # Replace with your document
        if os.path.exists(document_path):
            result = model.analyze_document(document_path)
            print(f"Analysis Result:\n{result}")
        else:
            print(f"Document {document_path} not found. Skipping analysis example.")
        
        # Example 2: Text Extraction (OCR)
        print("\n" + "="*50)
        print("Example 2: Text Extraction")
        print("="*50)
        
        if os.path.exists(document_path):
            extracted_text = model.extract_text(document_path)
            print(f"Extracted Text:\n{extracted_text}")
        else:
            print(f"Document {document_path} not found. Skipping OCR example.")
        
        # Example 3: Question Answering
        print("\n" + "="*50)
        print("Example 3: Question Answering")
        print("="*50)
        
        if os.path.exists(document_path):
            questions = [
                "What type of document is this?",
                "What are the key information points?",
                "Are there any dates mentioned?",
                "What is the main purpose of this document?"
            ]
            
            for question in questions:
                answer = model.answer_question(document_path, question)
                print(f"Q: {question}")
                print(f"A: {answer}\n")
        else:
            print(f"Document {document_path} not found. Skipping QA example.")
            
        # Example 4: Custom Chat
        print("\n" + "="*50)
        print("Example 4: Custom Chat")
        print("="*50)
        
        if os.path.exists(document_path):
            custom_prompt = """
            Please analyze this document and provide:
            1. Document type classification
            2. Key entities mentioned
            3. Important dates and numbers
            4. Overall summary
            """
            
            response = model.chat(custom_prompt.strip(), document_path)
            print(f"Custom Analysis:\n{response}")
        else:
            print(f"Document {document_path} not found. Skipping custom chat example.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. You have a trained Parsit model")
        print("2. The model path is correct")
        print("3. You have sample documents to analyze")


if __name__ == "__main__":
    main()