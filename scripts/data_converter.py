#!/usr/bin/env python3
"""
Dataset Format Converter for Parsit

This script converts various conversation formats to be compatible with Parsit training.
Supports conversion from:
- Standard {"from": "human"/"gpt", "value": "..."} format (already compatible)
- Other common conversation formats to Parsit-compatible format

Usage:
    python scripts/data_converter.py --input data.json --output converted_data.json --format parsit
"""

import json
import argparse
from typing import List, Dict, Any
from pathlib import Path


def convert_to_parsit_format(conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert conversation format to Parsit-compatible format.
    
    Input format should be:
    [
        {"from": "human", "value": "<image>\nQuestion about the image"},
        {"from": "gpt", "value": "Answer about the image"}
    ]
    
    This format is already compatible with Parsit, but this function
    ensures consistency and can handle minor variations.
    """
    converted = []
    
    for turn in conversations:
        # Ensure 'from' field uses correct role names
        if turn.get("from") == "human":
            converted.append({
                "from": "human",
                "value": turn["value"]
            })
        elif turn.get("from") in ["gpt", "assistant"]:
            converted.append({
                "from": "gpt", 
                "value": turn["value"]
            })
        else:
            # Handle other role names
            role = turn.get("from", "").lower()
            if role in ["user", "human"]:
                converted.append({
                    "from": "human",
                    "value": turn["value"]
                })
            elif role in ["assistant", "gpt", "bot"]:
                converted.append({
                    "from": "gpt",
                    "value": turn["value"]
                })
            else:
                print(f"Warning: Unknown role '{role}', skipping turn")
                continue
    
    return converted


def convert_llava_to_parsit(data_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert LLaVA-style data to Parsit format.
    
    LLaVA format typically has 'conversations' key with role-value pairs.
    """
    if "conversations" in data_item:
        # Already in conversation format, just ensure role names are correct
        converted_conversations = convert_to_parsit_format(data_item["conversations"])
        
        result = {
            "conversations": converted_conversations
        }
        
        # Preserve other fields like 'image', 'id', etc.
        for key, value in data_item.items():
            if key != "conversations":
                result[key] = value
                
        return result
    else:
        raise ValueError("Data item must contain 'conversations' key")


def convert_sharegpt_to_parsit(data_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ShareGPT-style data to Parsit format.
    
    ShareGPT typically uses different field names.
    """
    conversations = []
    
    if "conversations" in data_item:
        for turn in data_item["conversations"]:
            role = turn.get("from", "").lower()
            if role == "human":
                conversations.append({
                    "from": "human",
                    "value": turn.get("value", "")
                })
            elif role in ["gpt", "chatgpt", "assistant"]:
                conversations.append({
                    "from": "gpt", 
                    "value": turn.get("value", "")
                })
    
    result = {
        "conversations": conversations
    }
    
    # Preserve image and other metadata
    for key, value in data_item.items():
        if key not in ["conversations"]:
            result[key] = value
            
    return result


def validate_parsit_format(data_item: Dict[str, Any]) -> bool:
    """
    Validate that data item is in correct Parsit format.
    """
    if "conversations" not in data_item:
        return False
        
    conversations = data_item["conversations"]
    if not isinstance(conversations, list) or len(conversations) == 0:
        return False
        
    # Check that conversation alternates between human and gpt
    expected_roles = ["human", "gpt"] * (len(conversations) // 2 + 1)
    
    for i, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            return False
        if "from" not in turn or "value" not in turn:
            return False
        if turn["from"] not in ["human", "gpt"]:
            return False
        if i < len(expected_roles) and turn["from"] != expected_roles[i]:
            # Allow starting with either human or gpt
            if i == 0:
                expected_roles = ["gpt", "human"] * (len(conversations) // 2 + 1)
                if turn["from"] != expected_roles[i]:
                    return False
            else:
                return False
                
    return True


def process_dataset(input_file: str, output_file: str, input_format: str = "auto"):
    """
    Process entire dataset file.
    """
    print(f"Loading dataset from {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.jsonl'):
            # Handle JSONL format
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        else:
            # Handle JSON format
            data = json.load(f)
    
    print(f"Loaded {len(data)} data items")
    
    converted_data = []
    skipped_count = 0
    
    for i, item in enumerate(data):
        try:
            if input_format == "auto":
                # Try to detect format automatically
                if validate_parsit_format(item):
                    # Already in correct format
                    converted_item = item
                else:
                    # Try LLaVA format conversion
                    converted_item = convert_llava_to_parsit(item)
            elif input_format == "llava":
                converted_item = convert_llava_to_parsit(item)
            elif input_format == "sharegpt":
                converted_item = convert_sharegpt_to_parsit(item)
            elif input_format == "parsit":
                converted_item = item
            else:
                raise ValueError(f"Unknown input format: {input_format}")
            
            # Validate converted item
            if validate_parsit_format(converted_item):
                converted_data.append(converted_item)
            else:
                print(f"Warning: Item {i} failed validation after conversion, skipping")
                skipped_count += 1
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            skipped_count += 1
            continue
    
    print(f"Successfully converted {len(converted_data)} items")
    print(f"Skipped {skipped_count} items due to errors")
    
    # Save converted data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if output_file.endswith('.jsonl'):
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved converted dataset to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert datasets to Parsit-compatible format")
    parser.add_argument("--input", "-i", required=True, help="Input dataset file (.json or .jsonl)")
    parser.add_argument("--output", "-o", required=True, help="Output dataset file (.json or .jsonl)")
    parser.add_argument("--format", "-f", choices=["auto", "llava", "sharegpt", "parsit"], 
                       default="auto", help="Input format (default: auto-detect)")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate format without conversion")
    
    args = parser.parse_args()
    
    if args.validate_only:
        # Just validate the input file
        with open(args.input, 'r', encoding='utf-8') as f:
            if args.input.endswith('.jsonl'):
                data = []
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            else:
                data = json.load(f)
        
        valid_count = 0
        for i, item in enumerate(data):
            if validate_parsit_format(item):
                valid_count += 1
            else:
                print(f"Item {i} is not in valid Parsit format")
        
        print(f"Validation complete: {valid_count}/{len(data)} items are valid")
    else:
        process_dataset(args.input, args.output, args.format)


if __name__ == "__main__":
    main()