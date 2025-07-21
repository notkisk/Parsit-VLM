import json
import os
import torch
import logging
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict

from parsit.inference import ParsitInference
from parsit.data.document_processor import DocumentDataProcessor

logger = logging.getLogger(__name__)


class DocumentEvaluator:
    """Evaluate Parsit model on document understanding tasks"""
    
    def __init__(self, model_path: str, model_base: str = None, device: str = "cuda"):
        """
        Initialize evaluator with Parsit model
        
        Args:
            model_path: Path to Parsit model
            model_base: Base model path for LoRA models
            device: Device to run evaluation on
        """
        self.model = ParsitInference(model_path, model_base, device=device)
        self.device = device
        
    def evaluate_qa_dataset(self, dataset_file: str, image_folder: str, output_file: str = None) -> Dict:
        """
        Evaluate on QA dataset
        
        Args:
            dataset_file: Path to QA dataset JSON
            image_folder: Folder containing images
            output_file: Optional output file for detailed results
            
        Returns:
            Evaluation metrics dict
        """
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        logger.info(f"Evaluating on {len(dataset)} QA samples...")
        
        for item in tqdm(dataset, desc="Evaluating QA"):
            image_path = os.path.join(image_folder, item["image"])
            
            # Extract question and answer from conversations
            conversations = item["conversations"]
            human_msg = conversations[0]["value"]
            expected_answer = conversations[1]["value"]
            
            # Remove <image> token for processing
            question = human_msg.replace("<image>\n", "").strip()
            
            try:
                # Get model prediction
                predicted_answer = self.model.chat(question, image_path, temperature=0.1)
                
                # Simple exact match evaluation (can be improved with better metrics)
                is_correct = self._evaluate_answer_similarity(predicted_answer, expected_answer)
                
                result = {
                    "id": item.get("id", f"sample_{len(results)}"),
                    "question": question,
                    "expected": expected_answer,
                    "predicted": predicted_answer,
                    "correct": is_correct,
                    "image": item["image"]
                }
                results.append(result)
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
            except Exception as e:
                logger.error(f"Error processing {item.get('id', 'unknown')}: {e}")
                continue
                
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "correct": correct_predictions,
            "total": total_predictions,
            "error_rate": 1 - accuracy
        }
        
        # Save detailed results if requested
        if output_file:
            detailed_results = {
                "metrics": metrics,
                "results": results
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
                
        logger.info(f"QA Evaluation Results:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Correct: {correct_predictions}/{total_predictions}")
        
        return metrics
    
    def evaluate_ocr_dataset(self, dataset_file: str, image_folder: str, output_file: str = None) -> Dict:
        """
        Evaluate on OCR/text extraction dataset
        
        Args:
            dataset_file: Path to OCR dataset JSON
            image_folder: Folder containing images
            output_file: Optional output file for detailed results
            
        Returns:
            Evaluation metrics dict
        """
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        results = []
        total_char_errors = 0
        total_chars = 0
        
        logger.info(f"Evaluating on {len(dataset)} OCR samples...")
        
        for item in tqdm(dataset, desc="Evaluating OCR"):
            image_path = os.path.join(image_folder, item["image"])
            
            conversations = item["conversations"]
            expected_text = conversations[1]["value"]
            
            try:
                # Get model prediction for text extraction
                predicted_text = self.model.extract_text(image_path)
                
                # Calculate character-level edit distance
                char_errors = self._calculate_edit_distance(predicted_text, expected_text)
                total_char_errors += char_errors
                total_chars += len(expected_text)
                
                result = {
                    "id": item.get("id", f"ocr_{len(results)}"),
                    "expected": expected_text,
                    "predicted": predicted_text,
                    "char_errors": char_errors,
                    "char_accuracy": 1 - (char_errors / len(expected_text)) if len(expected_text) > 0 else 0,
                    "image": item["image"]
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {item.get('id', 'unknown')}: {e}")
                continue
                
        # Calculate metrics
        char_error_rate = total_char_errors / total_chars if total_chars > 0 else 0
        char_accuracy = 1 - char_error_rate
        
        metrics = {
            "character_accuracy": char_accuracy,
            "character_error_rate": char_error_rate,
            "total_character_errors": total_char_errors,
            "total_characters": total_chars
        }
        
        # Save detailed results if requested
        if output_file:
            detailed_results = {
                "metrics": metrics,
                "results": results
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
                
        logger.info(f"OCR Evaluation Results:")
        logger.info(f"  Character Accuracy: {char_accuracy:.3f}")
        logger.info(f"  Character Error Rate: {char_error_rate:.3f}")
        
        return metrics
    
    def benchmark_inference_speed(self, test_images: List[str], num_runs: int = 10) -> Dict:
        """
        Benchmark inference speed
        
        Args:
            test_images: List of test image paths
            num_runs: Number of runs for each image
            
        Returns:
            Speed benchmark results
        """
        import time
        
        times = []
        question = "What do you see in this document?"
        
        logger.info(f"Benchmarking inference speed on {len(test_images)} images...")
        
        # Warmup
        for image_path in test_images:
            self.model.chat(question, image_path)
            
        # Actual benchmarking
        for image_path in test_images:
            image_times = []
            for _ in range(num_runs):
                start_time = time.time()
                self.model.chat(question, image_path, max_new_tokens=100)
                end_time = time.time()
                image_times.append(end_time - start_time)
            times.extend(image_times)
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        metrics = {
            "average_time_seconds": avg_time,
            "std_time_seconds": std_time,
            "min_time_seconds": min_time,
            "max_time_seconds": max_time,
            "throughput_samples_per_second": 1.0 / avg_time
        }
        
        logger.info(f"Speed Benchmark Results:")
        logger.info(f"  Average Time: {avg_time:.3f}s")
        logger.info(f"  Throughput: {metrics['throughput_samples_per_second']:.2f} samples/sec")
        
        return metrics
    
    def _evaluate_answer_similarity(self, predicted: str, expected: str, threshold: float = 0.7) -> bool:
        """
        Evaluate answer similarity using simple heuristics
        Can be improved with semantic similarity measures
        """
        # Simple token overlap for now
        pred_tokens = set(predicted.lower().split())
        exp_tokens = set(expected.lower().split())
        
        if len(exp_tokens) == 0:
            return len(pred_tokens) == 0
            
        overlap = len(pred_tokens & exp_tokens)
        similarity = overlap / len(exp_tokens)
        
        return similarity >= threshold
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]


def run_evaluation():
    """Run evaluation on sample dataset"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize evaluator (replace with your model path)
    evaluator = DocumentEvaluator("path/to/parsit/model")
    
    # Run QA evaluation
    qa_metrics = evaluator.evaluate_qa_dataset(
        "processed_data/document_qa.json",
        "sample_images",
        "qa_evaluation_results.json"
    )
    
    # Run OCR evaluation
    ocr_metrics = evaluator.evaluate_ocr_dataset(
        "processed_data/document_ocr.json", 
        "sample_images",
        "ocr_evaluation_results.json"
    )
    
    print("Evaluation completed!")
    return qa_metrics, ocr_metrics


if __name__ == "__main__":
    run_evaluation()