# Parsit: Vision-Language Model for Document Analysis

Parsit is a specialized vision-language model designed for document analysis tasks. It combines **Qwen3-1.7B language models** with **SigLIP vision encoders** to deliver high performance on OCR, question answering, and structured document understanding.

## Features

- **Document-Focused Architecture**: Tailored specifically for document analysis (no video or general vision tasks)
- **Modern Components**: Qwen3-1.7B LLM + SigLIP vision encoder + MLP projector
- **Flexible Training**: Supports full fine-tuning, LoRA, and multiple training modes
- **Simple Inference**: Python API for document processing and structured text extraction
- **Comprehensive Evaluation**: Built-in metrics for QA accuracy and OCR performance

## Quick Start

### Installation

```bash
git clone https://github.com/your-repo/parsit.git
cd parsit
pip install -e .
```

### Basic Usage

```python
from parsit.inference import ParsitInference

# Load model
model = ParsitInference("path/to/parsit/model")

# Analyze document
result = model.analyze_document("document.jpg")
print(result)

# Extract text
text = model.extract_text("document.jpg")
print(text)

# Answer specific questions
answer = model.answer_question("document.jpg", "What is the total amount?")
print(answer)
```

## Training

### Data Preparation

```python
from parsit.data.document_processor import DocumentDataProcessor

# Initialize processor
processor = DocumentDataProcessor("./images", "./processed_data")

# Process QA dataset
qa_data = [
    {"image_path": "invoice.jpg", "question": "What is the total?", "answer": "$1,250.00"}
]
processor.process_pdf_qa_dataset(qa_data, "qa_dataset.json")

# Process OCR dataset
ocr_data = [
    {"image_path": "document.jpg", "text": "Extracted text content..."}
]
processor.process_ocr_dataset(ocr_data, "ocr_dataset.json")
```

### Training Scripts

```bash
# Pre-training (vision-language alignment)
bash scripts/pretrain.sh

# Fine-tuning (instruction following)
bash scripts/finetune.sh

# LoRA training (memory efficient)
bash scripts/train_lora.sh

# Full document training
bash scripts/train_parsit_documents.sh
```

## Model Architecture

```
Document Image → SigLIP-2 Encoder → MLP Projector → Qwen3-1.7B LLM → Text Response
```

### Components

- **Vision Encoder**: SigLIP-2 (google/siglip-so400m-patch14-384)  
- **Language Model**: Qwen3-1.7B-Instruct  
- **Projector**: 2-layer MLP with GELU activation (`mlp2x_gelu`)  
- **DeepSpeed**: ZeRO-2 and ZeRO-3 configurations for efficient training

## Evaluation

```python
from parsit.eval.document_eval import DocumentEvaluator

# Initialize evaluator
evaluator = DocumentEvaluator("path/to/model")

# Evaluate QA performance
qa_metrics = evaluator.evaluate_qa_dataset(
    "test_qa.json", 
    "test_images/",
    "qa_results.json"
)

# Evaluate OCR performance  
ocr_metrics = evaluator.evaluate_ocr_dataset(
    "test_ocr.json",
    "test_images/", 
    "ocr_results.json"
)
```

## Examples

### Invoice Analysis

```python
model = ParsitInference("parsit-qwen-7b")

# Extract key information
response = model.chat(
    "Extract the invoice number, date, total amount, and vendor name",
    "invoice.pdf"
)
```

### Form Understanding

```python
# Analyze form structure
response = model.chat(
    "Describe the structure of this form and list all the fields",
    "application_form.jpg"
)
```

## Configuration

Key parameters for document analysis training:

```bash
--mm_projector_type "mlp2x_gelu"          # Projector architecture
--image_aspect_ratio "pad"                # Handle document aspect ratios  
--group_by_modality_length True           # Efficient batching
--mm_vision_select_layer -2               # SigLIP layer selection
--model_max_length 2048                   # Context length for documents
```

## License

This project is released under a permissive open-source license.

## Acknowledgments

- Uses [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B-Instruct) language model  
- Powered by [SigLIP-2](https://github.com/google-research/big_vision) vision encoder
