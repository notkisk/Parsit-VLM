# Parsit: Vision-Language Model for Document Analysis

Parsit is a specialized vision-language model designed for document analysis tasks. Built on top of the LLaVA-NeXT architecture, Parsit combines **Qwen2.5 language models** with **SigLIP-2 vision encoders** to provide state-of-the-art performance on document understanding, OCR, and analysis tasks.

## Features

- **Document-Focused Architecture**: Optimized specifically for document analysis (no video processing)
- **Modern Components**: Qwen2.5 LLM + SigLIP-2 vision encoder + MLP projector
- **Flexible Training**: Support for full fine-tuning, LoRA, and various training configurations
- **Easy Inference**: Simple Python API for document analysis and text extraction
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
# Full fine-tuning
bash scripts/train_parsit_documents.sh

# LoRA training (memory efficient)
bash scripts/train_lora.sh
```

## Model Architecture

```
Document Image → SigLIP-2 Encoder → MLP Projector → Qwen2.5 LLM → Text Response
```

### Components
- **Vision Encoder**: SigLIP-2 (google/siglip-so400m-patch14-384)  
- **Language Model**: Qwen2.5 (1.5B/7B/14B variants supported)
- **Projector**: Configurable MLP (linear, mlp2x_gelu, etc.)

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

This project builds upon LLaVA-NeXT and follows the same licensing terms.

## Acknowledgments

- Built on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) architecture
- Uses [Qwen2.5](https://github.com/QwenLM/Qwen2.5) language models  
- Powered by [SigLIP](https://github.com/google-research/big_vision) vision encoder