# Parsit Documentation

This documentation provides comprehensive guides for using, training, and deploying Parsit - a specialized Vision-Language Model for document analysis.

## Table of Contents

1. [Installation Guide](installation.md)
2. [Model Architecture](architecture.md)
3. [Training Guide](training.md)
4. [Inference Guide](inference.md)
5. [Data Preparation](data_preparation.md)
6. [Model Configurations](configurations.md)
7. [Multi-GPU Training](multi_gpu_training.md)
8. [Deployment Guide](deployment.md)
9. [API Reference](api_reference.md)
10. [Troubleshooting](troubleshooting.md)
11. [Contributing](contributing.md)
12. [**Changelog**](CHANGELOG.md) 📋

## Quick Links

- [Getting Started](#getting-started)
- [Model Variants](#model-variants)
- [Example Usage](#example-usage)
- [Recent Updates](#recent-updates) 🆕

## Recent Updates

### 🔥 Latest Improvements (2025-01-22)
- **Fixed critical training issues**: Resolved loss/grad_norm calculation problems with DeepSpeed-aware parameter management
- **Enhanced reproducibility**: Restored dataloader worker configuration for deterministic training
- **Improved debugging**: Added comprehensive parameter status logging
- **Better scalability**: Added tensor parallelism support for multi-GPU training

For detailed changes, see the [**Changelog**](CHANGELOG.md) 📋

## Getting Started

Parsit is designed for document analysis tasks including:
- Document OCR and text extraction
- Form understanding and parsing
- Invoice and receipt processing
- Multi-page document analysis
- Question answering on documents

### Model Variants

| Model | Parameters | Vision Encoder | Recommended Use |
|-------|------------|----------------|-----------------|
| parsit-qwen3-1.7b | 1.7B | SigLIP-2-400M | Fast inference, lightweight deployment |

### Architecture

```
Document Image → SigLIP-2 Encoder → MLP Projector → Qwen3-1.7B LLM → Text Response
```

### Training Infrastructure

- **Pre-training**: Vision-language alignment with SigLIP-2 encoder
- **Fine-tuning**: Instruction tuning for document analysis tasks
- **DeepSpeed**: Enhanced ZeRO-2/ZeRO-3 support with parameter-aware unfreezing
- **Data Processing**: Advanced dataset handling with YAML configuration support
- **Distributed Training**: Tensor parallelism support for multi-GPU scalability
- **Parameter Management**: Real-time parameter status monitoring and debugging

## Example Usage

```python
from parsit.inference import ParsitInference

# Initialize model
model = ParsitInference("parsit-qwen3-1.7b")

# Analyze document
result = model.analyze_document("invoice.jpg")
print(result)
```

For detailed guides, please refer to the specific documentation files listed above.