# Training Data Format for Qwen2.5-VL Fine-tuning

## 📋 Dataset Structure

The training data should be structured as a **JSON file** with an array of conversation objects. Each conversation contains an image and a dialogue between human and assistant.

### 🗂️ Directory Structure
```
your_dataset/
├── dataset.json          # Main conversation file
└── images/              # Directory containing all images
    ├── image1.jpg
    ├── image2.png
    ├── document1.pdf.jpg  # Converted document images
    └── ...
```

## 📄 JSON Format Specification

### Basic Conversation Format
```json
[
  {
    "image": "image_filename.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nYour question or instruction here"
      },
      {
        "from": "gpt", 
        "value": "Assistant's response here"
      }
    ]
  }
]
```

### 📝 Detailed Example

```json
[
  {
    "image": "document_analysis_001.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nAnalyze this document and extract the key information"
      },
      {
        "from": "gpt",
        "value": "This document appears to be a financial report with the following key information:\n\n1. **Revenue**: $2.3M for Q3 2024\n2. **Expenses**: $1.8M operational costs\n3. **Net Profit**: $500K (21.7% margin)\n4. **Key Metrics**: 15% growth compared to Q2\n\nThe document structure includes header information, financial tables, and summary charts at the bottom."
      }
    ]
  },
  {
    "image": "chart_visualization_002.png", 
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat trends do you see in this chart?"
      },
      {
        "from": "gpt",
        "value": "Looking at this chart, I can identify several key trends:\n\n1. **Upward trajectory**: There's a clear positive trend from January to December\n2. **Seasonal patterns**: Notable dips in March and August, likely seasonal effects\n3. **Acceleration**: The growth rate appears to increase in the final quarter\n4. **Peak performance**: December shows the highest values at approximately 85% of the maximum scale\n\nThe data suggests strong overall performance with predictable seasonal variations."
      }
    ]
  }
]
```

## 🎯 Key Components

### 1. **Image Reference**
- `"image"`: Filename of the image (relative to the images folder)
- Supported formats: JPG, PNG, GIF, WebP, BMP
- Recommended: High-quality images, 224x224 to 2048x2048 pixels

### 2. **Conversation Structure**
- `"conversations"`: Array of alternating human/assistant messages
- Always starts with `"from": "human"`
- Must include `<image>` token in the first human message
- Can have multiple back-and-forth exchanges

### 3. **Message Format**
- `"from"`: Either `"human"` or `"gpt"`
- `"value"`: The actual message content
- First human message must contain `<image>` token

## 🔧 Advanced Examples

### Multi-turn Conversation
```json
{
  "image": "complex_document.jpg",
  "conversations": [
    {
      "from": "human", 
      "value": "<image>\nWhat type of document is this?"
    },
    {
      "from": "gpt",
      "value": "This appears to be a technical specification document for a software system, containing architecture diagrams and API documentation."
    },
    {
      "from": "human",
      "value": "Can you explain the main components shown in the architecture?"
    },
    {
      "from": "gpt", 
      "value": "The architecture shows three main components:\n1. **Frontend Layer**: React-based user interface\n2. **API Gateway**: Handles routing and authentication\n3. **Backend Services**: Microservices for data processing\n\nThe diagram shows data flow from user requests through the gateway to various backend services."
    }
  ]
}
```

### Document Analysis Format
```json
{
  "image": "invoice_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nExtract all the key information from this invoice"
    },
    {
      "from": "gpt",
      "value": "**Invoice Details:**\n- Invoice #: INV-2024-001\n- Date: March 15, 2024\n- Due Date: April 15, 2024\n\n**Vendor Information:**\n- Company: Tech Solutions Inc.\n- Address: 123 Business St, City, State 12345\n- Phone: (555) 123-4567\n\n**Bill To:**\n- Customer: ABC Corporation\n- Address: 456 Corporate Ave, City, State 67890\n\n**Items:**\n1. Software License - $2,500.00\n2. Implementation Services - $1,200.00\n3. Training Sessions - $800.00\n\n**Total Amount: $4,500.00**\n**Tax: $360.00**\n**Grand Total: $4,860.00**"
    }
  ]
}
```

## 📊 Data Preparation Best Practices

### 1. **Image Quality**
- Use high-resolution images (at least 224x224, preferably larger)
- Ensure text in documents is clearly readable
- Convert PDFs to high-quality JPG/PNG images if needed

### 2. **Conversation Quality**
- Write detailed, informative responses
- Include specific details that require visual understanding
- Vary the types of questions and analysis requested

### 3. **Dataset Size**
- Minimum: 100-500 examples for basic fine-tuning
- Recommended: 1000-5000+ examples for robust performance
- Balance different types of documents and tasks

### 4. **Task Diversity**
Include various types of tasks:
- Document analysis and summarization
- Information extraction 
- Chart and graph interpretation
- Visual question answering
- OCR and text recognition
- Layout understanding

## 🛠️ Data Conversion Utilities

### Create Dataset from Existing Data
```python
import json
from pathlib import Path

def create_parsit_dataset(source_data, output_path):
    """Convert your existing data to Parsit format"""
    
    dataset = []
    for item in source_data:
        conversation = {
            "image": item["image_filename"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{item['question']}"
                },
                {
                    "from": "gpt", 
                    "value": item["answer"]
                }
            ]
        }
        dataset.append(conversation)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
```

### Validate Dataset Format
```python
def validate_dataset(dataset_path, images_path):
    """Validate your dataset format"""
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    for i, item in enumerate(data):
        # Check required fields
        assert "image" in item, f"Item {i}: Missing 'image' field"
        assert "conversations" in item, f"Item {i}: Missing 'conversations' field"
        
        # Check image exists
        img_path = Path(images_path) / item["image"]
        assert img_path.exists(), f"Item {i}: Image {item['image']} not found"
        
        # Check conversation format
        convs = item["conversations"]
        assert len(convs) >= 2, f"Item {i}: Need at least 2 messages"
        assert convs[0]["from"] == "human", f"Item {i}: First message must be from human"
        assert "<image>" in convs[0]["value"], f"Item {i}: First message must contain <image>"
    
    print(f"✅ Dataset validation passed! {len(data)} samples validated.")
```

## 🚀 Usage with Your Dataset

Once your data is prepared:

```bash
# Set your dataset paths
export DATA_PATH="/path/to/your/dataset.json"
export IMAGE_FOLDER="/path/to/your/images"

# Run training
./scripts/finetune_qwen2_vl.sh

# Or with custom settings
MODEL_SIZE=7B NUM_GPUS=4 DATA_PATH="/path/to/dataset.json" IMAGE_FOLDER="/path/to/images" ./scripts/finetune_qwen2_vl.sh
```

## 📋 Quick Checklist

Before training, ensure:
- ✅ JSON file follows the correct conversation format
- ✅ All images referenced in JSON exist in the images folder  
- ✅ First human message in each conversation contains `<image>`
- ✅ Conversations alternate between "human" and "gpt"
- ✅ Images are high quality and readable
- ✅ Dataset size is sufficient for your use case
- ✅ Paths are correctly set in environment variables

Your dataset is now ready for Qwen2.5-VL fine-tuning! 🎯