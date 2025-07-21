import torch
import requests
from PIL import Image
from io import BytesIO

from parsit.model.builder import load_pretrained_model
from parsit.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from parsit.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


class ParsitInference:
    def __init__(self, model_path, model_base=None, load_8bit=False, load_4bit=False, device="auto"):
        """
        Initialize Parsit model for inference
        
        Args:
            model_path: Path to the Parsit model
            model_base: Base model path (for LoRA models)  
            load_8bit: Whether to load in 8-bit mode
            load_4bit: Whether to load in 4-bit mode
            device: Device to load model on
        """
        self.model_path = model_path
        self.model_name = get_model_name_from_path(model_path)
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=device
        )
        
        self.device = next(self.model.parameters()).device
        
    def load_image(self, image_path):
        """Load image from path or URL"""
        if image_path.startswith('http'):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image
    
    def chat(self, message, image_path=None, temperature=0.2, top_p=0.7, max_new_tokens=512):
        """
        Chat with Parsit model
        
        Args:
            message: Text prompt/question
            image_path: Path to image file or URL (optional)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Prepare conversation prompt
        if image_path is not None:
            # Add image token to prompt
            qs = DEFAULT_IMAGE_TOKEN + '\n' + message
            image = self.load_image(image_path)
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            images = image_tensor.unsqueeze(0).half().to(self.device)
        else:
            qs = message
            images = None
        
        # Format with conversation template
        conv = self._get_conversation_template()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        
        # Decode response
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(conv.sep):
            outputs = outputs[:-len(conv.sep)]
        outputs = outputs.strip()
        
        # Extract only the response part
        response_start = prompt.rstrip()
        if outputs.startswith(response_start):
            response = outputs[len(response_start):].strip()
        else:
            response = outputs
            
        return response
    
    def _get_conversation_template(self):
        """Get conversation template for Parsit (CHATML format)"""
        from parsit.conversation import conv_parsit
        return conv_parsit.copy()
    
    def analyze_document(self, document_image_path, question=None):
        """
        Analyze document image with optional specific question
        
        Args:
            document_image_path: Path to document image
            question: Specific question about the document (optional)
            
        Returns:
            Analysis result
        """
        if question is None:
            question = "Please analyze this document and provide a detailed description of its content, structure, and key information."
        
        return self.chat(question, document_image_path)
    
    def extract_text(self, document_image_path):
        """Extract text from document image"""
        question = "Please extract all the text content from this document. Maintain the original structure and formatting as much as possible."
        return self.chat(question, document_image_path)
    
    def answer_question(self, document_image_path, question):
        """Answer specific question about document"""
        return self.chat(question, document_image_path)


def load_model(model_path, **kwargs):
    """Convenience function to load Parsit model"""
    return ParsitInference(model_path, **kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize model
    parsit = ParsitInference("path/to/parsit/model")
    
    # Analyze document
    result = parsit.analyze_document("path/to/document.jpg")
    print(result)
    
    # Extract text
    text = parsit.extract_text("path/to/document.jpg") 
    print(text)
    
    # Answer specific question
    answer = parsit.answer_question("path/to/document.jpg", "What is the total amount?")
    print(answer)