"""
Conversation handling and template management for Qwen2.5-VL
"""

import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConversationTemplate(Enum):
    """Supported conversation templates"""
    CHATML = "chatml"
    QWEN = "qwen"
    VICUNA = "vicuna"
    PLAIN = "plain"


@dataclass
class ConversationConfig:
    """Configuration for conversation handling"""
    template: ConversationTemplate = ConversationTemplate.CHATML
    system_message: str = "You are a helpful AI assistant."
    user_role: str = "user"
    assistant_role: str = "assistant"
    system_role: str = "system"
    image_token: str = "<|vision_start|><|image_pad|><|vision_end|>"
    video_token: str = "<|vision_start|><|video_pad|><|vision_end|>"
    start_token: str = "<|im_start|>"
    end_token: str = "<|im_end|>"
    sep_token: str = "\n"


class ConversationHandler:
    """Handle conversation formatting and tokenization for training"""
    
    def __init__(
        self,
        tokenizer,
        processor,
        config: Optional[ConversationConfig] = None
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or ConversationConfig()
        
        # Set up template-specific configurations
        self._setup_template_config()
        
        logger.info(f"ConversationHandler initialized with template: {self.config.template.value}")
        
    def _setup_template_config(self):
        """Set up template-specific configurations"""
        
        if self.config.template == ConversationTemplate.CHATML:
            self.config.start_token = "<|im_start|>"
            self.config.end_token = "<|im_end|>"
            self.config.sep_token = "\n"
            
        elif self.config.template == ConversationTemplate.QWEN:
            self.config.start_token = "<|im_start|>"
            self.config.end_token = "<|im_end|>"
            self.config.sep_token = "\n"
            
        elif self.config.template == ConversationTemplate.VICUNA:
            self.config.start_token = ""
            self.config.end_token = ""
            self.config.sep_token = "\n"
            self.config.user_role = "USER"
            self.config.assistant_role = "ASSISTANT"
            
        elif self.config.template == ConversationTemplate.PLAIN:
            self.config.start_token = ""
            self.config.end_token = ""
            self.config.sep_token = "\n"
            
    def format_conversation(
        self,
        conversations: List[Dict[str, str]],
        images: Optional[List] = None,
        videos: Optional[List] = None,
        system_message: Optional[str] = None
    ) -> str:
        """
        Format conversation according to the selected template
        
        Args:
            conversations: List of conversation turns with 'from' and 'value' keys
            images: List of image data/paths (optional)
            videos: List of video data/paths (optional)
            system_message: Custom system message (optional)
            
        Returns:
            Formatted conversation string
        """
        
        if self.config.template == ConversationTemplate.CHATML:
            return self._format_chatml(conversations, images, videos, system_message)
        elif self.config.template == ConversationTemplate.QWEN:
            return self._format_qwen(conversations, images, videos, system_message)
        elif self.config.template == ConversationTemplate.VICUNA:
            return self._format_vicuna(conversations, images, videos, system_message)
        elif self.config.template == ConversationTemplate.PLAIN:
            return self._format_plain(conversations, images, videos, system_message)
        else:
            raise ValueError(f"Unsupported template: {self.config.template}")
            
    def _format_chatml(
        self,
        conversations: List[Dict[str, str]],
        images: Optional[List] = None,
        videos: Optional[List] = None,
        system_message: Optional[str] = None
    ) -> str:
        """Format conversation using ChatML template"""
        
        formatted_parts = []
        
        # Add system message
        system_msg = system_message or self.config.system_message
        if system_msg:
            formatted_parts.append(
                f"{self.config.start_token}{self.config.system_role}{self.config.sep_token}"
                f"{system_msg}{self.config.end_token}{self.config.sep_token}"
            )
            
        # Track media usage
        image_idx = 0
        video_idx = 0
        
        for turn in conversations:
            role = turn.get("from", "user")
            content = turn.get("value", "")
            
            # Map role names
            if role in ["human", "user"]:
                role = self.config.user_role
            elif role in ["gpt", "assistant", "bot"]:
                role = self.config.assistant_role
                
            # Process media tokens - for Qwen2.5-VL, keep tokens as-is
            processed_content = content
            
            # Count image tokens (but don't replace them)
            if self.config.image_token in processed_content and images:
                image_count = processed_content.count(self.config.image_token)
                image_idx += image_count
                    
            # Count video tokens (but don't replace them)
            if self.config.video_token in processed_content and videos:
                video_count = processed_content.count(self.config.video_token)
                video_idx += video_count
                    
            # Add formatted turn
            formatted_parts.append(
                f"{self.config.start_token}{role}{self.config.sep_token}"
                f"{processed_content}{self.config.end_token}{self.config.sep_token}"
            )
            
        return "".join(formatted_parts)
        
    def _format_qwen(
        self,
        conversations: List[Dict[str, str]],
        images: Optional[List] = None,
        videos: Optional[List] = None,
        system_message: Optional[str] = None
    ) -> str:
        """Format conversation using Qwen template (similar to ChatML)"""
        return self._format_chatml(conversations, images, videos, system_message)
        
    def _format_vicuna(
        self,
        conversations: List[Dict[str, str]],
        images: Optional[List] = None,
        videos: Optional[List] = None,
        system_message: Optional[str] = None
    ) -> str:
        """Format conversation using Vicuna template"""
        
        formatted_parts = []
        
        # Add system message
        system_msg = system_message or self.config.system_message
        if system_msg:
            formatted_parts.append(f"{system_msg}\n\n")
            
        for i, turn in enumerate(conversations):
            role = turn.get("from", "user")
            content = turn.get("value", "")
            
            # Map role names
            if role in ["human", "user"]:
                role = self.config.user_role
            elif role in ["gpt", "assistant", "bot"]:
                role = self.config.assistant_role
                
            # Add turn
            if i == 0 and role == self.config.user_role:
                formatted_parts.append(f"{role}: {content}\n")
            else:
                formatted_parts.append(f"{role}: {content}\n")
                
        return "".join(formatted_parts)
        
    def _format_plain(
        self,
        conversations: List[Dict[str, str]],
        images: Optional[List] = None,
        videos: Optional[List] = None,
        system_message: Optional[str] = None
    ) -> str:
        """Format conversation using plain template"""
        
        formatted_parts = []
        
        for turn in conversations:
            content = turn.get("value", "")
            formatted_parts.append(f"{content}\n")
            
        return "".join(formatted_parts)
        
    def prepare_training_sample(
        self,
        conversations: List[Dict[str, str]],
        images: Optional[List] = None,
        videos: Optional[List] = None,
        system_message: Optional[str] = None,
        max_length: int = 2048
    ) -> Dict[str, Any]:
        """
        Prepare a complete training sample with tokenization
        
        Args:
            conversations: List of conversation turns
            images: List of images (optional)
            videos: List of videos (optional)
            system_message: Custom system message (optional)
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        
        # Convert to HuggingFace format for Qwen2.5-VL
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
            
        # Process conversations and add images
        for turn in conversations:
            role = turn.get("from", turn.get("role", "user"))
            content = turn.get("value", turn.get("content", ""))
            
            if role in ["human", "user"]:
                # For user messages, format with image if present
                # Only add image for the first user message that has an image placeholder
                if images and len(images) > 0 and ("<|vision_start|>" in content or "<image>" in content):
                    message_content = [
                        {"type": "image", "image": images[0]},  # Use first image
                        {"type": "text", "text": content}
                    ]
                else:
                    message_content = content
                messages.append({"role": "user", "content": message_content})
            elif role in ["gpt", "assistant", "bot"]:
                messages.append({"role": "assistant", "content": content})
        
        # Use processor's chat template
        try:
            formatted_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # Process with the model processor  
            inputs = self.processor(
                text=formatted_text,
                images=images if images else None,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
        except Exception as e:
            logger.warning(f"Processor failed, falling back to tokenizer: {e}")
            # Fallback to tokenizer only
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
        # Create labels for training (copy of input_ids)
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Mask system and user tokens in labels (only train on assistant responses)
        self._mask_non_assistant_tokens(inputs, conversations)
        
        return inputs
        
    def _mask_non_assistant_tokens(
        self,
        inputs: Dict[str, Any],
        conversations: List[Dict[str, str]]
    ):
        """Mask tokens that should not contribute to loss (user messages, system messages)"""
        
        if "labels" not in inputs:
            return
            
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        
        # For simplicity, mask everything except assistant responses
        # This is a basic implementation - more sophisticated masking could be added
        
        # Convert to text to find assistant responses
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        
        # Find assistant response sections
        assistant_sections = []
        
        if self.config.template == ConversationTemplate.CHATML:
            pattern = rf"{re.escape(self.config.start_token)}{re.escape(self.config.assistant_role)}{re.escape(self.config.sep_token)}(.*?){re.escape(self.config.end_token)}"
            matches = re.finditer(pattern, text, re.DOTALL)
            
            for match in matches:
                start_pos = match.start()
                # Find the start of the actual content (after role and separator)
                content_start = text.find(self.config.sep_token, start_pos) + len(self.config.sep_token)
                content_end = match.end() - len(self.config.end_token)
                assistant_sections.append((content_start, content_end))
                
        # Mask tokens outside assistant sections
        for i in range(labels.shape[1]):
            token_start = i
            token_end = i + 1
            
            # Check if this token is in an assistant section
            in_assistant_section = False
            for section_start, section_end in assistant_sections:
                if token_start >= section_start and token_end <= section_end:
                    in_assistant_section = True
                    break
                    
            if not in_assistant_section:
                labels[0, i] = -100  # Ignore token in loss calculation
                
    def validate_conversation(self, conversations: List[Dict[str, str]]) -> bool:
        """
        Validate conversation format
        
        Args:
            conversations: List of conversation turns
            
        Returns:
            True if valid, False otherwise
        """
        
        if not conversations:
            return False
            
        for turn in conversations:
            if not isinstance(turn, dict):
                return False
            if "from" not in turn or "value" not in turn:
                return False
            if not isinstance(turn["from"], str) or not isinstance(turn["value"], str):
                return False
                
        return True
        
    def get_conversation_length(self, conversations: List[Dict[str, str]]) -> int:
        """
        Estimate conversation length in tokens
        
        Args:
            conversations: List of conversation turns
            
        Returns:
            Estimated token count
        """
        
        formatted_text = self.format_conversation(conversations)
        
        # Rough estimate - actual tokenization would be more accurate
        tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
        return len(tokens)


class ChatMLTemplate:
    """Specialized ChatML template handler for Qwen2.5-VL"""
    
    def __init__(self, tokenizer, processor):
        self.tokenizer = tokenizer
        self.processor = processor
        
    def format_multimodal_conversation(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List] = None,
        system: str = "You are a helpful AI assistant."
    ) -> str:
        """
        Format multimodal conversation with proper image token placement
        
        Args:
            messages: List of conversation messages
            images: List of images
            system: System message
            
        Returns:
            Formatted conversation string
        """
        
        conversation = []
        
        # Add system message
        if system:
            conversation.append(f"<|im_start|>system\n{system}<|im_end|>\n")
            
        # Process messages
        for i, message in enumerate(messages):
            role = message.get("role", message.get("from", "user"))
            content = message.get("content", message.get("value", ""))
            
            # Map role names
            if role in ["human", "user"]:
                role = "user"
            elif role in ["gpt", "assistant", "bot"]:
                role = "assistant"
                
            # Handle image tokens for first user message
            if i == 0 and role == "user" and images:
                # Add image tokens at the beginning of first user message
                image_tokens = "<image>" * len(images)
                if not content.startswith("<image>"):
                    content = image_tokens + content
                    
            conversation.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            
        return "".join(conversation)
        
    def create_training_prompt(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List] = None
    ) -> Tuple[str, str]:
        """
        Create training prompt with proper separation of input and target
        
        Args:
            messages: Conversation messages
            images: List of images
            
        Returns:
            Tuple of (input_text, target_text)
        """
        
        if not messages:
            raise ValueError("Messages cannot be empty")
            
        # Separate input messages from final assistant response
        input_messages = []
        target_message = None
        
        for message in messages:
            role = message.get("role", message.get("from", "user"))
            if role in ["gpt", "assistant", "bot"] and not target_message:
                target_message = message
                break
            else:
                input_messages.append(message)
                
        if not target_message:
            # If no assistant message, use the last message as target
            target_message = input_messages.pop()
            
        # Format input
        input_text = self.format_multimodal_conversation(input_messages, images)
        input_text += "<|im_start|>assistant\n"
        
        # Format target
        target_content = target_message.get("content", target_message.get("value", ""))
        target_text = target_content + "<|im_end|>"
        
        return input_text, target_text