"""
Parsit-VLM integration for Qwen2.5-VL models.
Uses native Qwen2.5-VL multimodal architecture with fine-tuning capabilities.
"""

from typing import List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..parsit_arch import ParsitMetaModel, ParsitMetaForCausalLM

# Suppress transformers warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class ParsitQwen2VLConfig(Qwen2VLConfig):
    """
    Configuration class for Parsit Qwen2.5-VL integration.
    Extends Qwen2VLConfig with Parsit-specific parameters.
    """
    model_type = "parsit_qwen2_vl"

    def __init__(
        self,
        # Parsit-specific vision-language parameters
        mm_use_im_start_end=False,
        mm_use_im_patch_token=True,
        mm_patch_merge_type='flat',
        mm_projector_lr=None,
        mm_vision_tower_lr=None,
        mm_newline_position='one',
        tune_mm_mlp_adapter=False,
        freeze_mm_mlp_adapter=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Parsit vision-language integration parameters
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_patch_merge_type = mm_patch_merge_type
        self.mm_projector_lr = mm_projector_lr
        self.mm_vision_tower_lr = mm_vision_tower_lr
        self.mm_newline_position = mm_newline_position
        self.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        self.freeze_mm_mlp_adapter = freeze_mm_mlp_adapter


class ParsitQwen2VLForConditionalGeneration(
    Qwen2VLForConditionalGeneration, 
    ParsitMetaForCausalLM
):
    """
    Parsit Qwen2.5-VL model for fine-tuning.
    Combines native Qwen2.5-VL multimodal capabilities with Parsit training infrastructure.
    """
    config_class = ParsitQwen2VLConfig

    def __init__(self, config):
        super(Qwen2VLForConditionalGeneration, self).__init__(config)
        
        # Initialize Parsit metadata
        self.config = config
        
        # Qwen2.5-VL has native multimodal architecture
        # No need for separate vision tower initialization
        
        # Enable fine-tuning configurations
        self._setup_finetuning_config()
        
    def _setup_finetuning_config(self):
        """Setup fine-tuning specific configurations."""
        if hasattr(self.config, 'tune_mm_mlp_adapter'):
            if self.config.tune_mm_mlp_adapter:
                # Enable MLP adapter tuning for vision-language projection
                for param in self.visual.merger.parameters():
                    param.requires_grad = True
                    
        if hasattr(self.config, 'freeze_mm_mlp_adapter'):
            if self.config.freeze_mm_mlp_adapter:
                # Freeze MLP adapter parameters
                for param in self.visual.merger.parameters():
                    param.requires_grad = False

    def get_model(self):
        """Return the model instance for Parsit compatibility."""
        return self

    def encode_images(self, images):
        """
        Encode images using Qwen2.5-VL's native vision encoder.
        
        Args:
            images: Input images tensor
            
        Returns:
            Encoded image features
        """
        # Qwen2.5-VL handles image encoding internally
        # This method is for compatibility with Parsit training pipeline
        if hasattr(self, 'visual'):
            return self.visual(images)
        else:
            # Use the model's internal image processing
            return images
    
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None
    ):
        """
        Prepare inputs and labels for multimodal training.
        
        This method handles the integration of image and text inputs
        for the Qwen2.5-VL architecture during training.
        """
        # For Qwen2.5-VL, use native multimodal input processing
        if images is None:
            return {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'past_key_values': past_key_values,
                'labels': labels,
            }
        
        # Process images and text together using Qwen2.5-VL's native approach
        # The model handles multimodal fusion internally
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'labels': labels,
            'images': images,
            'image_sizes': image_sizes,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with multimodal inputs.
        
        Handles both text-only and vision-language inputs for fine-tuning.
        """
        
        if images is not None:
            # Prepare multimodal inputs
            prepared_inputs = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            
            # Update inputs with prepared multimodal data
            input_ids = prepared_inputs.get('input_ids', input_ids)
            position_ids = prepared_inputs.get('position_ids', position_ids)
            attention_mask = prepared_inputs.get('attention_mask', attention_mask)
            past_key_values = prepared_inputs.get('past_key_values', past_key_values)
            labels = prepared_inputs.get('labels', labels)
            images = prepared_inputs.get('images', images)
            image_sizes = prepared_inputs.get('image_sizes', image_sizes)

        # Use Qwen2.5-VL's native forward pass
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=images,  # Qwen2.5-VL uses pixel_values
            image_grid_thw=image_sizes,  # Qwen2.5-VL uses image_grid_thw
            return_dict=return_dict,
            **kwargs
        )

    @property
    def model(self):
        """Return the model instance for compatibility."""
        return self

# Register the configuration
ParsitQwen2VLConfig.register_for_auto_class()
ParsitQwen2VLForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")