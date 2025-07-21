import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from parsit.model import *
from parsit.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from parsit.utils import rank0_print


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype="float16", attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    if customized_config is not None:
        kwargs["config"] = customized_config

    if "multimodal" in kwargs:
        if kwargs["multimodal"] is True:
            is_multimodal = True
            kwargs.pop("multimodal")
    else:
        is_multimodal = False

    if "parsit" in model_name.lower() or is_multimodal:
        # Load Parsit model - specialized for Qwen models only
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument."
            )
        
        if "lora" in model_name.lower() and model_base is not None:
            # LoRA loading logic for Qwen
            from parsit.model.language_model.parsit_qwen import ParsitQwenConfig
            
            lora_cfg_pretrained = ParsitQwenConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            rank0_print("Loading Parsit from base model...")
            model = ParsitQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)

            token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))

            rank0_print("Loading additional Parsit weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
            else:
                # Load from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
            non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            rank0_print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            rank0_print("Merging LoRA weights...")
            model = model.merge_and_unload()
            rank0_print("Model is loaded...")
            
        elif model_base is not None:  # Loading projector with preset language model
            rank0_print(f"Loading Parsit from base model {model_base}...")
            from parsit.model.language_model.parsit_qwen import ParsitQwenConfig
            
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = ParsitQwenConfig.from_pretrained(model_path)
            model = ParsitQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
            
        else:
            # Direct model loading
            rank0_print(f"Loading Parsit model: {model_path}")
            from parsit.model.language_model.parsit_qwen import ParsitQwenConfig
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if customized_config is None:
                parsit_cfg = ParsitQwenConfig.from_pretrained(model_path)
            else:
                parsit_cfg = customized_config

            if overwrite_config is not None:
                rank0_print(f"Overwriting config with {overwrite_config}")
                for k, v in overwrite_config.items():
                    setattr(parsit_cfg, k, v)

            model = ParsitQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=parsit_cfg, **kwargs)
    else:
        # Standard language model loading
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **kwargs
        )

    # Setup image processing tokens if multimodal
    if hasattr(model, "get_vision_tower") and model.get_vision_tower() is not None:
        image_processor = model.get_vision_tower().image_processor
    else:
        image_processor = None

    if hasattr(model, "get_mm_projector") and model.get_mm_projector() is not None:
        vision_tower = model.get_vision_tower()
        vision_tower.to(device=model.device, dtype=model.dtype)
        mm_projector = model.get_mm_projector()
        mm_projector.to(device=model.device, dtype=model.dtype)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len