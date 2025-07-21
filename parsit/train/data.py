import copy
import json
import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
import yaml
from torch.utils.data import Dataset

from parsit.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from parsit.utils import rank0_print
from parsit.mm_utils import process_images, tokenizer_image_token


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or self.data_args.early_mix_text:
                cur_len = cur_len if cur_len > self.data_args.image_token_len else self.data_args.image_token_len
                length_list.append(1000 + cur_len)  # Multimodal inputs - longer
            else:
                length_list.append(cur_len)  # Text-only inputs
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= self.tokenizer.model_max_length:
            return {}
        
        sources = self.list_data_dict[i]
        
        # Deep copy to avoid modifying original data
        sources = copy.deepcopy(sources)

        # Preprocess the data based on model requirements
        data_dict = preprocess([sources], self.tokenizer, self.data_args)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # Multimodal data processing
        if "image" in sources:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            
            try:
                # Handle relative and absolute paths
                image_path = image_file if image_file.startswith('/') else f"{image_folder}/{image_file}"
                
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
                
                # Process images using Parsit's image processor
                if hasattr(self.data_args, 'image_processor'):
                    processed_images = process_images([image], self.data_args.image_processor, self.data_args)
                    # Format as expected by data collator: [(image_tensor, original_size, modality)]
                    data_dict["image"] = [
                        (processed_images[0], image.size, "image")
                    ]
                
            except Exception as e:
                print(f"Error loading image {image_file}: {e}")
                # Return empty dict on error to skip this sample
                return {}

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def preprocess(sources, tokenizer, data_args):
    """
    Preprocess conversation data for training.
    Surgically ported from LLaVA train.py
    """
    # Import conversation utilities
    from parsit.conversation import conv_parsit
    
    # Preprocess multimodal data if present
    if data_args.mm_use_im_start_end:
        preprocess_multimodal(sources, data_args)
    else:
        preprocess_multimodal(sources, data_args)
    
    # Use Qwen preprocessing for Parsit
    return preprocess_qwen(sources, tokenizer, data_args)


def preprocess_multimodal(sources, data_args):
    """Process multimodal tokens in conversations."""
    is_multimodal = data_args.is_multimodal
    image_token_len = data_args.image_token_len
    
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            
            # Replace image tokens for proper processing
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_qwen(sources, tokenizer, data_args):
    """
    Preprocess conversations for Qwen model.
    Surgically ported from LLaVA train.py
    """
    from parsit.conversation import conv_parsit
    
    conv = conv_parsit.copy()
    roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

    # Apply conversation template
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from user
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for prompt in conversations
        ]

    targets = copy.deepcopy(input_ids)
    
    # Mask human tokens for loss calculation
    for target, source in zip(targets, sources):
        tokenized_lens = []
        for conversation in source:
            conversation_len = len(tokenizer_image_token(conversation["value"], tokenizer, IMAGE_TOKEN_INDEX))
            tokenized_lens.append(conversation_len)

        speakers = [sentence["from"] for sentence in source]
        target[:] = IGNORE_INDEX
        
        cur = 0
        for i, tlen in enumerate(tokenized_lens):
            if i % 2 == 1:  # assistant responses
                target[cur:cur + tlen] = input_ids[cur:cur + tlen]
            cur += tlen

    return dict(input_ids=input_ids, labels=targets)