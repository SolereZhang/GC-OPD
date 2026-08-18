# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions for preparing reference model inputs when ref model uses a different tokenizer or prompt template.
Also includes critique distillation utilities for computing ref log probs with external vLLM critique.
"""

import logging
import os
import json
import requests
import concurrent.futures
from typing import Optional, List, Dict, Any

import torch
import numpy as np

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def prepare_ref_model_inputs(
    batch: DataProto,
    ref_tokenizer,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> DataProto:
    """Prepare input_ids, attention_mask, position_ids for reference model.
    
    When the reference model uses a different prompt template than the actor model,
    we need to re-tokenize the prompts with the ref model's tokenizer and chat template,
    then concatenate with the original response_ids.
    
    Args:
        batch (DataProto): The data batch containing:
            - raw_prompt (in non_tensor_batch): The original messages list (if return_raw_chat=True)
            - responses: The generated response token ids from rollout
            - input_ids, attention_mask, position_ids: Actor model inputs
        ref_tokenizer: The tokenizer used by the reference model (for encoding new prompts)
        apply_chat_template_kwargs (dict, optional): Additional kwargs for apply_chat_template
        
    Returns:
        DataProto: Updated batch with ref_input_ids, ref_attention_mask, ref_position_ids added
    """
    if apply_chat_template_kwargs is None:
        apply_chat_template_kwargs = {}
    
    # Check if raw_prompt is available
    if "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError(
            "raw_prompt not found in batch.non_tensor_batch. "
            "Please set data.return_raw_chat=True in config to enable re-tokenization for ref model."
        )
    
    batch_size = len(batch)
    raw_prompts = batch.non_tensor_batch["raw_prompt"]  # List of messages
    responses = batch.batch["responses"]  # (batch_size, response_length)
    response_length = responses.shape[1]
    
    # Step 1: Tokenize all prompts with ref tokenizer's chat template
    ref_prompt_ids_list = []
    
    for i in range(batch_size):
        # Get the raw messages for this sample
        prompt_spec = raw_prompts[i]
        if isinstance(prompt_spec, str):
            ref_prompt_str = prompt_spec
        else:
            if not isinstance(prompt_spec, (list, np.ndarray)):
                raise TypeError(f"raw_prompt must be a list, numpy array, or string, got {type(prompt_spec)}")
            messages = list(prompt_spec)

            # Apply chat template to get the prompt string using ref tokenizer
            ref_prompt_str = ref_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs
            )
        
        # Tokenize prompt
        ref_prompt_output = ref_tokenizer(
            ref_prompt_str, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        ref_prompt_ids = ref_prompt_output["input_ids"][0]
        ref_prompt_ids_list.append(ref_prompt_ids)
    
    # Step 2: Find max prompt length and left pad all prompts
    max_prompt_len = max(len(ids) for ids in ref_prompt_ids_list)
    
    ref_prompt_ids_padded = []
    ref_prompt_attention_mask_padded = []
    
    for ref_prompt_ids in ref_prompt_ids_list:
        prompt_len = len(ref_prompt_ids)
        pad_len = max_prompt_len - prompt_len
        
        if pad_len > 0:
            # Left pad prompt ids
            padding = torch.full((pad_len,), ref_tokenizer.pad_token_id, dtype=ref_prompt_ids.dtype)
            ref_prompt_ids_padded_item = torch.cat([padding, ref_prompt_ids], dim=0)
            # Create attention mask: 0 for padding, 1 for real tokens
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(prompt_len, dtype=torch.long)
            ], dim=0)
        else:
            ref_prompt_ids_padded_item = ref_prompt_ids
            attention_mask = torch.ones(prompt_len, dtype=torch.long)
        
        ref_prompt_ids_padded.append(ref_prompt_ids_padded_item)
        ref_prompt_attention_mask_padded.append(attention_mask)
    
    # Stack into tensors
    ref_prompt_ids_tensor = torch.stack(ref_prompt_ids_padded, dim=0)  # (batch_size, max_prompt_len)
    ref_prompt_attention_mask_tensor = torch.stack(ref_prompt_attention_mask_padded, dim=0)  # (batch_size, max_prompt_len)
    
    # Step 3: Concat prompt with original responses
    # responses are already right-padded from rollout
    ref_input_ids_tensor = torch.cat([ref_prompt_ids_tensor, responses], dim=1)  # (batch_size, max_prompt_len + response_length)
    
    # Create attention mask for responses (1 for non-padding tokens)
    # Response attention mask should be derived from the original attention mask or response_mask
    if "response_mask" in batch.batch:
        response_attention_mask = batch.batch["response_mask"]
    else:
        # Fallback: assume all response tokens are valid (no padding)
        response_attention_mask = torch.ones_like(responses, dtype=torch.long)
    
    ref_attention_mask_tensor = torch.cat([ref_prompt_attention_mask_tensor, response_attention_mask], dim=1)
    
    # Step 4: Compute position_ids
    ref_position_ids_tensor = compute_position_id_with_mask(ref_attention_mask_tensor)
    
    # Add to batch
    batch.batch["ref_input_ids"] = ref_input_ids_tensor
    batch.batch["ref_attention_mask"] = ref_attention_mask_tensor
    batch.batch["ref_position_ids"] = ref_position_ids_tensor
    
    print(
        f"Original model input_ids shape={batch.batch['input_ids'].shape}, "
        f"Original model attention_mask shape={batch.batch['attention_mask'].shape}, "
        f"Original model position_ids shape={batch.batch['position_ids'].shape}, "
        f"Prepared ref model inputs: ref_input_ids shape={ref_input_ids_tensor.shape}, "
        f"ref_attention_mask shape={ref_attention_mask_tensor.shape}, "
        f"ref_position_ids shape={ref_position_ids_tensor.shape}"
    )
    
    return batch
