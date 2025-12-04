"""
Custom Data Collator for RCT Training

Preserves text fields needed for relational coherence loss while handling tensors.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer


@dataclass
class RelationalDataCollator:
    """
    Data collator that preserves text fields for relational loss computation.

    Unlike DataCollatorForLanguageModeling, this keeps input_text and output_text
    strings intact so they can be used in compute_loss().
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate features into a batch.

        Handles:
        - Tensor fields (input_ids, attention_mask, labels): stack into batch
        - String fields (input_text, output_text): keep as list
        - Other fields: keep as list
        """
        batch = {}

        # Separate tensor and non-tensor features
        tensor_keys = ['input_ids', 'attention_mask', 'labels']
        text_keys = ['input_text', 'output_text']
        metadata_keys = ['has_presence', 'type']

        # Stack tensor fields
        for key in tensor_keys:
            if key in features[0]:
                # Convert to tensors and stack
                batch[key] = torch.stack([
                    torch.tensor(f[key]) if not isinstance(f[key], torch.Tensor) else f[key]
                    for f in features
                ])

        # Preserve text fields as lists
        for key in text_keys:
            if key in features[0]:
                batch[key] = [f[key] for f in features]

        # Preserve metadata fields as lists
        for key in metadata_keys:
            if key in features[0]:
                batch[key] = [f[key] for f in features]

        return batch
