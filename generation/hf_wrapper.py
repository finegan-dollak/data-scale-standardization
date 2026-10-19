"""Wrapper that gives Hugging Face models the interface the embedders expect.

Originally named PythiaWrapper; it is used for every pretrained model in the paper.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput


@dataclass
class HFModelOutput(ModelOutput):
    all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


class HFModelWrapper(nn.Module):
    """Wraps any HuggingFace causal LM (Pythia, Gemstone, BabyLM baselines, ...) to satisfy the similarity_scoring embedder interface.

    The embedders in similarity_scoring.py expect:
      - ``outputs.all_hidden_states``: tuple of per-layer tensors (embedding + each block)
      - ``model.config.max_seq_length``: optional sequence-length cap for truncation

    HuggingFace Pythia (GPT-NeoX) returns hidden states in ``outputs.hidden_states`` and
    uses ``config.max_position_embeddings`` instead of ``config.max_seq_length``.  This
    wrapper bridges both gaps.

    Args:
        model: A HuggingFace ``AutoModelForCausalLM`` instance (e.g. Pythia).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        if not hasattr(self.config, "max_seq_length"):
            self.config.max_seq_length = getattr(self.config, "max_position_embeddings", None)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states
        else:
            # Some models (e.g. certain BabyLM architectures) return a plain
            # tuple rather than a ModelOutput.  Scan for the first element that
            # looks like a hidden-state tuple: a sequence whose members are all
            # Tensors.
            hidden_states = None
            for elem in outputs:
                if isinstance(elem, (tuple, list)) and len(elem) > 0 and all(
                    isinstance(h, torch.Tensor) for h in elem
                ):
                    hidden_states = tuple(elem)
                    break
            if hidden_states is None:
                raise ValueError(
                    "Could not extract hidden states from model output. "
                    "The model does not expose a 'hidden_states' attribute and "
                    "no tuple-of-tensors was found in the output tuple. "
                    "Ensure output_hidden_states=True is supported by this model."
                )
        return HFModelOutput(all_hidden_states=hidden_states)


# Name used in the code that produced the paper's results.
PythiaWrapper = HFModelWrapper
