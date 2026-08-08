"""
Shared utilities for VLM Hallucination interpretability experiments.

Provides model loading, POPE data loading, prompt construction, and
common helper functions used by all experiment scripts.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: make the VCD experiment code importable
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VCD_EXP = os.path.join(_ROOT, "VCD", "experiments")
for _p in [_VCD_EXP, os.path.join(_VCD_EXP, "llava")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DEFAULT_POPE_FILE = os.path.join(_ROOT, "data", "coco_pope_adversarial_ground_truth.json")
DEFAULT_IMAGE_DIR = os.path.join(_ROOT, "data", "val2014")
RESULTS_DIR = os.path.join(_ROOT, "results")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class POPESample:
    """One POPE benchmark question."""
    question_id: int
    image_file: str
    question: str
    label: str  # "yes" or "no"


@dataclass
class ModelBundle:
    """Everything needed to run the model."""
    tokenizer: object
    model: object
    image_processor: object
    context_len: int
    device: torch.device = torch.device("cuda")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_llava(
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "cuda",
    load_4bit: bool = False,
) -> ModelBundle:
    """Load LLaVA-1.5-7b and return a ModelBundle."""
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, load_4bit=load_4bit, device=device
    )
    model.eval()
    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        context_len=context_len,
        device=torch.device(device),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pope_samples(
    pope_file: str = DEFAULT_POPE_FILE,
    max_samples: Optional[int] = None,
    image_dir: str = DEFAULT_IMAGE_DIR,
    filter_existing_images: bool = True,
) -> List[POPESample]:
    """Load POPE JSONL questions, optionally filtering to locally-available images."""
    samples = []
    with open(pope_file, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            s = POPESample(
                question_id=d["question_id"],
                image_file=d["image"],
                question=d["text"],
                label=d["label"],
            )
            if filter_existing_images:
                if not os.path.isfile(os.path.join(image_dir, s.image_file)):
                    continue
            samples.append(s)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(question: str, model: nn.Module, conv_mode: str = "llava_v1") -> str:
    """Construct the full chat prompt for a POPE question."""
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def tokenize_prompt(prompt: str, tokenizer) -> torch.LongTensor:
    """Tokenize a prompt with image tokens and return input_ids on CUDA."""
    return (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_and_process_image(
    image_file: str,
    image_processor,
    image_dir: str = DEFAULT_IMAGE_DIR,
) -> torch.Tensor:
    """Load an image and return the preprocessed tensor (1, C, H, W) on CUDA."""
    path = os.path.join(image_dir, image_file)
    image = Image.open(path).convert("RGB")
    tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return tensor.unsqueeze(0).half().cuda()


# ---------------------------------------------------------------------------
# Token IDs for Yes / No
# ---------------------------------------------------------------------------
def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """Return (yes_token_id, no_token_id) for the tokenizer.

    LLaMA tokenizer: 'Yes' -> id, 'No' -> id.
    We take the first token of each decoded word.
    """
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------
@torch.inference_mode()
def forward_pass(
    bundle: ModelBundle,
    input_ids: torch.LongTensor,
    image_tensor: torch.Tensor,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
) -> dict:
    """Run a single forward pass through the model and return outputs dict.

    This does a *single* forward (no generation), collecting logits and
    optionally hidden_states and attentions for the full input sequence.
    """
    # We need to go through prepare_inputs_labels_for_multimodal
    # to interleave image features into the embedding sequence.
    model = bundle.model
    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
        return_dict=True,
        use_cache=False,
    )
    return outputs


def logit_diff_yes_no(logits: torch.Tensor, yes_id: int, no_id: int) -> float:
    """Compute logit(Yes) - logit(No) at the last sequence position."""
    last_logits = logits[0, -1, :]  # (vocab,)
    return (last_logits[yes_id] - last_logits[no_id]).item()


# ---------------------------------------------------------------------------
# Identify token positions in the input
# ---------------------------------------------------------------------------
def locate_image_tokens(input_ids: torch.LongTensor) -> Tuple[int, int]:
    """Return (start, end) index of IMAGE_TOKEN_INDEX tokens in input_ids.
    These are replaced by 576 image patch embeddings during forward pass."""
    ids = input_ids[0].tolist()
    positions = [i for i, x in enumerate(ids) if x == IMAGE_TOKEN_INDEX]
    if len(positions) == 0:
        return (-1, -1)
    return (positions[0], positions[-1] + 1)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def compute_pope_metrics(predictions: List[Dict]) -> Dict:
    """Given list of {pred, label} dicts, compute POPE metrics."""
    tp = tn = fp = fn = 0
    for p in predictions:
        pred = p["pred"].lower().strip()
        label = p["label"].lower().strip()
        pred_yes = "yes" in pred
        label_yes = label == "yes"
        if pred_yes and label_yes:
            tp += 1
        elif not pred_yes and not label_yes:
            tn += 1
        elif pred_yes and not label_yes:
            fp += 1
        else:
            fn += 1
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------
def save_results(results: dict, filename: str) -> str:
    """Save results dict to JSON in RESULTS_DIR. Returns the full path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[saved] {path}")
    return path


def save_figure(fig, filename: str) -> str:
    """Save a matplotlib figure to RESULTS_DIR."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")
    return path
