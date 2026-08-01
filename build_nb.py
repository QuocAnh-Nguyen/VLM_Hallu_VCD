#!/usr/bin/env python3
"""Build notebook by appending sections. Run: python3 build_nb.py"""

import json

OUT = "VLM_Hallucination_Mechanistic_Analysis.ipynb"

NB = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": []
}

def M(text):
    NB["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def C(text):
    lines = [line + "\n" for line in text.split("\n")]
    NB["cells"].append({
        "cell_type": "code", "metadata": {},
        "source": lines, "outputs": [], "execution_count": None
    })

def save():
    with open(OUT, "w") as f:
        json.dump(NB, f, indent=1, ensure_ascii=False)
    md_n = sum(1 for c in NB["cells"] if c["cell_type"] == "markdown")
    cd_n = len(NB["cells"]) - md_n
    print(f"Saved {len(NB['cells'])} cells ({md_n} md, {cd_n} code) to {OUT}")

# ================================================================
# TITLE
# ================================================================
M("""# VLM Hallucination -- Mechanistic Analysis

## Why Do Vision-Language Models Hallucinate Objects?

**Model:** `liuhaotian/llava-v1.5-7b` (CLIP ViT-L/14 + Llama-7B, 32 layers)
**Dataset:** POPE Adversarial (3000 Yes/No on 500 COCO images)
**GPUs:** Kaggle T4 16GB VRAM
**Repos:** VCD + DoLa source code local importable

### Five Experiments

| # | Experiment | Question |
|---|-----------|----------|
| E1 | **Per-Layer Logit Lens** | At which layer do hallucination and truth diverge? |
| E2 | **VCD Noise Probing** | Which layers are most sensitive to visual perturbation? |
| E3 | **Visual Logit Lens** (CVPR 2026) | What does the model "see" in high-attention regions? |
| E4 | **Activation Patching** | Can we causally trace the vision->language pathway? |
| E5 | **DoLa Layer Contrast** | Does early-mature logit subtraction suppress hallucination? |

**Core metric:** `logit_diff = logit("Yes") - logit("No")` -- positive = model leans Yes

Baselines (POPE Adversarial): | VCD Accuracy 80.0% | DoLa Accuracy 83.5% |

### References
- VCD: Let al., CVPR 2024
- DoLAP: Chuang et al., ICLR 2024
- Visual Logit-Lens: Wang et al., CVPR 2026
- IOI Circuit: Wang et al., 2022
""")

# ===============================================================
# SECTION 0: Environment
# ===============================================================
M("## Section 0: Environment Setup")

M("### 0.1: Install Dependencies")

C("""\
import sys, subprocess, importlib

for pkg in ['transformers', 'accelerate', 'bitsandbytes', 'sentencepiece']:
    mn = pkg.replace('-', '_')
    try:
        importlib.import_module(mn)
        print(f'  OK  {pkg}')
    except ImportError:
        print(f'  Installing {pkg} ...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

print('Ready.')
""")

M("### 0.2: Imports and Paths")

C("""\
import os, sys, json, math
from pathlib import Path
from collections import defaultdict
from functools import partial
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

# ---- Paths ----
WORKSPACE_ROOT = Path.cwd().resolve()
VCD_ROOT   = WORKSPACE_ROOT / 'VCD'
EXP_ROOT   = VCD_ROOT / 'experiments'
LLAVA_ROOT = EXP_ROOT / 'llava'
DOLA_ROOT  = WORKSPACE_ROOT / 'DoLa'
DATA_DIR   = WORKSPACE_ROOT / 'data'
RESULTS_DIR = WORKSPACE_ROOT / 'results'

for p in [str(VCD_ROOT), str(EXP_ROOT), str(LLAVA_ROOT), str(DOLA_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

print(f'Workspace: {WORKSPACE_ROOT}')
print(f'Pri data:  {DATA_DIR}')
print(f'Ask results: {RESULTS_DIR}')
print(f'CUDA:       {torch.cuda.is_available()}'):
if torch.cuda.is_available():
    dev_props = torch.cuda.get_device_properties(0)
    print(f'GPU:        {dev_props.name}')
    print(f'VRAM: {dev_props.total_mem / 1e9:.1f} GB')
""")

M("### 0.3: Core Utilities")

C("""\
# ---- Color palette (CVD-safe) ----
CAT_COLORS = {
    'TP': '#2a78d6',   # correct Yes
    'TN': '#4caf50',   # correct No
    'FP': '#d32f2f',   # HALLUCINATION
    'FN': '#ff9800',   # missed detection
}

# ---- Token IDs ----
def get_yes_no_ids(tok):
    yes_id = tokenize.encode('Yes', add_special_tokens=False)[-1]
    no_id  = tok.encode('No',  add_special_tokens=False)[-1]
    return yes_id, no_id

# ---- Core metric ----
def logit_diff(logits, yes_id, no_id):
    return logits[..., yes_id] - logits[..., no_id]

# ---- Logit Lens projection ----
def project_logit_diff(hidden, lm_head, yes_id, no_id, ln=None):
    if ln is not None:
        hidden = ln(hidden)
    W = lm_head.weight if hasattr(lm_head, 'weight') else lm_head
    logits = hidden @ W.T
    return logits[..., yes_id] - logits[..., no_id]

# ---- Activation Cache via register_forward_hook ----
class ActCache:
    def __init__(self):
        self.data = {}
        self.handles = []

    def hook_layer(self, model, layer_idx):
        def fn(module, input, output):
            self.data[f'L{layer_idx}'] = output[0].detach().cpu()
        L = model.model.layers[layer_idx]
        self.handles.append(L.register_forward_hook(fn))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

print('Utilities ready.')
""")

# ===============================================================
# SAVE NOTEBOOK
# ===============================================================
save()
print("Done!")