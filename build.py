#!/usr/bin/env python3
"""Build the notebook incrementally. Each section adds cells."""

import json

# Global notebook state
NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": []
}

def M(source):
    """Add markdown cell."""
    nc["cells"].append({
        "cell_type": "markdown", "metadata": {},
        "source": [source]   # single string = one md cell
    })

def C(*lines):
    """Add code cell. Each argument is one line of code."""
    nc["cells"].append({
        "cell_type": "code", "metadata": {},
        "source": [l + "\n" for l in lines],
        "outputs": [], "execution_count": None
    })

def save(path):
    with open(path, "w") as f:
        json.dump(nc, f, indent=1, ensure_ascii=False)
    md_n = sum(1 for c in nc["cells"] if c["cell_type"] == "markdown")
    code_n = len(nc["cells"]) - md_n
    print(f"Saved {len(nc['cells'])} cells ({md_n} md, {code_n} code) -> {path}")

# =============================================================
# SECTION 0: TITLE & SETUP
# =============================================================
M("""# VLM Hallucination -- Mechanistic Analysis

## Why Do Vision-Language Models Hallucinate Objects?

**Model:** `liuhaotian/llava-v1.5-7b` (CLIP ViT-L/14 + LLaMA-7B, 32 layers)
**Dataset:** POPE Adversarial (3000 Yes/No questions, 500 images)
**Environment:** Kaggle GPU (T4, ~16 GB VRAM)
**Baselines:** VCD Acc 80.0% | DoLa Acc 83.5%

### Five Experiments

| # | Experiment | Key Question |
|---|-----------|-------------|
| E1 | Per-Layer **Logit Lens** | Where do hallucination vs truth diverge across layers? |
| 2 | **VCD Noise Probing** | Which layers are most sensitive to visual perturbation? |
| 3 | **Visual Logit Lens** (CVPR 2026) | What does the model "see" in high-attention image regions? |
| 4 | **Activation Patching** | Can we causally trace the vision-to-language pathway? |
| 5 | **DoLa Layer Contrast** | Does early-layer logit subtraction suppress hallucination? |

**Core Metric:** `logit_diff = logit("Yes") - logit("No")` -- positive = model leans Yes

---

**References:**
- VCD: Leng et al., CVPR 2024 -- Visual Contrastive Decoding
- DoLa: Chuang et al., ICLR 2024 -- Decoding by Contrasting Layers
- SADT: Wang et al., CVPR 2026 -- Logit-Lens over Visual Attention
- IOI: Wang et al., 2022 -- Interpretability in the Wild
""")

# ---- SECTION 0.1 install ----
M("## Section 0: Environment Setup & Imports")
M("### 0.1 Install Dependencies (run once)")

C(
"import sys, subprocess, importlib",
"",
"for pkg in ['transformers', 'accelerate', 'bitsandbytes', 'sentencepiece']:",
"    modname = pkg.replace('-', '_')",
"    try:",
"        importlib.import_module(modname)",
"        print(f'  OK  {pkg}')",
"    except ImportError:",
"        print(f'  Installing {pkg} ...')",
"        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])",
"",
"print('Dependencies ready.')",
)

# ---- SECTION 0.2 imports ----
M("### 0.2 Imports & Path Configuration")

C("""import os, sys, json, math",
"from pathlib import Path",
"from collections import defaultdict",
"from functools import partial",
"import warnings",
"warnings.filterwarnings('ignore')",
"",
"import torch",
"import torch.nn as nn",
"import torch.nn.functional as F",
"from torch import Tensor",
"",
"import numpy as np",
"import matplotlib.pyplot as plt",
"import seaborn as sns",
"from PIL import Image",
"from tqdm.auto import tqdm",
"",
"# Path setup",
"WORKSPACE_ROOT = Path.cwd().resolve()",
"VCD_ROOT       = WORKSPACE_ROOT / 'VCD'",
"EXP_ROOT       = VCD_ROOT / 'experiments'",
"LLAVA_ROOT     = EXP_ROOT / 'llava'",
"DOLA_ROOT      = WORKSPACE_ROOT / 'DoLa'",
"DATA_DIR       = WORKSPACE_ROOT / 'data'",
"RESULTS_DIR    = WORKSPACE_ROOT / 'results'",
"",
"for p in [str(VCD_ROOT), str(EXP_ROOT), str(LLAVA_ROOT), str(DOLA_ROOT)]:",
"    if p not in sys.path:",
"        sys.path.insert(0, p)",
"",
"print(f'Workspace: {WORKSPACE_ROOT}')",
"print(f'Data dir:  {DATA_DIR}')",
"print(f'CUDA:      {torch.cuda.is_available()}')",
"if torch.cuda.is_available():",
"    props = torch.cuda.get_device_properties(0)",
"    print(f'GPU:       {props.name}')",
"    print(f'VRAM:      {props.total_mem / 1e9:.1f} GB')",
)

# ---- SECTION 0.3 utilities ----
M("### 0.3 Core Utilities: Logit Diff, Logit Lens, Activation Hooks")

C("""# Color palette (CVD-safe)",
"CAT_COLORS = {",
"    'TP': '#2a78d6',",
"    'TN': '#4caf50',",
"    'FP': '#d32f2f',  # HALLUCINATION",
"    'FN': '#ff9800',  # missed detection",
"}",
"",
"# Token IDs for 'Yes' and 'No'",
"def get_yes_no_ids(tokenizer):",
"    yes_id = tokenizer.encode('Yes', add_special_tokens=False)[-1]",
"    no_id  = tokenizer.encode('No',  add_special_tokens=False)[-1]",
"    return yes_id, no_id",
"",
"# Core metric: logit('Yes') - logit('No')",
"def logit_diff(logits, yes_id, no_id):",
"    return logits[..., yes_id] - logits[..., no_id]",
"",
"# Logit Lens: hidden -> LN -> W_U -> logit_diff",
"def project_logit_diff(hidden, lm_head, yes_id, no_id, ln=None):",
"    if ln is not None:",
"        hidden = ln(hidden)",
"    W = lm_head.weight if hasattr(lm_head, 'weight') else lm_head",
"    logits = hidden @ W.T",
"    return logits[..., yes_id] - logits[..., no_id]",
"",
"# Activation cache using register_forward_hook",
"class ActCache:",
"    def __init__(self):",
"        self.data = {}",
"        self.handles = []",
"",
"    def hook_layer(self, model, layer_idx):",
"        def fn(module, input, output):",
"            self.data[f'L{layer_idx}'] = output[0].detach().cpu()",
"        L = model.model.layers[layer_idx]",
"        self.handles.append(L.register_forward_hook(fn))",
"",
"    def remove(self):",
"        for h in self.handles:",
"            h.remove()",
"        self.handles.clear()",
"        self.data.clear()",
"",
"print('Core utilities ready.')",
)

# =============================================================
# SAVE
# =============================================================
save("VLM_Hallucination_Mechanistic_Analysis.ipynb")
print("Done!")
JDPY

echo "Script written. Now running..."
python3 /mnt/b/Workspace/self_learning/VLM_Hallu_VCD/build.py 2>&1
