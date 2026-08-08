"""
Experiment 2 — Activation Patching: Visual Grounding of Hallucination
=====================================================================

*IOI Analogue*: Section 3 of the IOI notebook uses activation patching at each
(layer, position) to determine *where* and *when* the IOI signal is computed.
A corrupted input is given to the model, but one activation is replaced with
the clean-run value; the recovery of the logit diff reveals the importance
of that (layer, position).

*VLM Adaptation*: We construct *paired* inputs for each POPE question:
  - **Clean run**:  The original image + question (ground truth context).
  - **Corrupted run**: A *noise-distorted* image (using VCD's diffusion noise)
    + the same question.

We patch the residual stream at each (layer, position_group) from the clean
run into the corrupted run.  Position groups are:
  - SYSTEM: System prompt tokens
  - IMAGE:  The 576 image-patch embedding positions
  - QUESTION: The text question tokens
  - LAST: The final token (where logits are read)

This reveals which layer and which token-group is most responsible for
correctly grounding the answer in visual evidence.

*Key Question*: When the model hallucinates (FP), does patching image tokens
recover the correct answer?  If NOT, the hallucination is driven by text
priors, not by misreading the image.

*What this measures*:
  Patching recovery = (logit_diff_patched - logit_diff_corrupted) /
                      (logit_diff_clean   - logit_diff_corrupted)
  A value of 1.0 = fully recovered, 0.0 = no effect.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_llava, load_pope_samples, build_prompt, tokenize_prompt,
    load_and_process_image, get_yes_no_token_ids, save_results,
    save_figure, RESULTS_DIR,
)

# VCD noise
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VCD = os.path.join(_ROOT, "VCD")
sys.path.insert(0, _VCD)
from vcd_utils.vcd_add_noise import add_diffusion_noise


def identify_position_groups(input_ids, n_image_patches=576):
    """Map positions in the *expanded* embedding sequence to groups.

    After prepare_inputs_labels_for_multimodal, the IMAGE_TOKEN_INDEX (-200)
    is replaced by n_image_patches positions.  We estimate the groups.

    Returns dict of group_name -> list of position indices.
    """
    ids = input_ids[0].tolist()
    img_positions = [i for i, x in enumerate(ids) if x == -200]

    if img_positions:
        img_start = img_positions[0]
        # After expansion: positions [img_start, img_start + n_image_patches)
        # are image tokens.
        # Before img_start: system tokens.
        # After img_start + n_image_patches: question tokens.
        total_text_len = len(ids) - len(img_positions)
        total_expanded = total_text_len + n_image_patches
    else:
        img_start = 0
        total_expanded = len(ids)
        n_image_patches = 0

    groups = {
        "system": list(range(0, img_start)),
        "image": list(range(img_start, img_start + n_image_patches)),
        "question": list(range(img_start + n_image_patches, total_expanded - 1)),
        "last": [total_expanded - 1],
    }
    return groups, total_expanded


@torch.inference_mode()
def get_hidden_states_and_logit_diff(bundle, input_ids, image_tensor, yes_id, no_id):
    """Forward pass returning all hidden states and the final logit diff."""
    outputs = bundle.model(
        input_ids=input_ids,
        images=image_tensor,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    logits = outputs.logits
    diff = (logits[0, -1, yes_id] - logits[0, -1, no_id]).item()
    return hidden_states, diff


def patch_and_measure(bundle, input_ids, image_clean, image_noisy,
                      yes_id, no_id, target_layer, target_positions):
    """Patch hidden states at (target_layer, target_positions) from clean→noisy.

    Strategy: Use hooks to intercept the residual stream.
    - Run clean forward to cache hidden states.
    - Run noisy forward with a hook that replaces target positions at target layer.
    """
    # Step 1: Cache clean hidden states
    clean_hs, clean_diff = get_hidden_states_and_logit_diff(
        bundle, input_ids, image_clean, yes_id, no_id
    )
    clean_at_layer = clean_hs[target_layer].clone()  # (1, seq, hidden)

    # Step 2: Forward on noisy with patching hook
    model = bundle.model

    patched_result = {}

    def patch_hook(module, input, output):
        # Transformer layer output is a tuple; output[0] is the hidden state tensor
        if isinstance(output, tuple) and len(output) > 0:
            hs = output[0].clone()
        else:
            return output
        clean_src = clean_at_layer.to(hs.device)
        for pos in target_positions:
            if pos < hs.shape[1] and pos < clean_src.shape[1]:
                hs[0, pos, :] = clean_src[0, pos, :]
        return (hs,) + output[1:]

    # Only hook transformer layers (skip layer 0 / embedding — unreliable with VLM)
    if target_layer == 0:
        return None  # skip embedding layer
    else:
        # Patch output of transformer layer (target_layer - 1)
        layer_idx = target_layer - 1
        if layer_idx < len(model.model.layers):
            handle = model.model.layers[layer_idx].register_forward_hook(patch_hook)
        else:
            return None

    try:
        outputs_patched = model(
            input_ids=input_ids,
            images=image_noisy,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        patched_diff = (
            outputs_patched.logits[0, -1, yes_id] - outputs_patched.logits[0, -1, no_id]
        ).item()
    finally:
        handle.remove()

    return patched_diff


def run_experiment(args):
    """Main experiment loop."""
    print("=" * 60)
    print("Experiment 2: Activation Patching (Clean→Noisy)")
    print("=" * 60)

    bundle = load_llava(args.model_path, device=args.device)
    yes_id, no_id = get_yes_no_token_ids(bundle.tokenizer)
    samples = load_pope_samples(max_samples=args.max_samples)
    print(f"Loaded {len(samples)} POPE samples")

    n_layers_to_probe = args.n_layers  # sample every k-th layer to save time
    model_n_layers = bundle.model.config.num_hidden_layers  # 32 for 7b
    layer_indices = list(range(1, model_n_layers + 1, max(1, model_n_layers // n_layers_to_probe)))
    if layer_indices[-1] != model_n_layers:
        layer_indices.append(model_n_layers)

    group_names = ["system", "image", "question", "last"]

    # recovery[category][group][layer_idx] = list of recovery values
    recovery = {cat: {g: defaultdict(list) for g in group_names}
                for cat in ["TP", "TN", "FP", "FN"]}

    for sample in tqdm(samples, desc="ActPatch"):
        prompt = build_prompt(sample.question, bundle.model)
        input_ids = tokenize_prompt(prompt, bundle.tokenizer)

        # Load clean and noisy images
        img_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "val2014", sample.image_file
        )
        from PIL import Image as PILImage
        pil_image = PILImage.open(img_path).convert("RGB")
        clean_tensor_raw = bundle.image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"][0]
        noisy_tensor_raw = add_diffusion_noise(clean_tensor_raw, args.noise_step)

        image_clean = clean_tensor_raw.unsqueeze(0).half().cuda()
        image_noisy = noisy_tensor_raw.unsqueeze(0).half().cuda()

        # Baselines
        _, clean_diff = get_hidden_states_and_logit_diff(
            bundle, input_ids, image_clean, yes_id, no_id
        )
        _, noisy_diff = get_hidden_states_and_logit_diff(
            bundle, input_ids, image_noisy, yes_id, no_id
        )

        if abs(clean_diff - noisy_diff) < 0.01:
            continue  # noise didn't change anything, skip

        # Determine prediction & category
        pred_yes = clean_diff > 0
        label_yes = sample.label == "yes"
        if label_yes and pred_yes:
            cat = "TP"
        elif not label_yes and not pred_yes:
            cat = "TN"
        elif not label_yes and pred_yes:
            cat = "FP"
        else:
            cat = "FN"

        # Position groups
        pos_groups, total_len = identify_position_groups(input_ids)

        # Patch each (layer, group)
        for layer_idx in layer_indices:
            for group_name in group_names:
                positions = pos_groups.get(group_name, [])
                if not positions:
                    continue
                patched_diff = patch_and_measure(
                    bundle, input_ids, image_clean, image_noisy,
                    yes_id, no_id, layer_idx, positions
                )
                if patched_diff is None:
                    continue

                # Recovery metric
                denom = clean_diff - noisy_diff
                if abs(denom) > 0.01:
                    rec = (patched_diff - noisy_diff) / denom
                else:
                    rec = 0.0
                recovery[cat][group_name][layer_idx].append(rec)

    # ---- Aggregate ----
    result_data = {
        "experiment": "activation_patching",
        "n_samples": len(samples),
        "layer_indices": layer_indices,
        "noise_step": args.noise_step,
    }

    for cat in ["TP", "TN", "FP", "FN"]:
        result_data[cat] = {}
        for gname in group_names:
            layer_means = {}
            for l in layer_indices:
                vals = recovery[cat][gname].get(l, [])
                layer_means[str(l)] = {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "std": float(np.std(vals)) if vals else 0.0,
                    "n": len(vals),
                }
            result_data[cat][gname] = layer_means

    save_results(result_data, "exp2_activation_patching.json")

    # ---- Plot heatmaps ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    cat_axes = {"TP": axes[0, 0], "TN": axes[0, 1], "FP": axes[1, 0], "FN": axes[1, 1]}
    cat_titles = {
        "TP": "True Positive (Correct Yes)",
        "TN": "True Negative (Correct No)",
        "FP": "Hallucination (False Positive)",
        "FN": "Missed Object (False Negative)",
    }

    for cat, ax in cat_axes.items():
        matrix = np.zeros((len(group_names), len(layer_indices)))
        for gi, gname in enumerate(group_names):
            for li, l in enumerate(layer_indices):
                vals = recovery[cat][gname].get(l, [])
                matrix[gi, li] = np.mean(vals) if vals else 0.0

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(len(layer_indices)))
        ax.set_xticklabels(layer_indices, fontsize=8)
        ax.set_yticks(range(len(group_names)))
        ax.set_yticklabels(group_names)
        ax.set_xlabel("Layer")
        ax.set_title(cat_titles[cat])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Activation Patching Recovery (Clean→Noisy)\n"
                 "Which (layer, token-group) restores correct Yes/No?", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "exp2_activation_patching_heatmap.png")
    plt.close(fig)

    # ---- Summary bar chart: image-token patching recovery across categories ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    cats_order = ["TP", "TN", "FP", "FN"]
    cat_colors = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}
    for ci, cat in enumerate(cats_order):
        means = []
        for l in layer_indices:
            vals = recovery[cat]["image"].get(l, [])
            means.append(np.mean(vals) if vals else 0.0)
        ax2.plot(layer_indices, means, color=cat_colors[cat], marker="o",
                 markersize=4, linewidth=2, label=cat)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Patching Recovery")
    ax2.set_title("Image-Token Patching Recovery by Layer\n"
                   "Does restoring clean image features fix hallucinations?")
    ax2.legend()
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Full recovery")
    ax2.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    save_figure(fig2, "exp2_image_patching_by_layer.png")
    plt.close(fig2)

    print("\n--- Experiment 2 Summary ---")
    for cat in cats_order:
        n = sum(len(recovery[cat]["image"].get(l, [])) for l in layer_indices)
        print(f"  {cat}: {n // len(layer_indices) if layer_indices else 0} samples")
    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2: Activation Patching")
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=8,
                        help="Number of layers to sample (evenly spaced)")
    parser.add_argument("--noise-step", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_experiment(args)
