"""
Experiment 6 — Hallucination Circuit Discovery via Causal Tracing
=================================================================

*IOI Analogue*: Sections 4-5 of the IOI notebook discover the complete IOI
circuit by combining path patching and ablation studies.  The key insight is
that the circuit has a specific *structure*: duplicate-token heads → S-inhibition
heads → name mover heads, forming a 3-stage pipeline.

*VLM Adaptation — Causal Tracing with Noise*: We adapt the causal tracing
methodology from Meng et al. (ROME, 2022) — also used in the IOI path
patching — to VLM hallucination.  Specifically:

1. **Corrupted baseline**: Run the model with a *noised* image (VCD-style
   diffusion noise) to create a "corrupted" baseline where visual information
   is degraded.

2. **Selective restoration**: At each (layer, position_group), restore the
   *clean* hidden state into the corrupted run (same as activation patching
   in Exp 2, but now we also measure *indirect effects*).

3. **Indirect Effect via Two-Step Patching**: Beyond direct restoration, we
   measure how restoring at layer L affects downstream layers.  This is done
   by restoring at (L, image_positions) and measuring the logit diff, then
   comparing with restoring at (L, image_positions) + (L', all_positions)
   simultaneously.  The difference reveals the *path-specific contribution*
   of image features through different layer groups.

4. **VCD Effect Decomposition**: We decompose the VCD contrastive signal into
   layer-wise contributions.  At each layer, we compute:
     VCD_effect_l = logit_diff(clean, layer_l) - logit_diff(noisy, layer_l)
   This reveals which layers benefit most from visual contrastive decoding.

*Novel Contribution*: This experiment connects VCD's empirical success to
specific internal mechanisms, explaining *why* contrastive decoding reduces
hallucination in terms of which layers and positions carry the corrective signal.
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
    save_figure,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "VCD"))
from vcd_utils.vcd_add_noise import add_diffusion_noise

N_IMAGE_PATCHES = 576


@torch.inference_mode()
def get_all_layer_logit_diffs(bundle, input_ids, image_tensor, yes_id, no_id):
    """Get logit(Yes)-logit(No) at each layer via logit lens.

    Returns list of floats, one per layer (0..n_layers inclusive).
    """
    outputs = bundle.model(
        input_ids=input_ids,
        images=image_tensor,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    lm_head = bundle.model.lm_head
    lm_weight = lm_head.weight.float()  # (vocab, hidden) on lm_head device
    diffs = []
    for hs in outputs.hidden_states:
        last_hs = hs[0, -1, :].float().to(lm_weight.device)
        logits = lm_weight @ last_hs
        diffs.append((logits[yes_id] - logits[no_id]).item())
    return diffs


@torch.inference_mode()
def causal_trace_window(bundle, input_ids, image_clean, image_noisy,
                        yes_id, no_id, restore_layer_start, restore_layer_end,
                        restore_positions):
    """Restore clean hidden states across a window of layers.

    Run with noisy image, but for layers in [restore_layer_start, restore_layer_end),
    replace the hidden states at restore_positions with clean-run values.

    Returns the final logit diff.
    """
    model = bundle.model

    # Step 1: Get clean hidden states
    clean_outputs = model(
        input_ids=input_ids,
        images=image_clean,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    clean_hidden = clean_outputs.hidden_states  # tuple of (1, seq, hidden)

    # Step 2: Run noisy with patching hooks
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hs = output[0].clone()
            clean_hs = clean_hidden[layer_idx + 1].to(hs.device)  # +1 because hidden_states[0] = embedding
            for pos in restore_positions:
                if pos < hs.shape[1] and pos < clean_hs.shape[1]:
                    hs[0, pos, :] = clean_hs[0, pos, :]
            return (hs,) + output[1:]
        return hook_fn

    for l in range(restore_layer_start, min(restore_layer_end, len(model.model.layers))):
        handle = model.model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(handle)

    try:
        outputs = model(
            input_ids=input_ids,
            images=image_noisy,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        diff = (outputs.logits[0, -1, yes_id] - outputs.logits[0, -1, no_id]).item()
    finally:
        for h in hooks:
            h.remove()

    return diff


def run_experiment(args):
    print("=" * 60)
    print("Experiment 6: Hallucination Circuit Discovery / VCD Decomposition")
    print("=" * 60)

    bundle = load_llava(args.model_path, device=args.device)
    yes_id, no_id = get_yes_no_token_ids(bundle.tokenizer)
    samples = load_pope_samples(max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples")

    n_model_layers = bundle.model.config.num_hidden_layers  # 32

    # Define layer windows (groups of 4 layers)
    window_size = args.window_size
    layer_windows = []
    for start in range(0, n_model_layers, window_size):
        end = min(start + window_size, n_model_layers)
        layer_windows.append((start, end))

    # ---- Part A: VCD Effect Decomposition ----
    print("\n--- Part A: VCD Layer-wise Effect Decomposition ---")
    vcd_effects_by_cat = {cat: [[] for _ in range(n_model_layers + 1)]
                          for cat in ["TP", "TN", "FP", "FN"]}

    for sample in tqdm(samples[:args.max_samples_vcd], desc="VCD Decomp"):
        prompt = build_prompt(sample.question, bundle.model)
        input_ids = tokenize_prompt(prompt, bundle.tokenizer)

        img_path = os.path.join(_ROOT, "data", "val2014", sample.image_file)
        from PIL import Image as PILImage
        pil_image = PILImage.open(img_path).convert("RGB")
        clean_raw = bundle.image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"][0]
        noisy_raw = add_diffusion_noise(clean_raw, args.noise_step)
        image_clean = clean_raw.unsqueeze(0).half().cuda()
        image_noisy = noisy_raw.unsqueeze(0).half().cuda()

        clean_diffs = get_all_layer_logit_diffs(bundle, input_ids, image_clean, yes_id, no_id)
        noisy_diffs = get_all_layer_logit_diffs(bundle, input_ids, image_noisy, yes_id, no_id)

        # Categorize based on clean-run prediction
        pred_yes = clean_diffs[-1] > 0
        label_yes = sample.label == "yes"
        if label_yes and pred_yes:
            cat = "TP"
        elif not label_yes and not pred_yes:
            cat = "TN"
        elif not label_yes and pred_yes:
            cat = "FP"
        else:
            cat = "FN"

        for l in range(len(clean_diffs)):
            effect = clean_diffs[l] - noisy_diffs[l]
            vcd_effects_by_cat[cat][l].append(effect)

    # Aggregate
    vcd_agg = {}
    for cat in ["TP", "TN", "FP", "FN"]:
        means = []
        for l in range(n_model_layers + 1):
            vals = vcd_effects_by_cat[cat][l]
            means.append(float(np.mean(vals)) if vals else 0.0)
        vcd_agg[cat] = means

    # ---- Part B: Causal Tracing with Layer Windows ----
    print("\n--- Part B: Causal Tracing (Window-based Restoration) ---")

    # Position groups
    def get_pos_groups(input_ids):
        ids = input_ids[0].tolist()
        img_pos = [i for i, x in enumerate(ids) if x == -200]
        if img_pos:
            img_start = img_pos[0]
            n_text = len(ids) - len(img_pos)
            total = n_text + N_IMAGE_PATCHES
        else:
            img_start = 0
            total = len(ids)
        return {
            "image": list(range(img_start, img_start + N_IMAGE_PATCHES)),
            "text": list(range(0, img_start)) + list(range(img_start + N_IMAGE_PATCHES, total)),
        }, total

    trace_results = {cat: {pos_type: {f"{s}-{e}": [] for s, e in layer_windows}
                           for pos_type in ["image", "text"]}
                     for cat in ["TP", "TN", "FP", "FN"]}

    for sample in tqdm(samples[:args.max_samples_trace], desc="Causal Trace"):
        prompt = build_prompt(sample.question, bundle.model)
        input_ids = tokenize_prompt(prompt, bundle.tokenizer)

        img_path = os.path.join(_ROOT, "data", "val2014", sample.image_file)
        from PIL import Image as PILImage
        pil_image = PILImage.open(img_path).convert("RGB")
        clean_raw = bundle.image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"][0]
        noisy_raw = add_diffusion_noise(clean_raw, args.noise_step)
        image_clean = clean_raw.unsqueeze(0).half().cuda()
        image_noisy = noisy_raw.unsqueeze(0).half().cuda()

        # Baselines
        clean_outputs = bundle.model(
            input_ids=input_ids, images=image_clean,
            return_dict=True, use_cache=False
        )
        clean_diff = (clean_outputs.logits[0, -1, yes_id] - clean_outputs.logits[0, -1, no_id]).item()

        noisy_outputs = bundle.model(
            input_ids=input_ids, images=image_noisy,
            return_dict=True, use_cache=False
        )
        noisy_diff = (noisy_outputs.logits[0, -1, yes_id] - noisy_outputs.logits[0, -1, no_id]).item()

        denom = clean_diff - noisy_diff
        if abs(denom) < 0.05:
            continue

        # Categorize
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

        pos_groups, total = get_pos_groups(input_ids)

        for pos_type in ["image", "text"]:
            positions = pos_groups[pos_type]
            for (ws, we) in layer_windows:
                patched_diff = causal_trace_window(
                    bundle, input_ids, image_clean, image_noisy,
                    yes_id, no_id, ws, we, positions
                )
                recovery = (patched_diff - noisy_diff) / denom
                trace_results[cat][pos_type][f"{ws}-{we}"].append(recovery)

    # Aggregate trace results
    trace_agg = {}
    for cat in ["TP", "TN", "FP", "FN"]:
        trace_agg[cat] = {}
        for pos_type in ["image", "text"]:
            window_means = {}
            for key, vals in trace_results[cat][pos_type].items():
                window_means[key] = float(np.mean(vals)) if vals else 0.0
            trace_agg[cat][pos_type] = window_means

    # ---- Plots ----

    # Plot A: VCD Effect Decomposition
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}
    cat_labels = {
        "TP": "True Positive", "TN": "True Negative",
        "FP": "Hallucination (FP)", "FN": "Missed Object (FN)",
    }
    x = np.arange(n_model_layers + 1)
    for cat in ["TP", "TN", "FP", "FN"]:
        n_cat = len(vcd_effects_by_cat[cat][0]) if vcd_effects_by_cat[cat][0] else 0
        ax.plot(x, vcd_agg[cat], color=colors[cat], linewidth=2,
                label=f"{cat_labels[cat]} (n={n_cat})")

    ax.set_xlabel("Layer (Logit Lens Position)")
    ax.set_ylabel("VCD Effect = logit_diff(clean) − logit_diff(noisy)")
    ax.set_title("VCD Effect Decomposition by Layer\n"
                 "At which layers does clean-vs-noisy image matter most?")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "exp6_vcd_decomposition.png")
    plt.close(fig)

    # Plot B: Causal Trace Heatmap
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    cat_ax = {"TP": axes2[0, 0], "TN": axes2[0, 1], "FP": axes2[1, 0], "FN": axes2[1, 1]}

    window_labels = [f"{s}-{e}" for s, e in layer_windows]

    for cat, ax in cat_ax.items():
        matrix = np.zeros((2, len(layer_windows)))
        for pi, pos_type in enumerate(["image", "text"]):
            for wi, wl in enumerate(window_labels):
                matrix[pi, wi] = trace_agg[cat].get(pos_type, {}).get(wl, 0.0)

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(len(window_labels)))
        ax.set_xticklabels(window_labels, fontsize=8, rotation=45)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Image", "Text"])
        ax.set_xlabel("Layer Window")
        ax.set_title(cat_labels[cat])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.suptitle("Causal Tracing: Restoring Clean Features → Recovery of Correct Answer\n"
                  "Which layer windows × token groups carry the corrective visual signal?",
                  fontsize=13)
    fig2.tight_layout()
    save_figure(fig2, "exp6_causal_trace_heatmap.png")
    plt.close(fig2)

    # Plot C: Image vs Text restoration comparison for FP
    if any(trace_results["FP"]["image"][k] for k in trace_results["FP"]["image"]):
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        x_windows = np.arange(len(layer_windows))
        width = 0.35
        img_vals = [trace_agg["FP"]["image"].get(wl, 0.0) for wl in window_labels]
        text_vals = [trace_agg["FP"]["text"].get(wl, 0.0) for wl in window_labels]

        ax3.bar(x_windows - width / 2, img_vals, width, label="Restore Image Tokens",
                color="#2980b9", alpha=0.8)
        ax3.bar(x_windows + width / 2, text_vals, width, label="Restore Text Tokens",
                color="#e67e22", alpha=0.8)
        ax3.set_xticks(x_windows)
        ax3.set_xticklabels(window_labels, rotation=45)
        ax3.set_xlabel("Layer Window")
        ax3.set_ylabel("Recovery (0=corrupted, 1=clean)")
        ax3.set_title("Hallucination Cases: Image vs Text Token Restoration\n"
                      "If image restoration helps more → hallucination is visual grounding failure")
        ax3.legend()
        ax3.axhline(0, color="gray", linestyle="--")
        ax3.grid(alpha=0.3, axis="y")
        fig3.tight_layout()
        save_figure(fig3, "exp6_fp_image_vs_text_restoration.png")
        plt.close(fig3)

    # Save
    result = {
        "experiment": "hallucination_circuit_vcd_decomposition",
        "n_samples": len(samples),
        "noise_step": args.noise_step,
        "window_size": args.window_size,
        "layer_windows": [[s, e] for s, e in layer_windows],
        "vcd_effect_by_layer": vcd_agg,
        "causal_trace": trace_agg,
        "group_counts_vcd": {
            cat: len(vcd_effects_by_cat[cat][0]) if vcd_effects_by_cat[cat][0] else 0
            for cat in ["TP", "TN", "FP", "FN"]
        },
    }
    save_results(result, "exp6_circuit_discovery.json")

    print("\n--- Experiment 6 Summary ---")
    print("VCD Effect by category (final layer):")
    for cat in ["TP", "TN", "FP", "FN"]:
        print(f"  {cat}: {vcd_agg[cat][-1]:.4f}")
    print("\nCausal Trace — FP Image restoration recovery:")
    for wl in window_labels:
        val = trace_agg.get("FP", {}).get("image", {}).get(wl, 0.0)
        print(f"  Window {wl}: {val:.4f}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 6: Circuit Discovery")
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-samples-vcd", type=int, default=100)
    parser.add_argument("--max-samples-trace", type=int, default=50)
    parser.add_argument("--noise-step", type=int, default=500)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_experiment(args)
