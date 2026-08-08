"""
Experiment 5 — Visual-Textual Attention Flow Analysis
=====================================================

*IOI Analogue*: The IOI notebook analyzes attention patterns of key heads:
name mover heads attend from END→IO, S-inhibition heads attend from END→S2.
This reveals how information flows through the circuit.

*VLM Adaptation*: We analyze attention flow in LLaVA to understand how the
model routes visual vs. textual information when making Yes/No decisions.

We measure:
  1. **Image Attention Ratio**: What fraction of attention (from the last
     token) goes to image-patch positions vs. question-text positions, at
     each layer?
  2. **Object-Name Attention**: When the model predicts "Yes", does the last
     token attend to question tokens containing the object name?  Does this
     differ for correct vs. hallucinated answers?
  3. **Attention Entropy**: How "focused" vs. "diffuse" is the attention at
     each layer?  Hallucination might correlate with overly diffuse attention
     to image patches (no specific grounding) or overly focused attention
     on text (ignoring the image).
  4. **Layer-wise Information Routing**: Quantify the shift from image-dominated
     to text-dominated attention across layers.

*Key Hypothesis*: Hallucinating models attend *less* to image patches and
*more* to text tokens (especially the object name in the question), suggesting
the model is "reading the answer from the question" rather than "seeing it
in the image."
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
    load_and_process_image, get_yes_no_token_ids, forward_pass,
    save_results, save_figure,
)


N_IMAGE_PATCHES = 576  # LLaVA 1.5: 24x24 = 576 CLIP patches


def get_token_regions(input_ids):
    """Identify system, image, and question token regions.

    Returns (img_start, img_end_expanded, total_expanded_len)
    where image tokens span [img_start, img_start + 576) after expansion.
    """
    ids = input_ids[0].tolist()
    img_positions = [i for i, x in enumerate(ids) if x == -200]

    if img_positions:
        img_start = img_positions[0]
        n_text = len(ids) - len(img_positions)
        total = n_text + N_IMAGE_PATCHES
    else:
        img_start = 0
        total = len(ids)

    return img_start, img_start + N_IMAGE_PATCHES, total


@torch.inference_mode()
def analyze_attention_patterns(bundle, input_ids, image_tensor, yes_id, no_id):
    """Run forward pass and analyze attention patterns.

    Returns dict with per-layer attention statistics.
    """
    model = bundle.model
    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )

    logit_diff = (outputs.logits[0, -1, yes_id] - outputs.logits[0, -1, no_id]).item()
    attentions = outputs.attentions  # tuple of (1, n_heads, seq, seq)

    img_start, img_end, total_len = get_token_regions(input_ids)

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    layer_stats = []
    for l in range(n_layers):
        attn = attentions[l][0]  # (n_heads, seq, seq)
        seq_len = attn.shape[-1]

        # Attention from the LAST token to all other tokens
        last_token_attn = attn[:, -1, :]  # (n_heads, seq)

        # Partition attention by region
        img_s = min(img_start, seq_len)
        img_e = min(img_end, seq_len)

        attn_to_system = last_token_attn[:, :img_s].sum(dim=-1)  # (n_heads,)
        attn_to_image = last_token_attn[:, img_s:img_e].sum(dim=-1)
        attn_to_question = last_token_attn[:, img_e:].sum(dim=-1)

        # Average across heads
        mean_sys = attn_to_system.mean().item()
        mean_img = attn_to_image.mean().item()
        mean_q = attn_to_question.mean().item()

        # Attention entropy (how diffuse is the attention?)
        # H = -sum(p * log(p))
        eps = 1e-10
        p = last_token_attn.float().clamp(min=eps)
        entropy = -(p * p.log()).sum(dim=-1).mean().item()

        # Per-head image attention ratio
        total_attn = attn_to_system + attn_to_image + attn_to_question + 1e-10
        head_img_ratios = (attn_to_image / total_attn).cpu().numpy()

        layer_stats.append({
            "attn_system": mean_sys,
            "attn_image": mean_img,
            "attn_question": mean_q,
            "entropy": entropy,
            "head_img_ratios": head_img_ratios.tolist(),
        })

    return {
        "logit_diff": logit_diff,
        "pred_yes": logit_diff > 0,
        "layer_stats": layer_stats,
    }


def run_experiment(args):
    print("=" * 60)
    print("Experiment 5: Visual-Textual Attention Flow Analysis")
    print("=" * 60)

    bundle = load_llava(args.model_path, device=args.device)
    yes_id, no_id = get_yes_no_token_ids(bundle.tokenizer)
    samples = load_pope_samples(max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples")

    n_layers = bundle.model.config.num_hidden_layers
    n_heads = bundle.model.config.num_attention_heads

    # Collect stats by category
    group_stats = {cat: {
        "attn_image": [[] for _ in range(n_layers)],
        "attn_question": [[] for _ in range(n_layers)],
        "attn_system": [[] for _ in range(n_layers)],
        "entropy": [[] for _ in range(n_layers)],
        "head_img_ratios": [[] for _ in range(n_layers)],
    } for cat in ["TP", "TN", "FP", "FN"]}

    for sample in tqdm(samples, desc="Attention Analysis"):
        prompt = build_prompt(sample.question, bundle.model)
        input_ids = tokenize_prompt(prompt, bundle.tokenizer)
        image_tensor = load_and_process_image(sample.image_file, bundle.image_processor)

        result = analyze_attention_patterns(
            bundle, input_ids, image_tensor, yes_id, no_id
        )

        label_yes = sample.label == "yes"
        pred_yes = result["pred_yes"]
        if label_yes and pred_yes:
            cat = "TP"
        elif not label_yes and not pred_yes:
            cat = "TN"
        elif not label_yes and pred_yes:
            cat = "FP"
        else:
            cat = "FN"

        for l, ls in enumerate(result["layer_stats"]):
            group_stats[cat]["attn_image"][l].append(ls["attn_image"])
            group_stats[cat]["attn_question"][l].append(ls["attn_question"])
            group_stats[cat]["attn_system"][l].append(ls["attn_system"])
            group_stats[cat]["entropy"][l].append(ls["entropy"])
            group_stats[cat]["head_img_ratios"][l].append(ls["head_img_ratios"])

    # ---- Aggregate ----
    agg = {}
    for cat in ["TP", "TN", "FP", "FN"]:
        agg[cat] = {}
        for metric in ["attn_image", "attn_question", "attn_system", "entropy"]:
            means = []
            stds = []
            for l in range(n_layers):
                vals = group_stats[cat][metric][l]
                means.append(float(np.mean(vals)) if vals else 0.0)
                stds.append(float(np.std(vals)) if vals else 0.0)
            agg[cat][metric] = {"mean": means, "std": stds}
        agg[cat]["count"] = len(group_stats[cat]["attn_image"][0]) if group_stats[cat]["attn_image"][0] else 0

    # ---- Plot 1: Image attention across layers by category ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    colors = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}
    cat_labels = {
        "TP": "True Positive", "TN": "True Negative",
        "FP": "Hallucination (FP)", "FN": "Missed Object (FN)",
    }

    x = np.arange(n_layers)

    # Panel A: Image attention
    ax = axes[0]
    for cat in ["TP", "TN", "FP", "FN"]:
        mean = np.array(agg[cat]["attn_image"]["mean"])
        std = np.array(agg[cat]["attn_image"]["std"])
        ax.plot(x, mean, color=colors[cat], linewidth=2,
                label=f"{cat_labels[cat]} (n={agg[cat]['count']})")
        ax.fill_between(x, mean - std, mean + std, color=colors[cat], alpha=0.1)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention to Image Tokens (avg over heads)")
    ax.set_title("(a) Image Attention by Layer")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel B: Entropy
    ax = axes[1]
    for cat in ["TP", "TN", "FP", "FN"]:
        mean = np.array(agg[cat]["entropy"]["mean"])
        ax.plot(x, mean, color=colors[cat], linewidth=2, label=cat_labels[cat])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Entropy (nats)")
    ax.set_title("(b) Attention Entropy by Layer\n(Lower = more focused)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Visual-Textual Attention Flow Analysis\n"
                 "Do hallucinations correlate with less image attention?", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "exp5_attention_flow.png")
    plt.close(fig)

    # ---- Plot 2: Stacked area chart showing image vs text attention routing ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    cat_axes = {"TP": axes2[0, 0], "TN": axes2[0, 1], "FP": axes2[1, 0], "FN": axes2[1, 1]}

    for cat, ax in cat_axes.items():
        sys_m = np.array(agg[cat]["attn_system"]["mean"])
        img_m = np.array(agg[cat]["attn_image"]["mean"])
        q_m = np.array(agg[cat]["attn_question"]["mean"])

        # Normalize to sum to 1
        total = sys_m + img_m + q_m + 1e-10
        sys_n = sys_m / total
        img_n = img_m / total
        q_n = q_m / total

        ax.stackplot(x, sys_n, img_n, q_n,
                     labels=["System", "Image", "Question"],
                     colors=["#bdc3c7", "#2980b9", "#e67e22"], alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attention Share")
        ax.set_title(f"{cat_labels[cat]} (n={agg[cat]['count']})")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 1)

    fig2.suptitle("Attention Routing: System vs Image vs Question\n"
                  "Information flow from visual to textual modality", fontsize=14)
    fig2.tight_layout()
    save_figure(fig2, "exp5_attention_routing.png")
    plt.close(fig2)

    # ---- Plot 3: FP vs TP difference in image attention ----
    if agg["FP"]["count"] > 0 and agg["TP"]["count"] > 0:
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        tp_img = np.array(agg["TP"]["attn_image"]["mean"])
        fp_img = np.array(agg["FP"]["attn_image"]["mean"])
        diff = fp_img - tp_img

        colors_bar = ["#e74c3c" if d < 0 else "#2ecc71" for d in diff]
        ax3.bar(x, diff, color=colors_bar, alpha=0.8)
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Image Attention Difference (FP − TP)")
        ax3.set_title("Image Attention Gap: Hallucination vs Correct Detection\n"
                      "Red bars = hallucinations attend LESS to image")
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.grid(alpha=0.3, axis="y")
        fig3.tight_layout()
        save_figure(fig3, "exp5_fp_tp_image_attention_gap.png")
        plt.close(fig3)

    # ---- Save results ----
    save_data = {
        "experiment": "attention_flow",
        "n_samples": len(samples),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "aggregated_stats": {cat: {
            metric: agg[cat][metric] for metric in ["attn_image", "attn_question", "attn_system", "entropy"]
        } for cat in ["TP", "TN", "FP", "FN"]},
        "group_counts": {cat: agg[cat]["count"] for cat in ["TP", "TN", "FP", "FN"]},
    }
    save_results(save_data, "exp5_attention_flow.json")

    print("\n--- Experiment 5 Summary ---")
    for cat in ["TP", "TN", "FP", "FN"]:
        if agg[cat]["count"] > 0:
            avg_img = np.mean(agg[cat]["attn_image"]["mean"])
            avg_ent = np.mean(agg[cat]["entropy"]["mean"])
            print(f"  {cat} (n={agg[cat]['count']}): avg_img_attn={avg_img:.4f}, avg_entropy={avg_ent:.2f}")

    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 5: Attention Flow")
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_experiment(args)
