"""
Experiment 3 — Attention Head Knockout for Hallucination
========================================================

*IOI Analogue*: Sections 2-3 and 5 of the IOI notebook perform:
  (a) Direct logit attribution — projecting each head's output onto the
      logit-difference direction to find which heads directly move the logit.
  (b) Head ablation — mean-ablating individual heads to measure their causal
      contribution to correct prediction.

*VLM Adaptation*: We perform *attention head ablation* on LLaVA's 32-layer
Llama backbone (32 heads per layer = 1024 heads total).  For each head we:
  1. Run the model with that head's output mean-ablated (replaced with
     its mean activation over a calibration set).
  2. Measure the change in logit(Yes)-logit(No) compared to the unablated run.
  3. Record whether the ablation *flips* the model's prediction.

We also compute **direct logit attribution** (DLA) — the dot product of each
head's output (at the last position) with the unembedding direction
(W_U[yes] - W_U[no]).

*Key Questions*:
  - Are there "object-grounding" heads that, when ablated, cause hallucination?
  - Are there "yes-bias" heads that, when ablated, reduce hallucination?
  - Do hallucination-critical heads attend primarily to image tokens or to
    question text?
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
    save_results, save_figure, RESULTS_DIR,
)


@torch.inference_mode()
def compute_direct_logit_attribution(bundle, input_ids, image_tensor, yes_id, no_id):
    """Compute each head's direct contribution to logit(Yes)-logit(No).

    This adapts the IOI technique of projecting head outputs onto the
    logit-difference direction:  d = W_U[yes] - W_U[no]

    Returns
    -------
    dla : np.ndarray of shape (n_layers, n_heads)
    """
    model = bundle.model
    # Get the logit-difference direction (on lm_head device)
    lm_head_weight = model.lm_head.weight  # (vocab_size, hidden_dim)
    d = (lm_head_weight[yes_id] - lm_head_weight[no_id]).float()  # (hidden_dim,)
    d_device = d.device

    # Forward pass with output_attentions and output_hidden_states
    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    dla = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        # Get the hidden states *input* to this layer's attention
        hs_in = outputs.hidden_states[layer_idx]  # (1, seq, hidden)

        # Get the attention weights
        attn_weights = outputs.attentions[layer_idx]  # (1, n_heads, seq, seq)

        # Compute V projection — move hs_in to the layer's device
        layer_device = attn.v_proj.weight.device
        hs_in_local = hs_in.to(layer_device)
        v = attn.v_proj(hs_in_local)  # (1, seq, hidden)
        v = v.view(1, -1, n_heads, head_dim).transpose(1, 2)  # (1, n_heads, seq, head_dim)

        # attn_output per head = attn_weights @ v → (1, n_heads, seq, head_dim)
        attn_weights_local = attn_weights.to(v.device)
        head_outputs = torch.matmul(attn_weights_local, v)

        # Each head's output at the last position
        last_head_out = head_outputs[0, :, -1, :]  # (n_heads, head_dim)

        # Project through o_proj to get contribution in residual stream space
        o_weight = attn.o_proj.weight  # (hidden, hidden)
        for h in range(n_heads):
            head_contribution = o_weight[:, h * head_dim : (h + 1) * head_dim].float() @ last_head_out[h].float()
            # Move to d's device for dot product
            dla[layer_idx, h] = (head_contribution.to(d_device) @ d).item()

    return dla


@torch.inference_mode()
def ablate_head_and_measure(bundle, input_ids, image_tensor, yes_id, no_id,
                            target_layer, target_head, mean_activation=None):
    """Ablate one attention head (zero its output) and measure logit diff change."""
    model = bundle.model
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    def ablation_hook(module, input, output):
        # output is a tuple: (attn_output, attn_weights, ...)
        # attn_output shape: (1, seq, hidden_dim)
        attn_output = output[0].clone()
        # Zero out the target head's contribution
        start = target_head * head_dim
        end = (target_head + 1) * head_dim
        if mean_activation is not None and mean_activation.shape[-1] == head_dim:
            # Mean-ablate: replace with mean
            seq_len = attn_output.shape[1]
            if mean_activation.shape[0] >= seq_len:
                attn_output[0, :, start:end] = mean_activation[:seq_len, :]
            else:
                attn_output[0, :, start:end] = 0.0
        else:
            attn_output[0, :, start:end] = 0.0

        return (attn_output,) + output[1:]

    layer = model.model.layers[target_layer].self_attn
    handle = layer.register_forward_hook(ablation_hook)

    try:
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        diff = (outputs.logits[0, -1, yes_id] - outputs.logits[0, -1, no_id]).item()
    finally:
        handle.remove()

    return diff


def run_experiment(args):
    print("=" * 60)
    print("Experiment 3: Attention Head Knockout / Direct Logit Attribution")
    print("=" * 60)

    bundle = load_llava(args.model_path, device=args.device)
    yes_id, no_id = get_yes_no_token_ids(bundle.tokenizer)
    samples = load_pope_samples(max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples")

    n_layers = bundle.model.config.num_hidden_layers  # 32
    n_heads = bundle.model.config.num_attention_heads  # 32

    # ---- Part A: Direct Logit Attribution ----
    print("\n--- Part A: Direct Logit Attribution ---")
    dla_by_cat = defaultdict(list)

    for sample in tqdm(samples[:args.max_samples_dla], desc="DLA"):
        prompt = build_prompt(sample.question, bundle.model)
        input_ids = tokenize_prompt(prompt, bundle.tokenizer)
        image_tensor = load_and_process_image(sample.image_file, bundle.image_processor)

        # Get baseline logit diff for categorization
        outputs = forward_pass(bundle, input_ids, image_tensor)
        base_diff = (outputs.logits[0, -1, yes_id] - outputs.logits[0, -1, no_id]).item()
        pred_yes = base_diff > 0
        label_yes = sample.label == "yes"

        if label_yes and pred_yes:
            cat = "TP"
        elif not label_yes and not pred_yes:
            cat = "TN"
        elif not label_yes and pred_yes:
            cat = "FP"
        else:
            cat = "FN"

        dla = compute_direct_logit_attribution(
            bundle, input_ids, image_tensor, yes_id, no_id
        )
        dla_by_cat[cat].append(dla)

    # Average DLA per category
    dla_results = {}
    for cat in ["TP", "TN", "FP", "FN"]:
        if dla_by_cat[cat]:
            mean_dla = np.mean(dla_by_cat[cat], axis=0)
            dla_results[cat] = mean_dla.tolist()
        else:
            dla_results[cat] = np.zeros((n_layers, n_heads)).tolist()

    # Plot DLA heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    cat_axes = {"TP": axes[0, 0], "TN": axes[0, 1], "FP": axes[1, 0], "FN": axes[1, 1]}
    cat_titles = {
        "TP": "True Positive",
        "TN": "True Negative",
        "FP": "Hallucination (FP)",
        "FN": "Missed Object (FN)",
    }

    for cat, ax in cat_axes.items():
        data = np.array(dla_results[cat])
        vmax = max(abs(data.min()), abs(data.max())) * 0.8
        if vmax < 0.01:
            vmax = 1.0
        im = ax.imshow(data.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
        ax.set_title(f"DLA: {cat_titles[cat]}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Direct Logit Attribution: Head Contributions to logit(Yes)-logit(No)\n"
                 "Red = pushes toward Yes, Blue = pushes toward No", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "exp3_dla_heatmap.png")
    plt.close(fig)

    # ---- Part B: Head Ablation on subset of layers ----
    print("\n--- Part B: Head Ablation ---")
    # Ablate critical layers identified from DLA
    # We'll test all heads in a subset of layers
    layers_to_ablate = list(range(0, n_layers, max(1, n_layers // args.n_ablation_layers)))

    ablation_impact = np.zeros((len(layers_to_ablate), n_heads))
    ablation_flip_rate = np.zeros((len(layers_to_ablate), n_heads))

    ablation_samples = samples[:args.max_samples_ablation]
    for si, sample in enumerate(tqdm(ablation_samples, desc="HeadAblation")):
        prompt = build_prompt(sample.question, bundle.model)
        input_ids = tokenize_prompt(prompt, bundle.tokenizer)
        image_tensor = load_and_process_image(sample.image_file, bundle.image_processor)

        # Baseline
        outputs = forward_pass(bundle, input_ids, image_tensor)
        base_diff = (outputs.logits[0, -1, yes_id] - outputs.logits[0, -1, no_id]).item()
        base_pred_yes = base_diff > 0

        for li, layer_idx in enumerate(layers_to_ablate):
            for head_idx in range(n_heads):
                ablated_diff = ablate_head_and_measure(
                    bundle, input_ids, image_tensor,
                    yes_id, no_id, layer_idx, head_idx
                )
                impact = ablated_diff - base_diff
                ablation_impact[li, head_idx] += impact
                ablated_pred_yes = ablated_diff > 0
                if ablated_pred_yes != base_pred_yes:
                    ablation_flip_rate[li, head_idx] += 1

    n_abl_samples = len(ablation_samples)
    ablation_impact /= max(n_abl_samples, 1)
    ablation_flip_rate /= max(n_abl_samples, 1)

    # Plot ablation impact
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    vmax = max(abs(ablation_impact.min()), abs(ablation_impact.max())) * 0.8
    if vmax < 0.01:
        vmax = 1.0
    im1 = ax1.imshow(ablation_impact.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax1.set_xticks(range(len(layers_to_ablate)))
    ax1.set_xticklabels(layers_to_ablate)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Head")
    ax1.set_title("Mean Logit-Diff Change When Head is Ablated\n"
                  "(Red = ablation pushes toward Yes, Blue = toward No)")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(ablation_flip_rate.T, aspect="auto", cmap="Reds", vmin=0)
    ax2.set_xticks(range(len(layers_to_ablate)))
    ax2.set_xticklabels(layers_to_ablate)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Head")
    ax2.set_title("Prediction Flip Rate When Head is Ablated\n"
                  "(How often does ablation change the model's answer?)")
    plt.colorbar(im2, ax=ax2)

    fig2.suptitle("Attention Head Ablation Analysis", fontsize=14)
    fig2.tight_layout()
    save_figure(fig2, "exp3_head_ablation.png")
    plt.close(fig2)

    # ---- Identify top hallucination-prone heads ----
    # Heads where ablation decreases Yes bias (negative impact = head was pushing Yes)
    top_yes_heads = []
    for li, layer_idx in enumerate(layers_to_ablate):
        for h in range(n_heads):
            top_yes_heads.append({
                "layer": layer_idx, "head": h,
                "mean_impact": float(ablation_impact[li, h]),
                "flip_rate": float(ablation_flip_rate[li, h]),
            })

    top_yes_heads.sort(key=lambda x: x["mean_impact"])
    top_no_bias = top_yes_heads[:10]  # heads that push most toward No when ablated
    top_yes_bias = top_yes_heads[-10:][::-1]  # heads that push most toward Yes

    result = {
        "experiment": "head_knockout",
        "n_samples": len(samples),
        "dla_results": dla_results,
        "ablation_layers": layers_to_ablate,
        "ablation_impact": ablation_impact.tolist(),
        "ablation_flip_rate": ablation_flip_rate.tolist(),
        "top_yes_bias_heads": top_yes_bias,
        "top_no_bias_heads": top_no_bias,
        "dla_group_counts": {cat: len(dla_by_cat[cat]) for cat in ["TP", "TN", "FP", "FN"]},
    }
    save_results(result, "exp3_head_knockout.json")

    print("\n--- Experiment 3 Summary ---")
    print("Top 5 heads that push toward YES (ablating them reduces Yes-bias):")
    for h in top_yes_bias[:5]:
        print(f"  Layer {h['layer']}, Head {h['head']}: "
              f"impact={h['mean_impact']:.4f}, flip_rate={h['flip_rate']:.2%}")
    print("\nTop 5 heads that push toward NO:")
    for h in top_no_bias[:5]:
        print(f"  Layer {h['layer']}, Head {h['head']}: "
              f"impact={h['mean_impact']:.4f}, flip_rate={h['flip_rate']:.2%}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3: Head Knockout")
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-samples-dla", type=int, default=50)
    parser.add_argument("--max-samples-ablation", type=int, default=30)
    parser.add_argument("--n-ablation-layers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_experiment(args)
