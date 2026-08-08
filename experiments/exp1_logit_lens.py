"""
Experiment 1 — Logit Lens for VLM Hallucination
================================================

*IOI Analogue*: Section 2 of the IOI notebook applies logit lens by projecting
the accumulated residual stream at each layer onto the logit-difference
direction (u_IO - u_S).  Here we adapt this to VLM hallucination by measuring
the logit difference  logit(Yes) - logit(No)  at every intermediate layer of
the LLM backbone.

*Key Insight*: In IOI, the model cannot distinguish IO from S until layer ~7,
and the signal peaks around layer 9 before declining slightly.  We ask: at
which layers does LLaVA "decide" an object is present or absent?  And
critically — does the trajectory differ for *correct* answers vs.
*hallucinated* answers?

*What this measures*:
  - For each (question, image) pair, we run one forward pass with
    `output_hidden_states=True`.
  - At each layer l, we take hidden_states[l] at the *last* token position,
    project it through the LM head to get logits, then compute
    logit(Yes) - logit(No).
  - We group samples by (ground_truth, prediction_correct) and average the
    trajectories.  This produces 4 curves:
      1. True Positive  (label=yes, model says yes → correct detection)
      2. True Negative  (label=no,  model says no  → correct rejection)
      3. False Positive (label=no,  model says yes → HALLUCINATION)
      4. False Negative (label=yes, model says no  → missed object)

*Expected Findings*:
  - Correct Yes/No should show clean separation early.
  - Hallucinations (FP) should look like TP in early layers but diverge late,
    OR they never develop strong visual grounding (flat early trajectory).
  - Missed objects (FN) may show initial Yes signal that gets suppressed.
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


def logit_lens_trajectory(
    bundle, input_ids, image_tensor, yes_id, no_id
):
    """Return per-layer logit(Yes)-logit(No) at the last token position.

    Returns
    -------
    trajectory : list[float]
        Length = n_layers + 1  (embedding + each transformer layer)
    final_pred : str
        "yes" or "no" based on final-layer logit diff
    """
    outputs = forward_pass(
        bundle, input_ids, image_tensor,
        output_hidden_states=True, output_attentions=False,
    )

    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
    lm_head = bundle.model.lm_head          # nn.Linear(hidden_dim, vocab_size)

    trajectory = []
    # Get lm_head weight device and dtype
    lm_weight = lm_head.weight.float()  # (vocab_size, hidden_dim) on lm_head device

    for hs in hidden_states:
        last_hs = hs[0, -1, :].float().to(lm_weight.device)  # match device
        logits = lm_weight @ last_hs                          # (vocab_size,)
        diff = (logits[yes_id] - logits[no_id]).item()
        trajectory.append(diff)

    final_diff = trajectory[-1]
    final_pred = "yes" if final_diff > 0 else "no"
    return trajectory, final_pred


def run_experiment(args):
    """Main experiment loop."""
    print("=" * 60)
    print("Experiment 1: Logit Lens Trajectory for Yes/No Hallucination")
    print("=" * 60)

    # Load model
    bundle = load_llava(args.model_path, device=args.device)
    yes_id, no_id = get_yes_no_token_ids(bundle.tokenizer)
    print(f"Token IDs — Yes: {yes_id}, No: {no_id}")

    # Load data
    samples = load_pope_samples(max_samples=args.max_samples)
    print(f"Loaded {len(samples)} POPE samples")

    # Collect trajectories grouped by outcome
    groups = defaultdict(list)  # key: "TP" / "TN" / "FP" / "FN"
    all_records = []

    for sample in tqdm(samples, desc="Logit Lens"):
        prompt = build_prompt(sample.question, bundle.model)
        input_ids = tokenize_prompt(prompt, bundle.tokenizer)
        image_tensor = load_and_process_image(sample.image_file, bundle.image_processor)

        trajectory, pred = logit_lens_trajectory(
            bundle, input_ids, image_tensor, yes_id, no_id
        )

        label_yes = sample.label == "yes"
        pred_yes = pred == "yes"
        if label_yes and pred_yes:
            cat = "TP"
        elif not label_yes and not pred_yes:
            cat = "TN"
        elif not label_yes and pred_yes:
            cat = "FP"   # hallucination
        else:
            cat = "FN"   # missed object

        groups[cat].append(trajectory)
        all_records.append({
            "question_id": sample.question_id,
            "label": sample.label,
            "pred": pred,
            "category": cat,
            "trajectory": trajectory,
        })

    # ---- Compute statistics ----
    n_layers = len(all_records[0]["trajectory"])
    stats = {}
    for cat in ["TP", "TN", "FP", "FN"]:
        trajs = groups.get(cat, [])
        if not trajs:
            stats[cat] = {"mean": [0.0] * n_layers, "std": [0.0] * n_layers, "count": 0}
            continue
        arr = np.array(trajs)
        stats[cat] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": len(trajs),
        }

    # ---- Save numeric results ----
    result = {
        "experiment": "logit_lens_yes_no",
        "n_samples": len(all_records),
        "n_layers": n_layers,
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "group_counts": {cat: stats[cat]["count"] for cat in ["TP", "TN", "FP", "FN"]},
        "group_stats": stats,
    }
    save_results(result, "exp1_logit_lens.json")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}
    labels_map = {
        "TP": f"True Positive  (n={stats['TP']['count']})",
        "TN": f"True Negative  (n={stats['TN']['count']})",
        "FP": f"Hallucination (FP)  (n={stats['FP']['count']})",
        "FN": f"Missed Object (FN)  (n={stats['FN']['count']})",
    }

    x = np.arange(n_layers)
    for cat in ["TP", "TN", "FP", "FN"]:
        if stats[cat]["count"] == 0:
            continue
        mean = np.array(stats[cat]["mean"])
        std = np.array(stats[cat]["std"])
        ax.plot(x, mean, color=colors[cat], linewidth=2, label=labels_map[cat])
        ax.fill_between(x, mean - std, mean + std, color=colors[cat], alpha=0.15)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer Index (0 = embedding, 1..32 = transformer layers)", fontsize=12)
    ax.set_ylabel("Logit(Yes) − Logit(No)", fontsize=12)
    ax.set_title("Logit Lens: Layer-wise Yes/No Decision Trajectory\n"
                 "How does the model internally decide an object is present?", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "exp1_logit_lens_trajectory.png")
    plt.close(fig)

    # ---- Additional analysis: layer-wise accuracy ----
    # At each layer, if logit_diff > 0 → predict "yes", else "no".
    # Compute accuracy at each layer.
    layer_accuracy = []
    for l in range(n_layers):
        correct = 0
        total = len(all_records)
        for rec in all_records:
            layer_pred_yes = rec["trajectory"][l] > 0
            label_yes = rec["label"] == "yes"
            if layer_pred_yes == label_yes:
                correct += 1
        layer_accuracy.append(correct / total if total else 0)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(range(n_layers), layer_accuracy, color="#9b59b6", linewidth=2, marker="o", markersize=3)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Accuracy (Yes/No)")
    ax2.set_title("Layer-wise Classification Accuracy\nAt which layer does the model 'know' the answer?")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    save_figure(fig2, "exp1_layer_accuracy.png")
    plt.close(fig2)

    result["layer_accuracy"] = layer_accuracy
    save_results(result, "exp1_logit_lens.json")

    print("\n--- Experiment 1 Summary ---")
    for cat in ["TP", "TN", "FP", "FN"]:
        print(f"  {cat}: {stats[cat]['count']} samples")
    print(f"  Final-layer accuracy: {layer_accuracy[-1]:.4f}")
    print(f"  Best layer accuracy:  {max(layer_accuracy):.4f} (layer {np.argmax(layer_accuracy)})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 1: Logit Lens")
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_experiment(args)
