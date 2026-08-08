"""
Experiment 4 — Hallucination Probing: Linear Probes on Residual Stream
======================================================================

*IOI Analogue*: The IOI notebook projects intermediate residual streams onto
the logit-difference direction (a linear probe).  Section 5 uses mean-ablation
with carefully constructed calibration distributions to isolate circuit
components.

*Novel VLM Adaptation*: We train lightweight *linear probes* on the residual
stream at each layer to predict:
  (a) Whether the model will hallucinate (binary: FP vs non-FP).
  (b) Whether the object is truly present (binary: label=yes vs label=no).
  (c) Whether the model's answer will be correct (binary: correct vs wrong).

By comparing probe accuracy across layers, we discover:
  - At which layer hallucination becomes "committed" — i.e., the layer after
    which a probe can reliably predict the model will hallucinate.
  - Whether hallucination is more driven by failures in visual processing
    (early layers) or language-prior override (late layers).

*Additional Analysis*: We compare probe directions to the Yes/No logit
difference direction.  If the "hallucination probe" aligns with the "Yes
direction," it suggests hallucination is a simple bias; if orthogonal,
it's a more complex failure mode.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_llava, load_pope_samples, build_prompt, tokenize_prompt,
    load_and_process_image, get_yes_no_token_ids, forward_pass,
    save_results, save_figure,
)


@torch.inference_mode()
def extract_residual_streams(bundle, input_ids, image_tensor):
    """Extract the residual stream at every layer at the LAST token position.

    Returns
    -------
    residuals : list of np.ndarray, each of shape (hidden_dim,)
        One per layer (including embedding = index 0).
    """
    outputs = bundle.model(
        input_ids=input_ids,
        images=image_tensor,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    residuals = []
    for hs in outputs.hidden_states:
        last_token = hs[0, -1, :].float().cpu().numpy()
        residuals.append(last_token)
    return residuals


def run_experiment(args):
    print("=" * 60)
    print("Experiment 4: Hallucination Probing on Residual Stream")
    print("=" * 60)

    bundle = load_llava(args.model_path, device=args.device)
    yes_id, no_id = get_yes_no_token_ids(bundle.tokenizer)
    samples = load_pope_samples(max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples")

    # ---- Collect residual streams + labels ----
    n_layers = bundle.model.config.num_hidden_layers + 1  # +1 for embedding layer
    all_residuals = [[] for _ in range(n_layers)]

    labels_present = []      # 1 if object actually present (label=yes)
    labels_hallucinate = []  # 1 if model hallucinates (FP)
    labels_correct = []      # 1 if model's answer is correct
    labels_pred_yes = []     # 1 if model predicts yes

    for sample in tqdm(samples, desc="Extracting residuals"):
        prompt = build_prompt(sample.question, bundle.model)
        input_ids = tokenize_prompt(prompt, bundle.tokenizer)
        image_tensor = load_and_process_image(sample.image_file, bundle.image_processor)

        residuals = extract_residual_streams(bundle, input_ids, image_tensor)

        # Also get prediction
        outputs = forward_pass(bundle, input_ids, image_tensor)
        logit_diff = (outputs.logits[0, -1, yes_id] - outputs.logits[0, -1, no_id]).item()
        pred_yes = logit_diff > 0

        label_yes = sample.label == "yes"
        is_correct = (pred_yes == label_yes)
        is_hallucination = (pred_yes and not label_yes)  # FP

        labels_present.append(1 if label_yes else 0)
        labels_hallucinate.append(1 if is_hallucination else 0)
        labels_correct.append(1 if is_correct else 0)
        labels_pred_yes.append(1 if pred_yes else 0)

        for l in range(n_layers):
            all_residuals[l].append(residuals[l])

    # Convert to arrays
    labels_present = np.array(labels_present)
    labels_hallucinate = np.array(labels_hallucinate)
    labels_correct = np.array(labels_correct)
    labels_pred_yes = np.array(labels_pred_yes)

    # ---- Train probes at each layer ----
    probe_tasks = {
        "object_present": labels_present,
        "will_hallucinate": labels_hallucinate,
        "answer_correct": labels_correct,
        "predicts_yes": labels_pred_yes,
    }

    results = {
        "experiment": "hallucination_probing",
        "n_samples": len(samples),
        "n_layers": n_layers,
        "label_distributions": {
            "object_present": int(labels_present.sum()),
            "will_hallucinate": int(labels_hallucinate.sum()),
            "answer_correct": int(labels_correct.sum()),
            "predicts_yes": int(labels_pred_yes.sum()),
        },
        "probe_accuracy": {},
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {
        "object_present": "#3498db",
        "will_hallucinate": "#e74c3c",
        "answer_correct": "#2ecc71",
        "predicts_yes": "#9b59b6",
    }
    labels_nice = {
        "object_present": "Object actually present",
        "will_hallucinate": "Model will hallucinate (FP)",
        "answer_correct": "Model answers correctly",
        "predicts_yes": "Model predicts 'Yes'",
    }

    for task_name, task_labels in probe_tasks.items():
        print(f"\n  Probing: {task_name} (positive: {task_labels.sum()}/{len(task_labels)})")

        # Skip if degenerate (all same label)
        if task_labels.sum() == 0 or task_labels.sum() == len(task_labels):
            print(f"    Skipping — degenerate labels")
            results["probe_accuracy"][task_name] = [0.5] * n_layers
            continue

        layer_accuracies = []
        for l in tqdm(range(n_layers), desc=f"  {task_name}", leave=False):
            X = np.array(all_residuals[l])

            # Standardize features
            X_mean = X.mean(axis=0, keepdims=True)
            X_std = X.std(axis=0, keepdims=True) + 1e-8
            X_norm = (X - X_mean) / X_std

            # Cross-validated logistic regression
            clf = LogisticRegression(
                max_iter=500, C=1.0, solver="lbfgs", random_state=42
            )
            try:
                scores = cross_val_score(clf, X_norm, task_labels, cv=5, scoring="accuracy")
                acc = scores.mean()
            except Exception:
                acc = 0.5

            layer_accuracies.append(acc)

        results["probe_accuracy"][task_name] = layer_accuracies

        ax.plot(range(n_layers), layer_accuracies, color=colors[task_name],
                linewidth=2, marker="o", markersize=3, label=labels_nice[task_name])

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
    ax.set_xlabel("Layer Index (0=embedding, 1..32=transformer layers)", fontsize=12)
    ax.set_ylabel("5-Fold CV Accuracy", fontsize=12)
    ax.set_title("Linear Probe Accuracy Across Layers\n"
                 "At which layer does the model 'commit' to hallucinating?", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    save_figure(fig, "exp4_probing_accuracy.png")
    plt.close(fig)

    # ---- Probe Direction Analysis ----
    # Train probes on the final layer and compare directions
    print("\n--- Probe Direction Alignment Analysis ---")
    lm_head_weight = bundle.model.lm_head.weight.float().cpu()
    yes_no_direction = (lm_head_weight[yes_id] - lm_head_weight[no_id]).numpy()
    yes_no_direction = yes_no_direction / (np.linalg.norm(yes_no_direction) + 1e-8)

    alignment_results = {}
    for task_name, task_labels in probe_tasks.items():
        if task_labels.sum() == 0 or task_labels.sum() == len(task_labels):
            continue

        # Train on last layer
        X_last = np.array(all_residuals[-1])
        X_mean = X_last.mean(axis=0, keepdims=True)
        X_std = X_last.std(axis=0, keepdims=True) + 1e-8
        X_norm = (X_last - X_mean) / X_std

        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42)
        clf.fit(X_norm, task_labels)

        probe_direction = clf.coef_[0]  # (hidden_dim,)
        probe_direction = probe_direction / (np.linalg.norm(probe_direction) + 1e-8)

        # Cosine similarity with Yes/No direction
        # Note: probe is in standardized space, so we transform yes_no_direction too
        yes_no_norm = (yes_no_direction - X_mean[0]) / X_std[0]
        yes_no_norm = yes_no_norm / (np.linalg.norm(yes_no_norm) + 1e-8)

        cos_sim = np.dot(probe_direction, yes_no_norm)
        alignment_results[task_name] = {
            "cosine_with_yes_no": float(cos_sim),
            "probe_accuracy_last_layer": float(results["probe_accuracy"][task_name][-1]),
        }
        print(f"  {task_name}: cos(probe, Yes-No) = {cos_sim:.4f}")

    results["alignment_analysis"] = alignment_results
    save_results(results, "exp4_probing.json")

    # ---- Bar chart of alignment ----
    if alignment_results:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        tasks = list(alignment_results.keys())
        cos_vals = [alignment_results[t]["cosine_with_yes_no"] for t in tasks]
        task_colors = [colors.get(t, "#95a5a6") for t in tasks]
        bars = ax2.bar(range(len(tasks)), cos_vals, color=task_colors)
        ax2.set_xticks(range(len(tasks)))
        ax2.set_xticklabels([labels_nice.get(t, t) for t in tasks], rotation=15, ha="right", fontsize=9)
        ax2.set_ylabel("Cosine Similarity with Yes/No Direction")
        ax2.set_title("Alignment of Probe Directions with Yes/No Logit Direction\n"
                      "High alignment → hallucination = simple Yes-bias")
        ax2.axhline(0, color="gray", linestyle="--")
        ax2.grid(alpha=0.3, axis="y")
        fig2.tight_layout()
        save_figure(fig2, "exp4_probe_alignment.png")
        plt.close(fig2)

    print("\n--- Experiment 4 Summary ---")
    for task_name in probe_tasks:
        accs = results["probe_accuracy"].get(task_name, [])
        if accs:
            best_layer = int(np.argmax(accs))
            print(f"  {task_name}: best acc = {max(accs):.4f} at layer {best_layer}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: Hallucination Probing")
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_experiment(args)
