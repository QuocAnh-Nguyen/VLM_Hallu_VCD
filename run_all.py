"""
run_all.py — Orchestrator for VLM Hallucination Interpretability Experiments
============================================================================

Runs all 6 experiments sequentially, or a specific subset via --experiments.
Results are saved to results/ with per-experiment JSON data and PNG plots.

Usage:
    python run_all.py                             # Run all experiments
    python run_all.py --experiments 1 4 5          # Run only experiments 1, 4, 5
    python run_all.py --max-samples 50 --quick     # Quick test run
"""

import os
import sys
import argparse
import time
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_experiment(exp_num, args):
    """Import and run one experiment module."""
    start = time.time()
    print(f"\n{'#' * 70}")
    print(f"# EXPERIMENT {exp_num}")
    print(f"{'#' * 70}\n")

    try:
        if exp_num == 1:
            from experiments.exp1_logit_lens import run_experiment as exp_fn
            exp_args = argparse.Namespace(
                model_path=args.model_path,
                max_samples=args.max_samples,
                device=args.device,
            )
        elif exp_num == 2:
            from experiments.exp2_activation_patching import run_experiment as exp_fn
            exp_args = argparse.Namespace(
                model_path=args.model_path,
                max_samples=min(args.max_samples, 100),
                n_layers=8,
                noise_step=500,
                device=args.device,
            )
        elif exp_num == 3:
            from experiments.exp3_attention_knockout import run_experiment as exp_fn
            exp_args = argparse.Namespace(
                model_path=args.model_path,
                max_samples=args.max_samples,
                max_samples_dla=min(args.max_samples, 50),
                max_samples_ablation=min(args.max_samples, 30),
                n_ablation_layers=8,
                device=args.device,
            )
        elif exp_num == 4:
            from experiments.exp4_probing import run_experiment as exp_fn
            exp_args = argparse.Namespace(
                model_path=args.model_path,
                max_samples=args.max_samples,
                device=args.device,
            )
        elif exp_num == 5:
            from experiments.exp5_attention_flow import run_experiment as exp_fn
            exp_args = argparse.Namespace(
                model_path=args.model_path,
                max_samples=args.max_samples,
                device=args.device,
            )
        elif exp_num == 6:
            from experiments.exp6_circuit_discovery import run_experiment as exp_fn
            exp_args = argparse.Namespace(
                model_path=args.model_path,
                max_samples=min(args.max_samples, 100),
                max_samples_vcd=min(args.max_samples, 100),
                max_samples_trace=min(args.max_samples, 50),
                noise_step=500,
                window_size=4,
                device=args.device,
            )
        else:
            print(f"Unknown experiment number: {exp_num}")
            return None

        result = exp_fn(exp_args)
        elapsed = time.time() - start
        print(f"\n  Experiment {exp_num} completed in {elapsed:.1f}s")
        return result

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  Experiment {exp_num} FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Run VLM Hallucination Experiments")
    parser.add_argument("--experiments", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                        help="Which experiments to run (default: all)")
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max POPE samples per experiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: reduce samples for testing")
    args = parser.parse_args()

    if args.quick:
        args.max_samples = min(args.max_samples, 20)

    print("=" * 70)
    print("VLM Hallucination Mechanistic Interpretability — Experiment Suite")
    print("=" * 70)
    print(f"  Model:       {args.model_path}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Experiments: {args.experiments}")
    print(f"  Device:      {args.device}")
    print()

    total_start = time.time()
    results = {}
    for exp_num in args.experiments:
        results[exp_num] = run_experiment(exp_num, args)

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_elapsed:.1f}s")
    print(f"{'=' * 70}")
    for exp_num in args.experiments:
        status = "OK" if results[exp_num] is not None else "FAILED"
        print(f"  Exp {exp_num}: {status}")

    # Save summary
    summary = {
        "total_time_seconds": total_elapsed,
        "model": args.model_path,
        "max_samples": args.max_samples,
        "experiments_run": args.experiments,
        "experiment_status": {str(k): "OK" if v is not None else "FAILED"
                              for k, v in results.items()},
    }
    os.makedirs("results", exist_ok=True)
    with open("results/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to results/run_summary.json")


if __name__ == "__main__":
    main()
