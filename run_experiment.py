#!/usr/bin/env python3
"""Run the full 3-condition LLM-Encoder experiment.

Usage:
    python -m LLM_Encoder.run_experiment [--n-runs N] [--max-tokens T] [--prompts N]
    python -m LLM_Encoder.run_experiment --spatial --n-runs 5 --prompts 5

This script:
  1. Runs LLM-only, Bio-LLM, and Shadow-LLM conditions
  2. Measures C-Score and consciousness metrics per token
  3. Saves all data to HDF5
  4. Performs statistical analysis
  5. Prints scientific report
"""

import sys
import os
import time
import argparse

# Ensure parent directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_Encoder.experiment import ThreeConditionExperiment, EXPERIMENT_PROMPTS
from LLM_Encoder.analysis import ExperimentAnalyzer


def main():
    parser = argparse.ArgumentParser(description='3-Condition LLM-Encoder Experiment')
    parser.add_argument('--n-runs', type=int, default=5,
                        help='Runs per condition per prompt (default: 5)')
    parser.add_argument('--max-tokens', type=int, default=25,
                        help='Max tokens per generation (default: 25)')
    parser.add_argument('--prompts', type=int, default=None,
                        help='Number of prompts to use (default: all 10)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Neural blend weight (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='experiment_data',
                        help='Output directory (default: experiment_data)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--spatial', action='store_true', default=True,
                        help='Use spatial encoder v2 (default: True)')
    parser.add_argument('--no-spatial', dest='spatial', action='store_false',
                        help='Use rank-based encoder v1')

    args = parser.parse_args()

    prompts = EXPERIMENT_PROMPTS[:args.prompts] if args.prompts else EXPERIMENT_PROMPTS

    print(f"\n{'='*70}")
    print(f"  LLM-ENCODER 3-CONDITION EXPERIMENT")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"  Encoder:    {'Spatial v2' if args.spatial else 'Rank v1'}")
    print(f"  Substrate:  1000 Izhikevich neurons, 59 channels")
    print(f"  Prompts:    {len(prompts)}")
    print(f"  Runs/cond:  {args.n_runs}")
    print(f"  Tokens:     {args.max_tokens}")
    print(f"  Alpha:      {args.alpha}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {args.output_dir}/")
    total = len(prompts) * 3 * args.n_runs
    print(f"  Total runs: {total}")
    print(f"{'='*70}\n")

    t0 = time.time()

    experiment = ThreeConditionExperiment(
        alpha=args.alpha,
        max_tokens=args.max_tokens,
        n_runs_per_condition=args.n_runs,
        prompts=prompts,
        output_dir=args.output_dir,
        substrate_seed=args.seed,
        use_spatial=args.spatial,
    )

    h5_path = experiment.run(verbose=not args.quiet)

    elapsed = time.time() - t0
    print(f"\n  Experiment completed in {elapsed/60:.1f} minutes")

    # Analyze
    print(f"\n  Running statistical analysis...")
    analyzer = ExperimentAnalyzer(results=experiment.get_results())
    report = analyzer.analyze()
    analyzer.print_report(report)
    report_path = analyzer.save_report(report, args.output_dir)

    print(f"\n  Files:")
    print(f"    HDF5 data: {h5_path}")
    print(f"    Report:    {report_path}")
    print(f"    Duration:  {elapsed/60:.1f} minutes")

    return report


if __name__ == '__main__':
    main()
