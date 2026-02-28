"""3-Condition Experiment — LLM-only vs Bio-LLM vs Shadow-LLM.

Scientific protocol:
  - Multiple prompts x multiple runs per condition
  - All data saved to HDF5 with full provenance
  - Identical consciousness metrics computed for all conditions
  - Statistical analysis via Mann-Whitney U + Cohen's d

Hypothesis:
  H0: Neural substrate integration produces no measurable difference
      in consciousness metrics compared to controls.
  H1: Bio-LLM (closed-loop) shows significantly higher C-Score,
      transfer entropy, and temporal depth than LLM-only and Shadow-LLM.
"""

import os
import sys
import time
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional
from dataclasses import asdict

import h5py

from .encoder import TokenVotingEngine, TokenRecord
from .encoder_v2 import TokenVotingEngineV2
from .neural_substrate import IzhikevichSubstrate


# ---------------------------------------------------------------------------
# Experiment Prompts — designed to probe different cognitive domains
# ---------------------------------------------------------------------------

EXPERIMENT_PROMPTS = [
    "What is consciousness?",
    "The neurons are listening. What do they hear?",
    "Reflect on the nature of thought itself.",
    "What emerges at the edge of complexity?",
    "How does subjective experience arise from matter?",
    "Describe what it feels like to understand something.",
    "Is there a difference between knowing and feeling?",
    "What happens when a system becomes aware of itself?",
    "The boundary between self and world is",
    "To perceive is to",
]


# ---------------------------------------------------------------------------
# HDF5 Data Logger
# ---------------------------------------------------------------------------

class HDF5Logger:
    """Logs experiment data to HDF5 with full provenance.

    Structure:
        /metadata                     (attrs: timestamp, version, git_hash)
        /conditions/{condition}/
            runs/{run_id}/
                tokens/                (dataset: n_tokens x features)
                consciousness/         (dataset: n_tokens x metrics)
                summary/               (attrs: aggregated metrics)
                spike_matrices/        (dataset: per-token spike data)
    """

    def __init__(self, output_dir: str = "experiment_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(output_dir, f"experiment_{self.timestamp}.h5")
        self._init_file()

    def _init_file(self):
        with h5py.File(self.filepath, 'w') as f:
            f.attrs['timestamp'] = self.timestamp
            f.attrs['version'] = '1.0.0'
            f.attrs['created'] = time.strftime("%Y-%m-%dT%H:%M:%S")

            # Conditions
            for cond in ['llm_only', 'bio_llm', 'shadow_llm']:
                f.create_group(f'conditions/{cond}')

    def log_run(
        self,
        condition: str,
        run_id: int,
        prompt: str,
        records: List[TokenRecord],
        summary: Dict,
        spike_matrices: Optional[List[np.ndarray]] = None,
    ):
        """Log a single run's data to HDF5."""
        with h5py.File(self.filepath, 'a') as f:
            grp = f.create_group(f'conditions/{condition}/run_{run_id:04d}')
            grp.attrs['prompt'] = prompt
            grp.attrs['condition'] = condition
            grp.attrs['run_id'] = run_id
            grp.attrs['n_tokens'] = len(records)
            grp.attrs['generated_text'] = summary.get('generated_text', '')

            # Token-level data
            n = len(records)
            if n > 0:
                token_ids = np.array([r.token_id for r in records], dtype=np.int32)
                overrides = np.array([r.was_override for r in records], dtype=np.bool_)
                latencies = np.array([r.latency_ms for r in records], dtype=np.float32)
                total_spikes = np.array([r.total_spikes for r in records], dtype=np.int32)
                ch_entropy = np.array([r.channel_entropy for r in records], dtype=np.float32)

                grp.create_dataset('token_ids', data=token_ids)
                grp.create_dataset('overrides', data=overrides)
                grp.create_dataset('latencies_ms', data=latencies)
                grp.create_dataset('total_spikes', data=total_spikes)
                grp.create_dataset('channel_entropy', data=ch_entropy)

                # Token texts as variable-length strings
                dt = h5py.string_dtype()
                texts = np.array([r.token_text for r in records], dtype=object)
                grp.create_dataset('token_texts', data=texts, dtype=dt)

                # Consciousness metrics per token
                c_metrics = [r.consciousness_metrics for r in records if r.consciousness_metrics]
                if c_metrics:
                    metric_keys = ['cscore', 'closure', 'lambda2_norm', 'rho', 'lzc',
                                   'transfer_entropy', 'temporal_depth', 'granger_density',
                                   'sync_index', 'fano_factor', 'channel_entropy']
                    cm_array = np.zeros((len(c_metrics), len(metric_keys)), dtype=np.float64)
                    for i, m in enumerate(c_metrics):
                        for j, k in enumerate(metric_keys):
                            cm_array[i, j] = m.get(k, 0.0)
                    ds = grp.create_dataset('consciousness_metrics', data=cm_array)
                    ds.attrs['columns'] = metric_keys

            # Summary attributes
            for key, val in summary.items():
                if isinstance(val, (int, float, str, bool)):
                    grp.attrs[key] = val
                elif isinstance(val, list) and all(isinstance(v, (int, float)) for v in val):
                    grp.create_dataset(f'summary_{key}', data=np.array(val, dtype=np.float64))
                elif isinstance(val, dict) and all(isinstance(v, (int, float)) for v in val.values()):
                    # Store dicts as individual attrs with prefix
                    for dk, dv in val.items():
                        grp.attrs[f'{key}_{dk}'] = dv

            # Spike matrices (if provided, subsample for storage)
            if spike_matrices:
                sm_grp = grp.create_group('spike_matrices')
                for idx, sm in enumerate(spike_matrices[:5]):  # Store first 5 only
                    sm_grp.create_dataset(f'token_{idx:04d}', data=sm, compression='gzip')

    def log_experiment_metadata(self, config: Dict):
        """Log experiment-level configuration."""
        with h5py.File(self.filepath, 'a') as f:
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool)):
                    f.attrs[k] = v
                elif isinstance(v, list):
                    f.attrs[k] = json.dumps(v)

    def get_filepath(self) -> str:
        return self.filepath


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

class ThreeConditionExperiment:
    """Runs the full 3-condition experiment protocol.

    Protocol:
    1. For each prompt:
       a. Run LLM-only (n_runs times)
       b. Run Bio-LLM (n_runs times) — fresh substrate per prompt, persistent across runs
       c. Run Shadow-LLM (n_runs times) — same substrate, shuffled responses
    2. Log everything to HDF5
    3. Print running statistics

    The substrate is PERSISTENT within a condition's runs for a given prompt.
    This tests whether the closed loop develops over multiple generations.
    """

    def __init__(
        self,
        model_path: str = None,
        alpha: float = 0.3,
        max_tokens: int = 25,
        n_runs_per_condition: int = 5,
        prompts: List[str] = None,
        output_dir: str = "experiment_data",
        substrate_seed: int = 42,
        use_spatial: bool = True,
    ):
        if model_path is None:
            from .encoder import MODEL_PATH
            model_path = model_path or MODEL_PATH

        self.model_path = model_path
        self.alpha = alpha
        self.use_spatial = use_spatial
        self.max_tokens = max_tokens
        self.n_runs = n_runs_per_condition
        self.prompts = prompts or EXPERIMENT_PROMPTS
        self.output_dir = output_dir
        self.substrate_seed = substrate_seed

        self.logger = HDF5Logger(output_dir)
        self.all_results = []  # Accumulates for analysis

    def run(self, verbose: bool = True) -> str:
        """Execute the full experiment. Returns path to HDF5 file."""
        conditions = ['llm_only', 'bio_llm', 'shadow_llm']
        total_runs = len(self.prompts) * len(conditions) * self.n_runs
        run_counter = 0

        # Log metadata
        self.logger.log_experiment_metadata({
            'n_prompts': len(self.prompts),
            'n_conditions': len(conditions),
            'n_runs_per_condition': self.n_runs,
            'total_runs': total_runs,
            'alpha': self.alpha,
            'max_tokens': self.max_tokens,
            'substrate_seed': self.substrate_seed,
            'model_path': self.model_path,
        })

        print(f"\n{'='*70}")
        print(f"  3-CONDITION EXPERIMENT")
        print(f"  {len(self.prompts)} prompts x 3 conditions x {self.n_runs} runs = {total_runs} total")
        print(f"  Output: {self.logger.get_filepath()}")
        print(f"{'='*70}\n")

        # Create engine (LLM loaded once, substrate reset per-condition)
        if self.use_spatial:
            engine = TokenVotingEngineV2(
                model_path=self.model_path,
                alpha=self.alpha,
                substrate_seed=self.substrate_seed,
            )
            print(f"  Using SPATIAL encoder (v2) — token-specific MEA patterns")
        else:
            engine = TokenVotingEngine(
                model_path=self.model_path,
                alpha=self.alpha,
                substrate_seed=self.substrate_seed,
            )
            print(f"  Using rank-based encoder (v1)")

        for prompt_idx, prompt in enumerate(self.prompts):
            print(f"\n{'#'*70}")
            print(f"  PROMPT {prompt_idx+1}/{len(self.prompts)}: {prompt}")
            print(f"{'#'*70}")

            for condition in conditions:
                # Fresh substrate for each condition-prompt pair
                engine.substrate = IzhikevichSubstrate(
                    seed=self.substrate_seed + prompt_idx * 100 + hash(condition) % 1000
                )

                for run in range(self.n_runs):
                    run_counter += 1
                    run_id = prompt_idx * 1000 + conditions.index(condition) * 100 + run

                    print(f"\n  [{run_counter}/{total_runs}] {condition} run {run+1}/{self.n_runs}")

                    text, records, summary = engine.generate(
                        prompt=prompt,
                        max_tokens=self.max_tokens,
                        condition=condition,
                        measure_consciousness=True,
                        verbose=verbose,
                    )

                    # Collect spike matrices for first few runs
                    spike_matrices = []
                    for rec in records[:5]:
                        if rec.consciousness_metrics:
                            spike_matrices.append(
                                engine.substrate.get_last_spike_matrix()
                            )

                    # Log to HDF5
                    self.logger.log_run(
                        condition=condition,
                        run_id=run_id,
                        prompt=prompt,
                        records=records,
                        summary=summary,
                        spike_matrices=spike_matrices if spike_matrices else None,
                    )

                    # Accumulate for analysis
                    self.all_results.append({
                        'prompt_idx': prompt_idx,
                        'prompt': prompt,
                        'condition': condition,
                        'run': run,
                        'summary': summary,
                    })

                    # Print running stats
                    if 'mean_cscore' in summary:
                        print(f"    C={summary['mean_cscore']:.4f} "
                              f"TE={summary.get('mean_transfer_entropy', 0):.6f} "
                              f"TD={summary.get('mean_temporal_depth', 0):.1f} "
                              f"OR={summary['override_rate']:.2%}")

        # Final summary
        self._print_final_summary()

        print(f"\nExperiment data saved: {self.logger.get_filepath()}")
        return self.logger.get_filepath()

    def _print_final_summary(self):
        """Print aggregate statistics across all runs."""
        print(f"\n{'='*120}")
        print(f"  EXPERIMENT COMPLETE — AGGREGATE RESULTS")
        print(f"{'='*120}")
        print(f"{'Condition':<15} {'C-Score':>10} {'OR%':>8} {'TPC':>8} "
              f"{'Bl.Ent':>8} {'NL.Align':>8} {'TopBoost':>10} {'C↔Ent':>8}")
        print(f"{'-'*90}")

        for cond in ['llm_only', 'bio_llm', 'shadow_llm']:
            results = [r for r in self.all_results if r['condition'] == cond]
            if not results:
                continue

            def mean_metric(key):
                vals = [r['summary'].get(key, 0) for r in results]
                return np.mean(vals) if vals else 0.0

            print(f"{cond:<15} "
                  f"{mean_metric('mean_cscore'):>10.4f} "
                  f"{mean_metric('override_rate')*100:>7.1f}% "
                  f"{mean_metric('token_pattern_consistency'):>8.4f} "
                  f"{mean_metric('mean_blended_entropy'):>8.4f} "
                  f"{mean_metric('mean_neural_llm_alignment'):>8.4f} "
                  f"{mean_metric('mean_top_prob_boost'):>10.6f} "
                  f"{mean_metric('cscore_entropy_corr'):>8.4f}")

        print(f"{'='*120}")

    def get_results(self) -> List[Dict]:
        """Return accumulated results for external analysis."""
        return self.all_results
