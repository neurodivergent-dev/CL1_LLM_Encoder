"""Experiment 6: Perturbation-Recovery Analysis — Tests CAUSAL substrate-behavior coupling.

Scientific rationale:
    If the substrate's STDP-learned representations genuinely contribute to the
    Bio-LLM thought process, then DISRUPTING those representations mid-thought
    should produce measurable behavioral degradation that correlates with the
    substrate's consciousness metric changes.

    This is analogous to TMS perturbation in consciousness research (Casali 2013).
    PCI (Perturbational Complexity Index) measures the complexity of the brain's
    response to perturbation — we measure the Bio-LLM's behavioral response to
    substrate perturbation.

Design:
    3 conditions (Bio-LLM, Shadow-LLM, LLM-only) x 3 seeds x 20 thoughts
    Phase 1 (thoughts 1-10): Normal generation — STDP accumulates
    PERTURBATION at thought 10: Scramble 50% of excitatory weights
    Phase 2 (thoughts 11-20): Post-perturbation — measure degradation + recovery

Key predictions:
    - Bio-LLM: alignment drops, entropy rises, overrides spike after perturbation
      Then RECOVERY via STDP re-learning over thoughts 11-20
    - Shadow-LLM: NO behavioral change (shuffled spikes don't carry spatial info)
    - LLM-only: NO behavioral change (substrate doesn't influence selection)
    - Critical test: Bio perturbation response > Shadow perturbation response

This bridges the gap between functional and phenomenal integration by testing
whether consciousness-like metrics CAUSALLY predict behavioral output.
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_Encoder.encoder_v2 import TokenVotingEngineV2, MODEL_PATH
from LLM_Encoder.neural_substrate import IzhikevichSubstrate
from LLM_Encoder.consciousness import ConsciousnessAssessor


# ---------------------------------------------------------------------------
# Perturbation Types
# ---------------------------------------------------------------------------

def perturbation_weight_scramble(substrate: IzhikevichSubstrate, fraction: float = 0.5, seed: int = 99):
    """Scramble a fraction of excitatory synaptic weights.

    This destroys STDP-learned spatial representations while preserving
    overall network statistics (same total weight, same distribution).
    """
    rng = np.random.default_rng(seed)
    Ne = substrate.cfg.n_excitatory
    N = substrate.cfg.n_neurons

    # Get all excitatory weight values (nonzero)
    exc_weights = substrate.S[:, :Ne].copy()
    nonzero_mask = exc_weights != 0
    nonzero_indices = np.argwhere(nonzero_mask)

    n_to_scramble = int(len(nonzero_indices) * fraction)
    if n_to_scramble < 2:
        return {'n_scrambled': 0, 'fraction': 0.0}

    # Select random subset to scramble
    selected = rng.choice(len(nonzero_indices), size=n_to_scramble, replace=False)
    selected_indices = nonzero_indices[selected]

    # Extract values at those positions
    vals = np.array([exc_weights[r, c] for r, c in selected_indices])

    # Shuffle the values (preserves distribution, destroys spatial structure)
    rng.shuffle(vals)

    # Write back
    for idx, (r, c) in enumerate(selected_indices):
        substrate.S[r, c] = vals[idx]

    return {
        'n_scrambled': n_to_scramble,
        'total_exc_connections': len(nonzero_indices),
        'fraction': fraction,
        'perturbation_type': 'weight_scramble',
    }


# ---------------------------------------------------------------------------
# Per-Thought Metrics
# ---------------------------------------------------------------------------

def compute_thought_metrics(records, summary: Dict) -> Dict:
    """Extract key per-thought metrics from a generation run."""
    return {
        'mean_cscore': summary.get('mean_cscore', 0.0),
        'std_cscore': summary.get('std_cscore', 0.0),
        'override_rate': summary.get('override_rate', 0.0),
        'mean_neural_llm_alignment': summary.get('mean_neural_llm_alignment', 0.0),
        'mean_blended_entropy': summary.get('mean_blended_entropy', 0.0),
        'mean_top_prob_boost': summary.get('mean_top_prob_boost', 0.0),
        'token_pattern_consistency': summary.get('token_pattern_consistency', 0.0),
        'mean_transfer_entropy': summary.get('mean_transfer_entropy', 0.0),
        'mean_temporal_depth': summary.get('mean_temporal_depth', 0.0),
        'mean_granger_density': summary.get('mean_granger_density', 0.0),
        'weight_frobenius_divergence': summary.get('weight_frobenius_divergence', 0.0),
        'weight_fractional_change': summary.get('weight_fractional_change', 0.0),
        'total_spikes': sum(r.total_spikes for r in records),
        'n_tokens': len(records),
        'n_overrides': sum(1 for r in records if r.was_override),
        'cscore_entropy_corr': summary.get('cscore_entropy_corr', 0.0),
        'mean_closure': summary.get('mean_closure', 0.0),
        'mean_rho': summary.get('mean_rho', 0.0),
    }


# ---------------------------------------------------------------------------
# Perturbation Experiment
# ---------------------------------------------------------------------------

class PerturbationExperiment:
    """Runs the perturbation-recovery experiment.

    For each condition and seed:
    1. Create fresh substrate + engine
    2. Generate 10 thoughts (self-prompting) — Phase 1 (baseline)
    3. Perturbate substrate
    4. Generate 10 more thoughts — Phase 2 (post-perturbation)
    5. Record everything to HDF5
    """

    SEED_THOUGHT = "I am thinking about what it means to think. The neurons are firing and something emerges from the patterns. "

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        alpha: float = 0.5,
        tokens_per_thought: int = 50,
        n_baseline_thoughts: int = 10,
        n_recovery_thoughts: int = 10,
        perturbation_fraction: float = 0.5,
        seeds: List[int] = None,
        context_window: int = 300,
        output_dir: str = "experiment_data",
    ):
        self.model_path = model_path
        self.alpha = alpha
        self.tokens_per_thought = tokens_per_thought
        self.n_baseline = n_baseline_thoughts
        self.n_recovery = n_recovery_thoughts
        self.perturbation_fraction = perturbation_fraction
        self.seeds = seeds or [42, 137, 271]
        self.context_window = context_window
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.h5_path = os.path.join(output_dir, f"perturbation_{self.timestamp}.h5")

        self.all_trajectories = {}  # condition -> seed -> [thought_metrics]

    def _run_thought_stream(
        self,
        engine: TokenVotingEngineV2,
        condition: str,
        n_thoughts: int,
        context: str,
        verbose: bool = True,
    ) -> Tuple[str, List[Dict]]:
        """Generate a stream of thoughts, self-prompting."""
        thought_metrics = []
        seed_prefix = self.SEED_THOUGHT[:50]

        for thought_idx in range(n_thoughts):
            # Generate one thought
            text, records, summary = engine.generate(
                prompt=context,
                max_tokens=self.tokens_per_thought,
                condition=condition,
                measure_consciousness=True,
                verbose=False,
            )

            metrics = compute_thought_metrics(records, summary)
            metrics['thought_idx'] = thought_idx
            metrics['generated_text'] = text
            metrics['context_len'] = len(context)
            thought_metrics.append(metrics)

            # Self-prompt: slide context window
            context += text
            if len(context) > self.context_window:
                context = seed_prefix + "... " + context[-(self.context_window - 54):]

            if verbose:
                cs = metrics['mean_cscore']
                al = metrics['mean_neural_llm_alignment']
                ent = metrics['mean_blended_entropy']
                orr = metrics['override_rate']
                print(f"    Thought {thought_idx+1:2d}: C={cs:.3f} align={al:.3f} "
                      f"ent={ent:.3f} OR={orr:.1%} "
                      f"wt_div={metrics['weight_fractional_change']:.4f}")

        return context, thought_metrics

    def run(self, verbose: bool = True) -> str:
        """Execute the full perturbation-recovery experiment."""
        conditions = ['bio_llm', 'shadow_llm', 'llm_only']
        total_runs = len(conditions) * len(self.seeds)
        run_counter = 0
        total_thoughts = self.n_baseline + self.n_recovery

        print(f"\n{'='*78}")
        print(f"  EXPERIMENT 6: PERTURBATION-RECOVERY ANALYSIS")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*78}")
        print(f"  Conditions:      {conditions}")
        print(f"  Seeds:           {self.seeds}")
        print(f"  Tokens/thought:  {self.tokens_per_thought}")
        print(f"  Baseline:        {self.n_baseline} thoughts")
        print(f"  Recovery:        {self.n_recovery} thoughts")
        print(f"  Perturbation:    {self.perturbation_fraction:.0%} weight scramble")
        print(f"  Total thoughts:  {total_thoughts} x {total_runs} = {total_thoughts * total_runs}")
        print(f"  Output:          {self.h5_path}")
        print(f"{'='*78}\n")

        # Initialize HDF5
        with h5py.File(self.h5_path, 'w') as f:
            f.attrs['experiment'] = 'perturbation_recovery_v1'
            f.attrs['timestamp'] = self.timestamp
            f.attrs['alpha'] = self.alpha
            f.attrs['tokens_per_thought'] = self.tokens_per_thought
            f.attrs['n_baseline'] = self.n_baseline
            f.attrs['n_recovery'] = self.n_recovery
            f.attrs['perturbation_fraction'] = self.perturbation_fraction
            f.attrs['seeds'] = json.dumps(self.seeds)
            f.attrs['context_window'] = self.context_window

        t_experiment_start = time.time()

        for condition in conditions:
            self.all_trajectories[condition] = {}

            for seed in self.seeds:
                run_counter += 1
                t_run_start = time.time()

                print(f"\n{'#'*78}")
                print(f"  [{run_counter}/{total_runs}] {condition.upper()} seed={seed}")
                print(f"{'#'*78}")

                # Fresh substrate and engine for each run
                engine = TokenVotingEngineV2(
                    model_path=self.model_path,
                    alpha=self.alpha,
                    substrate_seed=seed,
                )

                context = self.SEED_THOUGHT

                # === PHASE 1: Baseline (STDP accumulation) ===
                print(f"\n  --- Phase 1: Baseline ({self.n_baseline} thoughts) ---")
                context, baseline_metrics = self._run_thought_stream(
                    engine, condition, self.n_baseline, context, verbose=verbose,
                )

                # Record pre-perturbation substrate state
                pre_perturbation_state = engine.substrate.get_state_snapshot()
                pre_perturbation_weights = engine.substrate.S.copy()

                # === PERTURBATION ===
                print(f"\n  >>> PERTURBATION: Scrambling {self.perturbation_fraction:.0%} of excitatory weights <<<")
                perturbation_info = perturbation_weight_scramble(
                    engine.substrate,
                    fraction=self.perturbation_fraction,
                    seed=seed + 1000,
                )
                print(f"    Scrambled {perturbation_info['n_scrambled']} / "
                      f"{perturbation_info['total_exc_connections']} connections")

                # Record post-perturbation substrate state (before recovery)
                post_perturbation_state = engine.substrate.get_state_snapshot()

                # Compute PCI-like measure: weight change complexity
                weight_delta = engine.substrate.S - pre_perturbation_weights
                perturbation_complexity = float(np.linalg.norm(weight_delta, 'fro'))
                nonzero_changes = np.sum(np.abs(weight_delta) > 1e-6)

                perturbation_info['weight_change_norm'] = perturbation_complexity
                perturbation_info['n_connections_affected'] = int(nonzero_changes)
                print(f"    Weight change norm: {perturbation_complexity:.4f}")

                # === PHASE 2: Recovery ===
                print(f"\n  --- Phase 2: Recovery ({self.n_recovery} thoughts) ---")
                context, recovery_metrics = self._run_thought_stream(
                    engine, condition, self.n_recovery, context, verbose=verbose,
                )

                # Record final substrate state
                final_state = engine.substrate.get_state_snapshot()

                # Combine all metrics
                all_metrics = baseline_metrics + recovery_metrics
                self.all_trajectories[condition][seed] = all_metrics

                # Compute perturbation response metrics
                baseline_align = np.mean([m['mean_neural_llm_alignment'] for m in baseline_metrics[-5:]])
                recovery_align_early = np.mean([m['mean_neural_llm_alignment'] for m in recovery_metrics[:3]])
                recovery_align_late = np.mean([m['mean_neural_llm_alignment'] for m in recovery_metrics[-5:]])

                baseline_entropy = np.mean([m['mean_blended_entropy'] for m in baseline_metrics[-5:]])
                recovery_entropy_early = np.mean([m['mean_blended_entropy'] for m in recovery_metrics[:3]])
                recovery_entropy_late = np.mean([m['mean_blended_entropy'] for m in recovery_metrics[-5:]])

                baseline_or = np.mean([m['override_rate'] for m in baseline_metrics[-5:]])
                recovery_or_early = np.mean([m['override_rate'] for m in recovery_metrics[:3]])
                recovery_or_late = np.mean([m['override_rate'] for m in recovery_metrics[-5:]])

                baseline_cs = np.mean([m['mean_cscore'] for m in baseline_metrics[-5:]])
                recovery_cs_early = np.mean([m['mean_cscore'] for m in recovery_metrics[:3]])
                recovery_cs_late = np.mean([m['mean_cscore'] for m in recovery_metrics[-5:]])

                perturbation_response = {
                    'alignment_drop': float(baseline_align - recovery_align_early),
                    'alignment_recovery': float(recovery_align_late - recovery_align_early),
                    'entropy_rise': float(recovery_entropy_early - baseline_entropy),
                    'entropy_recovery': float(recovery_entropy_early - recovery_entropy_late),
                    'override_spike': float(recovery_or_early - baseline_or),
                    'override_recovery': float(recovery_or_early - recovery_or_late),
                    'cscore_drop': float(baseline_cs - recovery_cs_early),
                    'cscore_recovery': float(recovery_cs_late - recovery_cs_early),
                }

                elapsed = time.time() - t_run_start
                print(f"\n  Run complete ({elapsed:.1f}s)")
                print(f"  Perturbation response:")
                print(f"    Alignment drop:    {perturbation_response['alignment_drop']:+.4f} "
                      f"(recovery: {perturbation_response['alignment_recovery']:+.4f})")
                print(f"    Entropy rise:      {perturbation_response['entropy_rise']:+.4f} "
                      f"(recovery: {perturbation_response['entropy_recovery']:+.4f})")
                print(f"    Override spike:    {perturbation_response['override_spike']:+.4f} "
                      f"(recovery: {perturbation_response['override_recovery']:+.4f})")
                print(f"    C-Score drop:      {perturbation_response['cscore_drop']:+.4f} "
                      f"(recovery: {perturbation_response['cscore_recovery']:+.4f})")

                # === Save to HDF5 ===
                self._save_run(
                    condition, seed, all_metrics,
                    perturbation_info, perturbation_response,
                    pre_perturbation_state, post_perturbation_state, final_state,
                )

        total_elapsed = time.time() - t_experiment_start

        # === Final Analysis ===
        self._analyze_and_report()

        print(f"\n{'='*78}")
        print(f"  EXPERIMENT 6 COMPLETE")
        print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
        print(f"  Data saved: {self.h5_path}")
        print(f"{'='*78}")

        return self.h5_path

    def _save_run(
        self,
        condition: str,
        seed: int,
        all_metrics: List[Dict],
        perturbation_info: Dict,
        perturbation_response: Dict,
        pre_state: Dict,
        post_state: Dict,
        final_state: Dict,
    ):
        """Save a single run to HDF5."""
        with h5py.File(self.h5_path, 'a') as f:
            grp = f.require_group(f'{condition}/seed_{seed}')

            # Metric trajectories (n_thoughts x n_metrics)
            metric_keys = [
                'mean_cscore', 'std_cscore', 'override_rate',
                'mean_neural_llm_alignment', 'mean_blended_entropy',
                'mean_top_prob_boost', 'token_pattern_consistency',
                'mean_transfer_entropy', 'mean_temporal_depth',
                'mean_granger_density', 'weight_frobenius_divergence',
                'weight_fractional_change', 'total_spikes',
                'cscore_entropy_corr', 'mean_closure', 'mean_rho',
            ]

            n_thoughts = len(all_metrics)
            trajectory = np.zeros((n_thoughts, len(metric_keys)), dtype=np.float64)
            for i, m in enumerate(all_metrics):
                for j, k in enumerate(metric_keys):
                    trajectory[i, j] = float(m.get(k, 0.0))

            ds = grp.create_dataset('trajectory', data=trajectory, compression='gzip')
            ds.attrs['columns'] = json.dumps(metric_keys)
            ds.attrs['perturbation_at'] = self.n_baseline  # thought index

            # Generated text per thought
            dt = h5py.string_dtype()
            texts = np.array([m.get('generated_text', '') for m in all_metrics], dtype=object)
            grp.create_dataset('generated_texts', data=texts, dtype=dt)

            # Perturbation info
            for k, v in perturbation_info.items():
                grp.attrs[f'perturbation_{k}'] = v if isinstance(v, (int, float, str)) else str(v)

            # Perturbation response
            for k, v in perturbation_response.items():
                grp.attrs[f'response_{k}'] = float(v)

            # Substrate states
            for prefix, state in [('pre', pre_state), ('post', post_state), ('final', final_state)]:
                for k, v in state.items():
                    if isinstance(v, (int, float)):
                        grp.attrs[f'{prefix}_{k}'] = v

    def _analyze_and_report(self):
        """Analyze perturbation response across conditions."""
        from scipy import stats

        print(f"\n{'='*78}")
        print(f"  PERTURBATION RESPONSE ANALYSIS")
        print(f"{'='*78}")

        # Collect perturbation responses per condition
        conditions_data = {}
        for condition in ['bio_llm', 'shadow_llm', 'llm_only']:
            if condition not in self.all_trajectories:
                continue

            alignment_drops = []
            entropy_rises = []
            override_spikes = []
            cscore_drops = []
            alignment_recoveries = []

            for seed, metrics in self.all_trajectories[condition].items():
                baseline = metrics[:self.n_baseline]
                recovery = metrics[self.n_baseline:]

                if len(baseline) >= 5 and len(recovery) >= 3:
                    bl_align = np.mean([m['mean_neural_llm_alignment'] for m in baseline[-5:]])
                    re_align = np.mean([m['mean_neural_llm_alignment'] for m in recovery[:3]])
                    re_align_late = np.mean([m['mean_neural_llm_alignment'] for m in recovery[-5:]])
                    alignment_drops.append(bl_align - re_align)
                    alignment_recoveries.append(re_align_late - re_align)

                    bl_ent = np.mean([m['mean_blended_entropy'] for m in baseline[-5:]])
                    re_ent = np.mean([m['mean_blended_entropy'] for m in recovery[:3]])
                    entropy_rises.append(re_ent - bl_ent)

                    bl_or = np.mean([m['override_rate'] for m in baseline[-5:]])
                    re_or = np.mean([m['override_rate'] for m in recovery[:3]])
                    override_spikes.append(re_or - bl_or)

                    bl_cs = np.mean([m['mean_cscore'] for m in baseline[-5:]])
                    re_cs = np.mean([m['mean_cscore'] for m in recovery[:3]])
                    cscore_drops.append(bl_cs - re_cs)

            conditions_data[condition] = {
                'alignment_drops': np.array(alignment_drops),
                'entropy_rises': np.array(entropy_rises),
                'override_spikes': np.array(override_spikes),
                'cscore_drops': np.array(cscore_drops),
                'alignment_recoveries': np.array(alignment_recoveries),
            }

        # Print per-condition perturbation responses
        print(f"\n  {'Condition':<15} {'Align Drop':>12} {'Ent Rise':>12} {'OR Spike':>12} {'CS Drop':>12} {'Align Recov':>12}")
        print(f"  {'-'*75}")
        for cond, data in conditions_data.items():
            print(f"  {cond:<15} "
                  f"{np.mean(data['alignment_drops']):+12.4f} "
                  f"{np.mean(data['entropy_rises']):+12.4f} "
                  f"{np.mean(data['override_spikes']):+12.4f} "
                  f"{np.mean(data['cscore_drops']):+12.4f} "
                  f"{np.mean(data['alignment_recoveries']):+12.4f}")

        # === Hypothesis Tests ===
        print(f"\n  PERTURBATION HYPOTHESIS TESTS (Bonferroni corrected, 4 tests, alpha=0.0125)")
        print(f"  {'-'*75}")

        tests = {}
        n_tests = 4
        alpha_corrected = 0.05 / n_tests

        bio = conditions_data.get('bio_llm', {})
        shadow = conditions_data.get('shadow_llm', {})

        # HP1: Bio alignment drop > Shadow alignment drop
        if len(bio.get('alignment_drops', [])) >= 2 and len(shadow.get('alignment_drops', [])) >= 2:
            U, p = stats.mannwhitneyu(bio['alignment_drops'], shadow['alignment_drops'], alternative='greater')
            d = self._cohens_d(bio['alignment_drops'], shadow['alignment_drops'])
            sig = p < alpha_corrected
            tests['HP1'] = {'U': U, 'p': p, 'd': d, 'significant': sig}
            print(f"\n  HP1: Bio alignment drop > Shadow alignment drop")
            print(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {'*** SIG' if sig else 'n.s.'}")
            print(f"    Bio: {np.mean(bio['alignment_drops']):+.4f} +/- {np.std(bio['alignment_drops']):.4f}")
            print(f"    Shadow: {np.mean(shadow['alignment_drops']):+.4f} +/- {np.std(shadow['alignment_drops']):.4f}")

        # HP2: Bio entropy rise > Shadow entropy rise
        if len(bio.get('entropy_rises', [])) >= 2 and len(shadow.get('entropy_rises', [])) >= 2:
            U, p = stats.mannwhitneyu(bio['entropy_rises'], shadow['entropy_rises'], alternative='greater')
            d = self._cohens_d(bio['entropy_rises'], shadow['entropy_rises'])
            sig = p < alpha_corrected
            tests['HP2'] = {'U': U, 'p': p, 'd': d, 'significant': sig}
            print(f"\n  HP2: Bio entropy rise > Shadow entropy rise")
            print(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {'*** SIG' if sig else 'n.s.'}")
            print(f"    Bio: {np.mean(bio['entropy_rises']):+.4f} +/- {np.std(bio['entropy_rises']):.4f}")
            print(f"    Shadow: {np.mean(shadow['entropy_rises']):+.4f} +/- {np.std(shadow['entropy_rises']):.4f}")

        # HP3: Bio override spike > Shadow override spike
        if len(bio.get('override_spikes', [])) >= 2 and len(shadow.get('override_spikes', [])) >= 2:
            U, p = stats.mannwhitneyu(bio['override_spikes'], shadow['override_spikes'], alternative='greater')
            d = self._cohens_d(bio['override_spikes'], shadow['override_spikes'])
            sig = p < alpha_corrected
            tests['HP3'] = {'U': U, 'p': p, 'd': d, 'significant': sig}
            print(f"\n  HP3: Bio override spike > Shadow override spike")
            print(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {'*** SIG' if sig else 'n.s.'}")
            print(f"    Bio: {np.mean(bio['override_spikes']):+.4f} +/- {np.std(bio['override_spikes']):.4f}")
            print(f"    Shadow: {np.mean(shadow['override_spikes']):+.4f} +/- {np.std(shadow['override_spikes']):.4f}")

        # HP4: Bio alignment recovery > Shadow alignment recovery (STDP re-learning)
        if len(bio.get('alignment_recoveries', [])) >= 2 and len(shadow.get('alignment_recoveries', [])) >= 2:
            U, p = stats.mannwhitneyu(bio['alignment_recoveries'], shadow['alignment_recoveries'], alternative='greater')
            d = self._cohens_d(bio['alignment_recoveries'], shadow['alignment_recoveries'])
            sig = p < alpha_corrected
            tests['HP4'] = {'U': U, 'p': p, 'd': d, 'significant': sig}
            print(f"\n  HP4: Bio alignment recovery > Shadow recovery (STDP re-learning)")
            print(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {'*** SIG' if sig else 'n.s.'}")
            print(f"    Bio: {np.mean(bio['alignment_recoveries']):+.4f} +/- {np.std(bio['alignment_recoveries']):.4f}")
            print(f"    Shadow: {np.mean(shadow['alignment_recoveries']):+.4f} +/- {np.std(shadow['alignment_recoveries']):.4f}")

        # === Cross-metric Coupling ===
        print(f"\n  CROSS-METRIC COUPLING (Bio-LLM only)")
        print(f"  {'-'*75}")

        # Does C-Score drop predict alignment drop within Bio?
        if len(bio.get('cscore_drops', [])) >= 3 and len(bio.get('alignment_drops', [])) >= 3:
            r, p = stats.pearsonr(bio['cscore_drops'], bio['alignment_drops'])
            print(f"  C-Score drop <-> Alignment drop: r={r:.3f}, p={p:.4f}")
            print(f"    (Positive r = when consciousness drops, behavior degrades)")

        if len(bio.get('cscore_drops', [])) >= 3 and len(bio.get('entropy_rises', [])) >= 3:
            r, p = stats.pearsonr(bio['cscore_drops'], bio['entropy_rises'])
            print(f"  C-Score drop <-> Entropy rise:   r={r:.3f}, p={p:.4f}")

        # === PCI-like: Perturbation Complexity ===
        print(f"\n  PERTURBATION COMPLEXITY INDEX (PCI analog)")
        print(f"  {'-'*75}")

        for cond, data in self.all_trajectories.items():
            pci_values = []
            for seed, metrics in data.items():
                recovery = metrics[self.n_baseline:]
                if len(recovery) < 3:
                    continue
                # PCI = how complex is the response to perturbation?
                # Use the entropy of the C-Score trajectory after perturbation
                cs_traj = [m['mean_cscore'] for m in recovery]
                if np.std(cs_traj) > 1e-6:
                    # Normalized LZ complexity of binned C-Score trajectory
                    binned = (np.array(cs_traj) > np.mean(cs_traj)).astype(int)
                    from LLM_Encoder.consciousness import _lz_complexity
                    L = len(binned)
                    c = _lz_complexity(binned)
                    pci = c / (L / np.log2(max(L, 2))) if L >= 2 else 0.0
                    pci_values.append(pci)
                else:
                    pci_values.append(0.0)

            if pci_values:
                print(f"  {cond:<15} PCI = {np.mean(pci_values):.4f} +/- {np.std(pci_values):.4f}")

        # === Evidence Summary ===
        n_sig = sum(1 for t in tests.values() if t.get('significant'))
        print(f"\n  {'='*75}")
        print(f"  CONCLUSION: {n_sig}/{len(tests)} perturbation tests significant")

        if n_sig >= 3:
            print(f"  STRONG: Perturbation causes DIFFERENTIAL behavioral degradation in Bio vs Shadow.")
            print(f"  This demonstrates CAUSAL substrate-behavior coupling — the substrate's learned")
            print(f"  representations directly influence the thought stream's behavioral output.")
        elif n_sig >= 2:
            print(f"  MODERATE: Perturbation shows some differential effect between Bio and Shadow.")
            print(f"  Partial evidence of causal substrate-behavior coupling.")
        elif n_sig >= 1:
            print(f"  WEAK: Limited perturbation differential. The substrate may influence behavior")
            print(f"  but the effect is not robust across all metrics.")
        else:
            print(f"  NULL: No significant differential perturbation response detected.")
            print(f"  The substrate's learned representations may not causally influence behavior,")
            print(f"  or the perturbation was insufficient to produce measurable effects.")

        print(f"  {'='*75}")

        # Save analysis to JSON
        analysis = {
            'timestamp': self.timestamp,
            'experiment': 'perturbation_recovery_v1',
            'n_seeds': len(self.seeds),
            'n_baseline': self.n_baseline,
            'n_recovery': self.n_recovery,
            'perturbation_fraction': self.perturbation_fraction,
            'tests': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                          for kk, vv in v.items()} for k, v in tests.items()},
            'n_significant': n_sig,
            'per_condition': {
                cond: {
                    k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)}
                    for k, v in data.items()
                }
                for cond, data in conditions_data.items()
            },
        }

        analysis_path = os.path.join(self.output_dir, f"perturbation_analysis_{self.timestamp}.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n  Analysis saved: {analysis_path}")

    @staticmethod
    def _cohens_d(g1, g2):
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2:
            return 0.0
        v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        sp = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
        if sp < 1e-10:
            return 0.0
        return float((np.mean(g1) - np.mean(g2)) / sp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 6: Perturbation-Recovery')
    parser.add_argument('--alpha', type=float, default=0.5, help='Neural blend weight')
    parser.add_argument('--tokens', type=int, default=50, help='Tokens per thought')
    parser.add_argument('--baseline', type=int, default=10, help='Baseline thoughts')
    parser.add_argument('--recovery', type=int, default=10, help='Recovery thoughts')
    parser.add_argument('--perturbation', type=float, default=0.5, help='Weight scramble fraction')
    parser.add_argument('--seeds', type=str, default='42,137,271', help='Comma-separated seeds')
    parser.add_argument('--context', type=int, default=300, help='Context window chars')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    experiment = PerturbationExperiment(
        alpha=args.alpha,
        tokens_per_thought=args.tokens,
        n_baseline_thoughts=args.baseline,
        n_recovery_thoughts=args.recovery,
        perturbation_fraction=args.perturbation,
        seeds=seeds,
        context_window=args.context,
    )

    experiment.run(verbose=not args.quiet)


if __name__ == '__main__':
    main()
