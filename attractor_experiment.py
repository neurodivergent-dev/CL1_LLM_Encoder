#!/usr/bin/env python3
"""Experiment v5: Attractor Formation via Amplified STDP

MOTIVATION:
Exp 10 demonstrated STDP creates pattern-specific weight changes (d=6.43, p=0.031)
but these changes don't manifest behaviorally. Weight changes (~0.01-0.02) are too
sparse at 5% connectivity to create emergent cross-channel dynamics.

This experiment amplifies STDP parameters to test whether FUNCTIONAL ATTRACTORS
can form — i.e., whether synaptic plasticity crosses the threshold for behavioral
expression.

KEY AMPLIFICATIONS (vs Exp 10):
  - 15% connectivity (3× increase from 5%) — more pathways for STDP
  - Stronger STDP (A_plus=0.01 vs 0.005) — larger per-event weight changes
  - 500 training repetitions per pattern (2.5× increase from 200) — more exposure
  - Explicit tests for attractor signatures

FALSIFIABLE PREDICTIONS:
  P1: Pattern completion emerges (partial stimulus → full pattern activation)
  P2: Response specificity increases (trained > novel response magnitude)
  P3: Reverberation increases for trained patterns (post-stim activity persists)
  P4: Network integration changes after training (C-Score proxy)
  P5: Blind decoder accuracy improves over training blocks
  P6: Weight specificity replicates + amplifies Exp 10 findings

DESIGN:
  Phase 1: Baseline (spontaneous + trained pattern + partial pattern responses)
  Phase 2: Training (500 reps per pattern, A/B interleaved, with feedback)
  Phase 3: Post-training test (trained + novel + partial responses)
  Phase 4: Reverberation probes (stim 200ms + silence 500ms, trained vs novel)
  Phase 5: Spontaneous replay (detect trained patterns in spontaneous activity)

CONTROLS:
  - Pre/post comparison (within-seed): each seed is its own control
  - Novel pattern (never trained): controls for non-specific STDP drift
  - Weight analysis (ground truth): direct measurement of synaptic changes

Author: Antekythera Project
Date: 2026-02-28
Experiment: 11 (v5)
"""

import json
import time
import logging
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
from scipy import stats
import h5py

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_substrate import IzhikevichSubstrate, IzhikevichConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AttractorConfig:
    """Experiment v5 configuration."""
    # Training
    n_training_reps: int = 500       # reps per pattern (A and B)
    n_baseline_trials: int = 30      # pre-training pattern response trials
    n_test_trials: int = 30          # post-training pattern response trials
    n_partial_trials: int = 20       # pattern completion probes
    n_reverb_trials: int = 20        # reverberation probes
    n_spontaneous_windows: int = 30  # spontaneous replay detection

    # Network (amplified for behavioral STDP expression)
    connection_prob: float = 0.15    # 3× default (0.05 in Exp 10, 0.02 default)
    stdp_A_plus: float = 0.015      # 3× default — LTP-biased for attractor formation
    stdp_A_minus: float = 0.010     # ~1.7× default — net potentiation from co-activation

    # Patterns
    n_pattern_channels: int = 8     # channels per pattern
    n_overlap_channels: int = 3     # shared channels between A and B
    stim_amplitude: float = 1.5     # µA stimulation

    # Windows
    stim_window_s: float = 0.3      # stimulation + recording
    reverb_stim_s: float = 0.2      # stim phase of reverb probe
    reverb_silence_s: float = 0.5   # silence phase of reverb probe
    spontaneous_window_s: float = 0.5
    feedback_window_s: float = 0.2  # reinforcement re-stimulation

    # Sequential training (directional STDP)
    # Note: tick_rate_hz=240, dt=0.5ms → 0.15s = 36 steps = 18ms effective sim time
    # Neurons need ~16 steps to first spike, so 36 steps gives ~2 spikes per phase
    sequential_phase_s: float = 0.15  # 36 steps per phase, enough for 2-3 spikes
    full_pattern_s: float = 0.2       # 48 steps full-pattern reinforcement

    # Seeds
    n_seeds: int = 5

    # Analysis
    alpha: float = 0.05
    n_hypotheses: int = 6           # Bonferroni correction


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def spike_counts_to_vec(counts: Dict, n: int = 59) -> np.ndarray:
    vec = np.zeros(n)
    for k, v in counts.items():
        ch = int(k)
        if 0 <= ch < n:
            vec[ch] = v
    return vec


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Simple Blind Decoder (inline to avoid cross-file dependency)
# ---------------------------------------------------------------------------

class BlindDecoder:
    """Online decoder that learns from labeled examples. No ground truth access."""

    def __init__(self, n_channels: int = 59, ema_alpha: float = 0.1):
        self.n_channels = n_channels
        self.ema_alpha = ema_alpha
        self.templates = {}
        self.counts = {}

    def update(self, response: np.ndarray, label: str):
        if label not in self.templates:
            self.templates[label] = response.copy()
            self.counts[label] = 1
        else:
            self.templates[label] = (
                (1 - self.ema_alpha) * self.templates[label] +
                self.ema_alpha * response
            )
            self.counts[label] += 1

    def predict(self, response: np.ndarray) -> Tuple[str, float]:
        if len(self.templates) < 2:
            return "A", 0.5
        sims = {label: cosine_similarity(response, tmpl)
                for label, tmpl in self.templates.items()}
        best = max(sims, key=sims.get)
        sorted_sims = sorted(sims.values(), reverse=True)
        confidence = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 0.0
        return best, confidence


# ---------------------------------------------------------------------------
# Pattern generation
# ---------------------------------------------------------------------------

def generate_patterns(rng, n_channels=59, n_pattern=8, n_overlap=3):
    """Generate non-overlapping pattern sets for training and testing."""
    available = list(range(n_channels))
    rng.shuffle(available)

    # Pattern A: first n_pattern channels
    pattern_a = sorted(available[:n_pattern])

    # Pattern B: shares n_overlap channels with A, rest unique
    shared = pattern_a[:n_overlap]
    unique_b = sorted(available[n_pattern:n_pattern + (n_pattern - n_overlap)])
    pattern_b = sorted(shared + unique_b)

    # Novel pattern: entirely separate channels
    start_novel = 2 * n_pattern - n_overlap
    novel = sorted(available[start_novel:start_novel + n_pattern])

    # Partial A: first half of A channels
    half = n_pattern // 2
    partial_a = pattern_a[:half]
    completion_a = pattern_a[half:]

    return {
        'A': pattern_a,
        'B': pattern_b,
        'novel': novel,
        'partial_a': partial_a,
        'completion_a': completion_a,
    }


# ---------------------------------------------------------------------------
# Integration metrics (fast C-Score proxy)
# ---------------------------------------------------------------------------

def compute_integration_metrics(ch_spike_matrix):
    """Compute fast integration metrics from (n_channels, n_timesteps) spike matrix.

    Returns dict with:
    - mean_correlation: mean abs pairwise correlation between active channels
    - spectral_entropy: normalized entropy of correlation eigenspectrum
    - active_channels: number of channels with any spikes
    - total_spikes: total spike count
    - integration_index: mean_correlation × spectral_entropy
    """
    active_mask = ch_spike_matrix.sum(axis=1) > 0
    n_active = int(active_mask.sum())
    total_spikes = int(ch_spike_matrix.sum())

    if n_active < 3:
        return {
            'mean_correlation': 0.0,
            'spectral_entropy': 0.0,
            'active_channels': n_active,
            'total_spikes': total_spikes,
            'integration_index': 0.0,
        }

    active_data = ch_spike_matrix[active_mask]

    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(active_data)
    corr = np.nan_to_num(corr, 0)

    n = corr.shape[0]
    offdiag_mask = ~np.eye(n, dtype=bool)
    mean_corr = float(np.mean(np.abs(corr[offdiag_mask])))

    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.maximum(eigvals, 1e-10)
    eigvals_norm = eigvals / eigvals.sum()
    entropy = -np.sum(eigvals_norm * np.log(eigvals_norm))
    max_entropy = np.log(n)
    spec_entropy = float(entropy / (max_entropy + 1e-10))

    return {
        'mean_correlation': mean_corr,
        'spectral_entropy': spec_entropy,
        'active_channels': n_active,
        'total_spikes': total_spikes,
        'integration_index': float(mean_corr * spec_entropy),
    }


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class AttractorExperiment:
    """Experiment v5: Attractor Formation via Amplified STDP.

    Tests whether amplified STDP parameters create functional attractors
    that manifest as behavioral changes (pattern completion, reverberation,
    response specificity).
    """

    def __init__(self, seed: int, config: AttractorConfig, logger=None):
        self.seed = seed
        self.config = config
        self.logger = logger or logging.getLogger()
        self.rng = np.random.default_rng(seed)

        # Create substrate with amplified STDP parameters
        iz_cfg = IzhikevichConfig(
            connection_prob=config.connection_prob,
            stdp_A_plus=config.stdp_A_plus,
            stdp_A_minus=config.stdp_A_minus,
        )
        self.substrate = IzhikevichSubstrate(iz_cfg, seed=seed)

        # Generate patterns
        self.patterns = generate_patterns(
            self.rng,
            n_pattern=config.n_pattern_channels,
            n_overlap=config.n_overlap_channels,
        )

        self.logger.info(
            f"  Patterns: A={self.patterns['A']}, B={self.patterns['B']}, "
            f"novel={self.patterns['novel']}, partial={self.patterns['partial_a']}, "
            f"completion={self.patterns['completion_a']}"
        )

    # --- Helper methods ---

    def _spike_vec(self, counts):
        return spike_counts_to_vec(counts, 59)

    def _stimulate(self, channels, amplitude=None, window_s=None):
        """Stimulate specific channels and record ALL 59 channels."""
        if amplitude is None:
            amplitude = self.config.stim_amplitude
        if window_s is None:
            window_s = self.config.stim_window_s
        # Record ALL channels (0 amplitude for unstimulated)
        # This is critical: we must record unstimulated channels to detect
        # pattern completion (activity propagating from stimulated → unstimulated)
        amps = {ch: 0.0 for ch in range(59)}
        for ch in channels:
            amps[ch] = amplitude
        counts = self.substrate.stimulate_and_record(amps, window_s)
        return self._spike_vec(counts)

    def _stimulate_with_reverb(self, channels, amplitude=None):
        """Stimulate then record silence to measure reverberation."""
        if amplitude is None:
            amplitude = self.config.stim_amplitude

        # Stim phase — record all 59 channels
        amps = {ch: 0.0 for ch in range(59)}
        for ch in channels:
            amps[ch] = amplitude
        stim_counts = self.substrate.stimulate_and_record(
            amps, self.config.reverb_stim_s
        )

        # Reverb phase: zero stim on all channels, network state persists
        zero_amps = {ch: 0.0 for ch in range(59)}
        reverb_counts = self.substrate.stimulate_and_record(
            zero_amps, self.config.reverb_silence_s
        )

        return self._spike_vec(stim_counts), self._spike_vec(reverb_counts)

    def _spontaneous(self):
        zero = {ch: 0.0 for ch in range(59)}
        counts = self.substrate.stimulate_and_record(
            zero, self.config.spontaneous_window_s
        )
        return self._spike_vec(counts)

    def _get_channel_spike_matrix(self):
        """Aggregate neuron-level spike matrix to channel level."""
        raw = self.substrate.get_last_spike_matrix()  # (N_neurons, n_steps)
        n_ch = self.substrate.cfg.n_channels
        n_steps = raw.shape[1] if raw.ndim == 2 else 1
        ch_matrix = np.zeros((n_ch, n_steps))
        for ch, neurons in self.substrate.channel_neurons.items():
            if ch < n_ch:
                for n_idx in neurons:
                    if n_idx < raw.shape[0]:
                        ch_matrix[ch] += raw[n_idx]
        return ch_matrix

    def _completion_spikes(self, response_vec):
        """Count spikes in completion channels (unstimulated half of pattern A)."""
        return float(sum(response_vec[ch] for ch in self.patterns['completion_a']))

    # --- Experiment phases ---

    def run(self):
        t0 = time.time()
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"ATTRACTOR EXPERIMENT — Seed {self.seed}")
        self.logger.info(f"{'=' * 60}")

        results = {
            'seed': self.seed,
            'config': asdict(self.config),
            'patterns': self.patterns,
        }

        # Phase 1: Baseline
        self.logger.info("\n--- Phase 1: Baseline ---")
        baseline = self._run_probe_phase('baseline')
        results['baseline'] = baseline

        # Weight snapshot
        wd_pre = self.substrate.get_weight_divergence()
        S_pre = self.substrate.S.copy()
        results['weight_div_pre'] = wd_pre

        self.logger.info(f"  Completion spikes (partial→completion): {baseline['completion_mean']:.1f}")
        self.logger.info(f"  Response A total: {baseline['response_A_total']:.1f}")
        self.logger.info(f"  Response consistency A: {baseline['consistency_A']:.3f}")
        self.logger.info(f"  Integration index: {baseline['integration_mean']:.4f}")

        # Phase 2: Training
        self.logger.info("\n--- Phase 2: Training (500 reps/pattern) ---")
        training = self._run_training()
        results['training'] = training
        self.logger.info(f"  {training['total_reps']} total reps completed")
        self.logger.info(f"  Decoder accuracy trajectory: {[f'{a:.2f}' for a in training['block_accs']]}")

        # Phase 3: Post-training test
        self.logger.info("\n--- Phase 3: Post-Training Test ---")
        post = self._run_probe_phase('post_test')
        results['post_test'] = post

        wd_post = self.substrate.get_weight_divergence()
        S_post = self.substrate.S.copy()
        results['weight_div_post'] = wd_post

        self.logger.info(f"  Completion spikes: {post['completion_mean']:.1f} (was {baseline['completion_mean']:.1f})")
        self.logger.info(f"  Response A total: {post['response_A_total']:.1f}")
        self.logger.info(f"  Response consistency A: {post['consistency_A']:.3f} (was {baseline['consistency_A']:.3f})")
        self.logger.info(f"  Novel response total: {post.get('response_novel_total', 0):.1f}")
        self.logger.info(f"  Integration index: {post['integration_mean']:.4f}")

        # Phase 4: Reverberation
        self.logger.info("\n--- Phase 4: Reverberation ---")
        reverb = self._run_reverberation()
        results['reverberation'] = reverb
        self.logger.info(f"  Trained reverb: {reverb['trained_mean']:.1f}")
        self.logger.info(f"  Novel reverb: {reverb['novel_mean']:.1f}")

        # Phase 5: Spontaneous replay
        self.logger.info("\n--- Phase 5: Spontaneous Replay ---")
        replay = self._run_replay()
        results['replay'] = replay
        self.logger.info(f"  Replay events: A={replay['count_A']}, B={replay['count_B']}")

        # Weight analysis (ground truth)
        self.logger.info("\n--- Weight Analysis ---")
        weight_analysis = self._analyze_weights(S_pre, S_post)
        results['weight_analysis'] = weight_analysis
        self.logger.info(f"  |ΔW| Within-A: {weight_analysis['within_a_mean']:.4f} ({weight_analysis['within_a_pct']:.1%} changed)")
        self.logger.info(f"  |ΔW| Within-B: {weight_analysis['within_b_mean']:.4f}")
        self.logger.info(f"  |ΔW| Between:  {weight_analysis['between_mean']:.4f}")
        self.logger.info(f"  |ΔW| Novel:    {weight_analysis['novel_mean']:.4f}")
        self.logger.info(f"  Specificity:    {weight_analysis['specificity']:.4f}")
        self.logger.info(f"  --- DIRECTIONAL STDP ---")
        self.logger.info(f"  Forward (partial→completion): {weight_analysis['forward_signed']:+.4f} ({'LTP' if weight_analysis['forward_signed'] > 0 else 'LTD'})")
        self.logger.info(f"  Reverse (completion→partial): {weight_analysis['reverse_signed']:+.4f} ({'LTP' if weight_analysis['reverse_signed'] > 0 else 'LTD'})")
        self.logger.info(f"  Directionality: {weight_analysis['directionality']:+.4f}")
        self.logger.info(f"  Forward weight: {weight_analysis['fwd_weight_pre']:.3f} → {weight_analysis['fwd_weight_post']:.3f}")

        # Per-seed hypothesis tests
        analysis = self._per_seed_analysis(results)
        results['analysis'] = analysis

        # Store S matrices for HDF5
        results['_S_pre'] = S_pre
        results['_S_post'] = S_post

        elapsed = time.time() - t0
        results['elapsed_s'] = elapsed
        self.logger.info(f"\nSeed {self.seed} complete in {elapsed:.1f}s")
        return results

    def _run_probe_phase(self, phase_name):
        """Run baseline or post-training probe phase."""
        cfg = self.config
        n = cfg.n_baseline_trials if phase_name == 'baseline' else cfg.n_test_trials

        # Pattern A responses
        responses_a = []
        for _ in range(n):
            r = self._stimulate(self.patterns['A'])
            responses_a.append(r)

        # Pattern B responses
        responses_b = []
        for _ in range(n):
            r = self._stimulate(self.patterns['B'])
            responses_b.append(r)

        # Partial pattern completion
        completion_values = []
        for _ in range(cfg.n_partial_trials):
            r = self._stimulate(self.patterns['partial_a'])
            completion_values.append(self._completion_spikes(r))

        # Integration metrics (5 measurements)
        integration_values = []
        for _ in range(5):
            self._stimulate(self.patterns['A'])
            ch_matrix = self._get_channel_spike_matrix()
            m = compute_integration_metrics(ch_matrix)
            integration_values.append(m)

        # Response consistency: mean pairwise cosine similarity between response vectors
        # High consistency = stereotyped responses (attractor-like)
        response_a_vecs = np.array(responses_a)  # (n_trials, 59)
        if len(responses_a) >= 2:
            pairs_sim = []
            for j in range(len(responses_a)):
                for k in range(j + 1, len(responses_a)):
                    pairs_sim.append(cosine_similarity(responses_a[j], responses_a[k]))
            consistency_A = float(np.mean(pairs_sim)) if pairs_sim else 0.0
        else:
            consistency_A = 0.0

        result = {
            'response_A_totals': [float(r.sum()) for r in responses_a],
            'response_B_totals': [float(r.sum()) for r in responses_b],
            'response_A_total': float(np.mean([r.sum() for r in responses_a])),
            'response_B_total': float(np.mean([r.sum() for r in responses_b])),
            'completion_values': completion_values,
            'completion_mean': float(np.mean(completion_values)),
            'completion_std': float(np.std(completion_values)),
            'consistency_A': consistency_A,
            'integration_values': integration_values,
            'integration_mean': float(np.mean([m['integration_index'] for m in integration_values])),
            'active_channels_mean': float(np.mean([m['active_channels'] for m in integration_values])),
            'mean_correlation': float(np.mean([m['mean_correlation'] for m in integration_values])),
        }

        # Post-test also measures novel pattern
        if phase_name == 'post_test':
            responses_novel = []
            for _ in range(n):
                r = self._stimulate(self.patterns['novel'])
                responses_novel.append(r)
            result['response_novel_totals'] = [float(r.sum()) for r in responses_novel]
            result['response_novel_total'] = float(np.mean([r.sum() for r in responses_novel]))

        return result

    def _run_training(self):
        """Phase 2: Sequential stimulation with active inhibition for clean STDP.

        For each training trial:
          1. Stimulate first_half channels (builds pre-synaptic traces)
          2. Stimulate second_half + INHIBIT first_half (LTP on first→second,
             inhibition prevents noise-driven LTD on forward connections)

        NO full-pattern phase — this was causing competing LTD that destroyed
        the directional learning signal. Full-pattern probes done every 50 reps
        for decoder measurement only.

        Why inhibition matters: During phase 2, if a first_half neuron fires
        (from noise), STDP's LTD rule depresses S[second, first] — exactly the
        connections we want to strengthen. Injecting negative current (-6pA)
        into first_half during phase 2 prevents this.
        """
        cfg = self.config
        decoder = BlindDecoder(n_channels=59, ema_alpha=0.1)

        # Build interleaved schedule
        schedule = []
        for _ in range(cfg.n_training_reps):
            schedule.extend(['A', 'B'])
        self.rng.shuffle(schedule)

        block_accs = []
        block_correct = 0
        block_total = 0
        block_size = 50
        probe_interval = 50  # full-pattern probe every 50 trials

        for i, label in enumerate(schedule):
            pattern = self.patterns[label]
            half = len(pattern) // 2
            first_half = pattern[:half]
            second_half = pattern[half:]

            # Phase 1: Stimulate first half only (builds pre-synaptic traces)
            amps1 = {ch: 0.0 for ch in range(59)}
            for ch in first_half:
                amps1[ch] = cfg.stim_amplitude
            self.substrate.stimulate_and_record(amps1, cfg.sequential_phase_s)

            # Phase 2: Stimulate second half + INHIBIT first half
            # Inhibition: -1.0 × stim_gain = -6pA cancels tonic noise (3pA)
            # This prevents first_half from firing, avoiding LTD on forward connections
            amps2 = {ch: 0.0 for ch in range(59)}
            for ch in second_half:
                amps2[ch] = cfg.stim_amplitude
            for ch in first_half:
                amps2[ch] = -1.0  # active inhibition
            self.substrate.stimulate_and_record(amps2, cfg.sequential_phase_s)

            # Periodic full-pattern probe for decoder measurement
            if (i + 1) % probe_interval == 0:
                response = self._stimulate(pattern, window_s=cfg.full_pattern_s)
                pred, conf = decoder.predict(response)
                decoder.update(response, label)
                block_correct += int(pred == label)
                block_total += 1

                if block_total >= 10:
                    block_accs.append(block_correct / block_total)
                    block_correct = 0
                    block_total = 0

            if (i + 1) % 200 == 0:
                acc_str = f"{block_accs[-1]:.2f}" if block_accs else "N/A"
                self.logger.info(f"  Training {i + 1}/{len(schedule)}, decoder acc: {acc_str}")

        if block_total > 0:
            block_accs.append(block_correct / block_total)

        return {
            'total_reps': len(schedule),
            'block_accs': block_accs,
            'decoder_final_acc': block_accs[-1] if block_accs else 0.0,
            'n_blocks': len(block_accs),
        }

    def _run_reverberation(self):
        """Phase 4: Measure post-stimulus activity for trained vs novel patterns."""
        cfg = self.config

        trained_reverbs = []
        novel_reverbs = []

        for _ in range(cfg.n_reverb_trials):
            # Trained pattern A
            _, rev_trained = self._stimulate_with_reverb(self.patterns['A'])
            trained_reverbs.append(float(rev_trained.sum()))

            # Novel pattern
            _, rev_novel = self._stimulate_with_reverb(self.patterns['novel'])
            novel_reverbs.append(float(rev_novel.sum()))

        return {
            'trained_reverbs': trained_reverbs,
            'novel_reverbs': novel_reverbs,
            'trained_mean': float(np.mean(trained_reverbs)),
            'novel_mean': float(np.mean(novel_reverbs)),
        }

    def _run_replay(self):
        """Phase 5: Detect spontaneous replay of trained patterns."""
        # Build templates from pattern channel masks
        template_a = np.zeros(59)
        template_b = np.zeros(59)
        for ch in self.patterns['A']:
            template_a[ch] = 1.0
        for ch in self.patterns['B']:
            template_b[ch] = 1.0

        sims_a, sims_b = [], []
        count_a = count_b = 0
        threshold = 0.3

        for _ in range(self.config.n_spontaneous_windows):
            r = self._spontaneous()
            if r.sum() > 0:
                r_n = r / (np.linalg.norm(r) + 1e-10)
                t_a_n = template_a / (np.linalg.norm(template_a) + 1e-10)
                t_b_n = template_b / (np.linalg.norm(template_b) + 1e-10)
                sa = float(np.dot(r_n, t_a_n))
                sb = float(np.dot(r_n, t_b_n))
            else:
                sa = sb = 0.0

            sims_a.append(sa)
            sims_b.append(sb)
            if sa > threshold:
                count_a += 1
            if sb > threshold:
                count_b += 1

        return {
            'sims_a': sims_a,
            'sims_b': sims_b,
            'count_A': count_a,
            'count_B': count_b,
            'mean_sim_A': float(np.mean(sims_a)),
            'mean_sim_B': float(np.mean(sims_b)),
        }

    def _analyze_weights(self, S_pre, S_post):
        """Direct weight matrix analysis — ground truth for STDP.

        Key: measures DIRECTIONAL connections separately.
        Sequential training (partial→completion) should produce:
          S[completion, partial] → LTP (forward, what we want)
          S[partial, completion] → LTD (reverse, expected)
        """
        delta_S = S_post - S_pre
        ch_neurons = self.substrate.channel_neurons

        def neurons_for(channels):
            out = set()
            for ch in channels:
                out.update(ch_neurons.get(ch, []))
            return sorted(out)

        nA = neurons_for(self.patterns['A'])
        nB = neurons_for(self.patterns['B'])
        nN = neurons_for(self.patterns['novel'])
        nPartial = neurons_for(self.patterns['partial_a'])
        nCompletion = neurons_for(self.patterns['completion_a'])

        all_pattern = set(nA) | set(nB) | set(nN)
        n_other = sorted(set(range(self.substrate.cfg.n_neurons)) - all_pattern)

        abs_delta = np.abs(delta_S)
        mean_w = float(np.mean(np.abs(S_pre[S_pre != 0]))) if np.any(S_pre != 0) else 0.1

        within_a = abs_delta[np.ix_(nA, nA)]
        within_b = abs_delta[np.ix_(nB, nB)]
        between = abs_delta[np.ix_(nA, nB)]
        novel = abs_delta[np.ix_(nN, nN)]
        n_other_sub = n_other[:100] if len(n_other) >= 100 else n_other
        other = abs_delta[np.ix_(n_other_sub, n_other_sub)]

        within_avg = (np.mean(within_a) + np.mean(within_b)) / 2

        # DIRECTIONAL STDP — the critical measurement
        # S[post, pre] convention in Izhikevich:
        # S[completion, partial] = connections from partial → completion
        forward = delta_S[np.ix_(nCompletion, nPartial)]   # expect LTP (+)
        reverse = delta_S[np.ix_(nPartial, nCompletion)]   # expect LTD (-)
        # Absolute weights for forward pathway
        fwd_pre = S_pre[np.ix_(nCompletion, nPartial)]
        fwd_post = S_post[np.ix_(nCompletion, nPartial)]

        return {
            'within_a_mean': float(np.mean(within_a)),
            'within_b_mean': float(np.mean(within_b)),
            'between_mean': float(np.mean(between)),
            'novel_mean': float(np.mean(novel)),
            'other_mean': float(np.mean(other)),
            'within_a_pct': float(np.mean(within_a > 0.1 * mean_w)),
            'specificity': float(within_avg - np.mean(between)),
            'fro_norm': float(np.linalg.norm(delta_S, 'fro')),
            # DIRECTIONAL STDP (the key measurements)
            'forward_signed': float(np.mean(forward)),       # partial→completion (expect +)
            'forward_abs': float(np.mean(np.abs(forward))),
            'reverse_signed': float(np.mean(reverse)),       # completion→partial (expect -)
            'directionality': float(np.mean(forward) - np.mean(reverse)),
            'fwd_weight_pre': float(np.mean(fwd_pre[fwd_pre != 0])) if np.any(fwd_pre != 0) else 0.0,
            'fwd_weight_post': float(np.mean(fwd_post[fwd_post != 0])) if np.any(fwd_post != 0) else 0.0,
            # Signed within-pattern changes
            'signed_within_a': float(np.mean(delta_S[np.ix_(nA, nA)])),
            'signed_within_b': float(np.mean(delta_S[np.ix_(nB, nB)])),
            'signed_novel': float(np.mean(delta_S[np.ix_(nN, nN)])),
        }

    def _per_seed_analysis(self, results):
        """Quick per-seed statistics."""
        bl = results['baseline']
        pt = results['post_test']

        # Completion delta
        comp_pre = bl['completion_mean']
        comp_post = pt['completion_mean']

        # Specificity
        trained_mean = pt['response_A_total']
        novel_mean = pt.get('response_novel_total', 0)

        # Reverberation
        rev = results.get('reverberation', {})

        return {
            'completion_delta': comp_post - comp_pre,
            'completion_pre': comp_pre,
            'completion_post': comp_post,
            'specificity_trained': trained_mean,
            'specificity_novel': novel_mean,
            'reverb_trained': rev.get('trained_mean', 0),
            'reverb_novel': rev.get('novel_mean', 0),
            'integration_pre': bl['integration_mean'],
            'integration_post': pt['integration_mean'],
        }


# ---------------------------------------------------------------------------
# Cross-seed analysis
# ---------------------------------------------------------------------------

def cross_seed_analysis(all_results, logger, config):
    """Statistical tests across seeds using Wilcoxon signed-rank."""
    n = len(all_results)
    alpha_corr = config.alpha / config.n_hypotheses

    # Collect per-seed metrics
    completion_pre = []
    completion_post = []
    trained_totals = []
    novel_totals = []
    reverb_trained = []
    reverb_novel = []
    integration_pre = []
    integration_post = []
    decoder_early = []
    decoder_late = []
    weight_within = []
    weight_between = []
    weight_novel = []
    weight_other = []

    for r in all_results:
        bl = r['baseline']
        pt = r['post_test']
        a = r['analysis']
        w = r['weight_analysis']
        rev = r.get('reverberation', {})
        tr = r['training']

        completion_pre.append(bl['completion_mean'])
        completion_post.append(pt['completion_mean'])
        trained_totals.append(pt['response_A_total'])
        novel_totals.append(pt.get('response_novel_total', 0))
        reverb_trained.append(rev.get('trained_mean', 0))
        reverb_novel.append(rev.get('novel_mean', 0))
        integration_pre.append(bl['integration_mean'])
        integration_post.append(pt['integration_mean'])

        accs = tr['block_accs']
        if len(accs) >= 2:
            mid = len(accs) // 2
            decoder_early.append(float(np.mean(accs[:mid])))
            decoder_late.append(float(np.mean(accs[mid:])))
        else:
            decoder_early.append(0.5)
            decoder_late.append(0.5)

        wa = (w['within_a_mean'] + w['within_b_mean']) / 2
        weight_within.append(wa)
        weight_between.append(w['between_mean'])
        weight_novel.append(w['novel_mean'])
        weight_other.append(w['other_mean'])

    results = {}

    # --- P1: Pattern Completion ---
    completion_delta = [post - pre for pre, post in zip(completion_pre, completion_post)]
    if n >= 5:
        try:
            W, p = stats.wilcoxon(completion_delta, alternative='greater')
        except ValueError:
            W, p = 0, 1.0
    else:
        W, p = 0, 1.0
    d = float(np.mean(completion_delta) / (np.std(completion_delta) + 1e-10))
    logger.info(f"\nP1 Pattern Completion:")
    logger.info(f"  Pre:  {[f'{x:.1f}' for x in completion_pre]}")
    logger.info(f"  Post: {[f'{x:.1f}' for x in completion_post]}")
    logger.info(f"  Delta: {[f'{x:.1f}' for x in completion_delta]}")
    logger.info(f"  W={W}, p={p:.6f}, d={d:.2f}")
    sig = bool(p < alpha_corr)
    logger.info(f"  {'*** SIGNIFICANT' if sig else 'n.s.'} (α_corr={alpha_corr:.4f})")
    results['P1_completion'] = {
        'pre': completion_pre, 'post': completion_post,
        'delta': completion_delta, 'W': float(W), 'p': float(p), 'd': d,
        'significant': sig,
    }

    # --- P2: Response Specificity (trained > novel) ---
    spec_diff = [t - n for t, n in zip(trained_totals, novel_totals)]
    if n >= 5:
        try:
            W, p = stats.wilcoxon(spec_diff, alternative='greater')
        except ValueError:
            W, p = 0, 1.0
    else:
        W, p = 0, 1.0
    d_spec = cohens_d(trained_totals, novel_totals)
    logger.info(f"\nP2 Response Specificity (trained > novel):")
    logger.info(f"  Trained: {[f'{x:.1f}' for x in trained_totals]}")
    logger.info(f"  Novel:   {[f'{x:.1f}' for x in novel_totals]}")
    logger.info(f"  Diff: {[f'{x:.1f}' for x in spec_diff]}")
    logger.info(f"  W={W}, p={p:.6f}, d={d_spec:.2f}")
    sig = bool(p < alpha_corr)
    logger.info(f"  {'*** SIGNIFICANT' if sig else 'n.s.'}")
    results['P2_specificity'] = {
        'trained': trained_totals, 'novel': novel_totals,
        'diff': spec_diff, 'W': float(W), 'p': float(p), 'd': d_spec,
        'significant': sig,
    }

    # --- P3: Reverberation (trained > novel) ---
    reverb_diff = [t - n for t, n in zip(reverb_trained, reverb_novel)]
    if n >= 5:
        try:
            W, p = stats.wilcoxon(reverb_diff, alternative='greater')
        except ValueError:
            W, p = 0, 1.0
    else:
        W, p = 0, 1.0
    d_rev = cohens_d(reverb_trained, reverb_novel)
    logger.info(f"\nP3 Reverberation (trained > novel):")
    logger.info(f"  Trained: {[f'{x:.1f}' for x in reverb_trained]}")
    logger.info(f"  Novel:   {[f'{x:.1f}' for x in reverb_novel]}")
    logger.info(f"  W={W}, p={p:.6f}, d={d_rev:.2f}")
    sig = bool(p < alpha_corr)
    logger.info(f"  {'*** SIGNIFICANT' if sig else 'n.s.'}")
    results['P3_reverberation'] = {
        'trained': reverb_trained, 'novel': reverb_novel,
        'diff': reverb_diff, 'W': float(W), 'p': float(p), 'd': d_rev,
        'significant': sig,
    }

    # --- P4: Integration Change ---
    integ_delta = [post - pre for pre, post in zip(integration_pre, integration_post)]
    if n >= 5:
        try:
            W, p = stats.wilcoxon(integ_delta, alternative='greater')
        except ValueError:
            W, p = 0, 1.0
    else:
        W, p = 0, 1.0
    logger.info(f"\nP4 Integration Change:")
    logger.info(f"  Pre:   {[f'{x:.4f}' for x in integration_pre]}")
    logger.info(f"  Post:  {[f'{x:.4f}' for x in integration_post]}")
    logger.info(f"  Delta: {[f'{x:.4f}' for x in integ_delta]}")
    logger.info(f"  W={W}, p={p:.6f}")
    sig = bool(p < alpha_corr)
    logger.info(f"  {'*** SIGNIFICANT' if sig else 'n.s.'}")
    results['P4_integration'] = {
        'pre': integration_pre, 'post': integration_post,
        'delta': integ_delta, 'W': float(W), 'p': float(p),
        'significant': sig,
    }

    # --- P5: Decoder Improvement ---
    dec_diff = [l - e for l, e in zip(decoder_late, decoder_early)]
    if n >= 5:
        try:
            W, p = stats.wilcoxon(dec_diff, alternative='greater')
        except ValueError:
            W, p = 0, 1.0
    else:
        W, p = 0, 1.0
    logger.info(f"\nP5 Decoder Improvement:")
    logger.info(f"  Early: {[f'{x:.3f}' for x in decoder_early]}")
    logger.info(f"  Late:  {[f'{x:.3f}' for x in decoder_late]}")
    logger.info(f"  Diff:  {[f'{x:.3f}' for x in dec_diff]}")
    logger.info(f"  W={W}, p={p:.6f}")
    sig = bool(p < alpha_corr)
    logger.info(f"  {'*** SIGNIFICANT' if sig else 'n.s.'}")
    results['P5_decoder'] = {
        'early': decoder_early, 'late': decoder_late,
        'diff': dec_diff, 'W': float(W), 'p': float(p),
        'significant': sig,
    }

    # --- P6: Weight Specificity ---
    weight_diff = [w - b for w, b in zip(weight_within, weight_between)]
    if n >= 5:
        try:
            W, p = stats.wilcoxon(weight_diff, alternative='greater')
        except ValueError:
            W, p = 0, 1.0
    else:
        W, p = 0, 1.0
    d_w = cohens_d(weight_within, weight_between)
    logger.info(f"\nP6 Weight Specificity (within > between):")
    logger.info(f"  Within:  {[f'{x:.4f}' for x in weight_within]}")
    logger.info(f"  Between: {[f'{x:.4f}' for x in weight_between]}")
    logger.info(f"  Novel:   {[f'{x:.4f}' for x in weight_novel]}")
    logger.info(f"  Other:   {[f'{x:.4f}' for x in weight_other]}")
    logger.info(f"  W={W}, p={p:.6f}, d={d_w:.2f}")
    sig = bool(p < alpha_corr)
    logger.info(f"  {'*** SIGNIFICANT' if sig else 'n.s.'}")
    results['P6_weight'] = {
        'within': weight_within, 'between': weight_between,
        'novel': weight_novel, 'other': weight_other,
        'diff': weight_diff, 'W': float(W), 'p': float(p), 'd': d_w,
        'significant': sig,
    }

    # --- Summary ---
    sig_count = sum(1 for v in results.values()
                    if isinstance(v, dict) and v.get('significant', False))
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY: {sig_count}/{config.n_hypotheses} hypotheses significant")
    logger.info(f"(Bonferroni-corrected α = {alpha_corr:.4f})")
    logger.info(f"{'=' * 60}")

    if sig_count >= 4:
        verdict = "STRONG: Attractor formation with behavioral expression demonstrated"
    elif sig_count >= 2:
        verdict = "MODERATE: Partial evidence of attractor formation"
    elif sig_count >= 1:
        verdict = "WEAK: Limited evidence — plasticity without behavioral expression"
    else:
        verdict = "NULL: No evidence of attractor formation or behavioral change"

    logger.info(f"VERDICT: {verdict}")
    results['verdict'] = verdict
    results['n_significant'] = sig_count

    return results


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_multi_seed(config=None):
    """Run experiment across seeds and save combined results."""
    if config is None:
        config = AttractorConfig()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path(__file__).parent / "experiment_data"
    data_dir.mkdir(exist_ok=True)

    # Logging
    log_path = data_dir / f"v5_{timestamp}.log"
    logger = logging.getLogger('attractor_v5')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(str(log_path)))
    logger.addHandler(logging.StreamHandler())

    logger.info("=" * 60)
    logger.info("EXPERIMENT v5: ATTRACTOR FORMATION VIA AMPLIFIED STDP")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Config: {json.dumps(asdict(config), indent=2)}")
    logger.info(f"Key amplifications vs Exp 10:")
    logger.info(f"  connectivity: 0.05 → {config.connection_prob}")
    logger.info(f"  STDP A+: 0.005 → {config.stdp_A_plus} (LTP-biased for attractor formation)")
    logger.info(f"  STDP A-: 0.006 → {config.stdp_A_minus} (A+/A- = {config.stdp_A_plus/config.stdp_A_minus:.2f}, net potentiation)")
    logger.info(f"  training reps: 200 → {config.n_training_reps}")

    all_results = []

    for seed in range(42, 42 + config.n_seeds):
        exp = AttractorExperiment(seed, config, logger)
        result = exp.run()

        # Save per-seed HDF5
        h5_path = data_dir / f"v5_{timestamp}_seed{seed}.h5"
        with h5py.File(str(h5_path), 'w') as f:
            f.attrs['seed'] = seed
            f.attrs['timestamp'] = timestamp
            f.attrs['config'] = json.dumps(asdict(config))
            if '_S_pre' in result:
                f.create_dataset('S_pre', data=result['_S_pre'], compression='gzip')
            if '_S_post' in result:
                f.create_dataset('S_post', data=result['_S_post'], compression='gzip')
            # Save completion values
            bl = result.get('baseline', {})
            pt = result.get('post_test', {})
            if 'completion_values' in bl:
                f.create_dataset('completion_pre', data=bl['completion_values'])
            if 'completion_values' in pt:
                f.create_dataset('completion_post', data=pt['completion_values'])

        logger.info(f"Saved {h5_path}")

        # Remove large arrays before appending to combined results
        result_clean = {k: v for k, v in result.items() if not k.startswith('_')}
        all_results.append(result_clean)

    # Cross-seed analysis
    logger.info(f"\n{'=' * 60}")
    logger.info("CROSS-SEED ANALYSIS")
    logger.info(f"{'=' * 60}")

    cross = cross_seed_analysis(all_results, logger, config)

    # Save combined
    combined = {
        'experiment': 'v5_attractor_formation',
        'timestamp': timestamp,
        'config': asdict(config),
        'per_seed': all_results,
        'cross_seed': cross,
    }

    json_path = data_dir / f"v5_{timestamp}_combined.json"
    with open(str(json_path), 'w') as f:
        json.dump(combined, f, cls=NumpyEncoder, indent=2)

    logger.info(f"\nResults: {json_path}")
    logger.info(f"Log: {log_path}")

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment v5: Attractor Formation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation (1 seed, 50 reps)')
    parser.add_argument('--reps', type=int, default=500,
                        help='Training reps per pattern (default 500)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds (default 5)')
    parser.add_argument('--connectivity', type=float, default=0.15,
                        help='Connection probability (default 0.15)')
    args = parser.parse_args()

    if args.quick:
        cfg = AttractorConfig(
            n_training_reps=50,
            n_seeds=1,
            n_baseline_trials=10,
            n_test_trials=10,
            n_partial_trials=10,
            n_reverb_trials=10,
            n_spontaneous_windows=10,
            connection_prob=args.connectivity,
        )
    else:
        cfg = AttractorConfig(
            n_training_reps=args.reps,
            n_seeds=args.seeds,
            connection_prob=args.connectivity,
        )

    run_multi_seed(cfg)
