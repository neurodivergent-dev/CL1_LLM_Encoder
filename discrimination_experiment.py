#!/usr/bin/env python3
"""Experiment v4: STDP Learning & Cross-Channel Influence

Quick-run validation (v4a) showed the binary discrimination task was TRIVIALLY
EASY — 100% accuracy from trial 1 because stimulation templates are perfectly
separable. No room for learning detection.

This redesigned experiment tests the RIGHT question:
  "Does repeated co-stimulation of channels change cross-channel connectivity?"

This directly tests for STDP — the biological mechanism for learning. If STDP
works, channels that are repeatedly co-stimulated should develop stronger
mutual influence (higher cross-channel spike propagation).

Design:
  Phase 1: Measure baseline cross-channel influence matrix
  Phase 2: Repeated co-stimulation training (token A pattern × N repetitions)
  Phase 3: Measure post-training cross-channel influence matrix
  Phase 4: Compute Δ-influence and test for pattern-specificity
  Phase 5: Repeat with different conditions for controls
  Phase 6: Spontaneous replay detection

Conditions:
  TRAINED:    Repeated A+B pattern training with reinforcement feedback
  PASSIVE:    Same patterns, no reinforcement
  NOVEL:      Novel patterns (never trained) — influence should NOT change

Key Metrics (direct STDP evidence):
  - Cross-channel influence matrix (pre/post training)
  - Pattern-specific Δ-influence (trained pairs vs untrained pairs)
  - Response consistency trajectory (does variability decrease?)
  - Blind decoder accuracy trajectory (learns from data only)
  - Weight divergence (Izhikevich only — ground truth for STDP)
  - Replay index (spontaneous pattern emergence)

Pre-registered Hypotheses:
  H1: Trained Δ-influence > Novel Δ-influence (STDP specificity)
  H2: Within-pattern Δ-influence > between-pattern Δ-influence (pattern specificity)
  H3: Response consistency increases over training (stabilization)
  H4: Blind decoder accuracy improves over training blocks (discrimination)
  H5: Feedback Δ-influence > Passive Δ-influence (feedback helps plasticity)
  H6: Spontaneous replay increases post-training (internalization)
  H7: Weight change is pattern-specific (Izhikevich ground truth)

Author: Antekythera Project
Date: 2026-02-28
"""

import json
import time
import logging
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
from scipy import stats
import h5py

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial_encoder import SpatialEncoder, SENSORY_CHANNELS, FEEDBACK_CHANNELS
from neural_substrate import IzhikevichSubstrate, IzhikevichConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExpConfig:
    """Experiment v4 configuration."""
    # Training
    n_training_reps: int = 200       # repetitions of each pattern during training
    n_influence_probes: int = 5      # probes per channel for influence measurement
    block_size: int = 20             # trials per analysis block
    n_patterns: int = 2              # A and B

    # Channel configuration
    n_pattern_channels: int = 8      # channels per pattern
    n_overlap_channels: int = 3      # shared channels between A and B

    # Stimulation parameters
    stim_window_s: float = 0.5       # stimulation + recording window
    feedback_window_s: float = 0.3   # reinforcement re-stimulation
    probe_window_s: float = 0.3      # single-channel probe window

    # Spontaneous recording
    spontaneous_windows: int = 40    # number of 0.5s windows

    # Replay
    replay_threshold: float = 0.3    # cosine similarity for replay event

    # Decoder
    decoder_history: int = 10        # trials to look back for blind decoder

    # Seeds
    n_seeds: int = 5

    # Substrate
    substrate_type: str = "izhikevich"
    connection_prob: float = 0.05    # higher than default 0.02 for detectable cross-channel influence

    # Analysis
    alpha: float = 0.05
    bonferroni_n: int = 7


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cohens_d(a, b):
    """Cohen's d effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def spike_counts_to_vector(spike_counts: Dict[str, int], n_channels: int = 59) -> np.ndarray:
    """Convert spike counts dict to fixed-size vector."""
    vec = np.zeros(n_channels)
    for ch_str, count in spike_counts.items():
        ch = int(ch_str)
        if 0 <= ch < n_channels:
            vec[ch] = count
    return vec


# ---------------------------------------------------------------------------
# Blind Online Decoder
# ---------------------------------------------------------------------------

class BlindDecoder:
    """Online decoder that learns discrimination from labeled examples only.

    No access to stimulation patterns. Builds class templates from
    observed responses and classifies by nearest-template.
    """

    def __init__(self, n_channels: int = 59, ema_alpha: float = 0.1):
        self.n_channels = n_channels
        self.ema_alpha = ema_alpha  # how fast templates update
        self.templates = {}  # class_label -> running mean vector
        self.counts = {}     # class_label -> count

    def update(self, response: np.ndarray, label: str):
        """Update class template with new observation."""
        if label not in self.templates:
            self.templates[label] = response.copy()
            self.counts[label] = 1
        else:
            # EMA update
            self.templates[label] = (
                (1 - self.ema_alpha) * self.templates[label] +
                self.ema_alpha * response
            )
            self.counts[label] += 1

    def predict(self, response: np.ndarray) -> Tuple[str, float]:
        """Predict class from response. Returns (class, confidence)."""
        if len(self.templates) < 2:
            # Not enough data — random guess
            return "A", 0.5

        sims = {label: cosine_similarity(response, tmpl)
                for label, tmpl in self.templates.items()}

        best = max(sims, key=sims.get)
        # Confidence = margin between top two
        sorted_sims = sorted(sims.values(), reverse=True)
        confidence = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 0.0

        return best, confidence

    def accuracy_trajectory(self, responses: List[np.ndarray], labels: List[str],
                           block_size: int = 20) -> List[float]:
        """Replay the decoder learning trajectory on stored data."""
        dec = BlindDecoder(self.n_channels, self.ema_alpha)
        accs = []
        block_correct = 0
        block_total = 0

        for i, (resp, label) in enumerate(zip(responses, labels)):
            pred, _ = dec.predict(resp)
            correct = (pred == label)
            block_correct += int(correct)
            block_total += 1

            # Update AFTER prediction (online learning)
            dec.update(resp, label)

            if block_total >= block_size:
                accs.append(block_correct / block_total)
                block_correct = 0
                block_total = 0

        if block_total > 0:
            accs.append(block_correct / block_total)

        return accs


# ---------------------------------------------------------------------------
# Cross-Channel Influence Matrix
# ---------------------------------------------------------------------------

class InfluenceMapper:
    """Measures cross-channel influence by probing individual channels.

    For each channel, stimulates ONLY that channel and measures spike
    propagation to all other channels. This creates a directed influence
    matrix showing how strongly channel i drives channel j.
    """

    def __init__(self, substrate, n_probes: int = 5, probe_window_s: float = 0.3,
                 probe_amplitude: float = 2.0):
        self.substrate = substrate
        self.n_probes = n_probes
        self.probe_window_s = probe_window_s
        self.probe_amplitude = probe_amplitude

    def measure(self, channels: List[int], logger: logging.Logger = None) -> np.ndarray:
        """Measure cross-channel influence matrix.

        Returns: (n_channels, n_channels) matrix where M[i,j] = mean spikes
        in channel j when channel i is stimulated alone.
        """
        n = len(channels)
        influence = np.zeros((n, n))

        for i, src_ch in enumerate(channels):
            responses = []
            for probe in range(self.n_probes):
                pattern = {src_ch: self.probe_amplitude}
                spikes = self.substrate.stimulate_and_record(
                    pattern, window_s=self.probe_window_s
                )
                vec = np.array([spikes.get(str(ch), 0) for ch in channels])
                responses.append(vec)

            mean_resp = np.mean(responses, axis=0)
            influence[i, :] = mean_resp

            if logger and (i + 1) % 10 == 0:
                logger.info(f"  Influence probe: {i+1}/{n} channels mapped")

        return influence


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

class STDPLearningExperiment:
    """Experiment v4: Direct test of STDP-induced cross-channel learning."""

    def __init__(self, substrate, config: ExpConfig = None, seed: int = 42,
                 logger: logging.Logger = None):
        self.substrate = substrate
        self.cfg = config or ExpConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.logger = logger or logging.getLogger("STDPExp")
        self.encoder = SpatialEncoder(seed=seed)

        # Select channels for patterns A and B
        self._select_patterns()

        # Tracking
        self.training_responses = {'A': [], 'B': []}
        self.training_labels = []  # sequence of "A" or "B"
        self.training_vectors = []  # corresponding response vectors
        self.block_metrics = []  # per-block summaries

    def _select_patterns(self):
        """Create stimulation patterns with controlled overlap."""
        # Use first N sensory channels (deterministic, reproducible)
        available = SENSORY_CHANNELS[:20]
        self.rng.shuffle(available)

        n_pat = self.cfg.n_pattern_channels
        n_overlap = self.cfg.n_overlap_channels

        # Shared channels
        shared = available[:n_overlap]
        # A-only channels
        a_only = available[n_overlap:n_overlap + (n_pat - n_overlap)]
        # B-only channels
        b_only = available[n_overlap + (n_pat - n_overlap):n_overlap + 2*(n_pat - n_overlap)]
        # Novel channels (for control)
        remaining = available[n_overlap + 2*(n_pat - n_overlap):]

        self.channels_a = sorted(list(shared) + list(a_only))
        self.channels_b = sorted(list(shared) + list(b_only))
        self.channels_novel = sorted(remaining[:n_pat])

        # Amplitude profiles (different for A and B even on shared channels)
        self.pattern_a = {}
        for i, ch in enumerate(self.channels_a):
            # Deterministic but different amplitude per channel
            self.pattern_a[ch] = 1.0 + 1.0 * np.sin(i * 2.3 + 0.5)

        self.pattern_b = {}
        for i, ch in enumerate(self.channels_b):
            self.pattern_b[ch] = 1.0 + 1.0 * np.cos(i * 1.7 + 0.3)

        self.pattern_novel = {}
        for i, ch in enumerate(self.channels_novel):
            self.pattern_novel[ch] = 1.5  # uniform

        # All channels involved in any pattern (for influence measurement)
        self.all_pattern_channels = sorted(set(
            self.channels_a + self.channels_b + self.channels_novel
        ))

        # Vectors for SRC computation
        self.stim_vec_a = self._pattern_to_full_vector(self.pattern_a)
        self.stim_vec_b = self._pattern_to_full_vector(self.pattern_b)

        self.logger.info(f"Pattern A channels: {self.channels_a}")
        self.logger.info(f"Pattern B channels: {self.channels_b}")
        self.logger.info(f"Novel channels: {self.channels_novel}")
        self.logger.info(f"Shared channels: {sorted(set(self.channels_a) & set(self.channels_b))}")
        self.logger.info(f"All pattern channels: {self.all_pattern_channels}")

    def _pattern_to_full_vector(self, pattern: Dict[int, float]) -> np.ndarray:
        """Convert pattern dict to 59-dim vector."""
        vec = np.zeros(59)
        for ch, amp in pattern.items():
            if 0 <= ch < 59:
                vec[ch] = amp
        return vec

    def _stimulate(self, pattern: Dict[int, float], window: float = None) -> Tuple[Dict, np.ndarray]:
        """Stimulate and return (spike_counts, response_vector)."""
        if window is None:
            window = self.cfg.stim_window_s
        spike_counts = self.substrate.stimulate_and_record(pattern, window_s=window)
        vec = spike_counts_to_vector(spike_counts)
        return spike_counts, vec

    def run(self) -> Dict:
        """Execute the full experiment."""
        start_time = time.time()

        results = {
            'experiment': 'v4_stdp_learning',
            'start_time': datetime.now().isoformat(),
            'config': asdict(self.cfg),
            'seed': self.seed,
            'channels_a': self.channels_a,
            'channels_b': self.channels_b,
            'channels_novel': self.channels_novel,
        }

        # === Phase 1: Baseline Influence ===
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: Baseline Cross-Channel Influence")
        self.logger.info("=" * 60)
        mapper = InfluenceMapper(
            self.substrate,
            n_probes=self.cfg.n_influence_probes,
            probe_window_s=self.cfg.probe_window_s,
        )
        baseline_influence = mapper.measure(self.all_pattern_channels, self.logger)
        results['baseline_influence'] = baseline_influence.tolist()
        self.logger.info(f"  Baseline influence matrix: {baseline_influence.shape}")
        self.logger.info(f"  Mean diagonal (self-response): {np.mean(np.diag(baseline_influence)):.1f}")
        self.logger.info(f"  Mean off-diagonal (cross-influence): {np.mean(baseline_influence[~np.eye(len(self.all_pattern_channels), dtype=bool)]):.2f}")

        # === Phase 2: Baseline Spontaneous ===
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: Baseline Spontaneous Activity")
        self.logger.info("=" * 60)
        baseline_spont = self._record_spontaneous("baseline")

        # === Phase 3: Training ===
        self.logger.info("=" * 60)
        self.logger.info("PHASE 3: Training with Reinforcement")
        self.logger.info("=" * 60)
        training_data = self._run_training(condition="feedback")
        results['training_feedback'] = training_data

        # === Phase 4: Post-Training Influence ===
        self.logger.info("=" * 60)
        self.logger.info("PHASE 4: Post-Training Cross-Channel Influence")
        self.logger.info("=" * 60)
        post_train_influence = mapper.measure(self.all_pattern_channels, self.logger)
        results['post_train_influence'] = post_train_influence.tolist()

        delta_influence = post_train_influence - baseline_influence
        results['delta_influence'] = delta_influence.tolist()
        self.logger.info(f"  Mean |Δ-influence|: {np.mean(np.abs(delta_influence)):.3f}")
        self.logger.info(f"  Max |Δ-influence|: {np.max(np.abs(delta_influence)):.3f}")

        # === Phase 5: Post-Training Spontaneous ===
        self.logger.info("=" * 60)
        self.logger.info("PHASE 5: Post-Training Spontaneous Activity")
        self.logger.info("=" * 60)
        post_train_spont = self._record_spontaneous("post_training")

        # === Phase 6: Passive Training (no feedback) ===
        self.logger.info("=" * 60)
        self.logger.info("PHASE 6: Passive Training (No Feedback)")
        self.logger.info("=" * 60)
        # Need fresh substrate to compare fairly
        # Instead, just record influence after passive exposure
        passive_data = self._run_training(condition="passive")
        results['training_passive'] = passive_data

        post_passive_influence = mapper.measure(self.all_pattern_channels, self.logger)
        results['post_passive_influence'] = post_passive_influence.tolist()
        delta_passive = post_passive_influence - post_train_influence
        results['delta_passive_influence'] = delta_passive.tolist()

        # === Phase 7: Novel Pattern Exposure ===
        self.logger.info("=" * 60)
        self.logger.info("PHASE 7: Novel Pattern Exposure (Control)")
        self.logger.info("=" * 60)
        novel_data = self._run_novel_exposure()
        results['novel_exposure'] = novel_data

        post_novel_influence = mapper.measure(self.all_pattern_channels, self.logger)
        results['post_novel_influence'] = post_novel_influence.tolist()

        # === Phase 8: Final Spontaneous ===
        self.logger.info("=" * 60)
        self.logger.info("PHASE 8: Final Spontaneous Activity")
        self.logger.info("=" * 60)
        final_spont = self._record_spontaneous("final")

        # === Phase 9: Analysis ===
        self.logger.info("=" * 60)
        self.logger.info("ANALYSIS")
        self.logger.info("=" * 60)

        analysis = self._full_analysis(
            baseline_influence, post_train_influence,
            post_passive_influence, post_novel_influence,
        )
        results['analysis'] = analysis

        # Weight analysis (Izhikevich only)
        if hasattr(self.substrate, 'get_weight_divergence'):
            results['weight_divergence'] = self.substrate.get_weight_divergence()
            weight_analysis = self._analyze_weights()
            results['weight_analysis'] = weight_analysis

        results['duration_s'] = time.time() - start_time
        return results

    def _run_training(self, condition: str = "feedback") -> Dict:
        """Run training phase with A and B patterns."""
        n_reps = self.cfg.n_training_reps
        block_size = self.cfg.block_size

        all_responses_a = []
        all_responses_b = []
        block_data = []

        # Interleave A and B presentations
        sequence = []
        for _ in range(n_reps):
            sequence.append("A")
            sequence.append("B")
        self.rng.shuffle(sequence)

        decoder = BlindDecoder(n_channels=59)
        decoder_correct = 0
        decoder_total = 0

        for trial_idx, label in enumerate(sequence):
            pattern = self.pattern_a if label == "A" else self.pattern_b

            # Present stimulus
            spike_counts, resp_vec = self._stimulate(pattern)

            # Blind decoder prediction BEFORE update
            pred, conf = decoder.predict(resp_vec)
            correct = (pred == label)
            decoder_correct += int(correct)
            decoder_total += 1

            # Update decoder with ground truth
            decoder.update(resp_vec, label)

            # Store response
            if label == "A":
                all_responses_a.append(resp_vec)
            else:
                all_responses_b.append(resp_vec)

            self.training_labels.append(label)
            self.training_vectors.append(resp_vec)

            # Apply reinforcement feedback
            if condition == "feedback":
                # Re-stimulate the correct pattern (STDP reinforcement)
                self._stimulate(pattern, window=self.cfg.feedback_window_s)

            # Block summary
            if (trial_idx + 1) % block_size == 0:
                block_idx = (trial_idx + 1) // block_size
                block_start = trial_idx - block_size + 1

                # Response consistency within this block
                block_a = [all_responses_a[i] for i in range(len(all_responses_a))
                          if len(all_responses_a) - i <= block_size][-block_size//2:]
                block_b = [all_responses_b[i] for i in range(len(all_responses_b))
                          if len(all_responses_b) - i <= block_size][-block_size//2:]

                cons_a = self._within_class_consistency(block_a)
                cons_b = self._within_class_consistency(block_b)

                # Population vector distance
                mean_a = np.mean(block_a, axis=0) if block_a else np.zeros(59)
                mean_b = np.mean(block_b, axis=0) if block_b else np.zeros(59)
                pop_dist = float(np.linalg.norm(mean_a - mean_b))

                # Blind decoder accuracy for this block
                block_dec_acc = decoder_correct / max(1, decoder_total)

                block_summary = {
                    'block_idx': block_idx,
                    'condition': condition,
                    'consistency_a': cons_a,
                    'consistency_b': cons_b,
                    'mean_consistency': (cons_a + cons_b) / 2,
                    'pop_vec_distance': pop_dist,
                    'decoder_accuracy': block_dec_acc,
                    'mean_spikes': float(np.mean([v.sum() for v in (block_a + block_b)])),
                }
                block_data.append(block_summary)
                decoder_correct = 0
                decoder_total = 0

                self.logger.info(
                    f"  [{condition}] Block {block_idx}: "
                    f"cons={block_summary['mean_consistency']:.3f}, "
                    f"dist={pop_dist:.1f}, "
                    f"dec_acc={block_summary['decoder_accuracy']:.3f}, "
                    f"spikes={block_summary['mean_spikes']:.0f}"
                )

        # Compute blind decoder full trajectory
        decoder_trajectory = decoder.accuracy_trajectory(
            self.training_vectors[-2*n_reps:],
            self.training_labels[-2*n_reps:],
            block_size=block_size,
        )

        return {
            'condition': condition,
            'n_trials': len(sequence),
            'block_data': block_data,
            'decoder_trajectory': decoder_trajectory,
            'n_responses_a': len(all_responses_a),
            'n_responses_b': len(all_responses_b),
        }

    def _run_novel_exposure(self) -> Dict:
        """Expose substrate to novel patterns (never trained) for control."""
        n_reps = self.cfg.n_training_reps // 2  # shorter
        block_size = self.cfg.block_size

        block_data = []

        for trial_idx in range(n_reps):
            self._stimulate(self.pattern_novel)

            if (trial_idx + 1) % block_size == 0:
                block_data.append({
                    'block_idx': (trial_idx + 1) // block_size,
                    'condition': 'novel',
                })
                self.logger.info(f"  [novel] Block {(trial_idx+1)//block_size}")

        return {
            'condition': 'novel',
            'n_trials': n_reps,
            'block_data': block_data,
        }

    def _record_spontaneous(self, label: str) -> np.ndarray:
        """Record spontaneous activity (no stimulation)."""
        self.logger.info(f"Recording {label} spontaneous activity...")
        vectors = []
        null_pattern = {ch: 0.0 for ch in SENSORY_CHANNELS[:4]}

        for w in range(self.cfg.spontaneous_windows):
            _, vec = self._stimulate(null_pattern)
            vectors.append(vec)

        matrix = np.array(vectors)
        total = matrix.sum()
        active = (matrix.sum(axis=0) > 0).sum()
        self.logger.info(f"  {label}: {len(vectors)} windows, {total:.0f} spikes, {active} active channels")

        # Store for replay analysis
        if not hasattr(self, '_spontaneous'):
            self._spontaneous = {}
        self._spontaneous[label] = matrix

        return matrix

    def _within_class_consistency(self, responses: List[np.ndarray]) -> float:
        """Mean pairwise cosine similarity within a class."""
        if len(responses) < 2:
            return 0.0
        sims = []
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                sims.append(cosine_similarity(responses[i], responses[j]))
        return float(np.mean(sims))

    def _full_analysis(
        self,
        baseline_inf: np.ndarray,
        post_train_inf: np.ndarray,
        post_passive_inf: np.ndarray,
        post_novel_inf: np.ndarray,
    ) -> Dict:
        """Comprehensive statistical analysis."""
        analysis = {}

        # --- Influence Analysis ---
        ch_map = {ch: i for i, ch in enumerate(self.all_pattern_channels)}

        # Get indices for pattern channels
        idx_a = [ch_map[ch] for ch in self.channels_a if ch in ch_map]
        idx_b = [ch_map[ch] for ch in self.channels_b if ch in ch_map]
        idx_novel = [ch_map[ch] for ch in self.channels_novel if ch in ch_map]

        # Delta influence for trained patterns
        delta_train = post_train_inf - baseline_inf
        delta_passive = post_passive_inf - post_train_inf
        delta_novel = post_novel_inf - post_passive_inf

        # Within-pattern A influence change
        if len(idx_a) >= 2:
            within_a_delta = np.mean([abs(delta_train[i, j])
                                     for i in idx_a for j in idx_a if i != j])
        else:
            within_a_delta = 0.0

        # Within-pattern B influence change
        if len(idx_b) >= 2:
            within_b_delta = np.mean([abs(delta_train[i, j])
                                     for i in idx_b for j in idx_b if i != j])
        else:
            within_b_delta = 0.0

        # Between A-B influence change
        if idx_a and idx_b:
            between_ab_delta = np.mean([abs(delta_train[i, j])
                                       for i in idx_a for j in idx_b])
        else:
            between_ab_delta = 0.0

        # Novel pattern influence change (control)
        if len(idx_novel) >= 2:
            within_novel_delta = np.mean([abs(delta_novel[i, j])
                                         for i in idx_novel for j in idx_novel if i != j])
        else:
            within_novel_delta = 0.0

        # Overall non-pattern influence change
        all_idx = set(range(len(self.all_pattern_channels)))
        pattern_idx = set(idx_a + idx_b)
        other_idx = list(all_idx - pattern_idx)
        if len(other_idx) >= 2:
            other_delta = np.mean([abs(delta_train[i, j])
                                   for i in other_idx for j in other_idx if i != j])
        else:
            other_delta = 0.0

        analysis['influence'] = {
            'within_a_delta': float(within_a_delta),
            'within_b_delta': float(within_b_delta),
            'between_ab_delta': float(between_ab_delta),
            'within_novel_delta': float(within_novel_delta),
            'other_delta': float(other_delta),
            'mean_trained_delta': float((within_a_delta + within_b_delta) / 2),
            'overall_mean_delta': float(np.mean(np.abs(delta_train))),
        }

        self.logger.info(f"\nInfluence Analysis:")
        self.logger.info(f"  Within-A Δ: {within_a_delta:.4f}")
        self.logger.info(f"  Within-B Δ: {within_b_delta:.4f}")
        self.logger.info(f"  Between A-B Δ: {between_ab_delta:.4f}")
        self.logger.info(f"  Novel Δ: {within_novel_delta:.4f}")
        self.logger.info(f"  Other Δ: {other_delta:.4f}")

        # --- Training Block Analysis ---
        for cond_key in ['training_feedback', 'training_passive']:
            cond_data = getattr(self, f'_last_{cond_key}', None)
            # Will be set by the training method storing its results

        # Block trajectory analysis from stored block data
        if hasattr(self, '_feedback_blocks') and len(self._feedback_blocks) >= 3:
            cons = [b['mean_consistency'] for b in self._feedback_blocks]
            x = np.arange(len(cons))
            slope, _, r, p, _ = stats.linregress(x, cons)
            analysis['consistency_trajectory_feedback'] = {
                'values': cons,
                'slope': float(slope),
                'r': float(r),
                'p': float(p),
            }

        # --- Hypothesis Tests ---
        analysis['hypotheses'] = self._test_hypotheses(analysis)

        # --- Replay Analysis ---
        analysis['replay'] = self._analyze_replay()

        return analysis

    def _test_hypotheses(self, analysis: Dict) -> Dict:
        """Test all pre-registered hypotheses."""
        alpha_c = self.cfg.alpha / self.cfg.bonferroni_n
        hyp = {}
        inf = analysis.get('influence', {})

        # H1: Trained Δ-influence > Novel Δ-influence
        trained_delta = inf.get('mean_trained_delta', 0)
        novel_delta = inf.get('within_novel_delta', 0)
        hyp['H1_trained_gt_novel'] = {
            'description': 'Trained pattern Δ-influence > Novel pattern Δ-influence',
            'trained_delta': float(trained_delta),
            'novel_delta': float(novel_delta),
            'ratio': float(trained_delta / max(1e-10, novel_delta)),
            'positive': trained_delta > novel_delta,
        }

        # H2: Within-pattern Δ > between-pattern Δ (pattern specificity)
        within = inf.get('mean_trained_delta', 0)
        between = inf.get('between_ab_delta', 0)
        hyp['H2_within_gt_between'] = {
            'description': 'Within-pattern Δ > Between-pattern Δ (specificity)',
            'within_delta': float(within),
            'between_delta': float(between),
            'specificity_index': float(within - between),
            'positive': within > between,
        }

        # H7: Weight change specificity (Izhikevich only)
        if hasattr(self, '_weight_specificity'):
            hyp['H7_weight_specificity'] = self._weight_specificity

        self.logger.info(f"\nHypothesis Tests (Bonferroni α={alpha_c:.4f}):")
        for key, h in hyp.items():
            self.logger.info(f"  {key}: {'POSITIVE' if h.get('positive', False) else 'NEGATIVE'}")
            for k, v in h.items():
                if k != 'description':
                    self.logger.info(f"    {k}: {v}")

        return hyp

    def _analyze_replay(self) -> Dict:
        """Detect spontaneous replay of trained patterns."""
        if not hasattr(self, '_spontaneous') or len(self._spontaneous) < 2:
            return {'error': 'insufficient_spontaneous_data'}

        # Build templates from training responses
        if not self.training_vectors:
            return {'error': 'no_training_data'}

        a_vecs = [v for v, l in zip(self.training_vectors, self.training_labels) if l == "A"]
        b_vecs = [v for v, l in zip(self.training_vectors, self.training_labels) if l == "B"]

        if len(a_vecs) < 5 or len(b_vecs) < 5:
            return {'error': 'insufficient_class_data'}

        template_a = np.mean(a_vecs, axis=0)
        template_b = np.mean(b_vecs, axis=0)

        results = {}
        for label, matrix in self._spontaneous.items():
            sims_a = [cosine_similarity(matrix[i], template_a) for i in range(len(matrix))]
            sims_b = [cosine_similarity(matrix[i], template_b) for i in range(len(matrix))]

            n_replay_a = sum(1 for s in sims_a if s > self.cfg.replay_threshold)
            n_replay_b = sum(1 for s in sims_b if s > self.cfg.replay_threshold)

            results[label] = {
                'n_windows': len(matrix),
                'replay_a': n_replay_a,
                'replay_b': n_replay_b,
                'replay_rate': (n_replay_a + n_replay_b) / max(1, len(matrix)),
                'mean_sim_a': float(np.mean(sims_a)),
                'mean_sim_b': float(np.mean(sims_b)),
            }

            self.logger.info(f"  Replay [{label}]: A={n_replay_a}, B={n_replay_b}, "
                           f"rate={results[label]['replay_rate']:.3f}")

        # H6: Post-training > baseline
        if 'baseline' in results and 'post_training' in results:
            base_rate = results['baseline']['replay_rate']
            post_rate = results['post_training']['replay_rate']
            results['H6_replay_increase'] = {
                'description': 'Post-training replay rate > baseline',
                'baseline_rate': float(base_rate),
                'post_training_rate': float(post_rate),
                'increase': float(post_rate - base_rate),
                'positive': post_rate > base_rate,
            }

        return results

    def _analyze_weights(self) -> Dict:
        """Analyze weight changes for pattern specificity (Izhikevich only).

        This is the GROUND TRUTH for STDP learning — directly measures whether
        synaptic weights changed in a pattern-specific way.
        """
        if not hasattr(self.substrate, 'S') or not hasattr(self.substrate, '_initial_S'):
            return {}

        delta_S = self.substrate.S - self.substrate._initial_S
        Ne = self.substrate.cfg.n_excitatory

        # Map pattern channels to neuron indices
        ch_neurons = self.substrate.channel_neurons

        neurons_a = set()
        for ch in self.channels_a:
            neurons_a.update(ch_neurons.get(ch, []))
        neurons_b = set()
        for ch in self.channels_b:
            neurons_b.update(ch_neurons.get(ch, []))
        neurons_novel = set()
        for ch in self.channels_novel:
            neurons_novel.update(ch_neurons.get(ch, []))

        # A-only, B-only, shared neuron groups
        neurons_shared = neurons_a & neurons_b
        neurons_a_only = neurons_a - neurons_shared
        neurons_b_only = neurons_b - neurons_shared

        idx_a = sorted(neurons_a)
        idx_b = sorted(neurons_b)
        idx_a_only = sorted(neurons_a_only)
        idx_b_only = sorted(neurons_b_only)
        idx_shared = sorted(neurons_shared)
        idx_n = sorted(neurons_novel)

        # All non-pattern neurons
        all_pattern = neurons_a | neurons_b | neurons_novel
        idx_other = sorted(set(range(self.substrate.cfg.n_neurons)) - all_pattern)

        def mean_abs_change(rows, cols):
            if not rows or not cols:
                return 0.0
            sub = delta_S[np.ix_(list(rows), list(cols))]
            return float(np.mean(np.abs(sub)))

        def sum_change(rows, cols):
            """Sum of absolute weight changes (more sensitive for sparse connectivity)."""
            if not rows or not cols:
                return 0.0
            sub = delta_S[np.ix_(list(rows), list(cols))]
            return float(np.sum(np.abs(sub)))

        def n_changed(rows, cols, threshold=0.01):
            """Count weights that changed by more than threshold."""
            if not rows or not cols:
                return 0
            sub = delta_S[np.ix_(list(rows), list(cols))]
            return int(np.sum(np.abs(sub) > threshold))

        def n_total(rows, cols):
            if not rows or not cols:
                return 0
            return len(rows) * len(cols)

        # === Comprehensive weight analysis ===

        # Mean absolute change per neuron-pair
        within_a = mean_abs_change(idx_a, idx_a)
        within_b = mean_abs_change(idx_b, idx_b)
        between_ab = mean_abs_change(idx_a, idx_b)
        within_novel = mean_abs_change(idx_n, idx_n)
        within_other = mean_abs_change(idx_other[:100], idx_other[:100])  # sample

        # Sum of changes (more sensitive for sparse matrices)
        sum_within_a = sum_change(idx_a, idx_a)
        sum_within_b = sum_change(idx_b, idx_b)
        sum_between = sum_change(idx_a, idx_b)
        sum_novel = sum_change(idx_n, idx_n)

        # Number of weights that changed
        n_changed_a = n_changed(idx_a, idx_a)
        n_changed_b = n_changed(idx_b, idx_b)
        n_changed_between = n_changed(idx_a, idx_b)
        n_changed_novel = n_changed(idx_n, idx_n)

        # Total possible connections
        total_a = n_total(idx_a, idx_a)
        total_b = n_total(idx_b, idx_b)
        total_between = n_total(idx_a, idx_b)
        total_novel = n_total(idx_n, idx_n)

        # Overall weight statistics
        total_change = float(np.sum(np.abs(delta_S)))
        total_exc_change = float(np.sum(np.abs(delta_S[:, :Ne])))
        total_n_changed = int(np.sum(np.abs(delta_S) > 0.01))
        frac_change = float(np.linalg.norm(delta_S, 'fro') /
                          (np.linalg.norm(self.substrate._initial_S, 'fro') + 1e-10))

        # Specificity metrics
        mean_trained = (within_a + within_b) / 2
        specificity_mean = mean_trained - between_ab
        specificity_sum = (sum_within_a + sum_within_b) / 2 - sum_between
        specificity_count = ((n_changed_a / max(1, total_a) + n_changed_b / max(1, total_b)) / 2 -
                           n_changed_between / max(1, total_between))

        self._weight_specificity = {
            'description': 'Weight change specificity (Izhikevich ground truth)',
            'within_a_mean': within_a,
            'within_b_mean': within_b,
            'between_ab_mean': between_ab,
            'within_novel_mean': within_novel,
            'within_other_mean': within_other,
            'sum_within_a': sum_within_a,
            'sum_within_b': sum_within_b,
            'sum_between_ab': sum_between,
            'sum_novel': sum_novel,
            'n_changed_a': n_changed_a,
            'n_changed_b': n_changed_b,
            'n_changed_between': n_changed_between,
            'n_changed_novel': n_changed_novel,
            'pct_changed_a': n_changed_a / max(1, total_a),
            'pct_changed_b': n_changed_b / max(1, total_b),
            'pct_changed_between': n_changed_between / max(1, total_between),
            'pct_changed_novel': n_changed_novel / max(1, total_novel),
            'total_abs_change': total_change,
            'total_exc_change': total_exc_change,
            'total_n_changed': total_n_changed,
            'fractional_change': frac_change,
            'specificity_mean': float(specificity_mean),
            'specificity_sum': float(specificity_sum),
            'specificity_count': float(specificity_count),
            'positive': specificity_mean > 0,
        }

        self.logger.info(f"\nWeight Analysis (H7 — STDP Ground Truth):")
        self.logger.info(f"  Overall: {total_n_changed} weights changed, "
                        f"frac={frac_change:.4f}")
        self.logger.info(f"  Within-A: mean={within_a:.6f}, sum={sum_within_a:.2f}, "
                        f"n={n_changed_a}/{total_a} ({n_changed_a/max(1,total_a)*100:.1f}%)")
        self.logger.info(f"  Within-B: mean={within_b:.6f}, sum={sum_within_b:.2f}, "
                        f"n={n_changed_b}/{total_b} ({n_changed_b/max(1,total_b)*100:.1f}%)")
        self.logger.info(f"  Between A-B: mean={between_ab:.6f}, sum={sum_between:.2f}, "
                        f"n={n_changed_between}/{total_between} ({n_changed_between/max(1,total_between)*100:.1f}%)")
        self.logger.info(f"  Novel: mean={within_novel:.6f}, sum={sum_novel:.2f}, "
                        f"n={n_changed_novel}/{total_novel} ({n_changed_novel/max(1,total_novel)*100:.1f}%)")
        self.logger.info(f"  Other: mean={within_other:.6f}")
        self.logger.info(f"  Specificity (mean): {specificity_mean:.6f}")
        self.logger.info(f"  Specificity (sum): {specificity_sum:.2f}")
        self.logger.info(f"  Specificity (count): {specificity_count:.6f}")
        self.logger.info(f"  Pattern-specific: {specificity_mean > 0}")

        return self._weight_specificity

    def save_hdf5(self, filepath: str):
        """Save experimental data to HDF5."""
        with h5py.File(filepath, 'w') as h5:
            h5.attrs['experiment'] = 'v4_stdp_learning'
            h5.attrs['timestamp'] = datetime.now().isoformat()
            h5.attrs['seed'] = self.seed
            h5.attrs['channels_a'] = self.channels_a
            h5.attrs['channels_b'] = self.channels_b

            # Stimulation vectors
            h5.create_dataset('stim_vec_a', data=self.stim_vec_a)
            h5.create_dataset('stim_vec_b', data=self.stim_vec_b)

            # Training responses
            if self.training_vectors:
                h5.create_dataset('training_vectors', data=np.array(self.training_vectors))
                h5.create_dataset('training_labels',
                                data=np.array([1 if l == "A" else 0 for l in self.training_labels]))

            # Spontaneous recordings
            if hasattr(self, '_spontaneous'):
                grp = h5.create_group('spontaneous')
                for label, matrix in self._spontaneous.items():
                    grp.create_dataset(label, data=matrix)


# ---------------------------------------------------------------------------
# Multi-seed Runner
# ---------------------------------------------------------------------------

def run_multi_seed(config: ExpConfig = None, seeds: List[int] = None,
                   output_dir: str = None) -> Dict:
    """Run experiment across multiple seeds."""
    cfg = config or ExpConfig()
    seeds = seeds or list(range(42, 42 + cfg.n_seeds))

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_data")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(output_dir, f"v4_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger("STDPExp")

    logger.info("=" * 70)
    logger.info("EXPERIMENT v4: STDP Learning & Cross-Channel Influence")
    logger.info("=" * 70)
    logger.info(f"Config: {asdict(cfg)}")
    logger.info(f"Seeds: {seeds}")

    all_results = []

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"SEED {seed_idx+1}/{len(seeds)} (seed={seed})")
        logger.info(f"{'#' * 70}")

        iz_cfg = IzhikevichConfig(connection_prob=cfg.connection_prob)
        substrate = IzhikevichSubstrate(cfg=iz_cfg, seed=seed)

        exp = STDPLearningExperiment(
            substrate=substrate,
            config=cfg,
            seed=seed,
            logger=logger,
        )

        result = exp.run()
        result['seed_idx'] = seed_idx
        all_results.append(result)

        h5_path = os.path.join(output_dir, f"v4_{timestamp}_seed{seed}.h5")
        exp.save_hdf5(h5_path)
        logger.info(f"Saved: {h5_path}")

    # Cross-seed analysis
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-SEED ANALYSIS")
    logger.info("=" * 70)

    cross_seed = _cross_seed_analysis(all_results, logger)

    combined = {
        'experiment': 'v4_stdp_learning',
        'timestamp': timestamp,
        'config': asdict(cfg),
        'seeds': seeds,
        'per_seed_results': all_results,
        'cross_seed_analysis': cross_seed,
    }

    json_path = os.path.join(output_dir, f"v4_{timestamp}_combined.json")

    # Custom JSON serializer for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(json_path, 'w') as f:
        json.dump(combined, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Combined results: {json_path}")

    return combined


def _cross_seed_analysis(results: List[Dict], logger: logging.Logger) -> Dict:
    """Aggregate results across seeds."""
    analysis = {}

    # === Weight-based metrics (GROUND TRUTH for STDP) ===
    weight_within_a = []
    weight_within_b = []
    weight_between = []
    weight_novel = []
    weight_other = []
    weight_specificity = []

    for r in results:
        wa = r.get('weight_analysis', {})
        if wa:
            weight_within_a.append(wa.get('within_a_mean', 0))
            weight_within_b.append(wa.get('within_b_mean', 0))
            weight_between.append(wa.get('between_ab_mean', 0))
            weight_novel.append(wa.get('within_novel_mean', 0))
            weight_other.append(wa.get('within_other_mean', 0))
            weight_specificity.append(wa.get('specificity_mean', 0))

    n = len(weight_within_a)

    if n >= 3:
        logger.info(f"\n=== Weight-Based STDP Analysis (n={n} seeds) ===")
        trained_mean = [(a+b)/2 for a,b in zip(weight_within_a, weight_within_b)]

        # H1: Trained > Novel weight change
        try:
            stat, p = stats.wilcoxon(trained_mean, weight_novel, alternative='greater')
        except ValueError:
            stat, p = 0.0, 1.0
        d = cohens_d(trained_mean, weight_novel)
        analysis['H1_cross_seed'] = {
            'description': 'Trained within-pattern weight Δ > Novel pattern weight Δ',
            'W': float(stat), 'p': float(p), 'd': float(d),
            'trained_mean': float(np.mean(trained_mean)),
            'novel_mean': float(np.mean(weight_novel)),
            'significant': p < 0.05,
        }
        logger.info(f"H1 (trained>novel): W={stat:.1f}, p={p:.4f}, d={d:.3f}")
        logger.info(f"  Trained: {np.mean(trained_mean):.6f} ± {np.std(trained_mean):.6f}")
        logger.info(f"  Novel:   {np.mean(weight_novel):.6f} ± {np.std(weight_novel):.6f}")

        # H2: Within-pattern > Between-pattern (specificity)
        try:
            stat, p = stats.wilcoxon(trained_mean, weight_between, alternative='greater')
        except ValueError:
            stat, p = 0.0, 1.0
        d = cohens_d(trained_mean, weight_between)
        analysis['H2_cross_seed'] = {
            'description': 'Within-pattern weight Δ > Between-pattern weight Δ',
            'W': float(stat), 'p': float(p), 'd': float(d),
            'within_mean': float(np.mean(trained_mean)),
            'between_mean': float(np.mean(weight_between)),
            'significant': p < 0.05,
        }
        logger.info(f"H2 (within>between): W={stat:.1f}, p={p:.4f}, d={d:.3f}")

        # H5: Trained > Non-pattern weight change
        try:
            stat, p = stats.wilcoxon(trained_mean, weight_other, alternative='greater')
        except ValueError:
            stat, p = 0.0, 1.0
        d = cohens_d(trained_mean, weight_other)
        analysis['H5_cross_seed'] = {
            'description': 'Trained weight Δ > Non-pattern weight Δ',
            'W': float(stat), 'p': float(p), 'd': float(d),
            'significant': p < 0.05,
        }
        logger.info(f"H5 (trained>other): W={stat:.1f}, p={p:.4f}, d={d:.3f}")

        # H7: Specificity index > 0
        try:
            stat, p = stats.wilcoxon(weight_specificity, alternative='greater')
        except ValueError:
            stat, p = 0.0, 1.0
        analysis['H7_cross_seed'] = {
            'description': 'Weight specificity (within - between) > 0',
            'W': float(stat), 'p': float(p),
            'mean_specificity': float(np.mean(weight_specificity)),
            'per_seed': [float(s) for s in weight_specificity],
            'all_positive': all(s > 0 for s in weight_specificity),
            'significant': p < 0.05 and np.mean(weight_specificity) > 0,
        }
        logger.info(f"H7 (specificity>0): W={stat:.1f}, p={p:.4f}")
        logger.info(f"  Mean: {np.mean(weight_specificity):.6f}, per-seed: {[f'{s:.6f}' for s in weight_specificity]}")
        logger.info(f"  All positive: {all(s > 0 for s in weight_specificity)}")

        # Summary gradient
        logger.info(f"\n  Weight change gradient:")
        logger.info(f"    Within-A:  {np.mean(weight_within_a):.6f}")
        logger.info(f"    Within-B:  {np.mean(weight_within_b):.6f}")
        logger.info(f"    Between:   {np.mean(weight_between):.6f}")
        logger.info(f"    Novel:     {np.mean(weight_novel):.6f}")
        logger.info(f"    Other:     {np.mean(weight_other):.6f}")

    # Decoder learning trajectories
    all_decoder_trajs = []
    for r in results:
        fb = r.get('training_feedback', {})
        traj = fb.get('decoder_trajectory', [])
        if traj:
            all_decoder_trajs.append(traj)

    if all_decoder_trajs:
        min_len = min(len(t) for t in all_decoder_trajs)
        aligned = np.array([t[:min_len] for t in all_decoder_trajs])
        mean_traj = np.mean(aligned, axis=0)

        # H4: Decoder accuracy improves
        if len(mean_traj) >= 3:
            x = np.arange(len(mean_traj))
            slope, _, r, p, _ = stats.linregress(x, mean_traj)
            analysis['H4_decoder_learning'] = {
                'description': 'Blind decoder accuracy improves over blocks',
                'mean_trajectory': mean_traj.tolist(),
                'slope': float(slope),
                'r': float(r),
                'p': float(p),
                'significant': p < 0.05 and slope > 0,
            }
            logger.info(f"H4 decoder: slope={slope:.4f}, p={p:.4f}")

    # Replay analysis
    replay_diffs = []
    for r in results:
        replay = r.get('analysis', {}).get('replay', {})
        if 'baseline' in replay and 'post_training' in replay:
            base = replay['baseline']['replay_rate']
            post = replay['post_training']['replay_rate']
            replay_diffs.append(post - base)

    if len(replay_diffs) >= 3:
        stat, p = stats.wilcoxon(replay_diffs, alternative='greater')
        analysis['H6_cross_seed'] = {
            'description': 'Post-training replay > baseline',
            'W': float(stat), 'p': float(p),
            'mean_diff': float(np.mean(replay_diffs)),
            'significant': p < 0.05 and np.mean(replay_diffs) > 0,
        }
        logger.info(f"H6 cross-seed: W={stat:.1f}, p={p:.4f}")

    # === FINAL VERDICT ===
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("=" * 70)

    evidence = {
        'stdp_trained_gt_novel': analysis.get('H1_cross_seed', {}).get('significant', False),
        'stdp_within_gt_between': analysis.get('H2_cross_seed', {}).get('significant', False),
        'stdp_trained_gt_other': analysis.get('H5_cross_seed', {}).get('significant', False),
        'decoder_learning': analysis.get('H4_cross_seed', {}).get('significant', False),
        'spontaneous_replay': analysis.get('H6_cross_seed', {}).get('significant', False),
        'weight_specificity': analysis.get('H7_cross_seed', {}).get('significant', False),
    }

    n_positive = sum(evidence.values())

    if n_positive >= 4:
        verdict = "STRONG: Multiple lines of evidence for STDP-mediated substrate learning"
    elif n_positive >= 2:
        verdict = "MODERATE: Some evidence for substrate plasticity"
    elif n_positive >= 1:
        verdict = "WEAK: Minimal plasticity evidence, likely noise"
    else:
        verdict = "NULL: No evidence for substrate learning"

    evidence['n_positive'] = n_positive
    evidence['verdict'] = verdict
    analysis['evidence'] = evidence

    for k, v in evidence.items():
        logger.info(f"  {k}: {v}")

    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment v4: STDP Learning")
    parser.add_argument("--substrate", default="izhikevich", choices=["izhikevich", "cl1"])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-reps", type=int, default=200, help="Repetitions per pattern")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 seed, 50 reps")
    args = parser.parse_args()

    cfg = ExpConfig(
        substrate_type=args.substrate,
        n_seeds=args.n_seeds,
        n_training_reps=args.n_reps,
    )

    if args.quick:
        cfg.n_seeds = 1
        cfg.n_training_reps = 50
        cfg.spontaneous_windows = 20
        cfg.n_influence_probes = 3

    results = run_multi_seed(config=cfg, output_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    verdict = results.get('cross_seed_analysis', {}).get('evidence', {}).get('verdict', 'UNKNOWN')
    print(f"Verdict: {verdict}")
