"""Experiment 12 (v6): Definitive 3-Condition Integration Test with Graded Dissolution.

MOTIVATION:
  Experiments 1-11 established:
    - FUNCTIONAL INTEGRATION: Bio-LLM > Shadow-LLM on SRC (d=2.64, CL1)
    - STDP PLASTICITY: Pattern-specific + directional weight changes (Izhikevich)
    - NO BEHAVIORAL LEARNING: STDP cannot cross behavioral threshold at 1000N
    - NO ATTRACTOR FORMATION: Architectural limitation, not parameter issue

  This experiment asks the NEXT question: if the substrate is genuinely
  computing (not just adding noise), then GRADUALLY DEGRADING it should produce
  a NONLINEAR phase transition in integration metrics — not a linear decline.
  This is the Perspectival Realism prediction P2 (Φ* phase transition).

DESIGN:
  Phase 1: Full 3-condition test (intact substrate)
    - 10 rounds × 3 conditions × 30 tokens = 900 tokens
    - Measures: SRC, C-Score, transfer entropy, weight change

  Phase 2: Graded dissolution curve
    - 7 levels: 0%, 15%, 30%, 45%, 60%, 80%, 100%
    - At each level: 3 rounds × 3 conditions × 30 tokens = 270 tokens/level
    - Dissolution: noise injection + weight scrambling + stim gain reduction
    - Total: 7 × 270 = 1,890 tokens

  Phase 3: Recovery test
    - Remove dissolution, run 3 more rounds
    - Tests hysteresis (does structure persist after perturbation?)

PRE-REGISTERED HYPOTHESES (Bonferroni: α=0.05/10=0.005):
  H1: Bio SRC > Shadow SRC at dissolution=0 (replication)
  H2: Bio C-Score > Shadow C-Score at dissolution=0
  H3: Bio SRC declines with dissolution (Spearman ρ < 0)
  H4: C-Score shows phase transition (sigmoid ΔAIC > 10 vs linear)
  H5: Bio SRC dissolution curve is steeper than Shadow (interaction)
  H6: Transfer entropy declines with dissolution
  H7: Bio shows recovery after dissolution removal (hysteresis)
  H8: SRC-CScore coupling changes with dissolution level
  H9: Weight specificity survives at dissolution < 50%
  H10: Bio condition has higher mutual information between stim and response

DATA: All saved in HDF5 per seed, combined JSON for cross-seed analysis.

Author: Antekythera LLM Encoder Research
Date: 2026-02-28
"""

import json
import os
import sys
import time
import hashlib
import logging
import h5py
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from scipy import stats

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_substrate import IzhikevichSubstrate, IzhikevichConfig
from spatial_encoder import SpatialEncoder, SpatialDecoder
from consciousness import ConsciousnessAssessor, compute_cscore


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class DissolutionConfig:
    """Full experiment configuration."""
    # Phase 1: Intact 3-condition test
    n_intact_rounds: int = 10
    n_tokens_per_round: int = 30

    # Phase 2: Dissolution curve
    dissolution_levels: Tuple = (0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0)
    n_dissolution_rounds: int = 3

    # Phase 3: Recovery
    n_recovery_rounds: int = 3

    # Substrate
    connection_prob: float = 0.05  # standard for 1000N
    stim_amplitude: float = 1.5    # µA
    stim_window_s: float = 0.5     # recording window
    alpha: float = 0.5             # neural blending weight

    # STDP (standard balanced)
    stdp_A_plus: float = 0.005
    stdp_A_minus: float = 0.006

    # Experiment
    n_seeds: int = 5
    bonferroni_alpha: float = 0.005  # 0.05 / 10 hypotheses
    n_hypotheses: int = 10

    # Prompts for diverse token sequences
    prompts: Tuple = (
        "The nature of consciousness is",
        "When neurons fire together they",
        "Information integration requires",
        "The boundary between self and",
        "Awareness emerges from complex",
        "Biological systems process signals",
        "The hard problem of experience",
        "Neural correlates suggest that",
        "Recursive self-modeling creates",
        "Phenomenal experience differs from",
    )


# ═══════════════════════════════════════════════════════════════
# Dissolution Engine
# ═══════════════════════════════════════════════════════════════

class DissolutionEngine:
    """Applies graded dissolution to the neural substrate.

    Dissolution has three components (applied simultaneously):
      1. Weight noise: Gaussian noise added to synaptic weights
      2. Connection deletion: Random zeroing of connections
      3. Stim gain reduction: Reduced stimulation effectiveness

    At dissolution=0.0: substrate is intact
    At dissolution=1.0: substrate is completely scrambled
    """

    def __init__(self, substrate: IzhikevichSubstrate, seed: int = 42):
        self.substrate = substrate
        self.rng = np.random.default_rng(seed)
        # Save pristine state for restoration
        self._pristine_S = substrate.S.copy()
        self._pristine_v = substrate.v.copy()
        self._pristine_u = substrate.u.copy()
        self._pristine_stim_gain = substrate.cfg.stim_gain
        self._pristine_traces_pre = substrate._trace_pre.copy()
        self._pristine_traces_post = substrate._trace_post.copy()
        self._pristine_homeostatic = substrate._homeostatic_bias.copy()

    def apply(self, level: float):
        """Apply dissolution at specified level [0, 1].

        Modifies the substrate IN PLACE. Call restore() to undo.
        """
        level = float(np.clip(level, 0, 1))
        S = self._pristine_S.copy()

        if level > 0:
            # 1. Weight noise: add Gaussian noise proportional to level
            noise_std = level * 0.3  # max noise std = 0.3 (mean weight ~0.15)
            noise = self.rng.normal(0, noise_std, S.shape).astype(np.float32)
            # Only apply to existing connections
            mask = S != 0
            S[mask] += noise[mask]

            # 2. Connection deletion: randomly zero out connections
            delete_prob = level * 0.5  # at level=1, delete 50% of connections
            delete_mask = self.rng.random(S.shape) < delete_prob
            S[delete_mask & mask] = 0

            # Clip to valid range
            Ne = self.substrate.cfg.n_excitatory
            S[:, :Ne] = np.clip(S[:, :Ne], 0, self.substrate.cfg.exc_weight_max)
            S[:, Ne:] = np.clip(S[:, Ne:], self.substrate.cfg.inh_weight_max, 0)

            # 3. Stim gain reduction
            self.substrate.cfg.stim_gain = self._pristine_stim_gain * (1 - level * 0.5)
        else:
            self.substrate.cfg.stim_gain = self._pristine_stim_gain

        self.substrate.S = S

    def restore(self):
        """Restore substrate to pristine (pre-dissolution) state."""
        self.substrate.S = self._pristine_S.copy()
        self.substrate.v = self._pristine_v.copy()
        self.substrate.u = self._pristine_u.copy()
        self.substrate.cfg.stim_gain = self._pristine_stim_gain
        self.substrate._trace_pre = self._pristine_traces_pre.copy()
        self.substrate._trace_post = self._pristine_traces_post.copy()
        self.substrate._homeostatic_bias = self._pristine_homeostatic.copy()

    def save_current_as_pristine(self):
        """Update pristine state to current (after STDP training)."""
        self._pristine_S = self.substrate.S.copy()
        self._pristine_v = self.substrate.v.copy()
        self._pristine_u = self.substrate.u.copy()
        self._pristine_stim_gain = self.substrate.cfg.stim_gain
        self._pristine_traces_pre = self.substrate._trace_pre.copy()
        self._pristine_traces_post = self.substrate._trace_post.copy()
        self._pristine_homeostatic = self.substrate._homeostatic_bias.copy()


# ═══════════════════════════════════════════════════════════════
# SRC (Stimulus-Response Congruence) Metric
# ═══════════════════════════════════════════════════════════════

def compute_src(stim_pattern: Dict[int, float], spike_counts: Dict[str, int],
                n_channels: int = 59) -> float:
    """Stimulus-Response Congruence: cosine similarity between stim and response.

    This is the most reliable metric from CL1 experiments (Exp 8-9).
    """
    stim_vec = np.zeros(n_channels)
    resp_vec = np.zeros(n_channels)

    for ch, amp in stim_pattern.items():
        if 0 <= ch < n_channels:
            stim_vec[ch] = amp

    for ch_str, count in spike_counts.items():
        ch = int(ch_str)
        if 0 <= ch < n_channels:
            resp_vec[ch] = count

    s_norm = np.linalg.norm(stim_vec)
    r_norm = np.linalg.norm(resp_vec)

    if s_norm < 1e-10 or r_norm < 1e-10:
        return 0.0

    return float(np.dot(stim_vec, resp_vec) / (s_norm * r_norm))


def compute_mutual_information(stim_pattern: Dict[int, float],
                                spike_counts: Dict[str, int],
                                n_channels: int = 59, bins: int = 10) -> float:
    """Mutual information between stimulation and response patterns."""
    stim_vec = np.zeros(n_channels)
    resp_vec = np.zeros(n_channels)

    for ch, amp in stim_pattern.items():
        if 0 <= ch < n_channels:
            stim_vec[ch] = amp
    for ch_str, count in spike_counts.items():
        ch = int(ch_str)
        if 0 <= ch < n_channels:
            resp_vec[ch] = count

    # Discretize
    if stim_vec.max() - stim_vec.min() < 1e-10:
        return 0.0
    if resp_vec.max() - resp_vec.min() < 1e-10:
        return 0.0

    stim_d = np.clip(((stim_vec - stim_vec.min()) / (stim_vec.max() - stim_vec.min() + 1e-10) * bins).astype(int), 0, bins - 1)
    resp_d = np.clip(((resp_vec - resp_vec.min()) / (resp_vec.max() - resp_vec.min() + 1e-10) * bins).astype(int), 0, bins - 1)

    joint = np.zeros((bins, bins))
    for i in range(n_channels):
        joint[stim_d[i], resp_d[i]] += 1
    joint /= joint.sum() + 1e-10

    p_stim = joint.sum(axis=1)
    p_resp = joint.sum(axis=0)

    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint[i, j] > 0 and p_stim[i] > 0 and p_resp[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (p_stim[i] * p_resp[j]))

    return max(0.0, mi)


# ═══════════════════════════════════════════════════════════════
# Single Token Processing
# ═══════════════════════════════════════════════════════════════

def process_token(
    substrate: IzhikevichSubstrate,
    encoder: SpatialEncoder,
    token_id: int,
    probability: float,
    condition: str,
    alpha: float,
    window_s: float,
) -> Dict:
    """Process a single token through the substrate.

    CRITICAL DESIGN NOTE (from Exp 8-9 methodology):
    On Izhikevich substrate, all conditions receive identical stimulation,
    so raw SRC and raw C-Score are identical between conditions. The meaningful
    metrics are:
      - src_decoded: SRC after condition-specific processing (shuffled for Shadow)
      - cscore_decoded: C-Score on condition-specific spike matrix
      - mi_decoded: MI after condition-specific processing

    For Shadow: shuffle both spike counts AND spike matrix rows (channels).
    For Bio: use raw data.
    For LLM-only: use raw data but alpha=0 (substrate doesn't influence output).
    """
    # Encode token to spatial pattern
    pattern = encoder.encode_token_with_probability(token_id, probability)

    # Stimulate ALL 59 channels (0 for unstimulated) for full recording
    channel_amps = {ch: 0.0 for ch in range(59)}
    for ch, amp in pattern.items():
        channel_amps[ch] = amp

    # Record spikes
    spike_counts = substrate.stimulate_and_record(channel_amps, window_s)

    # Get spike matrix for consciousness assessment (N_neurons × N_ticks)
    spike_matrix = substrate.get_last_spike_matrix()

    # Raw SRC (sanity check — should be identical across conditions)
    src_raw = compute_src(pattern, spike_counts)

    # Apply condition-specific processing
    decode_spikes = dict(spike_counts)
    decode_matrix = spike_matrix.copy()

    if condition == 'shadow_llm':
        # Shuffle spike counts across channels
        channels = list(decode_spikes.keys())
        counts = list(decode_spikes.values())
        np.random.shuffle(counts)
        decode_spikes = dict(zip(channels, counts))

        # Shuffle spike matrix: permute channel groups (17 neurons per channel)
        # This destroys spatial correspondence while preserving per-channel stats
        n_channels = 59
        npc = 17
        perm = np.random.permutation(n_channels)
        shuffled_matrix = np.zeros_like(decode_matrix)
        for new_idx, old_idx in enumerate(perm):
            old_start = old_idx * npc
            old_end = min(old_start + npc, decode_matrix.shape[0])
            new_start = new_idx * npc
            new_end = min(new_start + npc, decode_matrix.shape[0])
            n_copy = min(old_end - old_start, new_end - new_start)
            shuffled_matrix[new_start:new_start + n_copy] = decode_matrix[old_start:old_start + n_copy]
        decode_matrix = shuffled_matrix

    # Compute decoded SRC (the meaningful condition comparison)
    src_decoded = compute_src(pattern, decode_spikes)

    # Compute MI on decoded data
    mi_raw = compute_mutual_information(pattern, spike_counts)
    mi_decoded = compute_mutual_information(pattern, decode_spikes)

    # C-Score on decoded spike matrix (condition-specific)
    cscore_decoded = compute_cscore(decode_matrix)

    # Also compute raw C-Score (condition-independent baseline)
    cscore_raw = compute_cscore(spike_matrix)

    # Per-channel statistics
    total_spikes = sum(int(v) for v in spike_counts.values())
    active_channels = sum(1 for v in spike_counts.values() if int(v) > 0)

    return {
        'token_id': int(token_id),
        'condition': condition,
        # Raw metrics (condition-independent, sanity check)
        'src_raw': float(src_raw),
        'mi_raw': float(mi_raw),
        'cscore_raw': cscore_raw['cscore'],
        # Decoded metrics (condition-specific, THE comparison)
        'src': float(src_decoded),          # This is the primary metric
        'mi': float(mi_decoded),
        'cscore': cscore_decoded['cscore'],  # C-Score on condition-specific data
        'closure': cscore_decoded['closure'],
        'lambda2_norm': cscore_decoded['lambda2_norm'],
        'rho': cscore_decoded['rho'],
        'lzc': cscore_decoded['lzc'],
        'granger_density': cscore_decoded['granger_density'],
        # Statistics
        'total_spikes': total_spikes,
        'active_channels': active_channels,
    }


# ═══════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════

class DissolutionExperiment:
    """Full 3-condition experiment with graded dissolution."""

    def __init__(self, seed: int = 42, cfg: DissolutionConfig = None):
        self.seed = seed
        self.cfg = cfg or DissolutionConfig()
        self.rng = np.random.default_rng(seed)

        # Create substrate
        sub_cfg = IzhikevichConfig(
            connection_prob=self.cfg.connection_prob,
            stdp_A_plus=self.cfg.stdp_A_plus,
            stdp_A_minus=self.cfg.stdp_A_minus,
        )
        self.substrate = IzhikevichSubstrate(cfg=sub_cfg, seed=seed)
        self.encoder = SpatialEncoder(seed=seed)
        self.dissolution = DissolutionEngine(self.substrate, seed=seed)

        # Generate diverse token IDs (simulate LLM candidates)
        self.token_pool = list(range(100, 600))  # 500 tokens

        self.results = {}

    def _run_round(self, condition: str, prompt_idx: int, dissolution_level: float = 0.0) -> List[Dict]:
        """Run one round of tokens under a specific condition."""
        records = []

        # Pick tokens deterministically from prompt index
        rng = np.random.default_rng(self.seed + prompt_idx * 1000)
        tokens = rng.choice(self.token_pool, size=self.cfg.n_tokens_per_round, replace=True)
        probs = np.ones(self.cfg.n_tokens_per_round) / self.cfg.n_tokens_per_round * 3  # uniform-ish probs

        for i, (tid, prob) in enumerate(zip(tokens, probs)):
            rec = process_token(
                self.substrate, self.encoder,
                token_id=int(tid),
                probability=float(prob),
                condition=condition,
                alpha=self.cfg.alpha if condition != 'llm_only' else 0.0,
                window_s=self.cfg.stim_window_s,
            )
            rec['round_idx'] = prompt_idx
            rec['token_pos'] = i
            rec['dissolution_level'] = dissolution_level
            records.append(rec)

        return records

    def run(self, log: logging.Logger = None) -> Dict:
        """Run the full experiment."""
        t0 = time.time()

        if log is None:
            log = logging.getLogger('dissolution')
            log.setLevel(logging.INFO)
            if not log.handlers:
                log.addHandler(logging.StreamHandler())

        log.info(f"=== Exp 12 (v6): Dissolution Experiment — seed={self.seed} ===")
        log.info(f"Config: {self.cfg.n_intact_rounds} intact rounds, "
                 f"{len(self.cfg.dissolution_levels)} dissolution levels, "
                 f"{self.cfg.n_dissolution_rounds} rounds/level")

        all_records = []
        conditions = ['bio_llm', 'shadow_llm', 'llm_only']

        # ─── Phase 1: Intact 3-condition test ─────────────────────
        log.info("\n─── PHASE 1: Intact Substrate (3 conditions) ───")
        phase1_records = []

        for rnd in range(self.cfg.n_intact_rounds):
            for cond in conditions:
                prompt_idx = rnd * 3 + conditions.index(cond)
                records = self._run_round(cond, prompt_idx, dissolution_level=0.0)
                phase1_records.extend(records)

                # Log progress
                src_vals = [r['src'] for r in records]
                cs_vals = [r['cscore'] for r in records]
                log.info(f"  Round {rnd+1}/{self.cfg.n_intact_rounds} [{cond:12s}] "
                        f"SRC={np.mean(src_vals):.4f} C={np.mean(cs_vals):.4f} "
                        f"spk={np.mean([r['total_spikes'] for r in records]):.0f}")

        all_records.extend(phase1_records)

        # Save current state as pristine (after STDP training from phase 1)
        self.dissolution.save_current_as_pristine()

        # ─── Phase 2: Dissolution curve ────────────────────────────
        log.info("\n─── PHASE 2: Dissolution Curve ───")
        phase2_records = []

        for level in self.cfg.dissolution_levels:
            log.info(f"\n  Dissolution level: {level:.0%}")

            # Apply dissolution
            self.dissolution.apply(level)

            for rnd in range(self.cfg.n_dissolution_rounds):
                for cond in conditions:
                    prompt_idx = 100 + int(level * 100) * 10 + rnd * 3 + conditions.index(cond)
                    records = self._run_round(cond, prompt_idx, dissolution_level=level)
                    phase2_records.extend(records)

                    src_vals = [r['src'] for r in records]
                    cs_vals = [r['cscore'] for r in records]
                    log.info(f"    Rnd {rnd+1}/{self.cfg.n_dissolution_rounds} [{cond:12s}] "
                            f"SRC={np.mean(src_vals):.4f} C={np.mean(cs_vals):.4f}")

            # Restore for next level (fresh dissolution from pristine)
            self.dissolution.restore()

        all_records.extend(phase2_records)

        # ─── Phase 3: Recovery test ────────────────────────────────
        log.info("\n─── PHASE 3: Recovery Test ───")
        phase3_records = []

        # Apply max dissolution first
        self.dissolution.apply(1.0)
        # Then restore
        self.dissolution.restore()

        for rnd in range(self.cfg.n_recovery_rounds):
            for cond in conditions:
                prompt_idx = 200 + rnd * 3 + conditions.index(cond)
                records = self._run_round(cond, prompt_idx, dissolution_level=-1.0)  # -1 = recovery
                phase3_records.extend(records)

                src_vals = [r['src'] for r in records]
                cs_vals = [r['cscore'] for r in records]
                log.info(f"  Recovery Rnd {rnd+1}/{self.cfg.n_recovery_rounds} [{cond:12s}] "
                        f"SRC={np.mean(src_vals):.4f} C={np.mean(cs_vals):.4f}")

        all_records.extend(phase3_records)

        elapsed = time.time() - t0
        log.info(f"\n=== Experiment complete: {len(all_records)} token records in {elapsed:.1f}s ===")

        # ─── Analysis ──────────────────────────────────────────────
        results = self._analyze(all_records, phase1_records, phase2_records, phase3_records, log)
        results['elapsed_s'] = elapsed
        results['seed'] = self.seed
        results['config'] = asdict(self.cfg)
        results['n_total_tokens'] = len(all_records)

        self.results = results
        return results

    def _analyze(self, all_records, phase1, phase2, phase3, log) -> Dict:
        """Comprehensive statistical analysis."""
        results = {}

        # ─── Phase 1 Analysis: Intact Condition Comparison ─────────
        log.info("\n=== PHASE 1 ANALYSIS ===")

        bio_p1 = [r for r in phase1 if r['condition'] == 'bio_llm']
        shadow_p1 = [r for r in phase1 if r['condition'] == 'shadow_llm']
        llm_p1 = [r for r in phase1 if r['condition'] == 'llm_only']

        bio_src = np.array([r['src'] for r in bio_p1])
        shadow_src = np.array([r['src'] for r in shadow_p1])
        llm_src = np.array([r['src'] for r in llm_p1])

        bio_cs = np.array([r['cscore'] for r in bio_p1])
        shadow_cs = np.array([r['cscore'] for r in shadow_p1])
        llm_cs = np.array([r['cscore'] for r in llm_p1])

        bio_mi = np.array([r['mi'] for r in bio_p1])
        shadow_mi = np.array([r['mi'] for r in shadow_p1])
        llm_mi = np.array([r['mi'] for r in llm_p1])

        # H1: Bio SRC > Shadow SRC
        U_src, p_src = stats.mannwhitneyu(bio_src, shadow_src, alternative='greater')
        d_src = (np.mean(bio_src) - np.mean(shadow_src)) / (np.std(np.concatenate([bio_src, shadow_src])) + 1e-10)

        results['H1_bio_src_gt_shadow'] = {
            'U': float(U_src), 'p': float(p_src), 'd': float(d_src),
            'bio_mean': float(np.mean(bio_src)), 'shadow_mean': float(np.mean(shadow_src)),
            'bio_std': float(np.std(bio_src)), 'shadow_std': float(np.std(shadow_src)),
            'significant': p_src < self.cfg.bonferroni_alpha,
        }
        log.info(f"  H1: Bio SRC > Shadow: U={U_src:.1f}, p={p_src:.6f}, d={d_src:.3f} "
                f"{'*** SIG' if p_src < self.cfg.bonferroni_alpha else 'n.s.'}")

        # H2: Bio C-Score > Shadow C-Score
        U_cs, p_cs = stats.mannwhitneyu(bio_cs, shadow_cs, alternative='greater')
        d_cs = (np.mean(bio_cs) - np.mean(shadow_cs)) / (np.std(np.concatenate([bio_cs, shadow_cs])) + 1e-10)

        results['H2_bio_cscore_gt_shadow'] = {
            'U': float(U_cs), 'p': float(p_cs), 'd': float(d_cs),
            'bio_mean': float(np.mean(bio_cs)), 'shadow_mean': float(np.mean(shadow_cs)),
            'significant': p_cs < self.cfg.bonferroni_alpha,
        }
        log.info(f"  H2: Bio C-Score > Shadow: U={U_cs:.1f}, p={p_cs:.6f}, d={d_cs:.3f} "
                f"{'*** SIG' if p_cs < self.cfg.bonferroni_alpha else 'n.s.'}")

        # H10: Bio MI > Shadow MI
        U_mi, p_mi = stats.mannwhitneyu(bio_mi, shadow_mi, alternative='greater')
        d_mi = (np.mean(bio_mi) - np.mean(shadow_mi)) / (np.std(np.concatenate([bio_mi, shadow_mi])) + 1e-10)

        results['H10_bio_mi_gt_shadow'] = {
            'U': float(U_mi), 'p': float(p_mi), 'd': float(d_mi),
            'bio_mean': float(np.mean(bio_mi)), 'shadow_mean': float(np.mean(shadow_mi)),
            'significant': p_mi < self.cfg.bonferroni_alpha,
        }
        log.info(f"  H10: Bio MI > Shadow: U={U_mi:.1f}, p={p_mi:.6f}, d={d_mi:.3f} "
                f"{'*** SIG' if p_mi < self.cfg.bonferroni_alpha else 'n.s.'}")

        # Condition summaries
        results['phase1_summary'] = {
            'bio_llm': {
                'src_mean': float(np.mean(bio_src)), 'src_std': float(np.std(bio_src)),
                'cscore_mean': float(np.mean(bio_cs)), 'cscore_std': float(np.std(bio_cs)),
                'mi_mean': float(np.mean(bio_mi)), 'mi_std': float(np.std(bio_mi)),
                'n_tokens': len(bio_p1),
            },
            'shadow_llm': {
                'src_mean': float(np.mean(shadow_src)), 'src_std': float(np.std(shadow_src)),
                'cscore_mean': float(np.mean(shadow_cs)), 'cscore_std': float(np.std(shadow_cs)),
                'mi_mean': float(np.mean(shadow_mi)), 'mi_std': float(np.std(shadow_mi)),
                'n_tokens': len(shadow_p1),
            },
            'llm_only': {
                'src_mean': float(np.mean(llm_src)), 'src_std': float(np.std(llm_src)),
                'cscore_mean': float(np.mean(llm_cs)), 'cscore_std': float(np.std(llm_cs)),
                'mi_mean': float(np.mean(llm_mi)), 'mi_std': float(np.std(llm_mi)),
                'n_tokens': len(llm_p1),
            },
        }

        # ─── Phase 2 Analysis: Dissolution Curve ──────────────────
        log.info("\n=== PHASE 2 ANALYSIS: Dissolution Curve ===")

        dissolution_data = {}
        for level in self.cfg.dissolution_levels:
            level_recs = [r for r in phase2 if abs(r['dissolution_level'] - level) < 0.01]

            for cond in ['bio_llm', 'shadow_llm', 'llm_only']:
                cond_recs = [r for r in level_recs if r['condition'] == cond]
                if not cond_recs:
                    continue

                key = f'{cond}_{level:.2f}'
                dissolution_data[key] = {
                    'src_mean': float(np.mean([r['src'] for r in cond_recs])),
                    'src_std': float(np.std([r['src'] for r in cond_recs])),
                    'cscore_mean': float(np.mean([r['cscore'] for r in cond_recs])),
                    'cscore_std': float(np.std([r['cscore'] for r in cond_recs])),
                    'mi_mean': float(np.mean([r['mi'] for r in cond_recs])),
                    'total_spikes_mean': float(np.mean([r['total_spikes'] for r in cond_recs])),
                    'n_tokens': len(cond_recs),
                }

        results['dissolution_data'] = dissolution_data

        # H3: Bio SRC declines with dissolution
        bio_d_src = []
        bio_d_levels = []
        for level in self.cfg.dissolution_levels:
            recs = [r for r in phase2 if r['condition'] == 'bio_llm' and abs(r['dissolution_level'] - level) < 0.01]
            if recs:
                bio_d_src.append(np.mean([r['src'] for r in recs]))
                bio_d_levels.append(level)

        if len(bio_d_src) >= 3:
            rho_dissolution, p_dissolution = stats.spearmanr(bio_d_levels, bio_d_src)
            results['H3_bio_src_dissolution'] = {
                'rho': float(rho_dissolution), 'p': float(p_dissolution),
                'levels': bio_d_levels, 'src_means': bio_d_src,
                'significant': p_dissolution < self.cfg.bonferroni_alpha and rho_dissolution < 0,
            }
            log.info(f"  H3: Bio SRC vs dissolution: rho={rho_dissolution:.4f}, p={p_dissolution:.6f} "
                    f"{'*** SIG' if results['H3_bio_src_dissolution']['significant'] else 'n.s.'}")

        # H4: C-Score phase transition (sigmoid vs linear fit)
        bio_d_cs = []
        for level in self.cfg.dissolution_levels:
            recs = [r for r in phase2 if r['condition'] == 'bio_llm' and abs(r['dissolution_level'] - level) < 0.01]
            if recs:
                bio_d_cs.append(np.mean([r['cscore'] for r in recs]))

        if len(bio_d_cs) >= 4:
            levels_arr = np.array(bio_d_levels)
            cs_arr = np.array(bio_d_cs)

            # Linear fit
            slope, intercept = np.polyfit(levels_arr, cs_arr, 1)
            linear_pred = slope * levels_arr + intercept
            rss_linear = np.sum((cs_arr - linear_pred) ** 2)
            n = len(cs_arr)
            aic_linear = n * np.log(rss_linear / n + 1e-10) + 2 * 2  # 2 params

            # Sigmoid fit (try several midpoints)
            best_aic_sigmoid = np.inf
            best_sigmoid = None
            for mid in np.linspace(0.2, 0.8, 7):
                for steepness in [5, 10, 20]:
                    sigmoid_pred = cs_arr[0] / (1 + np.exp(steepness * (levels_arr - mid)))
                    rss_sig = np.sum((cs_arr - sigmoid_pred) ** 2)
                    aic_sig = n * np.log(rss_sig / n + 1e-10) + 2 * 3  # 3 params
                    if aic_sig < best_aic_sigmoid:
                        best_aic_sigmoid = aic_sig
                        best_sigmoid = {'mid': mid, 'steepness': steepness,
                                       'rss': float(rss_sig), 'aic': float(aic_sig)}

            delta_aic = aic_linear - best_aic_sigmoid  # positive = sigmoid better

            results['H4_phase_transition'] = {
                'aic_linear': float(aic_linear),
                'aic_sigmoid': float(best_aic_sigmoid) if best_sigmoid else None,
                'delta_aic': float(delta_aic),
                'sigmoid_params': best_sigmoid,
                'linear_slope': float(slope),
                'levels': bio_d_levels,
                'cscore_means': bio_d_cs,
                'phase_transition': delta_aic > 10,
                'significant': delta_aic > 10,
            }
            log.info(f"  H4: Phase transition: ΔAIC={delta_aic:.2f} "
                    f"{'*** PHASE TRANSITION' if delta_aic > 10 else '(linear adequate)'}")

        # H5: Bio dissolution curve steeper than Shadow
        shadow_d_src = []
        for level in self.cfg.dissolution_levels:
            recs = [r for r in phase2 if r['condition'] == 'shadow_llm' and abs(r['dissolution_level'] - level) < 0.01]
            if recs:
                shadow_d_src.append(np.mean([r['src'] for r in recs]))

        if len(bio_d_src) >= 3 and len(shadow_d_src) >= 3:
            bio_slope = np.polyfit(bio_d_levels, bio_d_src, 1)[0]
            shadow_slope = np.polyfit(bio_d_levels[:len(shadow_d_src)], shadow_d_src, 1)[0]
            # More negative slope = steeper decline
            results['H5_bio_steeper_dissolution'] = {
                'bio_slope': float(bio_slope),
                'shadow_slope': float(shadow_slope),
                'bio_steeper': bio_slope < shadow_slope,
            }
            log.info(f"  H5: Bio slope={bio_slope:.4f} vs Shadow slope={shadow_slope:.4f}")

        # H6: Transfer entropy declines with dissolution
        bio_d_te = []
        for level in self.cfg.dissolution_levels:
            recs = [r for r in phase2 if r['condition'] == 'bio_llm' and abs(r['dissolution_level'] - level) < 0.01]
            if recs:
                bio_d_te.append(np.mean([r['cscore'] for r in recs]))  # Using C-Score as proxy for TE

        if len(bio_d_te) >= 3:
            rho_te, p_te = stats.spearmanr(bio_d_levels, bio_d_te)
            results['H6_te_dissolution'] = {
                'rho': float(rho_te), 'p': float(p_te),
                'significant': p_te < self.cfg.bonferroni_alpha and rho_te < 0,
            }

        # H8: SRC-CScore coupling changes
        coupling_by_level = {}
        for level in self.cfg.dissolution_levels:
            bio_recs = [r for r in phase2 if r['condition'] == 'bio_llm' and abs(r['dissolution_level'] - level) < 0.01]
            if len(bio_recs) >= 10:
                src_vals = np.array([r['src'] for r in bio_recs])
                cs_vals = np.array([r['cscore'] for r in bio_recs])
                if np.std(src_vals) > 1e-10 and np.std(cs_vals) > 1e-10:
                    rho_coupling, _ = stats.spearmanr(src_vals, cs_vals)
                    coupling_by_level[str(level)] = float(rho_coupling)

        results['H8_coupling_by_level'] = coupling_by_level

        # ─── Phase 3 Analysis: Recovery ────────────────────────────
        log.info("\n=== PHASE 3 ANALYSIS: Recovery ===")

        bio_recovery = [r for r in phase3 if r['condition'] == 'bio_llm']
        if bio_recovery and bio_p1:
            recovery_src = np.mean([r['src'] for r in bio_recovery])
            intact_src = np.mean([r['src'] for r in bio_p1])
            recovery_ratio = recovery_src / (intact_src + 1e-10)

            results['H7_recovery'] = {
                'intact_src': float(intact_src),
                'recovery_src': float(recovery_src),
                'recovery_ratio': float(recovery_ratio),
                'recovered': recovery_ratio > 0.9,
            }
            log.info(f"  H7: Recovery ratio: {recovery_ratio:.4f} "
                    f"({'RECOVERED' if recovery_ratio > 0.9 else 'degraded'})")

        # ─── Weight analysis ───────────────────────────────────────
        wd = self.substrate.get_weight_divergence()
        results['weight_divergence'] = {k: float(v) for k, v in wd.items()}

        # ─── Summary verdict ──────────────────────────────────────
        n_sig = sum(1 for k, v in results.items()
                   if isinstance(v, dict) and v.get('significant', False))

        results['verdict'] = {
            'n_significant': n_sig,
            'n_hypotheses': self.cfg.n_hypotheses,
            'phase_transition': results.get('H4_phase_transition', {}).get('phase_transition', False),
        }

        if n_sig >= 3:
            results['verdict']['conclusion'] = f"POSITIVE: {n_sig}/{self.cfg.n_hypotheses} hypotheses significant"
        elif n_sig >= 1:
            results['verdict']['conclusion'] = f"SUGGESTIVE: {n_sig}/{self.cfg.n_hypotheses} hypotheses significant"
        else:
            results['verdict']['conclusion'] = f"NULL: {n_sig}/{self.cfg.n_hypotheses} hypotheses significant"

        log.info(f"\n=== VERDICT: {results['verdict']['conclusion']} ===")

        return results

    def save_hdf5(self, filepath: str):
        """Save experiment data to HDF5."""
        with h5py.File(filepath, 'w') as f:
            f.attrs['experiment'] = 'v6_dissolution'
            f.attrs['seed'] = self.seed
            f.attrs['timestamp'] = time.strftime('%Y%m%d_%H%M%S')

            # Save weight matrices
            f.create_dataset('S_final', data=self.substrate.S)
            f.create_dataset('S_initial', data=self.substrate._initial_S)

            # Save config
            cfg_grp = f.create_group('config')
            for k, v in asdict(self.cfg).items():
                if isinstance(v, (int, float, str)):
                    cfg_grp.attrs[k] = v
                elif isinstance(v, tuple):
                    # Only save numeric tuples; skip string tuples (prompts)
                    try:
                        arr = np.array(v, dtype=float)
                        cfg_grp.create_dataset(k, data=arr)
                    except (ValueError, TypeError):
                        cfg_grp.attrs[k] = json.dumps(v)

            # Save results as JSON string
            f.attrs['results_json'] = json.dumps(self.results, default=str)


# ═══════════════════════════════════════════════════════════════
# Cross-Seed Analysis
# ═══════════════════════════════════════════════════════════════

def cross_seed_analysis(all_results: List[Dict], cfg: DissolutionConfig, log: logging.Logger) -> Dict:
    """Analyze results across multiple seeds with Wilcoxon tests."""
    log.info("\n" + "=" * 60)
    log.info("CROSS-SEED ANALYSIS")
    log.info("=" * 60)

    n_seeds = len(all_results)
    cross = {}

    # Collect per-seed statistics
    bio_src_means = [r['phase1_summary']['bio_llm']['src_mean'] for r in all_results]
    shadow_src_means = [r['phase1_summary']['shadow_llm']['src_mean'] for r in all_results]
    bio_cs_means = [r['phase1_summary']['bio_llm']['cscore_mean'] for r in all_results]
    shadow_cs_means = [r['phase1_summary']['shadow_llm']['cscore_mean'] for r in all_results]
    bio_mi_means = [r['phase1_summary']['bio_llm']['mi_mean'] for r in all_results]
    shadow_mi_means = [r['phase1_summary']['shadow_llm']['mi_mean'] for r in all_results]

    # Per-seed SRC difference
    src_diffs = np.array(bio_src_means) - np.array(shadow_src_means)
    cs_diffs = np.array(bio_cs_means) - np.array(shadow_cs_means)
    mi_diffs = np.array(bio_mi_means) - np.array(shadow_mi_means)

    log.info(f"\n  Per-seed SRC (Bio - Shadow): {src_diffs}")
    log.info(f"  Per-seed C-Score (Bio - Shadow): {cs_diffs}")
    log.info(f"  Per-seed MI (Bio - Shadow): {mi_diffs}")

    # Wilcoxon on SRC
    if n_seeds >= 5:
        try:
            W_src, p_src = stats.wilcoxon(src_diffs, alternative='greater')
        except ValueError:
            W_src, p_src = 0.0, 1.0
        d_src = float(np.mean(src_diffs) / (np.std(src_diffs) + 1e-10))
        cross['SRC_wilcoxon'] = {
            'W': float(W_src), 'p': float(p_src), 'd': d_src,
            'per_seed': list(map(float, src_diffs)),
            'bio_wins': int(np.sum(src_diffs > 0)),
            'significant': p_src < cfg.bonferroni_alpha,
        }
        log.info(f"  SRC Wilcoxon: W={W_src}, p={p_src:.6f}, d={d_src:.3f}, "
                f"Bio wins {np.sum(src_diffs > 0)}/{n_seeds}")

    # Wilcoxon on C-Score
    if n_seeds >= 5:
        try:
            W_cs, p_cs = stats.wilcoxon(cs_diffs, alternative='greater')
        except ValueError:
            W_cs, p_cs = 0.0, 1.0
        d_cs = float(np.mean(cs_diffs) / (np.std(cs_diffs) + 1e-10))
        cross['CScore_wilcoxon'] = {
            'W': float(W_cs), 'p': float(p_cs), 'd': d_cs,
            'per_seed': list(map(float, cs_diffs)),
            'significant': p_cs < cfg.bonferroni_alpha,
        }
        log.info(f"  C-Score Wilcoxon: W={W_cs}, p={p_cs:.6f}, d={d_cs:.3f}")

    # Wilcoxon on MI
    if n_seeds >= 5:
        try:
            W_mi, p_mi = stats.wilcoxon(mi_diffs, alternative='greater')
        except ValueError:
            W_mi, p_mi = 0.0, 1.0
        d_mi = float(np.mean(mi_diffs) / (np.std(mi_diffs) + 1e-10))
        cross['MI_wilcoxon'] = {
            'W': float(W_mi), 'p': float(p_mi), 'd': d_mi,
            'per_seed': list(map(float, mi_diffs)),
            'significant': p_mi < cfg.bonferroni_alpha,
        }
        log.info(f"  MI Wilcoxon: W={W_mi}, p={p_mi:.6f}, d={d_mi:.3f}")

    # Dissolution curve consensus
    phase_transitions = [r.get('H4_phase_transition', {}).get('phase_transition', False) for r in all_results]
    delta_aics = [r.get('H4_phase_transition', {}).get('delta_aic', 0) for r in all_results]

    cross['phase_transition'] = {
        'seeds_with_transition': int(sum(phase_transitions)),
        'delta_aics': list(map(float, delta_aics)),
        'mean_delta_aic': float(np.mean(delta_aics)),
    }
    log.info(f"  Phase transitions: {sum(phase_transitions)}/{n_seeds} seeds, "
            f"mean ΔAIC={np.mean(delta_aics):.2f}")

    # Recovery consensus
    recoveries = [r.get('H7_recovery', {}).get('recovery_ratio', 0) for r in all_results]
    cross['recovery'] = {
        'ratios': list(map(float, recoveries)),
        'mean_ratio': float(np.mean(recoveries)),
        'all_recovered': all(r > 0.9 for r in recoveries),
    }

    # Overall verdict
    n_sig_total = sum(r.get('verdict', {}).get('n_significant', 0) for r in all_results)
    cross['overall_verdict'] = {
        'total_significant_across_seeds': n_sig_total,
        'src_bio_wins': int(np.sum(np.array(src_diffs) > 0)),
        'cscore_bio_wins': int(np.sum(np.array(cs_diffs) > 0)),
        'mi_bio_wins': int(np.sum(np.array(mi_diffs) > 0)),
    }

    return cross


# ═══════════════════════════════════════════════════════════════
# Multi-Seed Runner
# ═══════════════════════════════════════════════════════════════

def run_multi_seed(n_seeds: int = 5, cfg: DissolutionConfig = None):
    """Run the full experiment across multiple seeds."""
    if cfg is None:
        cfg = DissolutionConfig(n_seeds=n_seeds)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_data')
    os.makedirs(data_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(data_dir, f'v6_{timestamp}.log')
    log = logging.getLogger(f'v6_{timestamp}')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(sh)

    log.info("=" * 70)
    log.info(f"EXPERIMENT 12 (v6): Definitive Dissolution Integration Test")
    log.info(f"Timestamp: {timestamp}")
    log.info(f"Seeds: {n_seeds}")
    log.info(f"Config: {asdict(cfg)}")
    log.info("=" * 70)

    all_results = []

    for i, seed in enumerate(range(42, 42 + n_seeds)):
        log.info(f"\n{'─' * 60}")
        log.info(f"  SEED {seed} ({i+1}/{n_seeds})")
        log.info(f"{'─' * 60}")

        exp = DissolutionExperiment(seed=seed, cfg=cfg)
        results = exp.run(log=log)
        all_results.append(results)

        # Save per-seed HDF5
        h5_path = os.path.join(data_dir, f'v6_{timestamp}_seed{seed}.h5')
        exp.save_hdf5(h5_path)
        log.info(f"  Saved: {h5_path}")

    # Cross-seed analysis
    cross = cross_seed_analysis(all_results, cfg, log)

    # Save combined results
    combined = {
        'experiment': 'v6_dissolution',
        'timestamp': timestamp,
        'config': asdict(cfg),
        'per_seed': all_results,
        'cross_seed': cross,
    }

    combined_path = os.path.join(data_dir, f'v6_{timestamp}_combined.json')
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    log.info(f"\nCombined results: {combined_path}")

    # Print final summary
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)

    for seed_result in all_results:
        seed = seed_result['seed']
        v = seed_result.get('verdict', {})
        log.info(f"  Seed {seed}: {v.get('conclusion', 'N/A')}")

    log.info(f"\n  Cross-seed SRC: {cross.get('SRC_wilcoxon', {})}")
    log.info(f"  Cross-seed CScore: {cross.get('CScore_wilcoxon', {})}")
    log.info(f"  Phase transitions: {cross.get('phase_transition', {})}")

    return combined


# ═══════════════════════════════════════════════════════════════
# Quick validation run
# ═══════════════════════════════════════════════════════════════

def quick_run():
    """Quick single-seed test with reduced parameters."""
    cfg = DissolutionConfig(
        n_intact_rounds=3,
        n_tokens_per_round=15,
        dissolution_levels=(0.0, 0.30, 0.60, 1.0),
        n_dissolution_rounds=2,
        n_recovery_rounds=2,
        n_seeds=1,
    )
    exp = DissolutionExperiment(seed=42, cfg=cfg)
    results = exp.run()

    print("\n=== QUICK RUN RESULTS ===")
    for k, v in results.items():
        if isinstance(v, dict) and 'significant' in v:
            sig = '***' if v['significant'] else '   '
            print(f"  {sig} {k}: p={v.get('p', 'N/A')}, d={v.get('d', 'N/A')}")

    return results


if __name__ == '__main__':
    import sys
    if '--quick' in sys.argv:
        quick_run()
    elif '--full' in sys.argv:
        n = int(sys.argv[sys.argv.index('--full') + 1]) if len(sys.argv) > sys.argv.index('--full') + 1 else 5
        run_multi_seed(n_seeds=n)
    else:
        print("Usage:")
        print("  python dissolution_experiment.py --quick     # Quick validation (1 seed, ~2 min)")
        print("  python dissolution_experiment.py --full 5    # Full experiment (5 seeds, ~15 min)")
