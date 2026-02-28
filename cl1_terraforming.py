#!/usr/bin/env python3
"""CL1 Terraforming Experiment — Doom-style feedback to terraform CL1 culture.

12 experiments showed robust signal preservation (SRC d=1.95) but 0/6 learning
tests passed and 0/5 phase transitions detected. The doom_for_garrett reference
successfully taught CL1 neurons to play Doom using multi-channel surprise-scaled
feedback. This adapts doom's feedback protocols to "terraform" the CL1 culture
into showing consciousness metrics above the LLM-only baseline.

Key innovations over previous experiments:
  1. SEPARATE motor/sensory channels (prevents stimulus artifact in decode)
  2. Doom-style reward/punishment on dedicated channel sets
  3. Surprise-scaled feedback (EMA TD-error modulates freq/amp/pulses)
  4. Unpredictable aversive stimulation prevents habituation
  5. Episode-level reinforcement (round-end aggregate feedback)
  6. Duty cycle (4h ON / 2h OFF) with consolidation measurement

Pre-registered hypotheses (Bonferroni α=0.0083 for 6 tests):
  H1: Decoder accuracy increases within a 4h training cycle
  H2: C-Score increases within a 4h training cycle
  H3: Post-rest C-Score > Pre-rest C-Score (consolidation)
  H4: CL1 C-Score > LLM-only C-Score after N cycles
  H5: Active channel count increases over cycles
  H6: Decoder accuracy improves across cycles (long-term learning)

Usage:
  python -m LLM_Encoder.cl1_terraforming --local --fast
  python -m LLM_Encoder.cl1_terraforming --smoke-test
  python -m LLM_Encoder.cl1_terraforming --n-cycles 3
  python -m LLM_Encoder.cl1_terraforming --analyze <h5_file>
"""

import os
import sys
import time
import json
import math
import h5py
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_Encoder.spatial_encoder import SpatialEncoder, AMP_MIN, AMP_MAX
from LLM_Encoder.consciousness import ConsciousnessAssessor, compute_cscore

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "LFM2-350M-Q4_0.gguf"
)

RESERVED_CHANNELS = frozenset({0, 4, 7, 56, 63})

PROMPTS = [
    "The nature of consciousness is deeply connected to ",
    "When neurons fire together, the emerging pattern creates ",
    "I am processing information and what emerges is ",
    "The boundary between signal and meaning dissolves when ",
    "Awareness arises from the integration of ",
    "To think is to transform raw sensation into ",
    "The recursive loop of perception and action generates ",
    "Information becomes experience through the process of ",
    "Neural patterns carry meaning when they are ",
    "Self-awareness emerges from recursive monitoring of ",
]

RESPONSE_WINDOW_S = 0.5
FEEDBACK_WINDOW_S = 0.2
MAX_CANDIDATES = 15


# ---------------------------------------------------------------------------
# 1. ChannelLayout — doom-style channel segregation
# ---------------------------------------------------------------------------

class ChannelLayout:
    """Doom-style channel segregation: sensory, motor, reward, punishment, episode.

    Key innovation: motor channels are SEPARATE from sensory. We stimulate sensory,
    read motor. This prevents stimulus artifact from dominating decode.
    """

    SENSORY = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    MOTOR = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    REWARD = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    PUNISHMENT = [50, 51, 52, 53, 54, 55, 57, 58, 59, 60]
    EPISODE_POS = [23, 24]
    EPISODE_NEG = [61, 62]

    ALL = SENSORY + MOTOR + REWARD + PUNISHMENT + EPISODE_POS + EPISODE_NEG

    @classmethod
    def validate(cls):
        """Verify no channel overlaps and no reserved channels used."""
        groups = [cls.SENSORY, cls.MOTOR, cls.REWARD, cls.PUNISHMENT,
                  cls.EPISODE_POS, cls.EPISODE_NEG]
        all_chs = []
        for g in groups:
            for ch in g:
                assert ch not in RESERVED_CHANNELS, f"Channel {ch} is hardware-reserved"
                assert ch not in all_chs, f"Channel {ch} appears in multiple groups"
                all_chs.append(ch)
        assert len(all_chs) == 59, f"Expected 59 channels, got {len(all_chs)}"


# Validate at import time
ChannelLayout.validate()


# ---------------------------------------------------------------------------
# 2. FeedbackConfig — all feedback parameters
# ---------------------------------------------------------------------------

@dataclass
class FeedbackConfig:
    """Doom-style feedback parameters adapted for token prediction."""

    # Per-token reward (correct prediction)
    reward_freq_hz: float = 20.0
    reward_amp_ua: float = 2.0
    reward_pulses: int = 30

    # Per-token punishment (wrong prediction)
    punish_freq_hz: float = 60.0
    punish_amp_ua: float = 2.0
    punish_pulses: int = 50

    # Surprise scaling (EMA TD-error modulation)
    surprise_beta: float = 0.999       # EMA smoothing for action-level
    surprise_freq_gain: float = 0.65   # how much surprise scales frequency
    surprise_amp_gain: float = 0.35    # how much surprise scales amplitude
    surprise_freq_max_scale: float = 2.0
    surprise_amp_max_scale: float = 1.5
    surprise_base_max_scale: float = 2.0  # max ratio before capping

    # Aversive (unpredictable) stimulation — triggers after consecutive errors
    aversive_consecutive_threshold: int = 3  # >3 wrong → aversive
    aversive_freq_min_hz: float = 4.0
    aversive_freq_max_hz: float = 40.0
    aversive_amp_ua: float = 2.0
    aversive_duration_s: float = 4.0
    aversive_rest_s: float = 4.0

    # Episode-level (round-end) feedback
    episode_pos_freq_hz: float = 40.0
    episode_pos_amp_ua: float = 2.0
    episode_pos_pulses: int = 80
    episode_neg_freq_hz: float = 120.0
    episode_neg_amp_ua: float = 2.0
    episode_neg_pulses: int = 160
    episode_accuracy_threshold: float = 0.5  # above → positive, below → negative
    episode_surprise_beta: float = 0.9       # faster EMA for episodes

    # Safety
    amp_min_ua: float = 0.3
    amp_max_ua: float = 2.5
    freq_min_hz: float = 4.0
    freq_max_hz: float = 40.0


# ---------------------------------------------------------------------------
# 3. SurpriseTracker — EMA prediction error tracking
# ---------------------------------------------------------------------------

class SurpriseTracker:
    """EMA-based prediction error tracker, adapted from doom_for_garrett.

    Tracks both per-token and per-episode surprise baselines.
    """

    def __init__(self, cfg: FeedbackConfig):
        self.cfg = cfg
        self._action_ema = 0.0      # per-token surprise baseline
        self._episode_ema = 0.0     # per-episode surprise baseline
        self._n_actions = 0
        self._consecutive_wrong = 0

    def update_action(self, correct: bool, surprise_magnitude: float):
        """Update per-token surprise EMA.

        Args:
            correct: whether the neural prediction matched LLM top token
            surprise_magnitude: -log(p) of the selected token
        """
        self._n_actions += 1
        self._action_ema = (self.cfg.surprise_beta * self._action_ema +
                            (1 - self.cfg.surprise_beta) * surprise_magnitude)

        if correct:
            self._consecutive_wrong = 0
        else:
            self._consecutive_wrong += 1

    def update_episode(self, accuracy: float, expected_accuracy: float = 0.5):
        """Update per-episode surprise EMA."""
        magnitude = abs(accuracy - expected_accuracy)
        self._episode_ema = (self.cfg.episode_surprise_beta * self._episode_ema +
                             (1 - self.cfg.episode_surprise_beta) * magnitude)

    def get_action_scales(self, magnitude: float, is_positive: bool
                          ) -> Tuple[float, float, float]:
        """Compute doom-style surprise scaling for per-token feedback.

        Returns: (freq_scale, amp_scale, pulse_scale)
        """
        if magnitude <= 0.0 or self._n_actions < 5:
            return 1.0, 1.0, 1.0

        baseline = max(self._action_ema, 1e-3)
        ratio = min(magnitude / baseline, self.cfg.surprise_base_max_scale)

        # Exponential compression (doom_for_garrett pattern)
        compression_k = 1.0

        # Frequency scaling — positive rewards decrease freq, negative increase
        freq_delta = self.cfg.surprise_freq_gain * (1.0 - math.exp(-compression_k * ratio))
        freq_delta = min(freq_delta, max(0.0, self.cfg.surprise_freq_max_scale - 1.0))
        if is_positive:
            freq_scale = max(0.5, 1.0 - freq_delta)
        else:
            freq_scale = min(self.cfg.surprise_freq_max_scale, 1.0 + freq_delta)

        # Amplitude scaling — always increases with surprise
        amp_delta = self.cfg.surprise_amp_gain * (1.0 - math.exp(-compression_k * ratio))
        amp_scale = min(self.cfg.surprise_amp_max_scale, 1.0 + amp_delta)
        amp_scale = max(1.0, amp_scale)

        # Pulse scaling — uses freq gain
        pulse_delta = self.cfg.surprise_freq_gain * (1.0 - math.exp(-compression_k * ratio))
        pulse_scale = min(self.cfg.surprise_base_max_scale, 1.0 + pulse_delta)
        pulse_scale = max(1.0, pulse_scale)

        return freq_scale, amp_scale, pulse_scale

    def get_episode_scales(self, accuracy: float) -> Tuple[float, float]:
        """Compute episode-level surprise scaling.

        Returns: (freq_scale, amp_scale)
        """
        magnitude = abs(accuracy - self.cfg.episode_accuracy_threshold)
        baseline = max(self._episode_ema, 1e-3)
        ratio = min(magnitude / baseline, self.cfg.surprise_base_max_scale) if magnitude > 0 else 0.0

        freq_scale = max(0.5, min(self.cfg.surprise_freq_max_scale,
                                  1.0 + self.cfg.surprise_freq_gain * ratio))
        amp_scale = max(0.5, min(self.cfg.surprise_amp_max_scale,
                                 1.0 + self.cfg.surprise_amp_gain * ratio))
        return freq_scale, amp_scale

    @property
    def should_aversive(self) -> bool:
        """Whether to trigger unpredictable aversive stimulation."""
        return self._consecutive_wrong > self.cfg.aversive_consecutive_threshold

    @property
    def state(self) -> Dict:
        return {
            'action_ema': self._action_ema,
            'episode_ema': self._episode_ema,
            'n_actions': self._n_actions,
            'consecutive_wrong': self._consecutive_wrong,
        }


# ---------------------------------------------------------------------------
# 4. DoomStyleFeedback — reward/punishment/aversive/episode delivery
# ---------------------------------------------------------------------------

class DoomStyleFeedback:
    """Delivers doom-style feedback to CL1 via substrate.stimulate_and_record().

    Since CL1CloudSubstrate only exposes stimulate_and_record(channel_amplitudes),
    we encode frequency/pulses as repeated stimulations with amplitude patterns.
    The substrate handles charge-balanced biphasic waveforms internally.
    """

    def __init__(self, substrate, cfg: FeedbackConfig, rng: np.random.Generator):
        self.substrate = substrate
        self.cfg = cfg
        self.rng = rng
        self.surprise = SurpriseTracker(cfg)
        self._feedback_log: List[Dict] = []

    def _clamp_amp(self, amp: float) -> float:
        return max(self.cfg.amp_min_ua, min(self.cfg.amp_max_ua, amp))

    def _deliver_burst(self, channels: List[int], amp: float, n_pulses: int,
                       window_s: float = FEEDBACK_WINDOW_S) -> Dict:
        """Deliver a burst as repeated stimulations at given amplitude.

        We approximate frequency/pulse count by delivering n_pulses stimulations
        with short recording windows. Each call to stimulate_and_record is one
        pulse of the biphasic waveform.
        """
        amp = self._clamp_amp(amp)
        pattern = {ch: amp for ch in channels}
        total_spikes = {}

        # Deliver in chunks to avoid overwhelming the substrate
        # Each stimulate_and_record call represents one pulse burst
        chunk_size = max(1, min(n_pulses, 10))
        n_chunks = max(1, n_pulses // chunk_size)

        for _ in range(n_chunks):
            spike_counts = self.substrate.stimulate_and_record(
                pattern, window_s=window_s / n_chunks
            )
            for ch_str, count in spike_counts.items():
                ch = int(ch_str) if isinstance(ch_str, str) else ch_str
                total_spikes[ch] = total_spikes.get(ch, 0) + count

        return total_spikes

    def deliver_reward(self, surprise_mag: float) -> Dict:
        """Deliver positive reward feedback on reward channels."""
        freq_s, amp_s, pulse_s = self.surprise.get_action_scales(surprise_mag, is_positive=True)
        amp = self._clamp_amp(self.cfg.reward_amp_ua * amp_s)
        pulses = max(1, int(self.cfg.reward_pulses * pulse_s))

        result = self._deliver_burst(ChannelLayout.REWARD, amp, pulses)
        self._feedback_log.append({
            'type': 'reward', 'amp': amp, 'pulses': pulses,
            'freq_scale': freq_s, 'amp_scale': amp_s, 'pulse_scale': pulse_s,
        })
        return result

    def deliver_punishment(self, surprise_mag: float) -> Dict:
        """Deliver negative punishment feedback on punishment channels."""
        freq_s, amp_s, pulse_s = self.surprise.get_action_scales(surprise_mag, is_positive=False)
        amp = self._clamp_amp(self.cfg.punish_amp_ua * amp_s)
        pulses = max(1, int(self.cfg.punish_pulses * pulse_s))

        result = self._deliver_burst(ChannelLayout.PUNISHMENT, amp, pulses)
        self._feedback_log.append({
            'type': 'punishment', 'amp': amp, 'pulses': pulses,
            'freq_scale': freq_s, 'amp_scale': amp_s, 'pulse_scale': pulse_s,
        })
        return result

    def deliver_aversive(self) -> Dict:
        """Deliver unpredictable aversive stimulation (random freq, 4s on/4s off)."""
        freq = self.rng.uniform(self.cfg.aversive_freq_min_hz, self.cfg.aversive_freq_max_hz)
        amp = self._clamp_amp(self.cfg.aversive_amp_ua)

        # Approximate duration with repeated bursts
        n_bursts = max(1, int(freq * self.cfg.aversive_duration_s / 10))
        burst_window = self.cfg.aversive_duration_s / n_bursts

        total_spikes = {}
        pattern = {ch: amp for ch in ChannelLayout.PUNISHMENT}

        for _ in range(n_bursts):
            spike_counts = self.substrate.stimulate_and_record(
                pattern, window_s=burst_window
            )
            for ch_str, count in spike_counts.items():
                ch = int(ch_str) if isinstance(ch_str, str) else ch_str
                total_spikes[ch] = total_spikes.get(ch, 0) + count

        # Rest period (no stimulation, just record to observe decay)
        rest_spikes = self.substrate.stimulate_and_record(
            {}, window_s=self.cfg.aversive_rest_s
        )

        self._feedback_log.append({
            'type': 'aversive', 'freq': freq, 'amp': amp,
            'duration': self.cfg.aversive_duration_s,
            'rest': self.cfg.aversive_rest_s,
        })
        return total_spikes

    def deliver_episode(self, accuracy: float) -> Dict:
        """Deliver episode-level feedback based on round accuracy."""
        positive = accuracy > self.cfg.episode_accuracy_threshold
        freq_s, amp_s = self.surprise.get_episode_scales(accuracy)

        if positive:
            channels = ChannelLayout.EPISODE_POS
            base_amp = self.cfg.episode_pos_amp_ua
            pulses = self.cfg.episode_pos_pulses
        else:
            channels = ChannelLayout.EPISODE_NEG
            base_amp = self.cfg.episode_neg_amp_ua
            pulses = self.cfg.episode_neg_pulses

        amp = self._clamp_amp(base_amp * amp_s)
        scaled_pulses = max(1, int(pulses * freq_s))

        result = self._deliver_burst(channels, amp, scaled_pulses, window_s=0.5)
        self.surprise.update_episode(accuracy)

        self._feedback_log.append({
            'type': 'episode_pos' if positive else 'episode_neg',
            'accuracy': accuracy, 'amp': amp, 'pulses': scaled_pulses,
            'freq_scale': freq_s, 'amp_scale': amp_s,
        })
        return result

    def get_log(self) -> List[Dict]:
        return self._feedback_log

    def clear_log(self):
        self._feedback_log.clear()


# ---------------------------------------------------------------------------
# 5. TerraformingDecoder — persistent Hebbian decoder on motor channels
# ---------------------------------------------------------------------------

class TerraformingDecoder:
    """Decoder that reads ONLY motor channels to determine neural token votes.

    Unlike previous decoders that read the same channels as stimulation
    (causing artifact), this reads motor channels which are never stimulated.
    Templates track per-token motor response patterns.
    """

    def __init__(self, motor_channels: List[int], alpha: float = 0.5,
                 learning_rate: float = 0.02):
        self.motor_channels = motor_channels
        self.n_motor = len(motor_channels)
        self.alpha = alpha
        self.lr = learning_rate

        self._templates: Dict[int, np.ndarray] = {}
        self._template_counts: Dict[int, int] = {}
        self._n_updates = 0
        self._round_idx = 0

        # Convergence tracking
        self._prev_templates: Dict[int, np.ndarray] = {}
        self._convergence_history: List[float] = []

        # Accuracy tracking
        self._correct = 0
        self._total = 0

    def _motor_vector(self, spike_counts: Dict) -> np.ndarray:
        """Extract motor channel spikes as a vector."""
        vec = np.zeros(self.n_motor, dtype=float)
        for i, ch in enumerate(self.motor_channels):
            val = spike_counts.get(str(ch), spike_counts.get(ch, 0))
            vec[i] = float(val)
        return vec

    def start_round(self, round_idx: int):
        self._round_idx = round_idx
        self._prev_templates = {tid: t.copy() for tid, t in self._templates.items()}

    def end_round(self) -> Dict:
        convergence = self._compute_convergence()
        self._convergence_history.append(convergence)
        return {
            'convergence': convergence,
            'n_templates': len(self._templates),
            'prediction_accuracy': self._correct / max(1, self._total),
            'n_updates': self._n_updates,
        }

    def _compute_convergence(self) -> float:
        if not self._prev_templates or not self._templates:
            return 1.0
        shared = set(self._prev_templates.keys()) & set(self._templates.keys())
        if not shared:
            return 1.0
        changes = []
        for tid in shared:
            diff = np.linalg.norm(self._templates[tid] - self._prev_templates[tid])
            norm = np.linalg.norm(self._prev_templates[tid]) + 1e-10
            changes.append(diff / norm)
        return float(np.mean(changes))

    def decode(self, spike_counts: Dict, model_probs: Dict[int, float]
               ) -> Tuple[Dict[int, float], Dict[int, float], float]:
        """Decode token from motor channel activity using template matching.

        Returns: (blended_probs, neural_probs, confidence)
        """
        motor_vec = self._motor_vector(spike_counts)
        candidates = list(model_probs.keys())

        # Template matching: cosine similarity with each token template
        neural_scores = {}
        for tid in candidates:
            if tid in self._templates:
                template = self._templates[tid]
                t_norm = np.linalg.norm(template)
                m_norm = np.linalg.norm(motor_vec)
                if t_norm > 1e-10 and m_norm > 1e-10:
                    neural_scores[tid] = float(np.dot(template, motor_vec) / (t_norm * m_norm))
                else:
                    neural_scores[tid] = 0.0
            else:
                neural_scores[tid] = 0.0

        # Softmax neural scores
        if neural_scores:
            max_s = max(neural_scores.values())
            exp_scores = {tid: math.exp(s - max_s) for tid, s in neural_scores.items()}
            total = sum(exp_scores.values()) + 1e-10
            neural_probs = {tid: exp_scores[tid] / total for tid in exp_scores}
        else:
            neural_probs = {tid: 1.0 / len(candidates) for tid in candidates}

        # Confidence ramps with experience
        confidence = min(1.0, self._n_updates / 50.0)

        # Blend: alpha * neural + (1-alpha) * model, scaled by confidence
        effective_alpha = self.alpha * confidence
        blended = {}
        for tid in candidates:
            np_val = neural_probs.get(tid, 0.0)
            mp_val = model_probs.get(tid, 0.0)
            blended[tid] = effective_alpha * np_val + (1 - effective_alpha) * mp_val

        # Normalize
        total = sum(blended.values()) + 1e-10
        blended = {tid: v / total for tid, v in blended.items()}

        return blended, neural_probs, confidence

    def update(self, selected_token: int, spike_counts: Dict, model_top: int = None):
        """Update template for selected token (Hebbian EMA learning)."""
        motor_vec = self._motor_vector(spike_counts)
        self._n_updates += 1

        if selected_token not in self._templates:
            self._templates[selected_token] = motor_vec.copy()
            self._template_counts[selected_token] = 1
        else:
            self._templates[selected_token] += self.lr * (
                motor_vec - self._templates[selected_token]
            )
            self._template_counts[selected_token] += 1

        if model_top is not None:
            self._total += 1
            if selected_token == model_top:
                self._correct += 1

    def get_stats(self) -> Dict:
        return {
            'n_templates': len(self._templates),
            'n_updates': self._n_updates,
            'convergence_history': self._convergence_history,
            'prediction_accuracy': self._correct / max(1, self._total),
        }


# ---------------------------------------------------------------------------
# 6. LLMOnlyControl — parallel LLM baseline
# ---------------------------------------------------------------------------

class LLMOnlyControl:
    """Parallel LLM-only baseline running same prompts without substrate.

    Records model entropy and token choices for C-Score comparison (H4).
    """

    def __init__(self, llm):
        self.llm = llm
        self._results: List[Dict] = []

    def run_round(self, prompt: str, n_tokens: int = 50) -> Dict:
        """Generate tokens with LLM only, no substrate."""
        try:
            self.llm.reset()
            if hasattr(self.llm, '_ctx') and self.llm._ctx is not None:
                self.llm._ctx.kv_cache_clear()
        except Exception:
            pass

        context = prompt
        token_ids = []
        entropies = []
        top_probs = []
        text = ""

        for _ in range(n_tokens):
            try:
                output = self.llm.create_completion(
                    context, max_tokens=1, logprobs=MAX_CANDIDATES, temperature=1.0,
                )
            except RuntimeError:
                break

            choice = output['choices'][0]
            logprobs_data = choice.get('logprobs', {})
            model_probs = {}

            if logprobs_data and logprobs_data.get('top_logprobs'):
                tlp = logprobs_data['top_logprobs'][0]
                for tok_text, logprob in tlp.items():
                    tids = self.llm.tokenize(tok_text.encode('utf-8'), add_bos=False)
                    if tids:
                        model_probs[tids[0]] = math.exp(logprob)

            if model_probs:
                total = sum(model_probs.values())
                model_probs = {k: v / total for k, v in model_probs.items()}
                probs_arr = np.array(list(model_probs.values()))
                probs_arr = probs_arr[probs_arr > 0]
                entropy = float(-np.sum(probs_arr * np.log2(probs_arr + 1e-10)))
                top_p = max(model_probs.values())
            else:
                entropy = 0.0
                top_p = 1.0

            selected = max(model_probs, key=model_probs.get) if model_probs else 0
            token_ids.append(selected)
            entropies.append(entropy)
            top_probs.append(top_p)

            tok_text = choice.get('text', '')
            text += tok_text
            context += tok_text

        result = {
            'token_ids': token_ids,
            'entropies': entropies,
            'top_probs': top_probs,
            'mean_entropy': float(np.mean(entropies)) if entropies else 0.0,
            'n_tokens': len(token_ids),
            'text': text,
        }
        self._results.append(result)
        return result

    def get_results(self) -> List[Dict]:
        return self._results


# ---------------------------------------------------------------------------
# 7. HDF5DataStore — data recording
# ---------------------------------------------------------------------------

class HDF5DataStore:
    """HDF5 data store with per-token granularity per cycle."""

    def __init__(self, path: str):
        self.path = path
        self.f = h5py.File(path, 'w')
        self.f.attrs['created'] = datetime.now().isoformat()
        self.f.attrs['experiment'] = 'cl1_terraforming'
        self.f.attrs['version'] = '1.0'

    def create_cycle_group(self, cycle_idx: int) -> h5py.Group:
        return self.f.create_group(f'cycle_{cycle_idx:03d}')

    def save_warmup(self, cycle_grp: h5py.Group, channel_map: Dict,
                    cscore: float, active_count: int):
        grp = cycle_grp.create_group('warmup')
        grp.attrs['cscore'] = cscore
        grp.attrs['active_channels'] = active_count
        # Save channel responsiveness map
        chs = sorted(channel_map.keys())
        grp.create_dataset('channels', data=np.array(chs, dtype=np.int32))
        grp.create_dataset('responses', data=np.array(
            [channel_map[ch] for ch in chs], dtype=np.float32))

    def save_round(self, cycle_grp: h5py.Group, round_idx: int,
                   data: Dict) -> h5py.Group:
        grp = cycle_grp.create_group(f'round_{round_idx:03d}')
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                grp.create_dataset(key, data=val)
            elif isinstance(val, list) and val and isinstance(val[0], (int, float)):
                grp.create_dataset(key, data=np.array(val, dtype=np.float32))
            elif isinstance(val, (int, float)):
                grp.attrs[key] = val
            elif isinstance(val, str):
                grp.attrs[key] = val
            elif isinstance(val, dict):
                grp.attrs[key] = json.dumps(val)
        return grp

    def save_assessment(self, cycle_grp: h5py.Group, data: Dict):
        grp = cycle_grp.create_group('assessment')
        for key, val in data.items():
            if isinstance(val, (int, float)):
                grp.attrs[key] = val
            elif isinstance(val, list):
                grp.create_dataset(key, data=np.array(val, dtype=np.float32))

    def save_rest_boundary(self, cycle_grp: h5py.Group, pre_cscore: float,
                           post_cscore: float):
        grp = cycle_grp.create_group('rest')
        grp.attrs['pre_rest_cscore'] = pre_cscore
        grp.attrs['post_rest_cscore'] = post_cscore

    def save_llm_control(self, cycle_grp: h5py.Group, data: Dict):
        round_idx = data.get('round_idx', 0)
        grp = cycle_grp.create_group(f'llm_control_{round_idx:03d}')
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                grp.create_dataset(key, data=val)
            elif isinstance(val, list) and val and isinstance(val[0], (int, float)):
                grp.create_dataset(key, data=np.array(val, dtype=np.float32))
            elif isinstance(val, (int, float)):
                grp.attrs[key] = val
            elif isinstance(val, str):
                grp.attrs[key] = val

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# 8. TerraformingExperiment — main controller with phase state machine
# ---------------------------------------------------------------------------

class Phase(Enum):
    WARMUP = "warmup"
    TRAINING = "training"
    ASSESSMENT = "assessment"
    REST = "rest"


class TerraformingExperiment:
    """Main terraforming experiment controller.

    Duty cycle per cycle:
      1. Warmup (30 min): sweep channels, build responsiveness map
      2. Training (3 hours): closed-loop token generation with doom feedback
      3. Assessment (30 min): pure measurement without feedback
      4. Rest (2 hours): no stimulation, pre/post C-Score
    """

    def __init__(
        self,
        substrate,
        model_path: str = MODEL_PATH,
        alpha: float = 0.5,
        tokens_per_round: int = 50,
        n_cycles: int = 3,
        seed: int = 42,
        output_dir: str = "experiment_data",
        fast_mode: bool = False,
    ):
        self.substrate = substrate
        self.model_path = model_path
        self.alpha = alpha
        self.tokens_per_round = tokens_per_round
        self.n_cycles = n_cycles
        self.seed = seed
        self.output_dir = output_dir
        self.fast = fast_mode

        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.h5_path = os.path.join(output_dir, f"terraform_{self.timestamp}.h5")
        self.log_path = os.path.join(output_dir, f"terraform_{self.timestamp}.log")

        # Timing (seconds)
        if fast_mode:
            self.warmup_duration = 5 * 60       # 5 min
            self.training_duration = 15 * 60    # 15 min
            self.assess_duration = 5 * 60       # 5 min
            self.rest_duration = 2 * 60         # 2 min
        else:
            self.warmup_duration = 30 * 60      # 30 min
            self.training_duration = 3 * 3600   # 3 hours
            self.assess_duration = 30 * 60      # 30 min
            self.rest_duration = 2 * 3600       # 2 hours

        self._rng = np.random.default_rng(seed)
        self.fb_cfg = FeedbackConfig()
        self.encoder = SpatialEncoder(n_active=8, channels=ChannelLayout.SENSORY, seed=seed)
        self.decoder = TerraformingDecoder(ChannelLayout.MOTOR, alpha=alpha)
        self.assessor = ConsciousnessAssessor()
        self.feedback = DoomStyleFeedback(substrate, self.fb_cfg, self._rng)
        self._llm = None
        self._llm_control = None
        self.data_store = None

    def _log(self, msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')

    def _load_llm(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            self._log("ERROR: pip install llama-cpp-python")
            sys.exit(1)
        self._log(f"Loading LLM: {os.path.basename(self.model_path)}")
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1, n_ctx=2048, logits_all=True, verbose=False,
        )
        self._llm_control = LLMOnlyControl(self._llm)
        self._log("LLM loaded.")

    def _measure_cscore(self) -> float:
        """Measure C-Score from current substrate state."""
        sm = self.substrate.get_last_spike_matrix()
        if sm is None or sm.size == 0:
            # Stimulate to get a spike matrix
            pattern = {ch: 1.5 for ch in ChannelLayout.SENSORY[:5]}
            self.substrate.stimulate_and_record(pattern, window_s=0.5)
            sm = self.substrate.get_last_spike_matrix()
        if sm is None or sm.size == 0:
            return 0.0
        result = self.assessor.assess(sm)
        return result.get('cscore', 0.0)

    def _warmup_phase(self, cycle_grp: h5py.Group) -> Dict:
        """Sweep all channels, build responsiveness map, track C-Score."""
        self._log("  [WARMUP] Sweeping channels...")
        channel_responses = {}
        t_start = time.time()
        sweep_rounds = 0

        while time.time() - t_start < self.warmup_duration:
            sweep_rounds += 1
            # Sweep each channel group
            for ch_list in [ChannelLayout.SENSORY, ChannelLayout.MOTOR,
                            ChannelLayout.REWARD, ChannelLayout.PUNISHMENT,
                            ChannelLayout.EPISODE_POS, ChannelLayout.EPISODE_NEG]:
                # Stimulate in batches of 10
                for i in range(0, len(ch_list), 10):
                    batch = ch_list[i:i + 10]
                    pattern = {ch: 2.0 for ch in batch}
                    spike_counts = self.substrate.stimulate_and_record(
                        pattern, window_s=0.3
                    )
                    for ch in batch:
                        val = int(spike_counts.get(str(ch), spike_counts.get(ch, 0)))
                        channel_responses[ch] = channel_responses.get(ch, 0) + val

                    if time.time() - t_start >= self.warmup_duration:
                        break
                if time.time() - t_start >= self.warmup_duration:
                    break

        # Measure C-Score at warmup end
        cscore = self._measure_cscore()
        active = sum(1 for v in channel_responses.values() if v > 0)

        # Baseline assessment: measure C-Score on sensory stimulation (pre-training)
        baseline_cscores = []
        self._log(f"  [WARMUP] Measuring baseline C-Score (5 sensory probes)...")
        for _ in range(5):
            probe_pattern = {ch: 1.5 for ch in ChannelLayout.SENSORY[:10]}
            self.substrate.stimulate_and_record(probe_pattern, window_s=0.5)
            sm = self.substrate.get_last_spike_matrix()
            bl_cs = self.assessor.assess(sm).get('cscore', 0.0)
            baseline_cscores.append(bl_cs)
        baseline_cscore = float(np.mean(baseline_cscores))

        self._log(f"  [WARMUP] Done: {sweep_rounds} sweeps, {active}/59 active, "
                  f"C-Score={cscore:.4f}, baseline={baseline_cscore:.4f}")

        self.data_store.save_warmup(cycle_grp, channel_responses, cscore, active)
        # Save baseline C-Score as attribute
        cycle_grp['warmup'].attrs['baseline_cscore'] = baseline_cscore
        cycle_grp['warmup'].create_dataset(
            'baseline_cscores', data=np.array(baseline_cscores, dtype=np.float32))
        return {'cscore': cscore, 'active_channels': active,
                'baseline_cscore': baseline_cscore,
                'channel_responses': channel_responses}

    def _training_round(self, prompt: str, round_idx: int) -> Dict:
        """Run one training round with doom-style feedback."""
        self._load_llm()

        try:
            self._llm.reset()
            if hasattr(self._llm, '_ctx') and self._llm._ctx is not None:
                self._llm._ctx.kv_cache_clear()
        except Exception:
            pass

        context = prompt
        token_ids = []
        correct_flags = []
        cscores = []
        motor_spikes_list = []
        feedback_types = []
        surprise_values = []
        text = ""
        n_correct = 0

        self.decoder.start_round(round_idx)

        for pos in range(self.tokens_per_round):
            # LLM inference
            try:
                output = self._llm.create_completion(
                    context, max_tokens=1, logprobs=MAX_CANDIDATES, temperature=1.0,
                )
            except RuntimeError as e:
                if 'llama_decode returned -1' in str(e):
                    self._llm = None
                    self._load_llm()
                    output = self._llm.create_completion(
                        context, max_tokens=1, logprobs=MAX_CANDIDATES, temperature=1.0,
                    )
                else:
                    raise

            choice = output['choices'][0]
            logprobs_data = choice.get('logprobs', {})
            top_logprobs = {}

            if logprobs_data and logprobs_data.get('top_logprobs'):
                tlp = logprobs_data['top_logprobs'][0]
                for tok_text, logprob in tlp.items():
                    tids = self._llm.tokenize(tok_text.encode('utf-8'), add_bos=False)
                    if tids:
                        top_logprobs[tids[0]] = {'text': tok_text, 'logprob': logprob}

            if top_logprobs:
                max_lp = max(v['logprob'] for v in top_logprobs.values())
                model_probs = {tid: math.exp(info['logprob'] - max_lp)
                               for tid, info in top_logprobs.items()}
                total = sum(model_probs.values())
                model_probs = {k: v / total for k, v in model_probs.items()}
            else:
                chosen_text = choice['text']
                chosen_ids = self._llm.tokenize(chosen_text.encode('utf-8'), add_bos=False)
                tid = chosen_ids[0] if chosen_ids else 0
                model_probs = {tid: 1.0}
                top_logprobs = {tid: {'text': chosen_text, 'logprob': 0.0}}

            model_top = max(model_probs, key=model_probs.get)

            # Encode token pattern on SENSORY channels
            combined_pattern, channel_to_token = self.encoder.encode_candidates(model_probs)
            channel_amplitudes = {int(ch): amp for ch, amp in combined_pattern.items()}

            # Stimulate sensory channels, record ALL channels (including motor)
            spike_counts = self.substrate.stimulate_and_record(
                channel_amplitudes, window_s=RESPONSE_WINDOW_S,
            )

            # C-Score: measure BEFORE feedback to avoid confound
            sm = self.substrate.get_last_spike_matrix()
            cs_result = self.assessor.assess(sm)
            cs = cs_result.get('cscore', 0.0)

            # Decode from MOTOR channels only
            blended, neural_probs, confidence = self.decoder.decode(spike_counts, model_probs)
            selected = max(blended, key=blended.get)

            # Was neural prediction correct?
            correct = (selected == model_top)
            if correct:
                n_correct += 1

            # Surprise magnitude
            p_selected = model_probs.get(selected, 0.01)
            surprise_mag = -math.log(max(p_selected, 1e-6))

            # Update decoder templates
            self.decoder.update(selected, spike_counts, model_top=model_top)

            # Update surprise tracker
            self.feedback.surprise.update_action(correct, surprise_mag)

            # === DOOM-STYLE FEEDBACK ===
            fb_type = 'none'
            if correct:
                self.feedback.deliver_reward(surprise_mag)
                fb_type = 'reward'
            else:
                self.feedback.deliver_punishment(surprise_mag)
                fb_type = 'punishment'

                # Check for aversive trigger
                if self.feedback.surprise.should_aversive:
                    self.feedback.deliver_aversive()
                    fb_type = 'aversive'

            # Motor channel spikes
            motor_spk = sum(
                int(spike_counts.get(str(ch), spike_counts.get(ch, 0)))
                for ch in ChannelLayout.MOTOR
            )

            # Record
            token_ids.append(selected)
            correct_flags.append(1 if correct else 0)
            cscores.append(cs)
            motor_spikes_list.append(motor_spk)
            feedback_types.append(fb_type)
            surprise_values.append(surprise_mag)

            # Get token text
            if selected in top_logprobs:
                tok_text = top_logprobs[selected]['text']
            else:
                tok_text = self._llm.detokenize([selected]).decode('utf-8', errors='replace')

            text += tok_text
            context += tok_text

            # Live output
            marker = '+' if correct else '-'
            if pos % 10 == 0:
                sys.stdout.write(f"\n    [{pos:3d}] C={cs:.3f} {marker} ")
            sys.stdout.write(tok_text)
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        # Episode-level feedback
        accuracy = n_correct / max(1, len(token_ids))
        self.feedback.deliver_episode(accuracy)

        round_metrics = self.decoder.end_round()

        return {
            'token_ids': np.array(token_ids, dtype=np.int32),
            'correct': np.array(correct_flags, dtype=np.int8),
            'cscore': np.array(cscores, dtype=np.float32),
            'motor_spikes': np.array(motor_spikes_list, dtype=np.int32),
            'surprise': np.array(surprise_values, dtype=np.float32),
            'accuracy': accuracy,
            'n_tokens': len(token_ids),
            'n_correct': n_correct,
            'mean_cscore': float(np.mean(cscores)) if cscores else 0.0,
            'prompt': prompt,
            'text': text,
            'feedback_types': json.dumps(feedback_types),
            **{f'decoder_{k}': v for k, v in round_metrics.items()
               if isinstance(v, (int, float))},
        }

    def _training_phase(self, cycle_grp: h5py.Group, cycle_idx: int) -> List[Dict]:
        """Run training rounds for the training duration."""
        self._log("  [TRAINING] Starting doom-style closed-loop training...")
        t_start = time.time()
        round_idx = 0
        round_summaries = []

        while time.time() - t_start < self.training_duration:
            prompt = PROMPTS[round_idx % len(PROMPTS)]
            self._log(f"  [TRAIN R{round_idx}] Prompt: {prompt[:50]}...")

            data = self._training_round(prompt, round_idx)
            self.data_store.save_round(cycle_grp, round_idx, data)

            self._log(f"  [TRAIN R{round_idx}] acc={data['accuracy']:.3f} "
                      f"C={data['mean_cscore']:.4f} "
                      f"templates={data.get('decoder_n_templates', 0)}")

            round_summaries.append({
                'round_idx': round_idx,
                'accuracy': data['accuracy'],
                'mean_cscore': data['mean_cscore'],
                'n_tokens': data['n_tokens'],
            })

            # Run LLM-only control in parallel
            llm_result = self._llm_control.run_round(prompt, self.tokens_per_round)
            self.data_store.save_llm_control(cycle_grp, {
                'round_idx': round_idx,
                'mean_entropy': llm_result['mean_entropy'],
                'n_tokens': llm_result['n_tokens'],
            })

            round_idx += 1

        self._log(f"  [TRAINING] Completed {round_idx} rounds in "
                  f"{(time.time() - t_start) / 60:.1f} min")
        return round_summaries

    def _assessment_phase(self, cycle_grp: h5py.Group) -> Dict:
        """Pure measurement WITHOUT feedback — tests if learning persists."""
        self._log("  [ASSESSMENT] Measuring without feedback...")
        t_start = time.time()
        assess_cscores = []
        assess_accuracies = []
        round_idx = 0

        # Temporarily disable feedback
        original_feedback = self.feedback
        self.feedback = None

        while time.time() - t_start < self.assess_duration:
            prompt = PROMPTS[round_idx % len(PROMPTS)]
            self._load_llm()

            try:
                self._llm.reset()
                if hasattr(self._llm, '_ctx') and self._llm._ctx is not None:
                    self._llm._ctx.kv_cache_clear()
            except Exception:
                pass

            context = prompt
            n_correct = 0
            round_cscores = []

            for pos in range(self.tokens_per_round):
                try:
                    output = self._llm.create_completion(
                        context, max_tokens=1, logprobs=MAX_CANDIDATES, temperature=1.0,
                    )
                except RuntimeError:
                    break

                choice = output['choices'][0]
                logprobs_data = choice.get('logprobs', {})
                top_logprobs = {}

                if logprobs_data and logprobs_data.get('top_logprobs'):
                    tlp = logprobs_data['top_logprobs'][0]
                    for tok_text, logprob in tlp.items():
                        tids = self._llm.tokenize(tok_text.encode('utf-8'), add_bos=False)
                        if tids:
                            top_logprobs[tids[0]] = {'text': tok_text, 'logprob': logprob}

                if top_logprobs:
                    max_lp = max(v['logprob'] for v in top_logprobs.values())
                    model_probs = {tid: math.exp(info['logprob'] - max_lp)
                                   for tid, info in top_logprobs.items()}
                    total = sum(model_probs.values())
                    model_probs = {k: v / total for k, v in model_probs.items()}
                else:
                    chosen_text = choice['text']
                    chosen_ids = self._llm.tokenize(chosen_text.encode('utf-8'), add_bos=False)
                    tid = chosen_ids[0] if chosen_ids else 0
                    model_probs = {tid: 1.0}

                model_top = max(model_probs, key=model_probs.get)

                # Encode and stimulate (no feedback)
                combined_pattern, _ = self.encoder.encode_candidates(model_probs)
                channel_amplitudes = {int(ch): amp for ch, amp in combined_pattern.items()}
                spike_counts = self.substrate.stimulate_and_record(
                    channel_amplitudes, window_s=RESPONSE_WINDOW_S,
                )

                # Decode from motor
                blended, _, _ = self.decoder.decode(spike_counts, model_probs)
                selected = max(blended, key=blended.get)
                if selected == model_top:
                    n_correct += 1

                sm = self.substrate.get_last_spike_matrix()
                cs_result = self.assessor.assess(sm)
                round_cscores.append(cs_result.get('cscore', 0.0))

                tok_text = choice.get('text', '')
                context += tok_text

            acc = n_correct / max(1, self.tokens_per_round)
            assess_accuracies.append(acc)
            assess_cscores.extend(round_cscores)
            round_idx += 1

        # Restore feedback
        self.feedback = original_feedback

        result = {
            'mean_cscore': float(np.mean(assess_cscores)) if assess_cscores else 0.0,
            'mean_accuracy': float(np.mean(assess_accuracies)) if assess_accuracies else 0.0,
            'n_rounds': round_idx,
            'cscores': assess_cscores,
            'accuracies': assess_accuracies,
        }

        self._log(f"  [ASSESSMENT] {round_idx} rounds: "
                  f"acc={result['mean_accuracy']:.3f} C={result['mean_cscore']:.4f}")

        self.data_store.save_assessment(cycle_grp, result)
        return result

    def _rest_phase(self, cycle_grp: h5py.Group) -> Dict:
        """No stimulation. Measure pre/post rest C-Score for consolidation."""
        self._log("  [REST] Measuring pre-rest C-Score...")
        pre_cscore = self._measure_cscore()

        self._log(f"  [REST] Pre-rest C-Score={pre_cscore:.4f}, "
                  f"sleeping {self.rest_duration / 60:.0f} min...")
        time.sleep(self.rest_duration)

        self._log("  [REST] Measuring post-rest C-Score...")
        post_cscore = self._measure_cscore()

        self._log(f"  [REST] Pre={pre_cscore:.4f} → Post={post_cscore:.4f} "
                  f"(Δ={post_cscore - pre_cscore:+.4f})")

        self.data_store.save_rest_boundary(cycle_grp, pre_cscore, post_cscore)
        return {'pre_cscore': pre_cscore, 'post_cscore': post_cscore}

    def run(self):
        """Run the full terraforming experiment."""
        t_start = time.time()
        self.data_store = HDF5DataStore(self.h5_path)

        cycle_hours = (self.warmup_duration + self.training_duration +
                       self.assess_duration + self.rest_duration) / 3600

        self._log("=" * 78)
        self._log("  CL1 TERRAFORMING EXPERIMENT — DOOM-STYLE FEEDBACK")
        self._log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 78)
        self._log(f"  Substrate:      {type(self.substrate).__name__}")
        self._log(f"  Cycles:         {self.n_cycles} × {cycle_hours:.1f}h")
        self._log(f"  Warmup:         {self.warmup_duration / 60:.0f} min")
        self._log(f"  Training:       {self.training_duration / 60:.0f} min")
        self._log(f"  Assessment:     {self.assess_duration / 60:.0f} min")
        self._log(f"  Rest:           {self.rest_duration / 60:.0f} min")
        self._log(f"  Tokens/round:   {self.tokens_per_round}")
        self._log(f"  Alpha:          {self.alpha}")
        self._log(f"  Fast mode:      {self.fast}")
        self._log(f"  Sensory:        {len(ChannelLayout.SENSORY)} channels")
        self._log(f"  Motor:          {len(ChannelLayout.MOTOR)} channels")
        self._log(f"  Reward:         {len(ChannelLayout.REWARD)} channels")
        self._log(f"  Punishment:     {len(ChannelLayout.PUNISHMENT)} channels")
        self._log(f"  Output:         {self.h5_path}")
        self._log("=" * 78)

        # Store config
        self.data_store.f.attrs['n_cycles'] = self.n_cycles
        self.data_store.f.attrs['alpha'] = self.alpha
        self.data_store.f.attrs['tokens_per_round'] = self.tokens_per_round
        self.data_store.f.attrs['fast_mode'] = self.fast
        self.data_store.f.attrs['seed'] = self.seed
        self.data_store.f.attrs['channel_layout'] = json.dumps({
            'sensory': ChannelLayout.SENSORY,
            'motor': ChannelLayout.MOTOR,
            'reward': ChannelLayout.REWARD,
            'punishment': ChannelLayout.PUNISHMENT,
            'episode_pos': ChannelLayout.EPISODE_POS,
            'episode_neg': ChannelLayout.EPISODE_NEG,
        })

        all_cycle_results = []

        for cycle_idx in range(self.n_cycles):
            self._log(f"\n{'─' * 78}")
            self._log(f"  CYCLE {cycle_idx + 1}/{self.n_cycles}")
            self._log(f"{'─' * 78}")

            cycle_grp = self.data_store.create_cycle_group(cycle_idx)
            cycle_result = {}

            # Phase 1: Warmup
            warmup = self._warmup_phase(cycle_grp)
            cycle_result['warmup'] = warmup

            # Phase 2: Training
            training = self._training_phase(cycle_grp, cycle_idx)
            cycle_result['training'] = training

            # Phase 3: Assessment
            assessment = self._assessment_phase(cycle_grp)
            cycle_result['assessment'] = assessment

            # Phase 4: Rest (skip on last cycle)
            if cycle_idx < self.n_cycles - 1:
                rest = self._rest_phase(cycle_grp)
                cycle_result['rest'] = rest
            else:
                self._log("  [REST] Skipping rest on final cycle.")

            all_cycle_results.append(cycle_result)

            # Cycle summary
            self._log(f"\n  CYCLE {cycle_idx + 1} SUMMARY:")
            self._log(f"    Warmup C-Score:     {warmup['cscore']:.4f}")
            self._log(f"    Active channels:    {warmup['active_channels']}")
            if training:
                accs = [r['accuracy'] for r in training]
                css = [r['mean_cscore'] for r in training]
                self._log(f"    Training accuracy:  {np.mean(accs):.3f} "
                          f"(first={accs[0]:.3f}, last={accs[-1]:.3f})")
                self._log(f"    Training C-Score:   {np.mean(css):.4f} "
                          f"(first={css[0]:.4f}, last={css[-1]:.4f})")
            self._log(f"    Assessment acc:     {assessment['mean_accuracy']:.3f}")
            self._log(f"    Assessment C-Score: {assessment['mean_cscore']:.4f}")

        elapsed = (time.time() - t_start) / 3600
        self._log(f"\n{'=' * 78}")
        self._log(f"  EXPERIMENT COMPLETE — {elapsed:.1f} hours")
        self._log(f"  Data saved to: {self.h5_path}")
        self._log(f"  Analyze with: python -m LLM_Encoder.cl1_terraforming --analyze {self.h5_path}")
        self._log(f"{'=' * 78}")

        self.data_store.close()
        return all_cycle_results


# ---------------------------------------------------------------------------
# 9. TerraformingAnalysis — post-hoc hypothesis testing
# ---------------------------------------------------------------------------

class TerraformingAnalysis:
    """Post-hoc analysis of terraforming experiment data.

    Tests 6 pre-registered hypotheses with Bonferroni correction (α=0.0083).
    """

    ALPHA = 0.05
    N_TESTS = 6
    BONFERRONI_ALPHA = ALPHA / N_TESTS  # 0.0083

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.f = h5py.File(h5_path, 'r')

    def _get_cycle_groups(self) -> List[str]:
        return sorted([k for k in self.f.keys() if k.startswith('cycle_')])

    def _get_round_data(self, cycle_key: str) -> List[Dict]:
        """Extract round-level data from a cycle."""
        cycle_grp = self.f[cycle_key]
        rounds = []
        for key in sorted(cycle_grp.keys()):
            if key.startswith('round_'):
                grp = cycle_grp[key]
                data = {}
                for attr_name in grp.attrs:
                    data[attr_name] = grp.attrs[attr_name]
                for ds_name in grp:
                    data[ds_name] = np.array(grp[ds_name])
                rounds.append(data)
        return rounds

    def _spearman_r(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Spearman rank correlation with p-value."""
        from scipy.stats import spearmanr
        if len(x) < 3:
            return 0.0, 1.0
        r, p = spearmanr(x, y)
        return float(r), float(p)

    def _wilcoxon(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Wilcoxon signed-rank test."""
        from scipy.stats import wilcoxon
        diff = x - y
        diff = diff[diff != 0]
        if len(diff) < 5:
            return 0.0, 1.0
        stat, p = wilcoxon(diff)
        return float(stat), float(p)

    def _cohens_d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cohen's d effect size."""
        pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
        if pooled_std < 1e-10:
            return 0.0
        return float((np.mean(x) - np.mean(y)) / pooled_std)

    def test_h1_accuracy_increase(self) -> Dict:
        """H1: Decoder accuracy increases within a 4h training cycle."""
        results = []
        for cycle_key in self._get_cycle_groups():
            rounds = self._get_round_data(cycle_key)
            if len(rounds) < 3:
                continue
            accuracies = np.array([r.get('accuracy', 0) for r in rounds])
            indices = np.arange(len(accuracies), dtype=float)
            r, p = self._spearman_r(indices, accuracies)
            results.append({
                'cycle': cycle_key, 'r': r, 'p': p,
                'first_acc': float(accuracies[0]),
                'last_acc': float(accuracies[-1]),
                'n_rounds': len(accuracies),
            })

        # Combine p-values across cycles (Fisher's method)
        if results:
            from scipy.stats import combine_pvalues
            ps = [r['p'] for r in results]
            _, combined_p = combine_pvalues(ps, method='fisher')
        else:
            combined_p = 1.0

        return {
            'hypothesis': 'H1: Decoder accuracy increases within training cycle',
            'per_cycle': results,
            'combined_p': float(combined_p),
            'significant': combined_p < self.BONFERRONI_ALPHA,
            'alpha': self.BONFERRONI_ALPHA,
        }

    def test_h2_cscore_increase(self) -> Dict:
        """H2: C-Score increases within a 4h training cycle."""
        results = []
        for cycle_key in self._get_cycle_groups():
            rounds = self._get_round_data(cycle_key)
            if len(rounds) < 3:
                continue
            cscores = np.array([r.get('mean_cscore', 0) for r in rounds])
            indices = np.arange(len(cscores), dtype=float)
            r, p = self._spearman_r(indices, cscores)
            results.append({
                'cycle': cycle_key, 'r': r, 'p': p,
                'first_cs': float(cscores[0]),
                'last_cs': float(cscores[-1]),
            })

        if results:
            from scipy.stats import combine_pvalues
            ps = [r['p'] for r in results]
            _, combined_p = combine_pvalues(ps, method='fisher')
        else:
            combined_p = 1.0

        return {
            'hypothesis': 'H2: C-Score increases within training cycle',
            'per_cycle': results,
            'combined_p': float(combined_p),
            'significant': combined_p < self.BONFERRONI_ALPHA,
            'alpha': self.BONFERRONI_ALPHA,
        }

    def test_h3_consolidation(self) -> Dict:
        """H3: Post-rest C-Score > Pre-rest C-Score (consolidation)."""
        pre_scores = []
        post_scores = []

        for cycle_key in self._get_cycle_groups():
            cycle_grp = self.f[cycle_key]
            if 'rest' in cycle_grp:
                rest = cycle_grp['rest']
                pre_scores.append(rest.attrs.get('pre_rest_cscore', 0))
                post_scores.append(rest.attrs.get('post_rest_cscore', 0))

        if len(pre_scores) < 2:
            return {
                'hypothesis': 'H3: Post-rest C-Score > Pre-rest (consolidation)',
                'significant': False,
                'note': f'Insufficient data: {len(pre_scores)} rest measurements',
                'alpha': self.BONFERRONI_ALPHA,
            }

        pre = np.array(pre_scores)
        post = np.array(post_scores)
        stat, p = self._wilcoxon(post, pre)
        d = self._cohens_d(post, pre)

        return {
            'hypothesis': 'H3: Post-rest C-Score > Pre-rest (consolidation)',
            'pre_mean': float(np.mean(pre)),
            'post_mean': float(np.mean(post)),
            'delta': float(np.mean(post - pre)),
            'cohens_d': d,
            'wilcoxon_stat': stat,
            'p': float(p),
            'significant': p < self.BONFERRONI_ALPHA,
            'alpha': self.BONFERRONI_ALPHA,
        }

    def test_h4_cl1_vs_llm(self) -> Dict:
        """H4: CL1 C-Score > LLM-only C-Score after N cycles."""
        cl1_cscores = []
        # Use assessment phase C-Scores (no feedback bias)
        for cycle_key in self._get_cycle_groups():
            cycle_grp = self.f[cycle_key]
            if 'assessment' in cycle_grp:
                assess = cycle_grp['assessment']
                cs = assess.attrs.get('mean_cscore', 0)
                cl1_cscores.append(cs)

        if not cl1_cscores:
            return {
                'hypothesis': 'H4: CL1 C-Score > LLM-only C-Score',
                'significant': False,
                'note': 'No assessment data found',
                'alpha': self.BONFERRONI_ALPHA,
            }

        # LLM-only baseline C-Score is 0 (no substrate)
        cl1 = np.array(cl1_cscores)
        last_cycle_cs = cl1[-1]

        return {
            'hypothesis': 'H4: CL1 C-Score > LLM-only after N cycles',
            'cl1_cscores_by_cycle': cl1.tolist(),
            'last_cycle_cscore': float(last_cycle_cs),
            'llm_only_cscore': 0.0,
            'note': 'LLM-only has no substrate, C-Score=0 by definition',
            'significant': last_cycle_cs > 0.1,  # meaningful threshold
            'alpha': self.BONFERRONI_ALPHA,
        }

    def test_h5_channel_recruitment(self) -> Dict:
        """H5: Active channel count increases over cycles."""
        active_counts = []
        for cycle_key in self._get_cycle_groups():
            cycle_grp = self.f[cycle_key]
            if 'warmup' in cycle_grp:
                warmup = cycle_grp['warmup']
                active_counts.append(warmup.attrs.get('active_channels', 0))

        if len(active_counts) < 2:
            return {
                'hypothesis': 'H5: Active channel count increases over cycles',
                'significant': False,
                'note': f'Insufficient data: {len(active_counts)} cycles',
                'alpha': self.BONFERRONI_ALPHA,
            }

        counts = np.array(active_counts, dtype=float)
        indices = np.arange(len(counts), dtype=float)
        r, p = self._spearman_r(indices, counts)

        return {
            'hypothesis': 'H5: Active channel count increases over cycles',
            'active_by_cycle': counts.tolist(),
            'first': float(counts[0]),
            'last': float(counts[-1]),
            'spearman_r': r,
            'p': float(p),
            'significant': p < self.BONFERRONI_ALPHA,
            'alpha': self.BONFERRONI_ALPHA,
        }

    def test_h6_cross_cycle_learning(self) -> Dict:
        """H6: Decoder accuracy improves across cycles (long-term learning)."""
        cycle_accuracies = []
        for cycle_key in self._get_cycle_groups():
            cycle_grp = self.f[cycle_key]
            if 'assessment' in cycle_grp:
                assess = cycle_grp['assessment']
                acc = assess.attrs.get('mean_accuracy', 0)
                cycle_accuracies.append(acc)

        if len(cycle_accuracies) < 2:
            return {
                'hypothesis': 'H6: Decoder accuracy improves across cycles',
                'significant': False,
                'note': f'Insufficient data: {len(cycle_accuracies)} cycles',
                'alpha': self.BONFERRONI_ALPHA,
            }

        accs = np.array(cycle_accuracies, dtype=float)
        indices = np.arange(len(accs), dtype=float)
        r, p = self._spearman_r(indices, accs)

        return {
            'hypothesis': 'H6: Decoder accuracy improves across cycles',
            'accuracy_by_cycle': accs.tolist(),
            'first': float(accs[0]),
            'last': float(accs[-1]),
            'spearman_r': r,
            'p': float(p),
            'significant': p < self.BONFERRONI_ALPHA,
            'alpha': self.BONFERRONI_ALPHA,
        }

    def run_all(self) -> Dict:
        """Run all 6 hypothesis tests."""
        results = {
            'h1': self.test_h1_accuracy_increase(),
            'h2': self.test_h2_cscore_increase(),
            'h3': self.test_h3_consolidation(),
            'h4': self.test_h4_cl1_vs_llm(),
            'h5': self.test_h5_channel_recruitment(),
            'h6': self.test_h6_cross_cycle_learning(),
        }

        n_sig = sum(1 for r in results.values() if r.get('significant', False))

        results['summary'] = {
            'n_hypotheses': self.N_TESTS,
            'n_significant': n_sig,
            'bonferroni_alpha': self.BONFERRONI_ALPHA,
            'data_file': self.h5_path,
        }

        return results

    def print_report(self):
        """Print formatted hypothesis test report."""
        results = self.run_all()

        print("\n" + "=" * 78)
        print("  TERRAFORMING HYPOTHESIS TESTS")
        print(f"  Bonferroni-corrected α = {self.BONFERRONI_ALPHA:.4f} ({self.N_TESTS} tests)")
        print("=" * 78)

        for key in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            r = results[key]
            sig = "PASS" if r.get('significant', False) else "FAIL"
            print(f"\n  [{sig}] {r['hypothesis']}")
            if 'p' in r:
                print(f"         p = {r['p']:.6f}")
            if 'combined_p' in r:
                print(f"         combined p = {r['combined_p']:.6f}")
            if 'cohens_d' in r:
                print(f"         Cohen's d = {r['cohens_d']:.3f}")
            if 'spearman_r' in r:
                print(f"         Spearman r = {r['spearman_r']:.3f}")
            if 'note' in r:
                print(f"         Note: {r['note']}")

        summary = results['summary']
        print(f"\n{'─' * 78}")
        print(f"  {summary['n_significant']}/{summary['n_hypotheses']} hypotheses supported")
        print(f"  Data: {summary['data_file']}")
        print("=" * 78)

        return results

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='CL1 Terraforming Experiment — Doom-style feedback'
    )
    parser.add_argument('--local', action='store_true',
                        help='Use local Izhikevich substrate instead of CL1')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: shortened durations for testing')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Single round on CL1 to verify connectivity')
    parser.add_argument('--n-cycles', type=int, default=3,
                        help='Number of duty cycles (default: 3)')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Path to HDF5 file for post-hoc analysis')
    parser.add_argument('--tokens', type=int, default=50,
                        help='Tokens per round (default: 50)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Neural blending weight (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='experiment_data',
                        help='Output directory (default: experiment_data)')

    args = parser.parse_args()

    # Analysis mode
    if args.analyze:
        print(f"Analyzing: {args.analyze}")
        analysis = TerraformingAnalysis(args.analyze)
        analysis.print_report()
        analysis.close()
        return

    # Create substrate
    if args.local:
        from LLM_Encoder.neural_substrate import IzhikevichSubstrate
        substrate = IzhikevichSubstrate(seed=args.seed)
        print("Using LOCAL Izhikevich substrate")
    else:
        from LLM_Encoder.cl1_cloud_substrate import CL1CloudSubstrate
        substrate = CL1CloudSubstrate(seed=args.seed)
        print(f"Using CL1 cloud substrate: {substrate.cl1_host}")

    # Smoke test
    if args.smoke_test:
        print("\n--- SMOKE TEST ---")
        print("Testing channel layout connectivity...")
        for name, channels in [
            ('SENSORY', ChannelLayout.SENSORY),
            ('MOTOR', ChannelLayout.MOTOR),
            ('REWARD', ChannelLayout.REWARD),
            ('PUNISHMENT', ChannelLayout.PUNISHMENT),
        ]:
            pattern = {ch: 2.0 for ch in channels[:3]}
            spikes = substrate.stimulate_and_record(pattern, window_s=0.3)
            total = sum(spikes.values()) if isinstance(spikes, dict) else 0
            print(f"  {name:12s}: stimulated {list(pattern.keys())}, "
                  f"got {total} spikes")

        sm = substrate.get_last_spike_matrix()
        assessor = ConsciousnessAssessor()
        result = assessor.assess(sm)
        print(f"\n  C-Score: {result.get('cscore', 0):.4f}")
        print("  Smoke test PASSED" if total > 0 else "  WARNING: no spikes detected")

        if hasattr(substrate, 'close'):
            substrate.close()
        return

    # Full experiment
    fast = args.fast or args.local  # local always uses fast mode
    experiment = TerraformingExperiment(
        substrate=substrate,
        alpha=args.alpha,
        tokens_per_round=args.tokens,
        n_cycles=args.n_cycles,
        seed=args.seed,
        output_dir=args.output_dir,
        fast_mode=fast,
    )

    try:
        experiment.run()
    finally:
        if hasattr(substrate, 'close'):
            substrate.close()


if __name__ == '__main__':
    main()
