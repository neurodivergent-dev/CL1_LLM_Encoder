#!/usr/bin/env python3
"""CL1 Experiment v3 — Extended Learning + Persistent Decoder + Dose-Response.

Building on v2's confirmed results (SRC d=1.79, C-Score d=3.99), this experiment
tests whether the Bio-LLM system LEARNS over extended interaction.

Key innovations over v2:
  1. PERSISTENT HebbianDecoder across all rounds (not reset per round)
     → Tests whether the substrate forms stable token-specific representations
  2. 20 rounds (2x v2) for temporal learning detection
  3. Template convergence tracking (how much decoder templates change)
  4. Epoch-based analysis (early/mid/late) for developmental trajectory
  5. Surprise-scaled feedback (prediction error modulates feedback intensity)
  6. Phase 2: High-alpha (0.8) Bio-LLM to test dose-response

Additional innovation:
  7. CHANNEL RECRUITMENT — actively train inactive channels by applying
     repeated stimulations to channels with low/no spike activity. This
     recruits more of the MEA into the token representation space.

Hypotheses:
  H1: Bio SRC > Shadow SRC (replication from v2)
  H2: Raw SRC equal across conditions (sanity check)
  H3: Bio C-Score > Shadow C-Score (replication from v2)
  H4: Bio C-Score INCREASES over rounds (learning trajectory)
  H5: Bio SRC INCREASES over rounds (learning trajectory)
  H6: Template convergence rate differs Bio vs Shadow
  H7: Late-epoch Bio > Early-epoch Bio (developmental improvement)
  H8: Shuffling degrades SRC (replication from v2)
  H9: High-alpha Bio > Standard-alpha Bio (dose-response)
"""

import os
import sys
import time
import json
import h5py
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_Encoder.spatial_encoder import (
    SpatialEncoder, SpatialDecoder, SENSORY_CHANNELS, FEEDBACK_CHANNELS,
    AMP_MIN, AMP_MAX
)
from LLM_Encoder.consciousness import ConsciousnessAssessor

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "LFM2-350M-Q4_0.gguf"
)

CONDITIONS = ['bio_llm', 'shadow_llm', 'llm_only']

# Extended prompt set for 20 rounds
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
TRAINING_WINDOW_S = 0.3
MAX_CANDIDATES = 15

# Channel recruitment
RESERVED_CHANNELS = frozenset({0, 4, 7, 56, 63})
ALL_USABLE_CHANNELS = [ch for ch in range(64) if ch not in RESERVED_CHANNELS]


class ChannelRecruiter:
    """Actively trains inactive channels to expand the neural representation.

    Tracks per-channel spike activity across all stimulations. Between rounds,
    applies targeted training stimulation to channels with low/no activity,
    gradually increasing amplitude to recruit them into the network.

    This is analogous to neural rehabilitation — stimulating dormant pathways
    to encourage plasticity and responsiveness.
    """

    def __init__(self, substrate, min_activity_threshold: int = 5,
                 training_amplitude: float = 1.5, max_amplitude: float = 2.5,
                 amplitude_step: float = 0.2, n_training_pulses: int = 3):
        self.substrate = substrate
        self.min_activity = min_activity_threshold
        self.base_amp = training_amplitude
        self.max_amp = max_amplitude
        self.amp_step = amplitude_step
        self.n_pulses = n_training_pulses

        # Per-channel cumulative activity
        self._channel_activity: Dict[int, int] = {ch: 0 for ch in ALL_USABLE_CHANNELS}
        self._channel_stim_count: Dict[int, int] = {ch: 0 for ch in ALL_USABLE_CHANNELS}
        self._training_amplitudes: Dict[int, float] = {ch: training_amplitude for ch in ALL_USABLE_CHANNELS}
        self._recruited_channels: List[int] = []
        self._training_history: List[Dict] = []

    def record_activity(self, spike_counts: Dict):
        """Record spike activity from a stimulation."""
        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            if ch in self._channel_activity:
                self._channel_activity[ch] += count
                self._channel_stim_count[ch] += 1

    def get_inactive_channels(self) -> List[int]:
        """Return channels with activity below threshold."""
        inactive = []
        for ch in ALL_USABLE_CHANNELS:
            if self._channel_activity[ch] < self.min_activity:
                inactive.append(ch)
        return inactive

    def train_inactive_channels(self) -> Dict:
        """Apply training stimulation to inactive channels.

        Returns training report with channels stimulated and responses.
        """
        inactive = self.get_inactive_channels()
        if not inactive:
            return {'n_trained': 0, 'n_recruited': 0, 'channels': []}

        report = {
            'n_inactive': len(inactive),
            'n_trained': 0,
            'n_recruited': 0,
            'channels': [],
        }

        # Train in small batches (don't overwhelm the MEA)
        batch_size = min(8, len(inactive))
        for i in range(0, len(inactive), batch_size):
            batch = inactive[i:i + batch_size]

            for pulse in range(self.n_pulses):
                # Build training stimulation
                train_amps = {}
                for ch in batch:
                    amp = self._training_amplitudes[ch]
                    train_amps[ch] = min(amp, self.max_amp)

                # Stimulate and check response
                spike_counts = self.substrate.stimulate_and_record(
                    train_amps, window_s=TRAINING_WINDOW_S
                )

                # Record responses
                for ch in batch:
                    ch_spikes = int(spike_counts.get(str(ch), spike_counts.get(ch, 0)))
                    if ch_spikes > 0:
                        self._channel_activity[ch] += ch_spikes
                        if ch not in self._recruited_channels:
                            self._recruited_channels.append(ch)
                            report['n_recruited'] += 1

            # Increase amplitude for still-inactive channels
            for ch in batch:
                if self._channel_activity[ch] < self.min_activity:
                    self._training_amplitudes[ch] = min(
                        self._training_amplitudes[ch] + self.amp_step,
                        self.max_amp
                    )

            report['n_trained'] += len(batch)
            report['channels'].extend(batch)

        self._training_history.append(report)
        return report

    def get_stats(self) -> Dict:
        """Return recruitment statistics."""
        active = sum(1 for ch in ALL_USABLE_CHANNELS
                     if self._channel_activity[ch] >= self.min_activity)
        return {
            'total_channels': len(ALL_USABLE_CHANNELS),
            'active_channels': active,
            'inactive_channels': len(ALL_USABLE_CHANNELS) - active,
            'recruited_channels': len(self._recruited_channels),
            'recruited_list': self._recruited_channels[:],
            'training_rounds': len(self._training_history),
            'channel_activity': dict(self._channel_activity),
        }


def stimulus_response_congruence(stim_pattern: Dict[int, float],
                                  spike_counts: Dict, n_channels: int = 59) -> float:
    """Compute cosine similarity between stimulation pattern and spike response."""
    stim_vec = np.zeros(n_channels, dtype=float)
    resp_vec = np.zeros(n_channels, dtype=float)

    for ch, amp in stim_pattern.items():
        ch = int(ch)
        if 0 <= ch < n_channels:
            stim_vec[ch] = float(amp)

    for ch_str, count in spike_counts.items():
        ch = int(ch_str)
        if 0 <= ch < n_channels:
            resp_vec[ch] = float(count)

    s_norm = np.linalg.norm(stim_vec)
    r_norm = np.linalg.norm(resp_vec)

    if s_norm < 1e-10 or r_norm < 1e-10:
        return 0.0

    return float(np.dot(stim_vec, resp_vec) / (s_norm * r_norm))


class PersistentHebbianDecoder:
    """Hebbian decoder that persists across rounds with convergence tracking.

    Key difference from v2's HebbianDecoder: templates are NOT reset between
    rounds, allowing cross-round learning. Also tracks template convergence
    rate to detect when representations stabilize.
    """

    def __init__(self, encoder: SpatialEncoder, alpha: float = 0.5,
                 learning_rate: float = 0.02):
        self.encoder = encoder
        self.alpha = alpha
        self.lr = learning_rate

        self._templates: Dict[int, np.ndarray] = {}
        self._template_counts: Dict[int, int] = {}
        self._n_channels = 59
        self._n_updates = 0
        self._round_idx = 0

        # Convergence tracking
        self._prev_templates: Dict[int, np.ndarray] = {}
        self._convergence_history: List[float] = []
        self._template_norm_history: List[float] = []

        # Prediction accuracy tracking
        self._correct_predictions = 0
        self._total_predictions = 0

    def start_round(self, round_idx: int):
        """Mark the start of a new round, snapshot templates for convergence."""
        self._round_idx = round_idx
        self._prev_templates = {tid: t.copy() for tid, t in self._templates.items()}

    def end_round(self) -> Dict:
        """Mark end of round, compute convergence metrics."""
        convergence = self._compute_convergence()
        self._convergence_history.append(convergence)

        # Track template norms (measure of representation strength)
        norms = [np.linalg.norm(t) for t in self._templates.values()]
        mean_norm = float(np.mean(norms)) if norms else 0.0
        self._template_norm_history.append(mean_norm)

        return {
            'convergence': convergence,
            'mean_template_norm': mean_norm,
            'n_templates': len(self._templates),
            'prediction_accuracy': (self._correct_predictions / max(1, self._total_predictions)),
        }

    def _compute_convergence(self) -> float:
        """Compute template change between rounds (lower = more converged)."""
        if not self._prev_templates or not self._templates:
            return 1.0

        shared_tokens = set(self._prev_templates.keys()) & set(self._templates.keys())
        if not shared_tokens:
            return 1.0

        changes = []
        for tid in shared_tokens:
            old = self._prev_templates[tid]
            new = self._templates[tid]
            diff = np.linalg.norm(new - old)
            norm = np.linalg.norm(old) + 1e-10
            changes.append(diff / norm)

        return float(np.mean(changes))

    def decode(
        self,
        spike_counts: Dict,
        model_probs: Dict[int, float],
        channel_to_token: Dict[int, int],
    ) -> Tuple[Dict[int, float], Dict[int, float], float]:
        """Decode via template matching + spatial ownership."""
        resp_vec = np.zeros(self._n_channels, dtype=float)
        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            if 0 <= ch < self._n_channels:
                resp_vec[ch] = float(count)

        token_scores = {}

        for tid in model_probs:
            score = 0.0

            # Spatial ownership score
            pattern = self.encoder.encode_token(tid)
            stim_vec = np.zeros(self._n_channels, dtype=float)
            for ch, amp in pattern.items():
                ch = int(ch)
                if 0 <= ch < self._n_channels:
                    stim_vec[ch] = amp

            s_norm = np.linalg.norm(stim_vec)
            r_norm = np.linalg.norm(resp_vec)
            if s_norm > 1e-10 and r_norm > 1e-10:
                src = np.dot(stim_vec, resp_vec) / (s_norm * r_norm)
                score += src * 0.5

            # Template matching score (learned over rounds)
            if tid in self._templates:
                template = self._templates[tid]
                t_norm = np.linalg.norm(template)
                if t_norm > 1e-10 and r_norm > 1e-10:
                    template_match = np.dot(template, resp_vec) / (t_norm * r_norm)
                    score += template_match * 0.5

            token_scores[tid] = score

        # Softmax conversion
        if token_scores:
            max_score = max(token_scores.values())
            neural_probs = {tid: np.exp(2.0 * (s - max_score))
                           for tid, s in token_scores.items()}
            total = sum(neural_probs.values())
            if total > 0:
                neural_probs = {k: v / total for k, v in neural_probs.items()}
        else:
            neural_probs = {}

        # Blend
        all_tokens = set(model_probs.keys()) | set(neural_probs.keys())
        blended = {}
        for tok in all_tokens:
            mp = model_probs.get(tok, 0.0)
            np_ = neural_probs.get(tok, 0.0)
            blended[tok] = (1 - self.alpha) * mp + self.alpha * np_

        total = sum(blended.values())
        if total > 0:
            blended = {tok: p / total for tok, p in blended.items()}

        confidence = min(1.0, self._n_updates / 30.0)
        return blended, neural_probs, confidence

    def update(self, selected_token: int, spike_counts: Dict,
               model_top: int = None):
        """Update template for selected token (EMA of observed response)."""
        resp_vec = np.zeros(self._n_channels, dtype=float)
        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            if 0 <= ch < self._n_channels:
                resp_vec[ch] = float(count)

        if selected_token not in self._templates:
            self._templates[selected_token] = resp_vec.copy()
            self._template_counts[selected_token] = 1
        else:
            old = self._templates[selected_token]
            self._templates[selected_token] = old + self.lr * (resp_vec - old)
            self._template_counts[selected_token] += 1

        self._n_updates += 1

        # Track prediction accuracy (did neural decoder match LLM top?)
        if model_top is not None:
            self._total_predictions += 1
            if selected_token == model_top:
                self._correct_predictions += 1

    def get_stats(self) -> Dict:
        return {
            'n_templates': len(self._templates),
            'n_updates': self._n_updates,
            'convergence_history': self._convergence_history,
            'template_norm_history': self._template_norm_history,
            'prediction_accuracy': (self._correct_predictions / max(1, self._total_predictions)),
        }


class CL1ExperimentV3:
    """Extended 3-condition experiment with persistent learning + dose-response."""

    def __init__(
        self,
        substrate,
        model_path: str = MODEL_PATH,
        alpha: float = 0.5,
        high_alpha: float = 0.8,
        tokens_per_thought: int = 50,
        n_rounds_phase1: int = 15,
        n_rounds_phase2: int = 5,
        seed: int = 42,
        feedback_enabled: bool = True,
        output_dir: str = "experiment_data",
    ):
        self.substrate = substrate
        self.model_path = model_path
        self.alpha = alpha
        self.high_alpha = high_alpha
        self.tokens_per_thought = tokens_per_thought
        self.n_rounds_p1 = n_rounds_phase1
        self.n_rounds_p2 = n_rounds_phase2
        self.n_rounds_total = n_rounds_phase1 + n_rounds_phase2
        self.seed = seed
        self.feedback_enabled = feedback_enabled
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.h5_path = os.path.join(output_dir, f"cl1_v3_{self.timestamp}.h5")
        self.log_path = os.path.join(output_dir, f"cl1_v3_{self.timestamp}.log")

        self.spatial_encoder = SpatialEncoder(seed=seed)
        self.assessor = ConsciousnessAssessor()
        self._llm = None
        self._rng = np.random.default_rng(seed)

        # PERSISTENT decoders — survive across rounds
        self._decoders = {
            'bio_llm': PersistentHebbianDecoder(self.spatial_encoder, alpha=alpha),
            'shadow_llm': PersistentHebbianDecoder(self.spatial_encoder, alpha=alpha),
            'bio_llm_high': PersistentHebbianDecoder(self.spatial_encoder, alpha=high_alpha),
        }

        # Channel recruitment — train inactive channels
        self._recruiter = ChannelRecruiter(substrate)

        # Surprise tracking for feedback modulation
        self._prediction_errors = []

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
        self._log("LLM loaded.")

    def _compute_surprise(self, selected_token: int,
                           model_probs: Dict[int, float]) -> float:
        """Compute surprise (negative log probability) for feedback scaling."""
        p = model_probs.get(selected_token, 0.01)
        surprise = -np.log(max(p, 1e-6))
        self._prediction_errors.append(surprise)

        # EMA baseline for relative surprise
        if len(self._prediction_errors) > 10:
            ema = np.mean(self._prediction_errors[-20:])
            return min(2.0, surprise / max(ema, 0.1))
        return 1.0

    def _generate_tokens(self, prompt: str, condition: str,
                          h5_group: h5py.Group, round_idx: int) -> Dict:
        """Generate tokens with persistent decoder and surprise-scaled feedback."""
        self._load_llm()

        use_neurons = condition in ('bio_llm', 'shadow_llm', 'bio_llm_high')
        shuffle_spikes = condition == 'shadow_llm'
        is_high_alpha = condition == 'bio_llm_high'

        # Select the right persistent decoder
        if condition == 'bio_llm_high':
            decoder_key = 'bio_llm_high'
        elif condition in ('bio_llm', 'shadow_llm'):
            decoder_key = condition
        else:
            decoder_key = None

        decoder = self._decoders.get(decoder_key) if decoder_key else None

        effective_alpha = (self.high_alpha if is_high_alpha
                          else self.alpha if use_neurons
                          else 0.0)

        try:
            self._llm.reset()
            if hasattr(self._llm, '_ctx') and self._llm._ctx is not None:
                self._llm._ctx.kv_cache_clear()
        except:
            pass

        text = ""
        context = prompt
        override_count = 0
        all_cs = []
        all_alignment = []
        all_entropy = []
        all_spikes = []
        all_src = []
        all_src_raw = []
        all_latencies = []
        token_ids = []
        was_overrides = []
        all_surprises = []
        prev_token_id = None

        # Mark round start for convergence tracking
        if decoder:
            decoder.start_round(round_idx)

        for pos in range(self.tokens_per_thought):
            t0 = time.time()

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
                model_probs = {tid: np.exp(info['logprob'] - max_lp)
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

            # Spatial encoding with temporal binding
            combined_pattern, channel_to_token = self.spatial_encoder.encode_candidates(model_probs)

            if prev_token_id is not None and use_neurons:
                prev_pattern = self.spatial_encoder.encode_token(prev_token_id)
                for ch, amp in prev_pattern.items():
                    if ch in combined_pattern:
                        combined_pattern[ch] = min(AMP_MAX, combined_pattern[ch] + amp * 0.2)
                    else:
                        combined_pattern[ch] = amp * 0.2

            # Stimulate CL1
            channel_amplitudes = {int(ch): amp for ch, amp in combined_pattern.items()}
            spike_counts = self.substrate.stimulate_and_record(
                channel_amplitudes, window_s=RESPONSE_WINDOW_S,
            )

            # Record activity for channel recruitment
            self._recruiter.record_activity(spike_counts)

            # === RAW SRC (same for all conditions) ===
            raw_src = stimulus_response_congruence(combined_pattern, spike_counts)
            all_src_raw.append(raw_src)

            # Consciousness assessment
            sm = self.substrate.get_last_spike_matrix()
            consciousness = self.assessor.assess(sm)
            cs = consciousness.get('cscore', 0)
            all_cs.append(cs)

            token_spikes = sum(spike_counts.values()) if isinstance(spike_counts, dict) else 0
            all_spikes.append(token_spikes)

            # Condition-specific processing
            if use_neurons and decoder and len(model_probs) > 1:
                decode_spikes = dict(spike_counts)
                if shuffle_spikes:
                    channels = list(decode_spikes.keys())
                    counts = list(decode_spikes.values())
                    np.random.shuffle(counts)
                    decode_spikes = dict(zip(channels, counts))

                # SRC after condition-specific processing
                processed_src = stimulus_response_congruence(combined_pattern, decode_spikes)
                all_src.append(processed_src)

                blended, neural_probs, confidence = decoder.decode(
                    decode_spikes, model_probs, channel_to_token
                )
                selected = max(blended, key=blended.get)
                was_override = selected != model_top

                # Update persistent decoder with model_top for accuracy tracking
                decoder.update(selected, decode_spikes, model_top=model_top)

                # === SURPRISE-SCALED FEEDBACK ===
                surprise = self._compute_surprise(selected, model_probs)
                all_surprises.append(surprise)

                if self.feedback_enabled and condition in ('bio_llm', 'bio_llm_high'):
                    fb_pattern = self.spatial_encoder.encode_token(selected)
                    fb_amps = {}
                    for i, fch in enumerate(FEEDBACK_CHANNELS):
                        if i < len(self.spatial_encoder.channels):
                            sch = self.spatial_encoder.channels[i]
                            if sch in fb_pattern:
                                # Scale feedback by surprise (higher surprise → stronger feedback)
                                base_amp = 1.5 * fb_pattern[sch] / AMP_MAX
                                scaled_amp = min(AMP_MAX, base_amp * surprise)
                                fb_amps[fch] = scaled_amp
                    if fb_amps:
                        self.substrate.stimulate_and_record(fb_amps, window_s=FEEDBACK_WINDOW_S)

                # Alignment
                shared = set(neural_probs.keys()) & set(model_probs.keys())
                if shared:
                    m_vec = np.array([model_probs[t] for t in shared])
                    n_vec = np.array([neural_probs[t] for t in shared])
                    mn, nn = np.linalg.norm(m_vec), np.linalg.norm(n_vec)
                    if mn > 1e-10 and nn > 1e-10:
                        all_alignment.append(float(np.dot(m_vec, n_vec) / (mn * nn)))
                    else:
                        all_alignment.append(0.0)
                else:
                    all_alignment.append(0.0)

                bp = np.array(list(blended.values()), dtype=float)
                bp = bp[bp > 0]
                bp = bp / (bp.sum() + 1e-10)
                all_entropy.append(float(-np.sum(bp * np.log2(bp + 1e-10))))
            else:
                selected = model_top
                was_override = False
                all_alignment.append(0.0)
                all_src.append(raw_src)
                all_surprises.append(0.0)
                bp = np.array(list(model_probs.values()), dtype=float)
                bp = bp[bp > 0]
                bp = bp / (bp.sum() + 1e-10)
                all_entropy.append(float(-np.sum(bp * np.log2(bp + 1e-10))))

            if was_override:
                override_count += 1

            if selected in top_logprobs:
                tok_text = top_logprobs[selected]['text']
            else:
                tok_text = self._llm.detokenize([selected]).decode('utf-8', errors='replace')

            latency = (time.time() - t0) * 1000
            all_latencies.append(latency)
            token_ids.append(selected)
            was_overrides.append(1 if was_override else 0)

            text += tok_text
            context += tok_text
            prev_token_id = selected

            # Live output
            marker = '*' if was_override else ' '
            if pos % 10 == 0:
                sys.stdout.write(f"\n    [{pos:3d}] C={cs:.3f} SRC={all_src[-1]:.3f} spk={token_spikes:3d} {marker}")
            sys.stdout.write(tok_text)
            sys.stdout.flush()

            if tok_text.strip() == '' and pos > 10 and text.endswith('\n\n'):
                break

        sys.stdout.write('\n')
        sys.stdout.flush()

        # End-of-round convergence metrics
        round_metrics = decoder.end_round() if decoder else {}

        # Save to HDF5
        n = len(token_ids)
        h5_group.create_dataset('token_ids', data=np.array(token_ids, dtype=np.int32))
        h5_group.create_dataset('was_override', data=np.array(was_overrides, dtype=np.int8))
        h5_group.create_dataset('cscore', data=np.array(all_cs[:n], dtype=np.float32))
        h5_group.create_dataset('alignment', data=np.array(all_alignment[:n], dtype=np.float32))
        h5_group.create_dataset('entropy', data=np.array(all_entropy[:n], dtype=np.float32))
        h5_group.create_dataset('spikes', data=np.array(all_spikes[:n], dtype=np.int32))
        h5_group.create_dataset('src', data=np.array(all_src[:n], dtype=np.float32))
        h5_group.create_dataset('src_raw', data=np.array(all_src_raw[:n], dtype=np.float32))
        h5_group.create_dataset('surprise', data=np.array(all_surprises[:n], dtype=np.float32))
        h5_group.create_dataset('latency_ms', data=np.array(all_latencies[:n], dtype=np.float32))
        h5_group.attrs['condition'] = condition
        h5_group.attrs['prompt'] = prompt
        h5_group.attrs['generated_text'] = text
        h5_group.attrs['n_tokens'] = n
        h5_group.attrs['override_count'] = override_count
        h5_group.attrs['round_idx'] = round_idx
        h5_group.attrs['effective_alpha'] = effective_alpha
        for k, v in round_metrics.items():
            if isinstance(v, (int, float)):
                h5_group.attrs[f'decoder_{k}'] = v

        summary = {
            'condition': condition,
            'round_idx': round_idx,
            'n_tokens': n,
            'override_count': override_count,
            'override_rate': override_count / max(1, n),
            'mean_cscore': float(np.mean(all_cs)) if all_cs else 0.0,
            'mean_alignment': float(np.mean(all_alignment)) if all_alignment else 0.0,
            'mean_entropy': float(np.mean(all_entropy)) if all_entropy else 0.0,
            'mean_spikes': float(np.mean(all_spikes)) if all_spikes else 0.0,
            'total_spikes': sum(all_spikes),
            'mean_src': float(np.mean(all_src)) if all_src else 0.0,
            'std_src': float(np.std(all_src)) if all_src else 0.0,
            'mean_src_raw': float(np.mean(all_src_raw)) if all_src_raw else 0.0,
            'mean_surprise': float(np.mean(all_surprises)) if all_surprises else 0.0,
            'mean_latency_ms': float(np.mean(all_latencies)) if all_latencies else 0.0,
            'generated_text': text,
            'effective_alpha': effective_alpha,
            **round_metrics,
        }
        return summary

    def run(self):
        """Run the extended experiment in two phases."""
        t_start = time.time()

        self._log("=" * 78)
        self._log("  CL1 EXPERIMENT v3 — EXTENDED LEARNING + DOSE-RESPONSE")
        self._log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 78)
        self._log(f"  Substrate:      {type(self.substrate).__name__}")
        self._log(f"  Phase 1:        {self.n_rounds_p1} rounds × 3 conditions (α={self.alpha})")
        self._log(f"  Phase 2:        {self.n_rounds_p2} rounds × Bio-LLM only (α={self.high_alpha})")
        self._log(f"  Tokens/round:   {self.tokens_per_thought}")
        self._log(f"  Feedback:       {'SURPRISE-SCALED (Bio only)' if self.feedback_enabled else 'OFF'}")
        self._log(f"  Decoder:        PERSISTENT (survives across rounds)")
        self._log(f"  Key question:   Does the Bio-LLM system LEARN over time?")
        self._log(f"  Output:         {self.h5_path}")
        self._log("=" * 78)

        h5 = h5py.File(self.h5_path, 'w')
        h5.attrs['experiment'] = 'CL1 v3 — Extended Learning + Dose-Response'
        h5.attrs['timestamp'] = self.timestamp
        h5.attrs['substrate'] = type(self.substrate).__name__
        h5.attrs['alpha'] = self.alpha
        h5.attrs['high_alpha'] = self.high_alpha
        h5.attrs['tokens_per_thought'] = self.tokens_per_thought
        h5.attrs['n_rounds_phase1'] = self.n_rounds_p1
        h5.attrs['n_rounds_phase2'] = self.n_rounds_p2
        h5.attrs['seed'] = self.seed
        h5.attrs['feedback_type'] = 'surprise_scaled'
        h5.attrs['decoder_type'] = 'persistent_hebbian'

        condition_summaries = defaultdict(list)

        # === PHASE 1: 3-condition interleaved ===
        self._log(f"\n{'='*78}")
        self._log(f"  PHASE 1: 3-condition interleaved ({self.n_rounds_p1} rounds, α={self.alpha})")
        self._log(f"{'='*78}")

        for round_idx in range(self.n_rounds_p1):
            prompt = PROMPTS[round_idx % len(PROMPTS)]

            self._log(f"\n{'#'*78}")
            self._log(f"  ROUND {round_idx + 1}/{self.n_rounds_total} (Phase 1)")
            self._log(f"  Prompt: {prompt[:60]}...")
            self._log(f"{'#'*78}")

            for cond in CONDITIONS:
                self._log(f"\n  --- {cond.upper()} (round {round_idx + 1}) ---")

                grp = h5.create_group(f"phase1/{cond}/round_{round_idx:03d}")
                try:
                    summary = self._generate_tokens(prompt, cond, grp, round_idx)
                    condition_summaries[cond].append(summary)

                    convergence_str = (f", conv={summary.get('convergence', 0):.4f}"
                                       if 'convergence' in summary else "")
                    self._log(f"  {cond}: {summary['n_tokens']} tok, "
                              f"SRC={summary['mean_src']:.4f}, "
                              f"C={summary['mean_cscore']:.4f}"
                              f"{convergence_str}")
                except Exception as e:
                    self._log(f"  ERROR in {cond}: {e}")
                    import traceback
                    traceback.print_exc()
                    condition_summaries[cond].append({'error': str(e), 'round_idx': round_idx})

            h5.flush()

            # === CHANNEL RECRUITMENT between rounds ===
            if (round_idx + 1) % 3 == 0:  # Every 3 rounds
                self._log(f"\n  [CHANNEL TRAINING] Round {round_idx + 1}...")
                recruit_report = self._recruiter.train_inactive_channels()
                if recruit_report['n_trained'] > 0:
                    stats = self._recruiter.get_stats()
                    self._log(f"    Trained {recruit_report['n_trained']} channels, "
                              f"recruited {recruit_report['n_recruited']} new")
                    self._log(f"    Active: {stats['active_channels']}/{stats['total_channels']} channels")
                else:
                    self._log(f"    All channels active — no training needed")

            # Periodic progress
            if (round_idx + 1) % 5 == 0:
                self._log_progress(condition_summaries, round_idx + 1)

        # === PHASE 2: High-alpha Bio-LLM ===
        self._log(f"\n{'='*78}")
        self._log(f"  PHASE 2: High-alpha Bio-LLM ({self.n_rounds_p2} rounds, α={self.high_alpha})")
        self._log(f"{'='*78}")

        for i in range(self.n_rounds_p2):
            round_idx = self.n_rounds_p1 + i
            prompt = PROMPTS[round_idx % len(PROMPTS)]

            self._log(f"\n{'#'*78}")
            self._log(f"  ROUND {round_idx + 1}/{self.n_rounds_total} (Phase 2 — HIGH ALPHA)")
            self._log(f"  Prompt: {prompt[:60]}...")
            self._log(f"{'#'*78}")

            grp = h5.create_group(f"phase2/bio_llm_high/round_{round_idx:03d}")
            try:
                summary = self._generate_tokens(prompt, 'bio_llm_high', grp, round_idx)
                condition_summaries['bio_llm_high'].append(summary)

                self._log(f"  bio_llm_high: {summary['n_tokens']} tok, "
                          f"SRC={summary['mean_src']:.4f}, "
                          f"C={summary['mean_cscore']:.4f}, "
                          f"conv={summary.get('convergence', 0):.4f}")
            except Exception as e:
                self._log(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                condition_summaries['bio_llm_high'].append({'error': str(e), 'round_idx': round_idx})

            h5.flush()

        # === ANALYSIS ===
        self._log(f"\n{'='*78}")
        self._log("  COMPREHENSIVE ANALYSIS")
        self._log(f"{'='*78}")

        analysis = self._analyze(condition_summaries, h5)

        h5.close()

        elapsed = time.time() - t_start
        self._log(f"\n{'='*78}")
        self._log(f"  EXPERIMENT COMPLETE")
        self._log(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        self._log(f"  Data saved: {self.h5_path}")
        self._log(f"{'='*78}")

        return analysis

    def _log_progress(self, summaries: Dict, n_rounds: int):
        """Log progress after every 5 rounds."""
        self._log(f"\n  --- PROGRESS ({n_rounds} rounds complete) ---")
        for cond in ['bio_llm', 'shadow_llm', 'llm_only']:
            valid = [s for s in summaries[cond] if 'error' not in s]
            if valid:
                recent = valid[-5:]
                mean_src = np.mean([s['mean_src'] for s in recent])
                mean_cs = np.mean([s['mean_cscore'] for s in recent])
                self._log(f"  {cond}: SRC={mean_src:.4f}, C={mean_cs:.4f} (last 5 rounds)")

    def _analyze(self, summaries: Dict, h5: h5py.File) -> Dict:
        """Comprehensive analysis with learning trajectories."""
        from scipy import stats

        analysis = {'timestamp': self.timestamp, 'hypotheses': [], 'conditions': {}}

        # Aggregate condition summaries
        for cond in list(summaries.keys()):
            valid = [s for s in summaries[cond] if 'error' not in s]
            if not valid:
                continue
            analysis['conditions'][cond] = {
                'n_rounds': len(valid),
                'mean_src': float(np.mean([s['mean_src'] for s in valid])),
                'std_src': float(np.std([s['mean_src'] for s in valid])),
                'mean_src_raw': float(np.mean([s['mean_src_raw'] for s in valid])),
                'mean_cscore': float(np.mean([s['mean_cscore'] for s in valid])),
                'std_cscore': float(np.std([s['mean_cscore'] for s in valid])),
                'mean_alignment': float(np.mean([s['mean_alignment'] for s in valid])),
                'mean_entropy': float(np.mean([s['mean_entropy'] for s in valid])),
                'mean_override_rate': float(np.mean([s['override_rate'] for s in valid])),
                'effective_alpha': valid[0].get('effective_alpha', 0),
            }

        self._log(f"\n  CONDITION SUMMARY")
        self._log(f"  {'Metric':<25s} {'Bio-LLM':>12s} {'Shadow-LLM':>12s} {'LLM-Only':>12s} {'Bio-High':>12s}")
        self._log(f"  {'-'*73}")

        for metric in ['mean_src', 'mean_src_raw', 'mean_cscore', 'mean_alignment',
                       'mean_entropy', 'mean_override_rate']:
            vals = []
            for cond in ['bio_llm', 'shadow_llm', 'llm_only', 'bio_llm_high']:
                v = analysis['conditions'].get(cond, {}).get(metric, 0)
                vals.append(f"{v:12.4f}")
            self._log(f"  {metric:<25s} {''.join(vals)}")

        # Extract valid summaries
        bio = [s for s in summaries.get('bio_llm', []) if 'error' not in s]
        shadow = [s for s in summaries.get('shadow_llm', []) if 'error' not in s]
        llm = [s for s in summaries.get('llm_only', []) if 'error' not in s]
        bio_high = [s for s in summaries.get('bio_llm_high', []) if 'error' not in s]

        tests = []
        n_total_tests = 9  # Total pre-registered tests
        bonferroni_alpha = 0.05 / n_total_tests

        self._log(f"\n  HYPOTHESIS TESTS (Bonferroni α = {bonferroni_alpha:.4f} for {n_total_tests} tests)")

        # H1: Bio SRC > Shadow SRC
        if bio and shadow:
            bio_src = [s['mean_src'] for s in bio]
            sha_src = [s['mean_src'] for s in shadow]
            U, p = stats.mannwhitneyu(bio_src, sha_src, alternative='greater')
            d = self._cohens_d(bio_src, sha_src)
            sig = self._sig_str(p, bonferroni_alpha)
            tests.append(('H1: Bio SRC > Shadow SRC', U, p, d, sig))
            self._log(f"\n  H1: Bio SRC > Shadow SRC (replication)")
            self._log(f"    U={U:.1f}, p={p:.6f}, d={d:.3f} {sig}")
            self._log(f"    Bio: {np.mean(bio_src):.4f} ± {np.std(bio_src):.4f}")
            self._log(f"    Shadow: {np.mean(sha_src):.4f} ± {np.std(sha_src):.4f}")

        # H2: Raw SRC equal (sanity)
        if bio and shadow:
            bio_raw = [s['mean_src_raw'] for s in bio]
            sha_raw = [s['mean_src_raw'] for s in shadow]
            _, p = stats.mannwhitneyu(bio_raw, sha_raw, alternative='two-sided')
            d = self._cohens_d(bio_raw, sha_raw)
            sig = "PASS" if p > 0.05 else "FAIL"
            tests.append(('H2: Raw SRC equal (sanity)', 0, p, d, sig))
            self._log(f"\n  H2: Raw SRC equal (sanity)")
            self._log(f"    p={p:.4f}, d={d:.3f} → {sig}")

        # H3: Bio C-Score > Shadow C-Score
        if bio and shadow:
            bio_cs = [s['mean_cscore'] for s in bio]
            sha_cs = [s['mean_cscore'] for s in shadow]
            t, p = stats.ttest_rel(bio_cs[:min(len(bio_cs), len(sha_cs))],
                                    sha_cs[:min(len(bio_cs), len(sha_cs))],
                                    alternative='greater')
            d = self._cohens_d(bio_cs, sha_cs)
            sig = self._sig_str(p, bonferroni_alpha)
            tests.append(('H3: Bio C-Score > Shadow C-Score', t, p, d, sig))
            self._log(f"\n  H3: Bio C-Score > Shadow C-Score (replication)")
            self._log(f"    t={t:.2f}, p={p:.6f}, d={d:.3f} {sig}")
            self._log(f"    Bio: {np.mean(bio_cs):.4f} ± {np.std(bio_cs):.4f}")
            self._log(f"    Shadow: {np.mean(sha_cs):.4f} ± {np.std(sha_cs):.4f}")

        # H4: Bio C-Score INCREASES over rounds (LEARNING)
        if bio and len(bio) >= 5:
            bio_cs_seq = [s['mean_cscore'] for s in bio]
            x = np.arange(len(bio_cs_seq))
            slope, intercept, r, p, se = stats.linregress(x, bio_cs_seq)
            sig = self._sig_str(p / 2 if slope > 0 else 1.0, bonferroni_alpha)  # one-tailed
            tests.append(('H4: Bio C-Score increases over rounds', slope, p / 2 if slope > 0 else 1.0, r, sig))
            self._log(f"\n  H4: Bio C-Score INCREASES over rounds (LEARNING)")
            self._log(f"    slope={slope:.6f}/round, r={r:.3f}, p={p/2:.6f} (one-tailed) {sig}")
            self._log(f"    Trajectory: {[f'{s:.3f}' for s in bio_cs_seq]}")

            # Also check shadow for comparison
            sha_cs_seq = [s['mean_cscore'] for s in shadow] if shadow else []
            if sha_cs_seq and len(sha_cs_seq) >= 5:
                slope_s, _, r_s, p_s, _ = stats.linregress(np.arange(len(sha_cs_seq)), sha_cs_seq)
                self._log(f"    Shadow trajectory: slope={slope_s:.6f}, r={r_s:.3f}, p={p_s/2:.4f}")

        # H5: Bio SRC INCREASES over rounds
        if bio and len(bio) >= 5:
            bio_src_seq = [s['mean_src'] for s in bio]
            x = np.arange(len(bio_src_seq))
            slope, intercept, r, p, se = stats.linregress(x, bio_src_seq)
            sig = self._sig_str(p / 2 if slope > 0 else 1.0, bonferroni_alpha)
            tests.append(('H5: Bio SRC increases over rounds', slope, p / 2 if slope > 0 else 1.0, r, sig))
            self._log(f"\n  H5: Bio SRC INCREASES over rounds (LEARNING)")
            self._log(f"    slope={slope:.6f}/round, r={r:.3f}, p={p/2:.6f} (one-tailed) {sig}")
            self._log(f"    Trajectory: {[f'{s:.4f}' for s in bio_src_seq]}")

        # H6: Template convergence differs Bio vs Shadow
        if bio and shadow:
            bio_conv = [s.get('convergence', 1.0) for s in bio if 'convergence' in s]
            sha_conv = [s.get('convergence', 1.0) for s in shadow if 'convergence' in s]
            if bio_conv and sha_conv and len(bio_conv) >= 3 and len(sha_conv) >= 3:
                # Bio should converge MORE (lower convergence value)
                U, p = stats.mannwhitneyu(bio_conv, sha_conv, alternative='less')
                d = self._cohens_d(sha_conv, bio_conv)  # reversed for positive d
                sig = self._sig_str(p, bonferroni_alpha)
                tests.append(('H6: Bio templates converge faster', U, p, d, sig))
                self._log(f"\n  H6: Bio templates converge faster than Shadow")
                self._log(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {sig}")
                self._log(f"    Bio convergence: {np.mean(bio_conv):.4f} ± {np.std(bio_conv):.4f}")
                self._log(f"    Shadow convergence: {np.mean(sha_conv):.4f} ± {np.std(sha_conv):.4f}")
            else:
                tests.append(('H6: Bio templates converge faster', 0, 1.0, 0.0, 'INSUFFICIENT'))
                self._log(f"\n  H6: INSUFFICIENT convergence data")

        # H7: Late-epoch Bio > Early-epoch Bio (DEVELOPMENTAL IMPROVEMENT)
        if bio and len(bio) >= 10:
            n_half = len(bio) // 2
            early_cs = [s['mean_cscore'] for s in bio[:n_half]]
            late_cs = [s['mean_cscore'] for s in bio[n_half:]]
            t, p = stats.ttest_ind(late_cs, early_cs, alternative='greater')
            d = self._cohens_d(late_cs, early_cs)
            sig = self._sig_str(p, bonferroni_alpha)
            tests.append(('H7: Late Bio > Early Bio', t, p, d, sig))
            self._log(f"\n  H7: Late-epoch Bio C-Score > Early-epoch (DEVELOPMENTAL)")
            self._log(f"    t={t:.2f}, p={p:.4f}, d={d:.3f} {sig}")
            self._log(f"    Early: {np.mean(early_cs):.4f} ± {np.std(early_cs):.4f}")
            self._log(f"    Late:  {np.mean(late_cs):.4f} ± {np.std(late_cs):.4f}")

            # Same for SRC
            early_src = [s['mean_src'] for s in bio[:n_half]]
            late_src = [s['mean_src'] for s in bio[n_half:]]
            self._log(f"    SRC: Early={np.mean(early_src):.4f}, Late={np.mean(late_src):.4f}")

        # H8: Shuffling degrades SRC
        if shadow:
            sha_src = [s['mean_src'] for s in shadow]
            sha_raw = [s['mean_src_raw'] for s in shadow]
            src_drop = [r - s for r, s in zip(sha_raw, sha_src)]
            if len(src_drop) >= 3 and np.std(src_drop) > 1e-10:
                t, p = stats.ttest_1samp(src_drop, 0, alternative='greater')
                d = float(np.mean(src_drop) / np.std(src_drop))
            else:
                t, p, d = 0, 1.0, 0.0
            sig = self._sig_str(p, bonferroni_alpha)
            tests.append(('H8: Shuffling degrades SRC', t, p, d, sig))
            self._log(f"\n  H8: Shuffling degrades SRC (replication)")
            self._log(f"    t={t:.2f}, p={p:.6f}, d={d:.3f} {sig}")
            self._log(f"    Mean SRC drop: {np.mean(src_drop):.4f} ± {np.std(src_drop):.4f}")

        # H9: High-alpha Bio > Standard-alpha Bio (DOSE-RESPONSE)
        if bio_high and bio:
            high_cs = [s['mean_cscore'] for s in bio_high]
            std_cs = [s['mean_cscore'] for s in bio[-len(bio_high):]]  # Compare with last N rounds
            if len(high_cs) >= 2 and len(std_cs) >= 2:
                U, p = stats.mannwhitneyu(high_cs, std_cs, alternative='greater')
                d = self._cohens_d(high_cs, std_cs)
                sig = self._sig_str(p, bonferroni_alpha)
                tests.append(('H9: High-α Bio > Std-α Bio', U, p, d, sig))
                self._log(f"\n  H9: High-α Bio C-Score > Standard-α Bio (DOSE-RESPONSE)")
                self._log(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {sig}")
                self._log(f"    High-α: {np.mean(high_cs):.4f} ± {np.std(high_cs):.4f}")
                self._log(f"    Std-α:  {np.mean(std_cs):.4f} ± {np.std(std_cs):.4f}")

                # SRC comparison
                high_src = [s['mean_src'] for s in bio_high]
                std_src = [s['mean_src'] for s in bio[-len(bio_high):]]
                self._log(f"    SRC: High-α={np.mean(high_src):.4f}, Std-α={np.mean(std_src):.4f}")

        # === SUMMARY ===
        n_sig = sum(1 for _, _, p, _, s in tests if 'SIG' in s or '***' in s or '**' in s)
        n_tests = len(tests)

        self._log(f"\n  {'='*73}")
        self._log(f"  RESULTS: {n_sig}/{n_tests} hypothesis tests significant")

        # Categorized verdict
        replication_tests = [t for t in tests if 'replication' in t[0].lower() or t[0].startswith(('H1', 'H3', 'H8'))]
        learning_tests = [t for t in tests if 'LEARN' in t[0].upper() or 'DEVELOP' in t[0].upper() or t[0].startswith(('H4', 'H5', 'H7'))]
        novel_tests = [t for t in tests if t[0].startswith(('H6', 'H9'))]

        n_replication = sum(1 for _, _, p, _, s in replication_tests if 'SIG' in s or '***' in s or '**' in s)
        n_learning = sum(1 for _, _, p, _, s in learning_tests if 'SIG' in s or '***' in s or '**' in s)
        n_novel = sum(1 for _, _, p, _, s in novel_tests if 'SIG' in s or '***' in s or '**' in s)

        self._log(f"    Replication (H1,H3,H8): {n_replication}/{len(replication_tests)}")
        self._log(f"    Learning (H4,H5,H7):    {n_learning}/{len(learning_tests)}")
        self._log(f"    Novel (H6,H9):           {n_novel}/{len(novel_tests)}")

        # Overall verdict
        if n_learning >= 2:
            verdict = "STRONG LEARNING — Bio-LLM develops learned representations over time"
        elif n_learning >= 1:
            verdict = "SUGGESTIVE LEARNING — partial evidence of developmental improvement"
        elif n_replication >= 2:
            verdict = "REPLICATION CONFIRMED — functional integration replicated, no learning detected"
        elif n_replication >= 1:
            verdict = "PARTIAL REPLICATION — some functional integration, no learning"
        else:
            verdict = "NO SIGNIFICANT EFFECTS"

        self._log(f"\n  VERDICT: {verdict}")

        # Decoder statistics
        self._log(f"\n  DECODER STATISTICS")
        for cond in ['bio_llm', 'shadow_llm', 'bio_llm_high']:
            decoder = self._decoders.get(cond)
            if decoder:
                st = decoder.get_stats()
                self._log(f"    {cond}: {st['n_templates']} templates, "
                          f"{st['n_updates']} updates, "
                          f"pred_acc={st['prediction_accuracy']:.3f}")
                if st['convergence_history']:
                    conv = st['convergence_history']
                    self._log(f"      Convergence: {[f'{c:.3f}' for c in conv]}")

        # Channel recruitment statistics
        recruit_stats = self._recruiter.get_stats()
        self._log(f"\n  CHANNEL RECRUITMENT")
        self._log(f"    Active channels:    {recruit_stats['active_channels']}/{recruit_stats['total_channels']}")
        self._log(f"    Recruited channels: {recruit_stats['recruited_channels']}")
        self._log(f"    Training rounds:    {recruit_stats['training_rounds']}")
        if recruit_stats['recruited_list']:
            self._log(f"    Recruited: {recruit_stats['recruited_list']}")

        self._log(f"  {'='*73}")

        # Save analysis
        analysis_grp = h5.create_group('analysis')
        for cond, s in analysis['conditions'].items():
            cg = analysis_grp.create_group(cond)
            for k, v in s.items():
                cg.attrs[k] = v

        for i, (name, stat, p, d, sig) in enumerate(tests):
            tg = analysis_grp.create_group(f'test_{i}')
            tg.attrs['name'] = name
            tg.attrs['statistic'] = float(stat)
            tg.attrs['p'] = float(p)
            tg.attrs['effect_size'] = float(d)
            tg.attrs['significant'] = 'SIG' in sig or '***' in sig or '**' in sig

        # Save decoder convergence histories
        for cond in ['bio_llm', 'shadow_llm', 'bio_llm_high']:
            decoder = self._decoders.get(cond)
            if decoder and decoder._convergence_history:
                analysis_grp.create_dataset(
                    f'{cond}_convergence',
                    data=np.array(decoder._convergence_history, dtype=np.float32)
                )
            if decoder and decoder._template_norm_history:
                analysis_grp.create_dataset(
                    f'{cond}_template_norms',
                    data=np.array(decoder._template_norm_history, dtype=np.float32)
                )

        # Save recruitment data
        recruit_stats = self._recruiter.get_stats()
        rg = analysis_grp.create_group('channel_recruitment')
        rg.attrs['active_channels'] = recruit_stats['active_channels']
        rg.attrs['total_channels'] = recruit_stats['total_channels']
        rg.attrs['recruited_channels'] = recruit_stats['recruited_channels']
        activity_arr = np.array([recruit_stats['channel_activity'].get(ch, 0)
                                 for ch in ALL_USABLE_CHANNELS], dtype=np.int32)
        rg.create_dataset('channel_activity', data=activity_arr)

        analysis_path = os.path.join(self.output_dir,
                                     f"cl1_v3_analysis_{self.timestamp}.json")
        analysis['tests'] = [(n, float(s), float(p), float(d), sig)
                             for n, s, p, d, sig in tests]
        analysis['n_significant'] = n_sig
        analysis['n_tests'] = n_tests
        analysis['verdict'] = verdict
        analysis['categories'] = {
            'replication': n_replication,
            'learning': n_learning,
            'novel': n_novel,
        }

        analysis['channel_recruitment'] = {
            'active': recruit_stats['active_channels'],
            'total': recruit_stats['total_channels'],
            'recruited': recruit_stats['recruited_channels'],
        }

        # Save learning trajectories
        for cond in ['bio_llm', 'shadow_llm', 'llm_only', 'bio_llm_high']:
            valid = [s for s in summaries.get(cond, []) if 'error' not in s]
            if valid:
                analysis[f'{cond}_trajectory'] = {
                    'cscore': [s['mean_cscore'] for s in valid],
                    'src': [s['mean_src'] for s in valid],
                    'convergence': [s.get('convergence', None) for s in valid],
                }

        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        self._log(f"  Analysis saved: {analysis_path}")

        return analysis

    @staticmethod
    def _cohens_d(a, b):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        if pooled < 1e-10:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / pooled)

    @staticmethod
    def _sig_str(p, alpha):
        if p < alpha / 10:
            return "*** SIGNIFICANT"
        elif p < alpha:
            return "** SIGNIFICANT"
        elif p < 0.05:
            return "* (nominal only)"
        return "n.s."


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CL1 Experiment v3 — Extended Learning')
    parser.add_argument('--local', action='store_true',
                        help='Use local Izhikevich instead of CL1')
    parser.add_argument('--rounds-p1', type=int, default=15,
                        help='Phase 1 rounds (3-condition)')
    parser.add_argument('--rounds-p2', type=int, default=5,
                        help='Phase 2 rounds (high-alpha Bio)')
    parser.add_argument('--tokens', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--high-alpha', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-feedback', action='store_true')

    args = parser.parse_args()

    if args.local:
        from LLM_Encoder.neural_substrate import IzhikevichSubstrate
        substrate = IzhikevichSubstrate(seed=args.seed)
        print("  Using local Izhikevich substrate")
    else:
        from LLM_Encoder.cl1_cloud_substrate import CL1CloudSubstrate
        substrate = CL1CloudSubstrate(seed=args.seed)
        if not substrate.is_connected:
            print("ERROR: Cannot connect to CL1. Use --local for Izhikevich.")
            sys.exit(1)

    exp = CL1ExperimentV3(
        substrate=substrate,
        alpha=args.alpha,
        high_alpha=args.high_alpha,
        tokens_per_thought=args.tokens,
        n_rounds_phase1=args.rounds_p1,
        n_rounds_phase2=args.rounds_p2,
        seed=args.seed,
        feedback_enabled=not args.no_feedback,
    )

    analysis = exp.run()

    if hasattr(substrate, 'close'):
        substrate.close()

    return analysis


if __name__ == '__main__':
    main()
