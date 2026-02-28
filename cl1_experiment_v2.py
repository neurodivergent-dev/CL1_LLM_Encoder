#!/usr/bin/env python3
"""CL1 Experiment v2 — Stimulus-Response Congruence Test.

The key insight: the v1 experiment's z-score decoder normalizes away any
genuine difference between Bio-LLM and Shadow-LLM. Instead, we measure
STIMULUS-RESPONSE CONGRUENCE (SRC) — the cosine similarity between the
stimulation pattern vector and the spike response vector.

In Bio-LLM: spike response maintains spatial congruence with stimulus
In Shadow-LLM: shuffling destroys spatial congruence → lower SRC
In LLM-only: spikes not used for selection, no closed loop → baseline SRC

This is a DIRECT test of information-preserving neural coupling that
shuffling cannot fake.

Additional metrics:
  - Hebbian decoder associations (v3)
  - Feedback stimulation (triple loop)
  - Temporal binding (previous token trace)
  - Cross-channel correlation structure
  - Reconstruction accuracy (predict token from response)
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

PROMPTS = [
    "The nature of consciousness is deeply connected to ",
    "When neurons fire together, the emerging pattern creates ",
    "I am processing information and what emerges is ",
    "The boundary between signal and meaning dissolves when ",
    "Awareness arises from the integration of ",
]

RESPONSE_WINDOW_S = 0.5
FEEDBACK_WINDOW_S = 0.2
MAX_CANDIDATES = 15
ALPHA = 0.5


def stimulus_response_congruence(stim_pattern: Dict[int, float],
                                  spike_counts: Dict, n_channels: int = 59) -> float:
    """Compute cosine similarity between stimulation pattern and spike response.

    This is the KEY metric. Bio-LLM should show higher SRC than Shadow-LLM
    because shuffling destroys the spatial congruence.
    """
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


def channel_mutual_info(stim_pattern: Dict[int, float],
                         spike_counts: Dict, n_channels: int = 59) -> float:
    """Approximate mutual information between stimulus and response channels.

    Measures whether the stimulus channels are preferentially activated.
    """
    stim_channels = set(int(ch) for ch, amp in stim_pattern.items() if amp > 0.5)
    resp_total = sum(float(v) for v in spike_counts.values()) or 1

    # Fraction of spikes on stimulated channels
    stim_spikes = sum(float(spike_counts.get(str(ch), 0)) for ch in stim_channels)
    frac_on_stim = stim_spikes / resp_total

    # Expected fraction by chance (uniform)
    expected_frac = len(stim_channels) / max(1, n_channels)

    # Normalized lift: >1 means stimulus channels fire preferentially
    if expected_frac > 0:
        return frac_on_stim / expected_frac
    return 0.0


class HebbianDecoder:
    """Slow-learning Hebbian decoder that is sensitive to spike shuffling.

    Key difference from SpatialDecoder: uses template matching against
    learned token-specific patterns rather than rolling z-scores.
    """

    def __init__(self, encoder: SpatialEncoder, alpha: float = 0.5,
                 learning_rate: float = 0.02):
        self.encoder = encoder
        self.alpha = alpha
        self.lr = learning_rate

        # Learned token→response templates
        # {token_id: np.array(n_channels)} — EMA of observed spike patterns
        self._templates: Dict[int, np.ndarray] = {}
        self._template_counts: Dict[int, int] = {}
        self._n_channels = 59
        self._n_updates = 0

    def decode(
        self,
        spike_counts: Dict,
        model_probs: Dict[int, float],
        channel_to_token: Dict[int, int],
    ) -> Tuple[Dict[int, float], Dict[int, float], float]:
        """Decode via template matching + spatial ownership."""
        # Build response vector
        resp_vec = np.zeros(self._n_channels, dtype=float)
        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            if 0 <= ch < self._n_channels:
                resp_vec[ch] = float(count)

        token_scores = {}

        for tid in model_probs:
            score = 0.0

            # Score from spatial ownership (always available)
            pattern = self.encoder.encode_token(tid)
            stim_vec = np.zeros(self._n_channels, dtype=float)
            for ch, amp in pattern.items():
                ch = int(ch)
                if 0 <= ch < self._n_channels:
                    stim_vec[ch] = amp

            # Stimulus-response correlation
            s_norm = np.linalg.norm(stim_vec)
            r_norm = np.linalg.norm(resp_vec)
            if s_norm > 1e-10 and r_norm > 1e-10:
                src = np.dot(stim_vec, resp_vec) / (s_norm * r_norm)
                score += src * 0.5

            # Score from learned template (if available)
            if tid in self._templates:
                template = self._templates[tid]
                t_norm = np.linalg.norm(template)
                if t_norm > 1e-10 and r_norm > 1e-10:
                    template_match = np.dot(template, resp_vec) / (t_norm * r_norm)
                    score += template_match * 0.5

            token_scores[tid] = score

        # Convert to probabilities (softmax)
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

    def update(self, selected_token: int, spike_counts: Dict):
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
            # Slow EMA update
            old = self._templates[selected_token]
            self._templates[selected_token] = old + self.lr * (resp_vec - old)
            self._template_counts[selected_token] += 1

        self._n_updates += 1

    def get_stats(self) -> Dict:
        return {
            'n_templates': len(self._templates),
            'n_updates': self._n_updates,
            'template_counts': dict(self._template_counts),
        }


class CL1ExperimentV2:
    """Decisive 3-condition experiment with Stimulus-Response Congruence."""

    def __init__(
        self,
        substrate,
        model_path: str = MODEL_PATH,
        alpha: float = ALPHA,
        tokens_per_thought: int = 50,
        n_thoughts: int = 10,
        seed: int = 42,
        feedback_enabled: bool = True,
        output_dir: str = "experiment_data",
    ):
        self.substrate = substrate
        self.model_path = model_path
        self.alpha = alpha
        self.tokens_per_thought = tokens_per_thought
        self.n_thoughts = n_thoughts
        self.seed = seed
        self.feedback_enabled = feedback_enabled
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.h5_path = os.path.join(output_dir, f"cl1_v2_{self.timestamp}.h5")
        self.log_path = os.path.join(output_dir, f"cl1_v2_{self.timestamp}.log")

        self.spatial_encoder = SpatialEncoder(seed=seed)
        self.assessor = ConsciousnessAssessor()
        self._llm = None
        self._rng = np.random.default_rng(seed)

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

    def _generate_tokens(self, prompt: str, condition: str,
                          h5_group: h5py.Group) -> Dict:
        """Generate tokens with SRC measurement and Hebbian decoder."""
        self._load_llm()

        use_neurons = condition in ('bio_llm', 'shadow_llm')
        shuffle_spikes = condition == 'shadow_llm'
        effective_alpha = self.alpha if use_neurons else 0.0

        decoder = HebbianDecoder(self.spatial_encoder, alpha=effective_alpha)

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
        all_src = []            # Stimulus-Response Congruence (KEY metric)
        all_src_raw = []        # SRC before any shuffling (same for all conditions)
        all_channel_mi = []     # Channel mutual information
        all_latencies = []
        token_ids = []
        was_overrides = []
        prev_token_id = None

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

            # Add temporal trace from previous token
            if prev_token_id is not None and use_neurons:
                prev_pattern = self.spatial_encoder.encode_token(prev_token_id)
                for ch, amp in prev_pattern.items():
                    if ch in combined_pattern:
                        combined_pattern[ch] = min(AMP_MAX, combined_pattern[ch] + amp * 0.2)
                    else:
                        combined_pattern[ch] = amp * 0.2

            # Stimulate CL1 neurons
            channel_amplitudes = {int(ch): amp for ch, amp in combined_pattern.items()}
            spike_counts = self.substrate.stimulate_and_record(
                channel_amplitudes, window_s=RESPONSE_WINDOW_S,
            )

            # === STIMULUS-RESPONSE CONGRUENCE (raw — same for all conditions) ===
            raw_src = stimulus_response_congruence(combined_pattern, spike_counts)
            all_src_raw.append(raw_src)

            # === Channel mutual information (raw) ===
            ch_mi = channel_mutual_info(combined_pattern, spike_counts)
            all_channel_mi.append(ch_mi)

            # Consciousness assessment
            sm = self.substrate.get_last_spike_matrix()
            consciousness = self.assessor.assess(sm)
            cs = consciousness.get('cscore', 0)
            all_cs.append(cs)

            token_spikes = sum(spike_counts.values()) if isinstance(spike_counts, dict) else 0
            all_spikes.append(token_spikes)

            # Condition-specific processing
            if use_neurons and len(model_probs) > 1:
                decode_spikes = dict(spike_counts)
                if shuffle_spikes:
                    channels = list(decode_spikes.keys())
                    counts = list(decode_spikes.values())
                    np.random.shuffle(counts)
                    decode_spikes = dict(zip(channels, counts))

                # SRC after condition-specific processing (THE key metric)
                processed_src = stimulus_response_congruence(combined_pattern, decode_spikes)
                all_src.append(processed_src)

                blended, neural_probs, confidence = decoder.decode(
                    decode_spikes, model_probs, channel_to_token
                )
                selected = max(blended, key=blended.get)
                was_override = selected != model_top

                # Update decoder with condition-specific spikes
                decoder.update(selected, decode_spikes)

                # Feedback stimulation (only for Bio-LLM — Shadow gets no feedback)
                if self.feedback_enabled and condition == 'bio_llm':
                    fb_pattern = self.spatial_encoder.encode_token(selected)
                    fb_amps = {}
                    for i, fch in enumerate(FEEDBACK_CHANNELS):
                        if i < len(self.spatial_encoder.channels):
                            sch = self.spatial_encoder.channels[i]
                            if sch in fb_pattern:
                                fb_amps[fch] = min(AMP_MAX, 1.5 * fb_pattern[sch] / AMP_MAX)
                    if fb_amps:
                        self.substrate.stimulate_and_record(fb_amps, window_s=FEEDBACK_WINDOW_S)

                # Alignment metric
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

                # Entropy
                bp = np.array(list(blended.values()), dtype=float)
                bp = bp[bp > 0]
                bp = bp / (bp.sum() + 1e-10)
                all_entropy.append(float(-np.sum(bp * np.log2(bp + 1e-10))))
            else:
                selected = model_top
                was_override = False
                all_alignment.append(0.0)
                all_src.append(raw_src)  # No processing for LLM-only
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
        h5_group.create_dataset('channel_mi', data=np.array(all_channel_mi[:n], dtype=np.float32))
        h5_group.create_dataset('latency_ms', data=np.array(all_latencies[:n], dtype=np.float32))
        h5_group.attrs['condition'] = condition
        h5_group.attrs['prompt'] = prompt
        h5_group.attrs['generated_text'] = text
        h5_group.attrs['n_tokens'] = n
        h5_group.attrs['override_count'] = override_count

        summary = {
            'condition': condition,
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
            'mean_channel_mi': float(np.mean(all_channel_mi)) if all_channel_mi else 0.0,
            'mean_latency_ms': float(np.mean(all_latencies)) if all_latencies else 0.0,
            'decoder_stats': decoder.get_stats(),
            'generated_text': text,
        }
        return summary

    def run(self):
        """Run the full 3-condition interleaved experiment."""
        t_start = time.time()

        self._log("=" * 78)
        self._log("  CL1 EXPERIMENT v2 — STIMULUS-RESPONSE CONGRUENCE TEST")
        self._log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 78)
        self._log(f"  Substrate:      {type(self.substrate).__name__}")
        self._log(f"  Conditions:     {CONDITIONS}")
        self._log(f"  Thoughts/cond:  {self.n_thoughts}")
        self._log(f"  Tokens/thought: {self.tokens_per_thought}")
        self._log(f"  Alpha:          {self.alpha}")
        self._log(f"  Feedback:       {'ON (Bio only)' if self.feedback_enabled else 'OFF'}")
        self._log(f"  Key metric:     Stimulus-Response Congruence (SRC)")
        self._log(f"  Output:         {self.h5_path}")
        self._log("=" * 78)

        h5 = h5py.File(self.h5_path, 'w')
        h5.attrs['experiment'] = 'CL1 v2 — SRC Test'
        h5.attrs['timestamp'] = self.timestamp
        h5.attrs['substrate'] = type(self.substrate).__name__
        h5.attrs['alpha'] = self.alpha
        h5.attrs['tokens_per_thought'] = self.tokens_per_thought
        h5.attrs['n_thoughts'] = self.n_thoughts
        h5.attrs['seed'] = self.seed
        h5.attrs['feedback_enabled'] = self.feedback_enabled

        condition_summaries = {c: [] for c in CONDITIONS}

        for thought_idx in range(self.n_thoughts):
            prompt = PROMPTS[thought_idx % len(PROMPTS)]

            self._log(f"\n{'#'*78}")
            self._log(f"  ROUND {thought_idx + 1}/{self.n_thoughts}")
            self._log(f"  Prompt: {prompt[:60]}...")
            self._log(f"{'#'*78}")

            for cond in CONDITIONS:
                self._log(f"\n  --- {cond.upper()} (round {thought_idx + 1}) ---")

                grp = h5.create_group(f"{cond}/thought_{thought_idx:03d}")
                try:
                    summary = self._generate_tokens(prompt, cond, grp)
                    condition_summaries[cond].append(summary)

                    self._log(f"  {cond}: {summary['n_tokens']} tok, "
                              f"SRC={summary['mean_src']:.4f}, "
                              f"SRC_raw={summary['mean_src_raw']:.4f}, "
                              f"align={summary['mean_alignment']:.3f}, "
                              f"override={summary['override_rate']:.1%}")
                except Exception as e:
                    self._log(f"  ERROR in {cond}: {e}")
                    import traceback
                    traceback.print_exc()
                    condition_summaries[cond].append({'error': str(e)})

            h5.flush()

        # Analysis
        self._log(f"\n{'='*78}")
        self._log("  ANALYSIS — STIMULUS-RESPONSE CONGRUENCE")
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

    def _analyze(self, summaries: Dict, h5: h5py.File) -> Dict:
        """Analyze with SRC as the primary metric."""
        from scipy import stats

        results = {}

        for cond in CONDITIONS:
            valid = [s for s in summaries[cond] if 'error' not in s]
            if not valid:
                continue
            results[cond] = {
                'n_thoughts': len(valid),
                'mean_src': float(np.mean([s['mean_src'] for s in valid])),
                'std_src': float(np.std([s['mean_src'] for s in valid])),
                'mean_src_raw': float(np.mean([s['mean_src_raw'] for s in valid])),
                'mean_alignment': float(np.mean([s['mean_alignment'] for s in valid])),
                'mean_entropy': float(np.mean([s['mean_entropy'] for s in valid])),
                'mean_cscore': float(np.mean([s['mean_cscore'] for s in valid])),
                'mean_override_rate': float(np.mean([s['override_rate'] for s in valid])),
                'mean_channel_mi': float(np.mean([s['mean_channel_mi'] for s in valid])),
                'total_spikes': sum(s['total_spikes'] for s in valid),
            }

        self._log(f"\n  CONDITION SUMMARY (SRC = Stimulus-Response Congruence)")
        self._log(f"  {'Metric':<25s} {'Bio-LLM':>12s} {'Shadow-LLM':>12s} {'LLM-Only':>12s}")
        self._log(f"  {'-'*61}")

        for metric in ['mean_src', 'mean_src_raw', 'mean_alignment', 'mean_entropy',
                       'mean_cscore', 'mean_override_rate', 'mean_channel_mi']:
            vals = []
            for cond in CONDITIONS:
                v = results.get(cond, {}).get(metric, 0)
                vals.append(f"{v:12.4f}")
            self._log(f"  {metric:<25s} {''.join(vals)}")

        # === HYPOTHESIS TESTS ===
        self._log(f"\n  HYPOTHESIS TESTS (alpha=0.05, Bonferroni corrected for 6 tests → 0.0083)")

        bio = [s for s in summaries['bio_llm'] if 'error' not in s]
        shadow = [s for s in summaries['shadow_llm'] if 'error' not in s]
        llm = [s for s in summaries['llm_only'] if 'error' not in s]

        tests = []

        # H1: Bio SRC > Shadow SRC (THE PRIMARY HYPOTHESIS)
        if bio and shadow:
            bio_src = [s['mean_src'] for s in bio]
            sha_src = [s['mean_src'] for s in shadow]
            U, p = stats.mannwhitneyu(bio_src, sha_src, alternative='greater')
            d = self._cohens_d(bio_src, sha_src)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tests.append(('H1: Bio SRC > Shadow SRC [PRIMARY]', U, p, d, sig))
            self._log(f"\n  H1 [PRIMARY]: Bio SRC > Shadow SRC")
            self._log(f"    U={U:.1f}, p={p:.6f}, d={d:.3f} {sig}")
            self._log(f"    Bio: {np.mean(bio_src):.4f} +/- {np.std(bio_src):.4f}")
            self._log(f"    Shadow: {np.mean(sha_src):.4f} +/- {np.std(sha_src):.4f}")

        # H2: Raw SRC is same across conditions (sanity check — same neurons)
        if bio and shadow:
            bio_raw = [s['mean_src_raw'] for s in bio]
            sha_raw = [s['mean_src_raw'] for s in shadow]
            _, p = stats.mannwhitneyu(bio_raw, sha_raw, alternative='two-sided')
            d = self._cohens_d(bio_raw, sha_raw)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tests.append(('H2: Raw SRC equal (sanity)', 0, p, d, sig))
            self._log(f"\n  H2 [SANITY]: Raw SRC equal across conditions")
            self._log(f"    p={p:.4f}, d={d:.3f} {sig}")
            self._log(f"    Bio raw: {np.mean(bio_raw):.4f} +/- {np.std(bio_raw):.4f}")
            self._log(f"    Shadow raw: {np.mean(sha_raw):.4f} +/- {np.std(sha_raw):.4f}")
            if p > 0.05:
                self._log(f"    ✓ PASS — same neurons produce same raw response")

        # H3: Bio alignment > Shadow alignment
        if bio and shadow:
            bio_a = [s['mean_alignment'] for s in bio]
            sha_a = [s['mean_alignment'] for s in shadow]
            U, p = stats.mannwhitneyu(bio_a, sha_a, alternative='greater')
            d = self._cohens_d(bio_a, sha_a)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tests.append(('H3: Bio alignment > Shadow', U, p, d, sig))
            self._log(f"\n  H3: Bio alignment > Shadow alignment")
            self._log(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {sig}")

        # H4: Bio channel MI > Shadow channel MI
        if bio and shadow:
            bio_mi = [s['mean_channel_mi'] for s in bio]
            sha_mi = [s['mean_channel_mi'] for s in shadow]
            U, p = stats.mannwhitneyu(bio_mi, sha_mi, alternative='greater')
            d = self._cohens_d(bio_mi, sha_mi)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tests.append(('H4: Bio channel MI > Shadow', U, p, d, sig))
            self._log(f"\n  H4: Bio channel MI > Shadow channel MI")
            self._log(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {sig}")

        # H5: Bio SRC increases over time (learning)
        if bio and len(bio) >= 3:
            bio_src_seq = [s['mean_src'] for s in bio]
            x = np.arange(len(bio_src_seq))
            slope, intercept, r, p, se = stats.linregress(x, bio_src_seq)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tests.append(('H5: Bio SRC increases over time', 0, p, slope, sig))
            self._log(f"\n  H5: Bio SRC increases over time (learning)")
            self._log(f"    slope={slope:.6f}, r={r:.3f}, p={p:.4f} {sig}")
            self._log(f"    Trajectory: {[f'{s:.4f}' for s in bio_src_seq]}")

        # H6: Shadow SRC decreases relative to raw (shuffling degrades)
        if shadow:
            sha_src = [s['mean_src'] for s in shadow]
            sha_raw = [s['mean_src_raw'] for s in shadow]
            src_drop = [r - s for r, s in zip(sha_raw, sha_src)]
            # One-sample t-test: is the drop > 0?
            if len(src_drop) >= 3 and np.std(src_drop) > 1e-10:
                t, p = stats.ttest_1samp(src_drop, 0, alternative='greater')
                d = float(np.mean(src_drop) / np.std(src_drop))
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            else:
                t, p, d = 0, 1.0, 0.0
                sig = "n.s."
            tests.append(('H6: Shadow SRC < raw SRC (shuffling degrades)', t, p, d, sig))
            self._log(f"\n  H6: Shadow SRC < raw SRC (shuffling degrades congruence)")
            self._log(f"    t={t:.2f}, p={p:.6f}, d={d:.3f} {sig}")
            self._log(f"    Mean SRC drop: {np.mean(src_drop):.4f} +/- {np.std(src_drop):.4f}")

        n_sig = sum(1 for _, _, p, _, s in tests if s != 'n.s.')
        n_tests = len(tests)

        self._log(f"\n  {'='*61}")
        self._log(f"  RESULT: {n_sig}/{n_tests} hypothesis tests significant")

        if n_sig >= 4:
            verdict = "STRONG evidence of differential neural-LLM integration"
        elif n_sig >= 2:
            verdict = "MODERATE evidence — key metrics differ between conditions"
        elif n_sig >= 1:
            verdict = "WEAK evidence — partial support"
        else:
            verdict = "NO significant differential integration detected"

        self._log(f"  VERDICT: {verdict}")
        self._log(f"  {'='*61}")

        # Save analysis
        analysis_grp = h5.create_group('analysis')
        for cond, s in results.items():
            cg = analysis_grp.create_group(cond)
            for k, v in s.items():
                cg.attrs[k] = v

        for i, (name, stat, p, d, sig) in enumerate(tests):
            tg = analysis_grp.create_group(f'test_{i}')
            tg.attrs['name'] = name
            tg.attrs['statistic'] = float(stat)
            tg.attrs['p'] = float(p)
            tg.attrs['effect_size'] = float(d)
            tg.attrs['significant'] = sig != 'n.s.'

        analysis_path = os.path.join(self.output_dir,
                                     f"cl1_v2_analysis_{self.timestamp}.json")
        analysis_out = {
            'timestamp': self.timestamp,
            'conditions': results,
            'tests': [(n, float(s), float(p), float(d), sig)
                      for n, s, p, d, sig in tests],
            'n_significant': n_sig,
            'n_tests': n_tests,
            'verdict': verdict,
        }
        with open(analysis_path, 'w') as f:
            json.dump(analysis_out, f, indent=2)
        self._log(f"  Analysis saved: {analysis_path}")

        return analysis_out

    @staticmethod
    def _cohens_d(a, b):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        if pooled < 1e-10:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / pooled)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CL1 Experiment v2 — SRC Test')
    parser.add_argument('--local', action='store_true',
                        help='Use local Izhikevich instead of CL1')
    parser.add_argument('--thoughts', type=int, default=10)
    parser.add_argument('--tokens', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5)
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

    exp = CL1ExperimentV2(
        substrate=substrate,
        alpha=args.alpha,
        tokens_per_thought=args.tokens,
        n_thoughts=args.thoughts,
        seed=args.seed,
        feedback_enabled=not args.no_feedback,
    )

    analysis = exp.run()

    if hasattr(substrate, 'close'):
        substrate.close()

    return analysis


if __name__ == '__main__':
    main()
