"""Token Voting Encoder v3 — Feedback-Augmented Closed Loop.

Key improvements over v2:
  1. FEEDBACK STIMULATION: After token selection, the substrate receives a
     "reward" signal on the channels that voted for the winning token. This
     creates a genuine perception-action-feedback loop (like biological systems).

  2. LEARNED DECODER: Instead of rolling z-scores (which adapt too fast and
     mask perturbation effects), the decoder maintains a learned mapping from
     channel spike patterns to token probabilities that accumulates over time.

  3. CONTEXT-DEPENDENT ENCODING: The stimulation pattern includes not just
     the current token candidates but also a trace of the previous token,
     creating temporal binding across the substrate.

Architecture:
  LLM logprobs → SpatialEncoder(token, prob) → substrate → spikes
    → LearnedDecoder → blended probs → token selection
    → FeedbackStim(selected_token) → substrate (reinforcement)
    → next LLM inference (closed triple loop)

The triple loop (stimulus → response → selection → feedback) should produce
measurably tighter integration than v2's single loop, because the substrate
receives direct reinforcement about which of its responses "won".
"""

import os
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .neural_substrate import IzhikevichSubstrate, IzhikevichConfig
from .spatial_encoder import SpatialEncoder, SpatialDecoder, FEEDBACK_CHANNELS, AMP_MIN, AMP_MAX
from .consciousness import ConsciousnessAssessor

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "LFM2-350M-Q4_0.gguf"
)

RESPONSE_WINDOW_S = 0.5
FEEDBACK_WINDOW_S = 0.2  # shorter feedback pulse
DEFAULT_ALPHA = 0.5
MAX_CANDIDATES = 15


@dataclass
class TokenRecordV3:
    """Per-token record with feedback metrics."""
    position: int
    token_id: int
    token_text: str
    model_top_token: int
    was_override: bool
    alpha: float
    latency_ms: float
    total_spikes: int
    # Coupling metrics
    blended_entropy: float = 0.0
    neural_llm_alignment: float = 0.0
    model_top_prob_boost: float = 0.0
    # Feedback metrics
    feedback_spikes: int = 0
    feedback_channels: int = 0
    # Consciousness
    consciousness_metrics: Optional[Dict] = None
    # Decoder learning
    decoder_confidence: float = 0.0  # how confident the learned decoder is


class LearnedDecoder:
    """Token probability decoder with persistent learned mappings.

    Unlike SpatialDecoder (rolling z-scores), this decoder learns a
    channel→token mapping that persists across the session. This means
    perturbation effects are NOT absorbed by an adaptive baseline.

    The learned mapping is updated by Hebbian association: when token T
    is selected and channel C had high spike count, strengthen C→T.
    """

    def __init__(self, encoder: SpatialEncoder, alpha: float = 0.5, learning_rate: float = 0.01):
        self.encoder = encoder
        self.alpha = alpha
        self.lr = learning_rate

        # Learned channel→token association weights
        # Shape conceptually: {channel: {token_id: weight}}
        self._associations: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Channel baseline statistics (slowly updating)
        self._channel_means: Dict[int, float] = {}
        self._channel_vars: Dict[int, float] = {}
        self._channel_n: Dict[int, int] = {}
        self._ema_alpha = 0.01  # very slow EMA for baseline (NOT adaptive)

        # Confidence tracking
        self._n_updates = 0

    def decode(
        self,
        spike_counts: Dict[str, int],
        model_probs: Dict[int, float],
        channel_to_token: Dict[int, int],
    ) -> Tuple[Dict[int, float], Dict[int, float], float]:
        """Decode spike response into blended token probabilities.

        Returns (blended_probs, neural_probs, decoder_confidence).
        """
        # 1. Update channel baselines (slow EMA)
        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            c = float(count)
            if ch not in self._channel_means:
                self._channel_means[ch] = c
                self._channel_vars[ch] = 1.0
                self._channel_n[ch] = 1
            else:
                old_mean = self._channel_means[ch]
                self._channel_means[ch] += self._ema_alpha * (c - old_mean)
                self._channel_vars[ch] += self._ema_alpha * ((c - old_mean)**2 - self._channel_vars[ch])
                self._channel_n[ch] += 1

        # 2. Compute neural probs from LEARNED associations
        token_scores = defaultdict(float)

        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            # Z-score against slow baseline
            mean = self._channel_means.get(ch, 0)
            std = max(1.0, np.sqrt(self._channel_vars.get(ch, 1.0)))
            z = (float(count) - mean) / std

            # Score from channel-token ownership (spatial encoder)
            owner_token = channel_to_token.get(ch)
            if owner_token is not None:
                token_scores[owner_token] += z * 0.5  # spatial contribution

            # Score from learned associations (Hebbian)
            if ch in self._associations:
                for tid, weight in self._associations[ch].items():
                    if tid in model_probs:  # only score candidate tokens
                        token_scores[tid] += z * weight

        # 3. Convert scores to probabilities (softmax)
        neural_probs = {}
        if token_scores:
            max_score = max(token_scores.values())
            for tid, score in token_scores.items():
                neural_probs[tid] = np.exp(score - max_score)
            total = sum(neural_probs.values())
            if total > 0:
                neural_probs = {k: v / total for k, v in neural_probs.items()}

        # 4. Blend with model probabilities
        all_tokens = set(model_probs.keys()) | set(neural_probs.keys())
        blended = {}
        for tok in all_tokens:
            mp = model_probs.get(tok, 0.0)
            np_ = neural_probs.get(tok, 0.0)
            blended[tok] = (1 - self.alpha) * mp + self.alpha * np_

        total = sum(blended.values())
        if total > 0:
            blended = {tok: p / total for tok, p in blended.items()}

        # Confidence: how well-established are the associations?
        confidence = min(1.0, self._n_updates / 50.0)  # ramps up over first 50 tokens

        return blended, neural_probs, confidence

    def update(self, selected_token: int, spike_counts: Dict[str, int], channel_to_token: Dict[int, int]):
        """Hebbian update: strengthen association between active channels and selected token.

        Called AFTER token selection to teach the decoder which channels
        "correctly" voted for the winning token.
        """
        if not spike_counts:
            return

        # Find which channels had above-average spikes
        counts = np.array([float(v) for v in spike_counts.values()])
        threshold = np.mean(counts) if len(counts) > 0 else 0

        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            if float(count) > threshold:
                # Strengthen association: this channel → selected token
                self._associations[ch][selected_token] += self.lr * (float(count) - threshold) / (threshold + 1)

                # Weak anti-Hebbian: slightly weaken associations to NON-selected tokens
                for tid in list(self._associations[ch].keys()):
                    if tid != selected_token:
                        self._associations[ch][tid] *= (1 - self.lr * 0.1)

        # Clip association weights
        for ch in self._associations:
            for tid in list(self._associations[ch].keys()):
                self._associations[ch][tid] = np.clip(self._associations[ch][tid], -2.0, 2.0)
                # Prune near-zero weights
                if abs(self._associations[ch][tid]) < 0.001:
                    del self._associations[ch][tid]

        self._n_updates += 1

    def get_association_stats(self) -> Dict:
        """Return statistics about learned associations."""
        n_assoc = sum(len(v) for v in self._associations.values())
        all_weights = [w for ch_dict in self._associations.values() for w in ch_dict.values()]
        return {
            'n_associations': n_assoc,
            'n_channels_with_associations': len(self._associations),
            'mean_weight': float(np.mean(all_weights)) if all_weights else 0.0,
            'max_weight': float(np.max(all_weights)) if all_weights else 0.0,
            'n_updates': self._n_updates,
        }


class TokenVotingEngineV3:
    """LLM + Neural Substrate with feedback-augmented closed loop.

    v3 improvements:
    - Feedback stimulation after token selection (triple loop)
    - Learned decoder with persistent associations (perturbation-sensitive)
    - Context-dependent encoding with temporal trace
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        alpha: float = DEFAULT_ALPHA,
        substrate_seed: int = 42,
        n_gpu_layers: int = -1,
        feedback_enabled: bool = True,
        feedback_amplitude: float = 1.5,  # µA for feedback stim
    ):
        self.model_path = model_path
        self.alpha = alpha
        self.n_gpu_layers = n_gpu_layers
        self.substrate_seed = substrate_seed
        self.feedback_enabled = feedback_enabled
        self.feedback_amplitude = feedback_amplitude

        self.substrate = IzhikevichSubstrate(seed=substrate_seed)
        self.spatial_encoder = SpatialEncoder(seed=substrate_seed)

        self._llm = None
        self._prev_token_id = None  # for temporal binding

    def _load_model(self, force_reload: bool = False):
        if self._llm is not None and not force_reload:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            print("ERROR: pip install llama-cpp-python")
            sys.exit(1)

        if not os.path.exists(self.model_path):
            print(f"ERROR: Model not found at {self.model_path}")
            sys.exit(1)

        if force_reload and self._llm is not None:
            del self._llm
            self._llm = None

        print(f"  Loading LLM: {os.path.basename(self.model_path)}")
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=2048,
            logits_all=True,
            verbose=False,
        )
        print(f"  LLM loaded.")

    def _send_feedback(self, selected_token: int, spike_counts: Dict[str, int],
                       channel_to_token: Dict[int, int]):
        """Send feedback stimulation for the selected token.

        The feedback signal goes to FEEDBACK_CHANNELS (separate from
        sensory channels) and encodes:
        - WHICH token was selected (spatial pattern on feedback channels)
        - HOW much the substrate's response contributed (scaled amplitude)
        """
        if not self.feedback_enabled:
            return 0

        # Get the spatial pattern for the selected token
        pattern = self.spatial_encoder.encode_token(selected_token)

        # Map sensory pattern to feedback channels
        # Each sensory channel maps to a corresponding feedback channel
        feedback_pattern = {}
        sensory_to_feedback = {}
        for i, fch in enumerate(FEEDBACK_CHANNELS):
            if i < len(self.spatial_encoder.channels):
                sensory_to_feedback[self.spatial_encoder.channels[i]] = fch

        for sch, amp in pattern.items():
            fch = sensory_to_feedback.get(sch)
            if fch is not None:
                # Scale feedback by the original amplitude
                feedback_pattern[fch] = min(AMP_MAX, self.feedback_amplitude * (amp / AMP_MAX))

        if not feedback_pattern:
            return 0

        # Stimulate feedback channels (shorter window)
        feedback_counts = self.substrate.stimulate_and_record(
            feedback_pattern, window_s=FEEDBACK_WINDOW_S,
        )

        return sum(feedback_counts.values()) if isinstance(feedback_counts, dict) else 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        condition: str = "bio_llm",
        measure_consciousness: bool = True,
        verbose: bool = True,
    ) -> Tuple[str, List[TokenRecordV3], Dict]:
        """Generate text with feedback-augmented closed loop."""
        self._load_model()

        use_neurons = condition in ('bio_llm', 'shadow_llm')
        shuffle_spikes = condition == 'shadow_llm'
        effective_alpha = self.alpha if use_neurons else 0.0

        decoder = LearnedDecoder(self.spatial_encoder, alpha=effective_alpha)
        assessor = ConsciousnessAssessor()

        # Reset KV cache
        try:
            self._llm.reset()
            if hasattr(self._llm, '_ctx') and self._llm._ctx is not None:
                self._llm._ctx.kv_cache_clear()
        except Exception:
            pass

        records = []
        text = ""
        context = prompt
        override_count = 0
        all_consciousness = []
        self._prev_token_id = None

        if verbose:
            print(f"  [{condition.upper()} v3] prompt={prompt[:50]}... alpha={effective_alpha}"
                  f" feedback={'ON' if self.feedback_enabled else 'OFF'}")

        for pos in range(max_tokens):
            t0 = time.time()

            # LLM inference
            try:
                output = self._llm.create_completion(
                    context, max_tokens=1, logprobs=MAX_CANDIDATES, temperature=1.0,
                )
            except RuntimeError as e:
                if 'llama_decode returned -1' in str(e):
                    self._load_model(force_reload=True)
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

            # Convert to probabilities
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

            # SPATIAL ENCODING with temporal context
            combined_pattern, channel_to_token = self.spatial_encoder.encode_candidates(model_probs)

            # Add temporal trace: previous token's pattern at reduced amplitude
            if self._prev_token_id is not None and use_neurons:
                prev_pattern = self.spatial_encoder.encode_token(self._prev_token_id)
                for ch, amp in prev_pattern.items():
                    if ch in combined_pattern:
                        combined_pattern[ch] = min(AMP_MAX, combined_pattern[ch] + amp * 0.2)
                    else:
                        combined_pattern[ch] = amp * 0.2

            # Stimulate substrate
            channel_amplitudes = {int(ch): amp for ch, amp in combined_pattern.items()}
            spike_counts = self.substrate.stimulate_and_record(
                channel_amplitudes, window_s=RESPONSE_WINDOW_S,
            )

            # Consciousness measurement
            consciousness_metrics = None
            if measure_consciousness:
                sm = self.substrate.get_last_spike_matrix()
                consciousness_metrics = assessor.assess(sm)
                all_consciousness.append(consciousness_metrics)

            if use_neurons and len(model_probs) > 1:
                # Shadow: shuffle spike counts
                decode_spikes = dict(spike_counts)
                if shuffle_spikes:
                    channels = list(decode_spikes.keys())
                    counts = list(decode_spikes.values())
                    np.random.shuffle(counts)
                    decode_spikes = dict(zip(channels, counts))

                blended, neural_probs, confidence = decoder.decode(
                    decode_spikes, model_probs, channel_to_token
                )
                selected = max(blended, key=blended.get)
                was_override = selected != model_top

                # FEEDBACK: teach the decoder and substrate what was selected
                decoder.update(selected, decode_spikes, channel_to_token)
                feedback_spikes = self._send_feedback(selected, spike_counts, channel_to_token)

                # Coupling metrics
                shared = set(neural_probs.keys()) & set(model_probs.keys())
                if shared:
                    m_vec = np.array([model_probs[t] for t in shared])
                    n_vec = np.array([neural_probs[t] for t in shared])
                    mn, nn = np.linalg.norm(m_vec), np.linalg.norm(n_vec)
                    nl_align = float(np.dot(m_vec, n_vec) / (mn * nn)) if mn > 1e-10 and nn > 1e-10 else 0.0
                else:
                    nl_align = 0.0

                bp = np.array(list(blended.values()), dtype=float)
                bp = bp[bp > 0]
                bp = bp / (bp.sum() + 1e-10)
                blended_ent = float(-np.sum(bp * np.log2(bp + 1e-10)))

                top_boost = blended.get(model_top, 0.0) - model_probs.get(model_top, 0.0)
            else:
                selected = model_top
                was_override = False
                nl_align = 0.0
                blended_ent = 0.0
                top_boost = 0.0
                feedback_spikes = 0
                confidence = 0.0
                blended = model_probs

            if was_override:
                override_count += 1

            # Get text
            if selected in top_logprobs:
                tok_text_out = top_logprobs[selected]['text']
            else:
                tok_text_out = self._llm.detokenize([selected]).decode('utf-8', errors='replace')

            latency = (time.time() - t0) * 1000
            total_spk = sum(int(v) for v in spike_counts.values()) if isinstance(spike_counts, dict) else 0

            rec = TokenRecordV3(
                position=pos, token_id=selected, token_text=tok_text_out,
                model_top_token=model_top, was_override=was_override,
                alpha=effective_alpha, latency_ms=round(latency, 1),
                total_spikes=total_spk,
                blended_entropy=blended_ent,
                neural_llm_alignment=nl_align,
                model_top_prob_boost=top_boost,
                feedback_spikes=feedback_spikes,
                feedback_channels=len(FEEDBACK_CHANNELS) if self.feedback_enabled else 0,
                consciousness_metrics=consciousness_metrics,
                decoder_confidence=confidence,
            )
            records.append(rec)

            self._prev_token_id = selected
            text += tok_text_out
            context += tok_text_out

            if verbose and pos % 10 == 0:
                cs = consciousness_metrics['cscore'] if consciousness_metrics else 0.0
                print(f"    [{pos:3d}] {tok_text_out!r:12s} spk={total_spk:4d} "
                      f"C={cs:.3f} align={nl_align:.3f} conf={confidence:.2f} "
                      f"fb={feedback_spikes:3d} {'*' if was_override else ''}")

            if tok_text_out.strip() == '' and pos > 5 and text.endswith('\n\n'):
                break

        # Summary
        weight_div = self.substrate.get_weight_divergence()
        assoc_stats = decoder.get_association_stats()

        summary = {
            'condition': condition,
            'prompt': prompt,
            'generated_text': text,
            'n_tokens': len(records),
            'override_count': override_count,
            'override_rate': override_count / max(1, len(records)),
            'encoding': 'spatial_v3_feedback',
            'feedback_enabled': self.feedback_enabled,
            'weight_frobenius_divergence': weight_div['frobenius_divergence'],
            'weight_fractional_change': weight_div['fractional_change'],
            'pct_weights_changed': weight_div['pct_weights_changed'],
            'decoder_n_associations': assoc_stats['n_associations'],
            'decoder_mean_weight': assoc_stats['mean_weight'],
        }

        if all_consciousness:
            cs_vals = [m['cscore'] for m in all_consciousness]
            summary.update({
                'mean_cscore': float(np.mean(cs_vals)),
                'std_cscore': float(np.std(cs_vals)),
                'max_cscore': float(np.max(cs_vals)),
                'mean_closure': float(np.mean([m['closure'] for m in all_consciousness])),
                'mean_lambda2_norm': float(np.mean([m['lambda2_norm'] for m in all_consciousness])),
                'mean_rho': float(np.mean([m['rho'] for m in all_consciousness])),
                'mean_lzc': float(np.mean([m['lzc'] for m in all_consciousness])),
                'mean_transfer_entropy': float(np.mean([m['transfer_entropy'] for m in all_consciousness])),
                'mean_temporal_depth': float(np.mean([m['temporal_depth'] for m in all_consciousness])),
                'mean_granger_density': float(np.mean([m['granger_density'] for m in all_consciousness])),
                'cscore_trajectory': cs_vals,
            })

            # Coupling metrics
            ent_vals = [r.blended_entropy for r in records]
            align_vals = [r.neural_llm_alignment for r in records if r.neural_llm_alignment != 0.0]
            boost_vals = [r.model_top_prob_boost for r in records]
            conf_vals = [r.decoder_confidence for r in records]
            fb_vals = [r.feedback_spikes for r in records]

            summary['mean_blended_entropy'] = float(np.mean(ent_vals))
            summary['mean_neural_llm_alignment'] = float(np.mean(align_vals)) if align_vals else 0.0
            summary['mean_top_prob_boost'] = float(np.mean(boost_vals))
            summary['mean_decoder_confidence'] = float(np.mean(conf_vals))
            summary['total_feedback_spikes'] = int(np.sum(fb_vals))

            # C-Score↔entropy coupling
            cs_arr = np.array(cs_vals[:len(ent_vals)])
            ent_arr = np.array(ent_vals[:len(cs_vals)])
            if len(cs_arr) >= 5 and np.std(ent_arr) > 1e-10 and np.std(cs_arr) > 1e-10:
                corr = float(np.corrcoef(cs_arr, ent_arr)[0, 1])
                summary['cscore_entropy_corr'] = corr if not np.isnan(corr) else 0.0
            else:
                summary['cscore_entropy_corr'] = 0.0

            # Token pattern consistency
            token_patterns = defaultdict(list)
            for rec in records:
                pattern = np.zeros(59)
                for i in range(59):
                    pattern[i] = float(spike_counts.get(str(i), 0))
                norm = np.linalg.norm(pattern)
                if norm > 0:
                    pattern /= norm
                token_patterns[rec.token_id].append(pattern)

            consistencies = []
            for tid, patterns in token_patterns.items():
                if len(patterns) >= 2:
                    sims = [float(np.dot(patterns[i], patterns[j]))
                            for i in range(len(patterns))
                            for j in range(i+1, len(patterns))]
                    if sims:
                        consistencies.append(np.mean(sims))
            summary['token_pattern_consistency'] = float(np.mean(consistencies)) if consistencies else 0.0

        if verbose:
            print(f"  [{condition.upper()} v3] {len(records)} tokens, {override_count} overrides")
            if 'mean_cscore' in summary:
                print(f"    C-Score: {summary['mean_cscore']:.4f} ± {summary['std_cscore']:.4f}")
                print(f"    Alignment: {summary.get('mean_neural_llm_alignment', 0):.4f}")
                print(f"    Decoder associations: {assoc_stats['n_associations']}")
                print(f"    Total feedback spikes: {summary.get('total_feedback_spikes', 0)}")

        return text, records, summary
