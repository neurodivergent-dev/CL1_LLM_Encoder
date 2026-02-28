"""Token Voting Encoder — LLM + Neural Substrate Integration.

Consolidation of neural_token_voting_remote.py into a clean module.
Supports three experiment conditions:
  1. llm_only:   Pure LLM (alpha=0, no substrate)
  2. bio_llm:    Closed-loop neural voting (alpha=0.3)
  3. shadow_llm: Shuffled neural responses (alpha=0.3, spikes decorrelated)
"""

import os
import sys
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

from .neural_substrate import IzhikevichSubstrate, IzhikevichConfig
from .consciousness import ConsciousnessAssessor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOTING_CHANNELS = list(range(15))  # Use first 15 of 59 channels for voting
RESPONSE_WINDOW_S = 0.5            # Shorter for local (no network latency)
DEFAULT_ALPHA = 0.5                 # Strong enough for meaningful overrides
MAX_CANDIDATES = 15
AMP_MIN = 0.5
AMP_MAX = 2.5
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "LFM2-350M-Q4_0.gguf"
)


# ---------------------------------------------------------------------------
# Neural Vote Decoder
# ---------------------------------------------------------------------------

class NeuralVoteDecoder:
    """Decodes spike responses into token probabilities via z-score + sigmoid."""

    def __init__(self, alpha: float = DEFAULT_ALPHA, baseline_window: int = 20):
        self.alpha = alpha
        self._history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=baseline_window))

    def decode(
        self,
        spike_counts: Dict[str, int],
        model_probs: Dict[int, float],
        channel_to_token: Dict[int, int],
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """Decode spikes into blended probabilities.

        Returns (blended_probs, neural_probs, z_scores).
        """
        z_scores = {}
        raw_neural = {}

        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            history = self._history.get(ch)

            if history and len(history) >= 3:
                mean = np.mean(list(history))
                std = np.std(list(history))
                z = (count - mean) / std if std > 0 else 0.0
            else:
                z = 0.0

            z_scores[ch] = round(z, 3)
            raw_neural[ch] = 1.0 / (1.0 + np.exp(-z))

        # Normalize
        total = sum(raw_neural.values())
        neural_probs = {}
        if total > 0:
            for ch, val in raw_neural.items():
                token_id = channel_to_token.get(ch)
                if token_id is not None:
                    neural_probs[token_id] = val / total

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

        # Update baseline
        for ch_str, count in spike_counts.items():
            self._history[int(ch_str)].append(count)

        return blended, neural_probs, z_scores


# ---------------------------------------------------------------------------
# Token data for logging
# ---------------------------------------------------------------------------

@dataclass
class TokenRecord:
    """Per-token voting record for experiment logging."""
    position: int
    token_id: int
    token_text: str
    model_top_token: int
    was_override: bool
    alpha: float
    latency_ms: float
    model_probs: Dict[int, float]
    neural_probs: Dict[int, float]
    blended_probs: Dict[int, float]
    spike_counts: Dict[str, int]
    z_scores: Dict[int, float]
    total_spikes: int
    fano_factor: float
    channel_entropy: float
    sync_index: float
    consciousness_metrics: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Token Voting Engine (Core Encoder)
# ---------------------------------------------------------------------------

class TokenVotingEngine:
    """Integrates LLM inference with local Izhikevich neural voting.

    The closed loop:
      LLM logprobs → stimulation amplitudes → Izhikevich spikes →
      blended probabilities → token selection → next context → LLM logprobs
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        alpha: float = DEFAULT_ALPHA,
        substrate_seed: int = 42,
        n_gpu_layers: int = -1,
    ):
        self.model_path = model_path
        self.alpha = alpha
        self.n_gpu_layers = n_gpu_layers
        self.substrate_seed = substrate_seed

        # Create persistent substrate
        self.substrate = IzhikevichSubstrate(seed=substrate_seed)

        # Components (reset per-run)
        self.decoder = None
        self.assessor = None

        # LLM (lazy)
        self._llm = None

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

        if force_reload:
            del self._llm
            self._llm = None

        print(f"  Loading LLM: {self.model_path}")
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=2048,
            logits_all=True,
            verbose=False,
        )
        print(f"  LLM loaded.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 25,
        condition: str = "bio_llm",
        measure_consciousness: bool = True,
        verbose: bool = True,
    ) -> Tuple[str, List[TokenRecord], Dict]:
        """Generate text under specified condition.

        Parameters
        ----------
        prompt : str
        max_tokens : int
        condition : str
            'llm_only', 'bio_llm', or 'shadow_llm'
        measure_consciousness : bool
            If True, compute C-Score on spike data per token.
        verbose : bool

        Returns
        -------
        text : str
        records : list of TokenRecord
        summary : dict with aggregated metrics
        """
        self._load_model()

        use_neurons = condition in ('bio_llm', 'shadow_llm')
        shuffle_spikes = condition == 'shadow_llm'
        effective_alpha = self.alpha if use_neurons else 0.0

        self.decoder = NeuralVoteDecoder(alpha=effective_alpha)
        self.assessor = ConsciousnessAssessor()

        # Reset KV cache (critical: prevents llama_decode -1 overflow)
        try:
            self._llm.reset()  # reset n_tokens counter
            if hasattr(self._llm, '_ctx') and self._llm._ctx is not None:
                self._llm._ctx.kv_cache_clear()  # clear actual KV cache
        except Exception as e:
            print(f"    [WARN] KV cache clear failed: {e}")

        records = []
        text = ""
        context = prompt
        override_count = 0
        all_consciousness = []

        if verbose:
            print(f"  [{condition.upper()}] prompt={prompt[:50]}... alpha={effective_alpha}")

        for pos in range(max_tokens):
            t0 = time.time()

            # LLM inference (with KV cache overflow retry)
            try:
                output = self._llm.create_completion(
                    context, max_tokens=1, logprobs=MAX_CANDIDATES, temperature=1.0,
                )
            except RuntimeError as e:
                if 'llama_decode returned -1' in str(e):
                    # KV cache overflow — force reload model
                    print(f"    [WARN] KV cache overflow at token {pos}, reloading model...")
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
                for token_text_str, logprob in tlp.items():
                    token_ids = self._llm.tokenize(token_text_str.encode('utf-8'), add_bos=False)
                    if token_ids:
                        top_logprobs[token_ids[0]] = {'text': token_text_str, 'logprob': logprob}

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

            # ALL conditions stimulate the substrate for C-Score measurement.
            # The difference is how neural responses affect token selection.
            sorted_tokens = sorted(model_probs.items(), key=lambda x: -x[1])
            n_cand = min(len(sorted_tokens), len(VOTING_CHANNELS))

            candidates = []
            ch_to_tok = {}
            for rank, (tid, prob) in enumerate(sorted_tokens[:n_cand]):
                ch = VOTING_CHANNELS[rank]
                amp = AMP_MIN + (AMP_MAX - AMP_MIN) * float(prob)
                candidates.append({'channel': ch, 'amplitude': round(float(amp), 3)})
                ch_to_tok[ch] = tid

            vote_result = self.substrate.vote(candidates, window_s=RESPONSE_WINDOW_S)
            spike_counts = vote_result['spike_counts']

            # Consciousness measurement (all conditions)
            consciousness_metrics = None
            if measure_consciousness:
                sm = self.substrate.get_last_spike_matrix()
                consciousness_metrics = self.assessor.assess(sm)
                all_consciousness.append(consciousness_metrics)

            if use_neurons and len(model_probs) > 1:
                # Shadow condition: shuffle spikes to destroy channel mapping
                decode_spikes = dict(spike_counts)
                if shuffle_spikes:
                    channels = list(decode_spikes.keys())
                    counts = list(decode_spikes.values())
                    np.random.shuffle(counts)
                    decode_spikes = dict(zip(channels, counts))

                blended, neural_probs, z_scores = self.decoder.decode(
                    decode_spikes, model_probs, ch_to_tok
                )
                selected = max(blended, key=blended.get)
                was_override = selected != model_top
            else:
                # LLM-only: substrate observed but doesn't influence selection
                selected = model_top
                neural_probs = {}
                blended = model_probs
                z_scores = {}
                was_override = False

            # Population stats
            counts_arr = np.array(list({int(k): v for k, v in spike_counts.items()}.values()), dtype=float)
            total_spk = int(counts_arr.sum())
            if total_spk > 0:
                p = counts_arr / total_spk
                p = p[p > 0]
                ch_entropy = float(-np.sum(p * np.log2(p + 1e-10)))
            else:
                ch_entropy = 0.0

            if was_override:
                override_count += 1

            # Get text
            if selected in top_logprobs:
                token_text_out = top_logprobs[selected]['text']
            else:
                token_text_out = self._llm.detokenize([selected]).decode('utf-8', errors='replace')

            latency = (time.time() - t0) * 1000

            rec = TokenRecord(
                position=pos, token_id=selected, token_text=token_text_out,
                model_top_token=model_top, was_override=was_override,
                alpha=effective_alpha, latency_ms=round(latency, 1),
                model_probs=model_probs, neural_probs=neural_probs,
                blended_probs=blended, spike_counts=spike_counts,
                z_scores=z_scores, total_spikes=total_spk,
                fano_factor=0.0, channel_entropy=ch_entropy,
                sync_index=0.0, consciousness_metrics=consciousness_metrics,
            )
            records.append(rec)

            text += token_text_out
            context += token_text_out

            if verbose and pos % 5 == 0:
                cs = consciousness_metrics['cscore'] if consciousness_metrics else 0.0
                print(f"    [{pos:3d}] {token_text_out!r:12s} spk={total_spk:4d} C={cs:.3f} {'*' if was_override else ''}")

            # Stop conditions
            if token_text_out.strip() == '' and pos > 5 and text.endswith('\n\n'):
                break

        # Build summary
        summary = {
            'condition': condition,
            'prompt': prompt,
            'generated_text': text,
            'n_tokens': len(records),
            'override_count': override_count,
            'override_rate': override_count / max(1, len(records)),
            'substrate_state': self.substrate.get_state_snapshot(),
        }

        if all_consciousness:
            cscore_vals = [m['cscore'] for m in all_consciousness]
            summary['mean_cscore'] = float(np.mean(cscore_vals))
            summary['std_cscore'] = float(np.std(cscore_vals))
            summary['max_cscore'] = float(np.max(cscore_vals))
            summary['mean_closure'] = float(np.mean([m['closure'] for m in all_consciousness]))
            summary['mean_lambda2_norm'] = float(np.mean([m['lambda2_norm'] for m in all_consciousness]))
            summary['mean_rho'] = float(np.mean([m['rho'] for m in all_consciousness]))
            summary['mean_lzc'] = float(np.mean([m['lzc'] for m in all_consciousness]))
            summary['mean_transfer_entropy'] = float(np.mean([m['transfer_entropy'] for m in all_consciousness]))
            summary['mean_temporal_depth'] = float(np.mean([m['temporal_depth'] for m in all_consciousness]))
            summary['mean_granger_density'] = float(np.mean([m['granger_density'] for m in all_consciousness]))
            summary['cscore_trajectory'] = cscore_vals

        if verbose:
            print(f"  [{condition.upper()}] {len(records)} tokens, {override_count} overrides")
            if 'mean_cscore' in summary:
                print(f"    C-Score: {summary['mean_cscore']:.4f} ± {summary['std_cscore']:.4f}")

        return text, records, summary
