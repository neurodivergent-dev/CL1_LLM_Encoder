"""Token Voting Encoder v2 — Spatial Encoding for Neural Substrate.

Key improvement over v1: uses SpatialEncoder for token-specific stimulation
patterns. Each token maps to a CONSISTENT spatial pattern across the MEA,
allowing the culture to develop token-specific neural representations via
STDP plasticity over the closed loop.

Architecture:
  LLM logprobs → SpatialEncoder(token_id, probability) → spatial pattern
    → Izhikevich substrate → spike response → SpatialDecoder → blended probs
    → token selection → next context → LLM logprobs (loop closes)

The critical hypothesis: in bio_llm mode, the closed loop allows STDP to
strengthen connections along stimulated pathways, creating genuine
token-specific representations. In shadow_llm mode, shuffled responses
break this coupling. In llm_only mode, the substrate is stimulated but
doesn't influence selection, so there's no adaptive pressure.
"""

import os
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .neural_substrate import IzhikevichSubstrate, IzhikevichConfig
from .spatial_encoder import SpatialEncoder, SpatialDecoder
from .consciousness import ConsciousnessAssessor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESPONSE_WINDOW_S = 0.5
DEFAULT_ALPHA = 0.5
MAX_CANDIDATES = 15
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "LFM2-350M-Q4_0.gguf"
)


@dataclass
class TokenRecord:
    """Per-token voting record."""
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
    n_active_channels: int = 0
    pattern_channels: List[int] = field(default_factory=list)
    # Coupling metrics (per-token)
    blended_entropy: float = 0.0         # Shannon entropy of blended distribution
    neural_llm_alignment: float = 0.0    # Cosine similarity between neural and model prob vectors
    model_top_prob_boost: float = 0.0    # blended_prob[model_top] - model_prob[model_top]


class TokenVotingEngineV2:
    """LLM + Neural Substrate integration with spatial encoding.

    Key differences from v1:
      - Uses SpatialEncoder for consistent token→pattern mapping
      - Each token has a fixed spatial fingerprint on the MEA
      - SpatialDecoder aggregates spikes by token ownership
      - STDP in the substrate creates genuine learned representations
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

        self.substrate = IzhikevichSubstrate(seed=substrate_seed)
        self.spatial_encoder = SpatialEncoder(seed=substrate_seed)

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

    def generate(
        self,
        prompt: str,
        max_tokens: int = 25,
        condition: str = "bio_llm",
        measure_consciousness: bool = True,
        verbose: bool = True,
    ) -> Tuple[str, List[TokenRecord], Dict]:
        """Generate text under specified condition with spatial encoding.

        Conditions:
          llm_only:   Substrate stimulated but doesn't influence selection
          bio_llm:    Full closed loop with spatial encoding
          shadow_llm: Spikes shuffled across channels (breaks spatial info)
        """
        self._load_model()

        use_neurons = condition in ('bio_llm', 'shadow_llm')
        shuffle_spikes = condition == 'shadow_llm'
        effective_alpha = self.alpha if use_neurons else 0.0

        decoder = SpatialDecoder(self.spatial_encoder, alpha=effective_alpha)
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

        if verbose:
            print(f"  [{condition.upper()}] prompt={prompt[:50]}... alpha={effective_alpha}")

        for pos in range(max_tokens):
            t0 = time.time()

            # LLM inference
            try:
                output = self._llm.create_completion(
                    context, max_tokens=1, logprobs=MAX_CANDIDATES, temperature=1.0,
                )
            except RuntimeError as e:
                if 'llama_decode returned -1' in str(e):
                    print(f"    [WARN] KV cache overflow, reloading...")
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

            # SPATIAL ENCODING: map token candidates to MEA stimulation pattern
            combined_pattern, channel_to_token = self.spatial_encoder.encode_candidates(model_probs)

            # Stimulate substrate (all conditions)
            channel_amplitudes = {int(ch): amp for ch, amp in combined_pattern.items()}
            spike_counts = self.substrate.stimulate_and_record(
                channel_amplitudes, window_s=RESPONSE_WINDOW_S,
            )

            # Consciousness measurement (all conditions)
            consciousness_metrics = None
            if measure_consciousness:
                sm = self.substrate.get_last_spike_matrix()
                consciousness_metrics = assessor.assess(sm)
                all_consciousness.append(consciousness_metrics)

            if use_neurons and len(model_probs) > 1:
                # Shadow: shuffle spike counts across channels
                decode_spikes = dict(spike_counts)
                if shuffle_spikes:
                    channels = list(decode_spikes.keys())
                    counts = list(decode_spikes.values())
                    np.random.shuffle(counts)
                    decode_spikes = dict(zip(channels, counts))

                blended, neural_probs, z_scores = decoder.decode(
                    decode_spikes, model_probs, channel_to_token
                )
                selected = max(blended, key=blended.get)
                was_override = selected != model_top
            else:
                selected = model_top
                neural_probs = {}
                blended = model_probs
                z_scores = {}
                was_override = False

            if was_override:
                override_count += 1

            # Get text
            if selected in top_logprobs:
                tok_text_out = top_logprobs[selected]['text']
            else:
                tok_text_out = self._llm.detokenize([selected]).decode('utf-8', errors='replace')

            latency = (time.time() - t0) * 1000

            # Stats
            counts_arr = np.array([v for v in spike_counts.values()], dtype=float)
            total_spk = int(counts_arr.sum())
            if total_spk > 0:
                p = counts_arr / total_spk
                p = p[p > 0]
                ch_ent = float(-np.sum(p * np.log2(p + 1e-10)))
            else:
                ch_ent = 0.0

            # Per-token coupling metrics
            # 1. Blended probability entropy (lower = more decisive)
            bp_vals = np.array(list(blended.values()), dtype=float)
            bp_vals = bp_vals[bp_vals > 0]
            if len(bp_vals) > 0:
                bp_vals = bp_vals / (bp_vals.sum() + 1e-10)
                blended_ent = float(-np.sum(bp_vals * np.log2(bp_vals + 1e-10)))
            else:
                blended_ent = 0.0

            # 2. Neural-LLM alignment: cosine similarity of prob vectors
            # over shared token IDs
            if neural_probs and model_probs:
                shared_tids = set(neural_probs.keys()) & set(model_probs.keys())
                if shared_tids:
                    m_vec = np.array([model_probs[t] for t in shared_tids])
                    n_vec = np.array([neural_probs[t] for t in shared_tids])
                    m_norm = np.linalg.norm(m_vec)
                    n_norm = np.linalg.norm(n_vec)
                    if m_norm > 1e-10 and n_norm > 1e-10:
                        nl_align = float(np.dot(m_vec, n_vec) / (m_norm * n_norm))
                    else:
                        nl_align = 0.0
                else:
                    nl_align = 0.0
            else:
                nl_align = 0.0

            # 3. Model top probability boost: how much did blending change
            # the probability of the LLM's top choice?
            model_top_p = model_probs.get(model_top, 0.0)
            blended_top_p = blended.get(model_top, 0.0)
            top_boost = blended_top_p - model_top_p

            rec = TokenRecord(
                position=pos, token_id=selected, token_text=tok_text_out,
                model_top_token=model_top, was_override=was_override,
                alpha=effective_alpha, latency_ms=round(latency, 1),
                model_probs=model_probs, neural_probs=neural_probs,
                blended_probs=blended, spike_counts=spike_counts,
                z_scores=z_scores, total_spikes=total_spk,
                fano_factor=0.0, channel_entropy=ch_ent,
                sync_index=consciousness_metrics.get('sync_index', 0) if consciousness_metrics else 0,
                consciousness_metrics=consciousness_metrics,
                n_active_channels=len(combined_pattern),
                pattern_channels=list(combined_pattern.keys()),
                blended_entropy=blended_ent,
                neural_llm_alignment=nl_align,
                model_top_prob_boost=top_boost,
            )
            records.append(rec)

            text += tok_text_out
            context += tok_text_out

            if verbose and pos % 5 == 0:
                cs = consciousness_metrics['cscore'] if consciousness_metrics else 0.0
                print(f"    [{pos:3d}] {tok_text_out!r:12s} spk={total_spk:4d} "
                      f"C={cs:.3f} ch={len(combined_pattern)} "
                      f"{'*' if was_override else ''}")

            if tok_text_out.strip() == '' and pos > 5 and text.endswith('\n\n'):
                break

        # Weight divergence (key STDP metric)
        weight_div = self.substrate.get_weight_divergence()

        # Summary
        summary = {
            'condition': condition,
            'prompt': prompt,
            'generated_text': text,
            'n_tokens': len(records),
            'override_count': override_count,
            'override_rate': override_count / max(1, len(records)),
            'substrate_state': self.substrate.get_state_snapshot(),
            'encoding': 'spatial_v2',
            # STDP weight divergence metrics
            'weight_frobenius_divergence': weight_div['frobenius_divergence'],
            'weight_fractional_change': weight_div['fractional_change'],
            'pct_weights_changed': weight_div['pct_weights_changed'],
        }

        if all_consciousness:
            cs_vals = [m['cscore'] for m in all_consciousness]
            summary['mean_cscore'] = float(np.mean(cs_vals))
            summary['std_cscore'] = float(np.std(cs_vals))
            summary['max_cscore'] = float(np.max(cs_vals))
            summary['mean_closure'] = float(np.mean([m['closure'] for m in all_consciousness]))
            summary['mean_lambda2_norm'] = float(np.mean([m['lambda2_norm'] for m in all_consciousness]))
            summary['mean_rho'] = float(np.mean([m['rho'] for m in all_consciousness]))
            summary['mean_lzc'] = float(np.mean([m['lzc'] for m in all_consciousness]))
            summary['mean_transfer_entropy'] = float(np.mean([m['transfer_entropy'] for m in all_consciousness]))
            summary['mean_temporal_depth'] = float(np.mean([m['temporal_depth'] for m in all_consciousness]))
            summary['mean_granger_density'] = float(np.mean([m['granger_density'] for m in all_consciousness]))
            summary['cscore_trajectory'] = cs_vals

            # --- CONSCIOUSNESS-BEHAVIOR COUPLING ---
            # Key metric: does C-Score predict override behavior?
            # In Bio-LLM, the substrate's "consciousness" state (C-Score)
            # directly influences token selection via the closed loop.
            # In Shadow-LLM, the substrate state doesn't predict behavior
            # because spikes are shuffled before decoding.

            override_vals = [1.0 if r.was_override else 0.0 for r in records]
            spike_vals = [float(r.total_spikes) for r in records]

            # 1. C-Score↔Override point-biserial correlation
            cs_arr = np.array(cs_vals[:len(override_vals)])
            or_arr = np.array(override_vals[:len(cs_vals)])
            if len(cs_arr) >= 5 and np.std(cs_arr) > 1e-10 and np.std(or_arr) > 1e-10:
                corr = float(np.corrcoef(cs_arr, or_arr)[0, 1])
                summary['cscore_override_corr'] = corr if not np.isnan(corr) else 0.0
            else:
                summary['cscore_override_corr'] = 0.0

            # 2. C-Score↔Spike count correlation (neural activity coupling)
            spk_arr = np.array(spike_vals[:len(cs_vals)])
            if len(cs_arr) >= 5 and np.std(spk_arr) > 1e-10:
                corr_spk = float(np.corrcoef(cs_arr, spk_arr)[0, 1])
                summary['cscore_spikes_corr'] = corr_spk if not np.isnan(corr_spk) else 0.0
            else:
                summary['cscore_spikes_corr'] = 0.0

            # 3. C-Score trajectory slope (is consciousness increasing over tokens?)
            if len(cs_arr) >= 5:
                positions = np.arange(len(cs_arr))
                slope = float(np.polyfit(positions, cs_arr, 1)[0])
                summary['cscore_slope'] = slope
            else:
                summary['cscore_slope'] = 0.0

            # 4. Token pattern consistency: for repeated tokens, how similar
            # are their spike responses? Bio-LLM should show HIGHER consistency
            # because STDP creates stable token-specific representations.
            token_spike_patterns = defaultdict(list)
            for rec in records:
                if rec.spike_counts:
                    # Create a vector of spike counts per channel
                    pattern = np.array([rec.spike_counts.get(str(ch), 0)
                                       for ch in range(59)], dtype=float)
                    if np.sum(pattern) > 0:
                        pattern = pattern / (np.linalg.norm(pattern) + 1e-10)
                    token_spike_patterns[rec.token_id].append(pattern)

            # Average intra-token cosine similarity
            consistencies = []
            for tid, patterns in token_spike_patterns.items():
                if len(patterns) >= 2:
                    # Pairwise cosine similarity
                    sims = []
                    for i in range(len(patterns)):
                        for j in range(i + 1, len(patterns)):
                            sim = float(np.dot(patterns[i], patterns[j]))
                            sims.append(sim)
                    if sims:
                        consistencies.append(np.mean(sims))

            summary['token_pattern_consistency'] = float(np.mean(consistencies)) if consistencies else 0.0
            summary['n_repeated_tokens'] = sum(1 for p in token_spike_patterns.values() if len(p) >= 2)

            # --- CONTINUOUS COUPLING METRICS (Exp 5 improvement) ---
            # These replace the flawed C-Score↔override correlation (H9)
            # which fails because Bio has ~0 overrides (no variance).

            # 5. Mean blended entropy: how decisive is the joint system?
            # Bio-LLM should have LOWER entropy (substrate reinforces LLM top choice)
            # Shadow should have HIGHER entropy (noise spreads probability mass)
            ent_vals = [r.blended_entropy for r in records]
            summary['mean_blended_entropy'] = float(np.mean(ent_vals))

            # 6. Neural-LLM alignment: how well do decoded neural probs match LLM probs?
            # Bio-LLM should have HIGHER alignment (STDP learns LLM patterns)
            # Shadow should have LOWER alignment (shuffled = random)
            align_vals = [r.neural_llm_alignment for r in records if r.neural_llm_alignment != 0.0]
            summary['mean_neural_llm_alignment'] = float(np.mean(align_vals)) if align_vals else 0.0

            # 7. Model top probability boost: how much does blending help the LLM's top choice?
            # Bio-LLM should have POSITIVE boost (substrate reinforces top choice)
            # Shadow should have ~0 or NEGATIVE boost (noise detracts from top choice)
            boost_vals = [r.model_top_prob_boost for r in records]
            summary['mean_top_prob_boost'] = float(np.mean(boost_vals))

            # 8. C-Score↔blended_entropy correlation (CONTINUOUS coupling)
            # In Bio-LLM, when C-Score is high (organized state), blended entropy
            # should be LOW (decisive output). In Shadow, no such coupling.
            if len(cs_arr) >= 5 and len(ent_vals) >= 5:
                ent_arr = np.array(ent_vals[:len(cs_arr)])
                if np.std(ent_arr) > 1e-10 and np.std(cs_arr) > 1e-10:
                    corr_ent = float(np.corrcoef(cs_arr, ent_arr)[0, 1])
                    summary['cscore_entropy_corr'] = corr_ent if not np.isnan(corr_ent) else 0.0
                else:
                    summary['cscore_entropy_corr'] = 0.0
            else:
                summary['cscore_entropy_corr'] = 0.0

        if verbose:
            print(f"  [{condition.upper()}] {len(records)} tokens, {override_count} overrides")
            if 'mean_cscore' in summary:
                print(f"    C-Score: {summary['mean_cscore']:.4f} ± {summary['std_cscore']:.4f}")

        return text, records, summary
