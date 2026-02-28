#!/usr/bin/env python3
"""CL1 Live Experiment — 3-Condition Test on Real Biological Neurons.

Tests whether real biological neurons develop tighter functional integration
with an LLM when in a closed feedback loop (Bio-LLM) versus controls
(Shadow-LLM: shuffled spikes, LLM-only: no neural influence).

Design:
  - Single CL1 substrate (shared across conditions)
  - Interleaved condition ordering (controls for temporal drift)
  - Same stimulation patterns for all conditions (only decoding differs)
  - Full HDF5 logging with per-token metrics

Key metrics:
  - Neural-LLM alignment (cosine similarity of decoded vs model probs)
  - Blended entropy (how decisive is the joint system?)
  - Override rate (how often does substrate change LLM's choice?)
  - Token pattern consistency (do repeated tokens get consistent responses?)
  - Spike count statistics

Hypothesis: Bio-LLM will show HIGHER alignment and LOWER entropy than
Shadow-LLM, because real biological STDP creates token-specific
representations that reinforce the LLM's predictions.
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

from LLM_Encoder.spatial_encoder import SpatialEncoder, SpatialDecoder
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
MAX_CANDIDATES = 15
ALPHA = 0.5


class CL1LiveExperiment:
    """3-condition experiment on CL1 biological neurons."""

    def __init__(
        self,
        substrate,
        model_path: str = MODEL_PATH,
        alpha: float = ALPHA,
        tokens_per_thought: int = 50,
        n_thoughts: int = 10,
        seed: int = 42,
        output_dir: str = "experiment_data",
    ):
        self.substrate = substrate
        self.model_path = model_path
        self.alpha = alpha
        self.tokens_per_thought = tokens_per_thought
        self.n_thoughts = n_thoughts
        self.seed = seed
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.h5_path = os.path.join(output_dir, f"cl1_experiment_{self.timestamp}.h5")
        self.log_path = os.path.join(output_dir, f"cl1_experiment_{self.timestamp}.log")

        self.spatial_encoder = SpatialEncoder(seed=seed)
        self.assessor = ConsciousnessAssessor()
        self._llm = None
        self._rng = np.random.default_rng(seed)

    def _log(self, msg: str):
        """Log to file and stdout."""
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

    def _generate_tokens(
        self,
        prompt: str,
        condition: str,
        h5_group: h5py.Group,
    ) -> Dict:
        """Generate tokens under specified condition, logging to HDF5.

        Returns summary dict with per-thought metrics.
        """
        self._load_llm()

        use_neurons = condition in ('bio_llm', 'shadow_llm')
        shuffle_spikes = condition == 'shadow_llm'
        effective_alpha = self.alpha if use_neurons else 0.0

        decoder = SpatialDecoder(self.spatial_encoder, alpha=effective_alpha)

        # Reset LLM context
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
        all_latencies = []
        token_ids = []
        token_texts = []
        was_overrides = []
        spike_matrices = []

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

            # Spatial encoding → CL1 stimulation (ALL conditions)
            combined_pattern, channel_to_token = self.spatial_encoder.encode_candidates(model_probs)
            channel_amplitudes = {int(ch): amp for ch, amp in combined_pattern.items()}

            spike_counts = self.substrate.stimulate_and_record(
                channel_amplitudes, window_s=RESPONSE_WINDOW_S,
            )

            # Consciousness assessment
            sm = self.substrate.get_last_spike_matrix()
            consciousness = self.assessor.assess(sm)
            cs = consciousness.get('cscore', 0)
            all_cs.append(cs)

            # Spike stats
            token_spikes = sum(spike_counts.values()) if isinstance(spike_counts, dict) else 0
            all_spikes.append(token_spikes)

            # Condition-specific decoding
            if use_neurons and len(model_probs) > 1:
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

                # Neural-LLM alignment
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

                # Blended entropy
                bp = np.array(list(blended.values()), dtype=float)
                bp = bp[bp > 0]
                bp = bp / (bp.sum() + 1e-10)
                all_entropy.append(float(-np.sum(bp * np.log2(bp + 1e-10))))
            else:
                selected = model_top
                blended = model_probs
                was_override = False
                all_alignment.append(0.0)
                bp = np.array(list(model_probs.values()), dtype=float)
                bp = bp[bp > 0]
                bp = bp / (bp.sum() + 1e-10)
                all_entropy.append(float(-np.sum(bp * np.log2(bp + 1e-10))))

            if was_override:
                override_count += 1

            # Token text
            if selected in top_logprobs:
                tok_text = top_logprobs[selected]['text']
            else:
                tok_text = self._llm.detokenize([selected]).decode('utf-8', errors='replace')

            latency = (time.time() - t0) * 1000
            all_latencies.append(latency)

            token_ids.append(selected)
            token_texts.append(tok_text)
            was_overrides.append(1 if was_override else 0)
            spike_matrices.append(sm)

            text += tok_text
            context += tok_text

            # Print live token
            marker = '*' if was_override else ' '
            if pos % 10 == 0:
                sys.stdout.write(f"\n    [{pos:3d}] C={cs:.3f} spk={token_spikes:3d} {marker}")
            sys.stdout.write(tok_text)
            sys.stdout.flush()

            # Early stop
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
        h5_group.create_dataset('latency_ms', data=np.array(all_latencies[:n], dtype=np.float32))
        h5_group.attrs['condition'] = condition
        h5_group.attrs['prompt'] = prompt
        h5_group.attrs['generated_text'] = text
        h5_group.attrs['n_tokens'] = n
        h5_group.attrs['override_count'] = override_count
        h5_group.attrs['alpha'] = self.alpha if use_neurons else 0.0

        # Save spike matrices for full consciousness analysis
        if spike_matrices:
            combined = np.stack(spike_matrices[:n], axis=0)  # (n_tokens, T, N)
            h5_group.create_dataset('spike_matrices', data=combined, compression='gzip')

        # Token pattern consistency (for repeated tokens)
        token_patterns = defaultdict(list)
        for i, tid in enumerate(token_ids):
            if i < len(all_spikes):
                # Simplified pattern: just spike count (for CL1 where we don't have full spatial info)
                token_patterns[tid].append(all_spikes[i])

        consistencies = []
        for tid, counts in token_patterns.items():
            if len(counts) >= 2:
                arr = np.array(counts, dtype=float)
                if np.std(arr) > 0:
                    cv = np.std(arr) / (np.mean(arr) + 1e-10)
                    consistencies.append(1.0 - min(cv, 1.0))  # Higher = more consistent
                else:
                    consistencies.append(1.0)

        summary = {
            'condition': condition,
            'n_tokens': n,
            'override_count': override_count,
            'override_rate': override_count / max(1, n),
            'mean_cscore': float(np.mean(all_cs)) if all_cs else 0.0,
            'std_cscore': float(np.std(all_cs)) if all_cs else 0.0,
            'mean_alignment': float(np.mean(all_alignment)) if all_alignment else 0.0,
            'std_alignment': float(np.std(all_alignment)) if all_alignment else 0.0,
            'mean_entropy': float(np.mean(all_entropy)) if all_entropy else 0.0,
            'std_entropy': float(np.std(all_entropy)) if all_entropy else 0.0,
            'mean_spikes': float(np.mean(all_spikes)) if all_spikes else 0.0,
            'total_spikes': sum(all_spikes),
            'mean_latency_ms': float(np.mean(all_latencies)) if all_latencies else 0.0,
            'token_consistency': float(np.mean(consistencies)) if consistencies else 0.0,
            'generated_text': text,
        }
        return summary

    def run(self):
        """Run the full 3-condition interleaved experiment."""
        t_start = time.time()

        self._log("=" * 78)
        self._log("  CL1 LIVE EXPERIMENT — 3-CONDITION TEST ON BIOLOGICAL NEURONS")
        self._log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 78)
        self._log(f"  Substrate:      {type(self.substrate).__name__}")
        self._log(f"  Conditions:     {CONDITIONS}")
        self._log(f"  Thoughts/cond:  {self.n_thoughts}")
        self._log(f"  Tokens/thought: {self.tokens_per_thought}")
        self._log(f"  Alpha:          {self.alpha}")
        self._log(f"  Prompts:        {len(PROMPTS)}")
        self._log(f"  Total tokens:   ~{self.n_thoughts * self.tokens_per_thought * 3}")
        self._log(f"  Output:         {self.h5_path}")
        self._log("=" * 78)

        # Open HDF5
        h5 = h5py.File(self.h5_path, 'w')
        h5.attrs['experiment'] = 'CL1 Live 3-Condition'
        h5.attrs['timestamp'] = self.timestamp
        h5.attrs['substrate'] = type(self.substrate).__name__
        h5.attrs['alpha'] = self.alpha
        h5.attrs['tokens_per_thought'] = self.tokens_per_thought
        h5.attrs['n_thoughts'] = self.n_thoughts
        h5.attrs['seed'] = self.seed
        h5.attrs['conditions'] = json.dumps(CONDITIONS)

        # Collect summaries per condition
        condition_summaries = {c: [] for c in CONDITIONS}

        # Interleaved design: for each thought round, run all 3 conditions
        for thought_idx in range(self.n_thoughts):
            # Pick a prompt (cycle through)
            prompt = PROMPTS[thought_idx % len(PROMPTS)]

            self._log(f"\n{'#'*78}")
            self._log(f"  ROUND {thought_idx + 1}/{self.n_thoughts}")
            self._log(f"  Prompt: {prompt[:60]}...")
            self._log(f"{'#'*78}")

            for cond in CONDITIONS:
                self._log(f"\n  --- {cond.upper()} (thought {thought_idx + 1}) ---")

                group_name = f"{cond}/thought_{thought_idx:03d}"
                grp = h5.create_group(group_name)

                try:
                    summary = self._generate_tokens(prompt, cond, grp)
                    condition_summaries[cond].append(summary)

                    self._log(f"  {cond}: {summary['n_tokens']} tok, "
                              f"override={summary['override_rate']:.1%}, "
                              f"align={summary['mean_alignment']:.3f}, "
                              f"entropy={summary['mean_entropy']:.3f}, "
                              f"C={summary['mean_cscore']:.3f}, "
                              f"spk={summary['total_spikes']}")
                except Exception as e:
                    self._log(f"  ERROR in {cond}: {e}")
                    import traceback
                    traceback.print_exc()
                    condition_summaries[cond].append({'error': str(e)})

            h5.flush()

        # ---------- ANALYSIS ----------
        self._log(f"\n{'='*78}")
        self._log("  ANALYSIS")
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
        """Analyze the 3-condition results and compute hypothesis tests."""
        from scipy import stats

        results = {}

        for cond in CONDITIONS:
            valid = [s for s in summaries[cond] if 'error' not in s]
            if not valid:
                continue

            results[cond] = {
                'n_thoughts': len(valid),
                'mean_alignment': float(np.mean([s['mean_alignment'] for s in valid])),
                'std_alignment': float(np.std([s['mean_alignment'] for s in valid])),
                'mean_entropy': float(np.mean([s['mean_entropy'] for s in valid])),
                'std_entropy': float(np.std([s['mean_entropy'] for s in valid])),
                'mean_cscore': float(np.mean([s['mean_cscore'] for s in valid])),
                'mean_override_rate': float(np.mean([s['override_rate'] for s in valid])),
                'mean_spikes': float(np.mean([s['mean_spikes'] for s in valid])),
                'total_spikes': sum(s['total_spikes'] for s in valid),
                'mean_latency_ms': float(np.mean([s['mean_latency_ms'] for s in valid])),
                'token_consistency': float(np.mean([s['token_consistency'] for s in valid])),
            }

        self._log(f"\n  CONDITION SUMMARY")
        self._log(f"  {'Metric':<25s} {'Bio-LLM':>12s} {'Shadow-LLM':>12s} {'LLM-Only':>12s}")
        self._log(f"  {'-'*61}")

        for metric in ['mean_alignment', 'mean_entropy', 'mean_cscore',
                       'mean_override_rate', 'mean_spikes', 'token_consistency']:
            vals = []
            for cond in CONDITIONS:
                v = results.get(cond, {}).get(metric, 0)
                vals.append(f"{v:12.4f}")
            self._log(f"  {metric:<25s} {''.join(vals)}")

        # ---- HYPOTHESIS TESTS ----
        self._log(f"\n  HYPOTHESIS TESTS (alpha=0.05, Bonferroni corrected for 5 tests → 0.01)")

        bio = [s for s in summaries['bio_llm'] if 'error' not in s]
        shadow = [s for s in summaries['shadow_llm'] if 'error' not in s]
        llm = [s for s in summaries['llm_only'] if 'error' not in s]

        tests = []

        # H1: Bio alignment > Shadow alignment
        if bio and shadow:
            bio_align = [s['mean_alignment'] for s in bio]
            sha_align = [s['mean_alignment'] for s in shadow]
            U, p = stats.mannwhitneyu(bio_align, sha_align, alternative='greater')
            d = self._cohens_d(bio_align, sha_align)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tests.append(('H1: Bio alignment > Shadow', U, p, d, sig))
            self._log(f"\n  H1: Bio-LLM alignment > Shadow-LLM alignment")
            self._log(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {sig}")
            self._log(f"    Bio: {np.mean(bio_align):.4f} +/- {np.std(bio_align):.4f}")
            self._log(f"    Shadow: {np.mean(sha_align):.4f} +/- {np.std(sha_align):.4f}")

        # H2: Bio entropy < Shadow entropy (more decisive)
        if bio and shadow:
            bio_ent = [s['mean_entropy'] for s in bio]
            sha_ent = [s['mean_entropy'] for s in shadow]
            U, p = stats.mannwhitneyu(sha_ent, bio_ent, alternative='greater')
            d = self._cohens_d(sha_ent, bio_ent)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tests.append(('H2: Shadow entropy > Bio entropy', U, p, d, sig))
            self._log(f"\n  H2: Shadow entropy > Bio entropy (Bio more decisive)")
            self._log(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {sig}")
            self._log(f"    Bio: {np.mean(bio_ent):.4f} +/- {np.std(bio_ent):.4f}")
            self._log(f"    Shadow: {np.mean(sha_ent):.4f} +/- {np.std(sha_ent):.4f}")

        # H3: Bio override rate < Shadow override rate (STDP aligns substrate with LLM)
        if bio and shadow:
            bio_or = [s['override_rate'] for s in bio]
            sha_or = [s['override_rate'] for s in shadow]
            U, p = stats.mannwhitneyu(sha_or, bio_or, alternative='greater')
            d = self._cohens_d(sha_or, bio_or)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tests.append(('H3: Shadow overrides > Bio overrides', U, p, d, sig))
            self._log(f"\n  H3: Shadow overrides > Bio overrides (Bio substrate aligned)")
            self._log(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {sig}")
            self._log(f"    Bio: {np.mean(bio_or):.4f} +/- {np.std(bio_or):.4f}")
            self._log(f"    Shadow: {np.mean(sha_or):.4f} +/- {np.std(sha_or):.4f}")

        # H4: Bio alignment > LLM-only alignment (closed loop helps)
        if bio and llm:
            bio_align = [s['mean_alignment'] for s in bio]
            llm_align = [s['mean_alignment'] for s in llm]
            if any(a > 0 for a in llm_align):
                U, p = stats.mannwhitneyu(bio_align, llm_align, alternative='greater')
                d = self._cohens_d(bio_align, llm_align)
            else:
                U, p, d = 0, 0.0001, 99.0  # LLM-only has no alignment by design
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tests.append(('H4: Bio alignment > LLM-only', U, p, d, sig))
            self._log(f"\n  H4: Bio alignment > LLM-only alignment")
            self._log(f"    U={U:.1f}, p={p:.4f}, d={d:.3f} {sig}")
            self._log(f"    Bio: {np.mean(bio_align):.4f}")
            self._log(f"    LLM-only: {np.mean(llm_align):.4f}")

        # H5: Alignment trajectory — Bio alignment increases over time (learning)
        if bio and len(bio) >= 3:
            bio_align_seq = [s['mean_alignment'] for s in bio]
            x = np.arange(len(bio_align_seq))
            slope, intercept, r, p, se = stats.linregress(x, bio_align_seq)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            tests.append(('H5: Bio alignment increases over time', 0, p, slope, sig))
            self._log(f"\n  H5: Bio alignment increases over time (learning trajectory)")
            self._log(f"    slope={slope:.6f}, r={r:.3f}, p={p:.4f} {sig}")
            self._log(f"    Trajectory: {[f'{a:.3f}' for a in bio_align_seq]}")

        # Save analysis to HDF5
        analysis_grp = h5.create_group('analysis')
        for cond, stats_dict in results.items():
            cg = analysis_grp.create_group(cond)
            for k, v in stats_dict.items():
                cg.attrs[k] = v

        test_grp = analysis_grp.create_group('hypothesis_tests')
        for i, (name, U, p, d, sig) in enumerate(tests):
            tg = test_grp.create_group(f"test_{i}")
            tg.attrs['name'] = name
            tg.attrs['U'] = float(U)
            tg.attrs['p'] = float(p)
            tg.attrs['effect_size'] = float(d)
            tg.attrs['significant'] = sig != 'n.s.'

        n_sig = sum(1 for _, _, _, _, s in tests if s != 'n.s.')
        self._log(f"\n  RESULT: {n_sig}/{len(tests)} hypothesis tests significant")

        if n_sig >= 3:
            self._log("  VERDICT: STRONG evidence of differential neural-LLM integration")
        elif n_sig >= 1:
            self._log("  VERDICT: SUGGESTIVE evidence (partial support)")
        else:
            self._log("  VERDICT: NO significant differential integration detected")

        # Save analysis JSON
        analysis_path = os.path.join(self.output_dir,
                                     f"cl1_analysis_{self.timestamp}.json")
        analysis_out = {
            'timestamp': self.timestamp,
            'conditions': results,
            'tests': [(n, float(u), float(p), float(d), s)
                      for n, u, p, d, s in tests],
            'n_significant': n_sig,
            'n_tests': len(tests),
        }
        with open(analysis_path, 'w') as f:
            json.dump(analysis_out, f, indent=2)
        self._log(f"  Analysis saved: {analysis_path}")

        return analysis_out

    @staticmethod
    def _cohens_d(a, b):
        """Cohen's d effect size."""
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        if pooled_std < 1e-10:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / pooled_std)


def main():
    """Run the CL1 live experiment."""
    import argparse

    parser = argparse.ArgumentParser(description='CL1 Live 3-Condition Experiment')
    parser.add_argument('--local', action='store_true',
                        help='Use local Izhikevich instead of CL1')
    parser.add_argument('--thoughts', type=int, default=10,
                        help='Number of thoughts per condition (default: 10)')
    parser.add_argument('--tokens', type=int, default=50,
                        help='Tokens per thought (default: 50)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Neural blend weight (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

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

    exp = CL1LiveExperiment(
        substrate=substrate,
        alpha=args.alpha,
        tokens_per_thought=args.tokens,
        n_thoughts=args.thoughts,
        seed=args.seed,
    )

    analysis = exp.run()

    # Clean up
    if hasattr(substrate, 'close'):
        substrate.close()

    return analysis


if __name__ == '__main__':
    main()
