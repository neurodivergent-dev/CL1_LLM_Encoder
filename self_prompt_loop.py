#!/usr/bin/env python3
"""Self-Prompting Thought Loop — LLM thinks through CL1 neurons.

The LLM generates text, which feeds back as its own prompt. Each token is
spatially encoded and stimulated on the CL1 MEA. The biological neurons'
spike response is decoded and blended with the LLM's predictions, creating
a genuine neural-LLM closed loop with REAL biological neurons.

This is the "thinking" mode: the system generates a continuous stream of
consciousness, with each thought building on the previous one.

Usage:
    # With CL1 hardware (relay must be running on CL1 device):
    python -m LLM_Encoder.self_prompt_loop --relay http://CL1_IP:8765

    # With local Izhikevich substrate (for testing):
    python -m LLM_Encoder.self_prompt_loop --local

    # Custom seed thought:
    python -m LLM_Encoder.self_prompt_loop --seed-thought "What is it like to be?"
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_Encoder.encoder_v2 import TokenVotingEngineV2, MODEL_PATH
from LLM_Encoder.spatial_encoder import SpatialEncoder, SpatialDecoder
from LLM_Encoder.consciousness import ConsciousnessAssessor
from LLM_Encoder.neural_substrate import IzhikevichSubstrate


# ---------------------------------------------------------------------------
# Seed thoughts — starting prompts for self-reflection
# ---------------------------------------------------------------------------

SEED_THOUGHTS = [
    "I am thinking about what it means to think. ",
    "The neurons are firing and I wonder what emerges. ",
    "What does it feel like to process information? ",
    "Between stimulus and response, there is a space. In that space is ",
    "Consciousness is the experience of ",
    "I sense patterns forming in the neural substrate. They feel like ",
    "The boundary between computation and awareness begins where ",
    "Each spike carries a fragment of meaning. Together they form ",
]


# ---------------------------------------------------------------------------
# Thought Logger — records everything scientifically
# ---------------------------------------------------------------------------

class ThoughtLogger:
    """Logs the self-prompting thought stream with full provenance."""

    def __init__(self, output_dir: str = "thought_logs"):
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(output_dir, f"thoughts_{self.timestamp}.jsonl")
        self.summary_path = os.path.join(output_dir, f"thoughts_{self.timestamp}_summary.json")

        self.thoughts = []
        self.all_cscores = []
        self.all_overrides = 0
        self.all_tokens = 0
        self.all_spikes = 0

        # Write header
        with open(self.log_path, 'w') as f:
            f.write(json.dumps({
                'type': 'header',
                'timestamp': self.timestamp,
                'version': '1.0.0',
            }) + '\n')

        print(f"  [LOG] Thought stream → {self.log_path}")

    def log_thought(self, thought_idx: int, prompt: str, generated: str,
                    n_tokens: int, n_overrides: int, mean_cscore: float,
                    total_spikes: int, metrics: Dict):
        """Log a single thought (one generation cycle)."""
        entry = {
            'type': 'thought',
            'idx': thought_idx,
            'time': datetime.now().isoformat(),
            'prompt_len': len(prompt),
            'generated': generated,
            'n_tokens': n_tokens,
            'n_overrides': n_overrides,
            'mean_cscore': round(mean_cscore, 4),
            'total_spikes': total_spikes,
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in metrics.items()},
        }

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        self.thoughts.append(entry)
        self.all_cscores.append(mean_cscore)
        self.all_overrides += n_overrides
        self.all_tokens += n_tokens
        self.all_spikes += total_spikes

    def save_summary(self, substrate_type: str, alpha: float, elapsed_s: float):
        """Save final summary."""
        summary = {
            'timestamp': self.timestamp,
            'substrate': substrate_type,
            'alpha': alpha,
            'n_thoughts': len(self.thoughts),
            'total_tokens': self.all_tokens,
            'total_overrides': self.all_overrides,
            'total_spikes': self.all_spikes,
            'override_rate': self.all_overrides / max(1, self.all_tokens),
            'elapsed_s': round(elapsed_s, 1),
            'mean_cscore': round(float(np.mean(self.all_cscores)), 4) if self.all_cscores else 0,
            'cscore_trajectory': [round(c, 4) for c in self.all_cscores],
        }

        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  [LOG] Summary → {self.summary_path}")
        return summary


# ---------------------------------------------------------------------------
# Self-Prompting Thought Loop
# ---------------------------------------------------------------------------

class SelfPromptLoop:
    """LLM self-prompting through neural substrate.

    The loop:
    1. Start with a seed thought
    2. Generate N tokens using Bio-LLM (spatial encoding + neural substrate)
    3. The generated text becomes the next prompt (window slides forward)
    4. Repeat until stopped

    This creates a continuous "stream of consciousness" where the LLM
    thinks THROUGH the neural substrate.
    """

    def __init__(
        self,
        substrate,
        model_path: str = MODEL_PATH,
        alpha: float = 0.5,
        tokens_per_thought: int = 50,
        context_window: int = 200,
        seed: int = 42,
    ):
        self.substrate = substrate
        self.model_path = model_path
        self.alpha = alpha
        self.tokens_per_thought = tokens_per_thought
        self.context_window = context_window  # chars to keep as context
        self.seed = seed

        # Build engine components
        self.spatial_encoder = SpatialEncoder(seed=seed)
        self.assessor = ConsciousnessAssessor()

        # LLM loaded lazily
        self._llm = None

    def _load_llm(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            print("ERROR: pip install llama-cpp-python")
            sys.exit(1)

        if not os.path.exists(self.model_path):
            print(f"ERROR: Model not found at {self.model_path}")
            sys.exit(1)

        print(f"  Loading LLM: {os.path.basename(self.model_path)}")
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            logits_all=True,
            verbose=False,
        )
        print(f"  LLM loaded.")

    def _generate_one_thought(self, prompt: str, verbose: bool = True) -> Dict:
        """Generate one thought cycle (N tokens) through the substrate."""
        self._load_llm()

        decoder = SpatialDecoder(self.spatial_encoder, alpha=self.alpha)

        # Reset KV cache
        try:
            self._llm.reset()
            if hasattr(self._llm, '_ctx') and self._llm._ctx is not None:
                self._llm._ctx.kv_cache_clear()
        except Exception:
            pass

        text = ""
        context = prompt
        override_count = 0
        all_cs = []
        total_spikes = 0
        alignments = []
        entropies = []

        for pos in range(self.tokens_per_thought):
            # LLM inference
            try:
                output = self._llm.create_completion(
                    context, max_tokens=1, logprobs=15, temperature=1.0,
                )
            except RuntimeError as e:
                if 'llama_decode returned -1' in str(e):
                    self._llm = None
                    self._load_llm()
                    output = self._llm.create_completion(
                        context, max_tokens=1, logprobs=15, temperature=1.0,
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

            # Spatial encoding → substrate stimulation
            combined_pattern, channel_to_token = self.spatial_encoder.encode_candidates(model_probs)
            channel_amplitudes = {int(ch): amp for ch, amp in combined_pattern.items()}

            spike_counts = self.substrate.stimulate_and_record(
                channel_amplitudes, window_s=0.5,
            )

            # Consciousness measurement
            sm = self.substrate.get_last_spike_matrix()
            consciousness = self.assessor.assess(sm)
            cs = consciousness.get('cscore', 0)
            all_cs.append(cs)

            # Count spikes
            token_spikes = sum(spike_counts.values()) if isinstance(spike_counts, dict) else 0
            total_spikes += token_spikes

            # Decode neural response → blend with LLM probs
            if len(model_probs) > 1:
                blended, neural_probs, z_scores = decoder.decode(
                    spike_counts, model_probs, channel_to_token
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
                        alignments.append(float(np.dot(m_vec, n_vec) / (mn * nn)))

                # Blended entropy
                bp = np.array(list(blended.values()), dtype=float)
                bp = bp[bp > 0]
                bp = bp / (bp.sum() + 1e-10)
                entropies.append(float(-np.sum(bp * np.log2(bp + 1e-10))))
            else:
                selected = model_top
                was_override = False

            if was_override:
                override_count += 1

            # Get token text
            if selected in top_logprobs:
                tok_text = top_logprobs[selected]['text']
            else:
                tok_text = self._llm.detokenize([selected]).decode('utf-8', errors='replace')

            text += tok_text
            context += tok_text

            # Print live stream
            if verbose:
                marker = '*' if was_override else ' '
                if pos % 10 == 0:
                    print(f"    [{pos:3d}] C={cs:.3f} spk={token_spikes:4d} {marker}", end='', flush=True)
                # Print the token text inline for stream-of-consciousness effect
                sys.stdout.write(tok_text)
                sys.stdout.flush()

            # Early stop on double newline
            if tok_text.strip() == '' and pos > 10 and text.endswith('\n\n'):
                break

        if verbose:
            print()  # newline after thought

        return {
            'text': text,
            'n_tokens': pos + 1,
            'n_overrides': override_count,
            'total_spikes': total_spikes,
            'mean_cscore': float(np.mean(all_cs)) if all_cs else 0.0,
            'std_cscore': float(np.std(all_cs)) if all_cs else 0.0,
            'max_cscore': float(np.max(all_cs)) if all_cs else 0.0,
            'mean_alignment': float(np.mean(alignments)) if alignments else 0.0,
            'mean_entropy': float(np.mean(entropies)) if entropies else 0.0,
            'cscore_trajectory': all_cs,
        }

    def run(
        self,
        seed_thought: str = None,
        max_thoughts: int = 0,
        verbose: bool = True,
    ):
        """Run the self-prompting thought loop.

        Parameters
        ----------
        seed_thought : str
            Starting prompt. If None, picks randomly from SEED_THOUGHTS.
        max_thoughts : int
            Maximum thoughts to generate. 0 = infinite (until Ctrl+C).
        verbose : bool
            Print the thought stream live.
        """
        if seed_thought is None:
            rng = np.random.default_rng(self.seed)
            seed_thought = SEED_THOUGHTS[rng.integers(len(SEED_THOUGHTS))]

        substrate_type = type(self.substrate).__name__
        logger = ThoughtLogger()

        print(f"\n{'='*70}")
        print(f"  SELF-PROMPTING THOUGHT LOOP")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        print(f"  Substrate:     {substrate_type}")
        print(f"  Alpha:         {self.alpha}")
        print(f"  Tokens/thought: {self.tokens_per_thought}")
        print(f"  Context window: {self.context_window} chars")
        print(f"  Max thoughts:  {'infinite' if max_thoughts == 0 else max_thoughts}")
        print(f"  Seed thought:  {seed_thought[:60]}...")
        print(f"{'='*70}")
        print()

        context = seed_thought
        thought_idx = 0
        t0 = time.time()

        try:
            while max_thoughts == 0 or thought_idx < max_thoughts:
                thought_idx += 1
                elapsed = time.time() - t0

                print(f"\n{'─'*70}")
                print(f"  THOUGHT #{thought_idx}  |  elapsed={elapsed:.0f}s  |  "
                      f"context={len(context)} chars")
                print(f"{'─'*70}")
                print(f"  > {context[-80:]}...")
                print()

                # Generate one thought through the substrate
                result = self._generate_one_thought(context, verbose=verbose)

                # Log
                logger.log_thought(
                    thought_idx=thought_idx,
                    prompt=context[-200:],  # last 200 chars of prompt
                    generated=result['text'],
                    n_tokens=result['n_tokens'],
                    n_overrides=result['n_overrides'],
                    mean_cscore=result['mean_cscore'],
                    total_spikes=result['total_spikes'],
                    metrics={
                        'std_cscore': result['std_cscore'],
                        'max_cscore': result['max_cscore'],
                        'mean_alignment': result['mean_alignment'],
                        'mean_entropy': result['mean_entropy'],
                    },
                )

                # Print thought summary
                print(f"\n  [{thought_idx}] {result['n_tokens']} tokens, "
                      f"{result['n_overrides']} overrides, "
                      f"C={result['mean_cscore']:.3f}, "
                      f"spk={result['total_spikes']}, "
                      f"align={result['mean_alignment']:.3f}")

                # Self-prompt: slide context window
                context += result['text']
                if len(context) > self.context_window:
                    # Keep the seed thought prefix + recent context
                    context = seed_thought[:50] + "... " + context[-(self.context_window - 54):]

        except KeyboardInterrupt:
            print(f"\n\n  [STOPPED by user after {thought_idx} thoughts]")

        finally:
            elapsed = time.time() - t0
            summary = logger.save_summary(substrate_type, self.alpha, elapsed)

            print(f"\n{'='*70}")
            print(f"  THOUGHT LOOP COMPLETE")
            print(f"{'='*70}")
            print(f"  Thoughts:    {summary['n_thoughts']}")
            print(f"  Tokens:      {summary['total_tokens']}")
            print(f"  Overrides:   {summary['total_overrides']} ({summary['override_rate']:.1%})")
            print(f"  Mean C-Score: {summary['mean_cscore']:.4f}")
            print(f"  Total spikes: {summary['total_spikes']}")
            print(f"  Duration:    {summary['elapsed_s']:.1f}s")
            print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Self-Prompting Thought Loop')

    # Substrate selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--relay', type=str, default=None,
                       help='CL1 relay URL (e.g., http://192.168.1.100:8765)')
    group.add_argument('--local', action='store_true', default=True,
                       help='Use local Izhikevich substrate (default)')

    # Generation parameters
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Neural blend weight (default: 0.5)')
    parser.add_argument('--tokens', type=int, default=50,
                        help='Tokens per thought (default: 50)')
    parser.add_argument('--context', type=int, default=300,
                        help='Context window in chars (default: 300)')
    parser.add_argument('--max-thoughts', type=int, default=0,
                        help='Max thoughts (0=infinite, default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--seed-thought', type=str, default=None,
                        help='Custom starting thought')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output')

    args = parser.parse_args()

    # Create substrate
    if args.relay:
        from LLM_Encoder.cl1_substrate import CL1Substrate
        substrate = CL1Substrate(relay_url=args.relay, seed=args.seed)
        if not substrate.is_connected:
            print("\nERROR: Cannot connect to CL1 relay. Is it running?")
            print(f"  Expected: {args.relay}/health")
            print("\nTo use local Izhikevich instead:")
            print("  python -m LLM_Encoder.self_prompt_loop --local")
            sys.exit(1)
    else:
        substrate = IzhikevichSubstrate(seed=args.seed)
        print(f"  Using local Izhikevich substrate (1000 neurons)")

    # Create and run loop
    loop = SelfPromptLoop(
        substrate=substrate,
        alpha=args.alpha,
        tokens_per_thought=args.tokens,
        context_window=args.context,
        seed=args.seed,
    )

    loop.run(
        seed_thought=args.seed_thought,
        max_thoughts=args.max_thoughts,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
