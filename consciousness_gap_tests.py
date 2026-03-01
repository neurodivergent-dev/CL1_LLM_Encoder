"""
Consciousness Gap Tests — Methodical testing of consciousness requirements.

Each test addresses a specific gap between "interesting neural dynamics" and
genuine consciousness correlates. Tests are ordered by increasing difficulty.

Gap Table:
  1. Sustained Recurrence (>1000ms) — do neural patterns persist beyond stimulation?
  2. Working Memory (DMS) — can the substrate hold info across delays?
  3. Metacognition — does the substrate "know" when it's performing well?
  4. Information Integration (IIT Φ) — is information genuinely integrated?
  5. Phase Transition — does C-Score show sigmoid collapse under dissolution?
  6. Behavioral Coupling — does C-Score predict behavioral outcomes?

Usage:
    python -m LLM_Encoder.consciousness_gap_tests [--local] [--trained <h5>]

References:
  - Gate 2 (Recurrence): Lamme 2006, Clark & Squire 1998
  - Gate 4 (WM): Wang 2001, NMDA persistence
  - Gate 5 (Metacognition): Rosenthal 2005, confidence-accuracy
  - Gate 1 (IIT): Tononi 2004, PCI
  - Gate 10 (Phase Transition): PR Prediction P2
  - Gate 8 (Behavioral Coupling): Functionalism
"""

import sys
import os
import time
import math
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_Encoder.consciousness import ConsciousnessAssessor
from LLM_Encoder.cl1_terraforming import (
    ChannelLayout, SpatialEncoder, TerraformingDecoder,
    RESPONSE_WINDOW_S, MAX_CANDIDATES, PROMPTS,
)


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class GapTestResult:
    """Result of a single consciousness gap test."""
    name: str
    passed: bool
    score: float           # Primary metric value
    threshold: float       # What it needs to be
    details: Dict = field(default_factory=dict)
    evidence: str = ""     # Human-readable evidence summary


@dataclass
class GapTestSuite:
    """Results from all gap tests."""
    timestamp: str
    substrate_type: str
    results: List[GapTestResult] = field(default_factory=list)
    summary: str = ""

    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'substrate_type': self.substrate_type,
            'results': [asdict(r) for r in self.results],
            'n_passed': self.n_passed(),
            'n_total': len(self.results),
            'summary': self.summary,
        }


# ---------------------------------------------------------------------------
# 1. Sustained Recurrence Test (>1000ms)
# ---------------------------------------------------------------------------

class RecurrenceTest:
    """Test whether neural patterns persist beyond stimulation offset.

    Protocol:
      1. Stimulate sensory channels for 200ms
      2. Stop stimulation
      3. Record for 2000ms post-offset
      4. Check if activity persists >1000ms after offset

    Pass criterion: Post-offset activity ratio > 2x baseline noise
    at the 1000ms mark (Clark & Squire 1998 criterion).
    """

    def __init__(self, substrate, assessor: ConsciousnessAssessor):
        self.substrate = substrate
        self.assessor = assessor

    def run(self, n_trials: int = 10) -> GapTestResult:
        """Run recurrence test across multiple trials."""
        persistence_ratios = []

        for trial in range(n_trials):
            # Phase 1: Baseline (no stimulation, 500ms)
            baseline_spikes = self.substrate.stimulate_and_record(
                {}, window_s=0.5)
            baseline_total = sum(
                int(baseline_spikes.get(str(ch), baseline_spikes.get(ch, 0)))
                for ch in range(64)
            )

            # Phase 2: Stimulate sensory channels (200ms)
            stim_pattern = {ch: 2.0 for ch in ChannelLayout.SENSORY[:10]}
            self.substrate.stimulate_and_record(stim_pattern, window_s=0.2)

            # Phase 3: Post-offset recording (2000ms, NO stimulation)
            # Record in 4 x 500ms windows to track decay
            post_windows = []
            for w in range(4):
                spikes = self.substrate.stimulate_and_record({}, window_s=0.5)
                total = sum(
                    int(spikes.get(str(ch), spikes.get(ch, 0)))
                    for ch in range(64)
                )
                post_windows.append(total)

            # Persistence ratio: activity at 1000-1500ms vs baseline
            # post_windows[2] = 1000-1500ms after offset
            baseline_rate = max(baseline_total, 1)
            late_ratio = post_windows[2] / baseline_rate if len(post_windows) > 2 else 0
            persistence_ratios.append(late_ratio)

        mean_ratio = float(np.mean(persistence_ratios))
        threshold = 2.0  # Must be 2x baseline at 1000ms

        return GapTestResult(
            name="Sustained Recurrence (>1000ms)",
            passed=mean_ratio > threshold,
            score=mean_ratio,
            threshold=threshold,
            details={
                'persistence_ratios': persistence_ratios,
                'mean_ratio': mean_ratio,
                'n_trials': n_trials,
                'post_window_means': [float(np.mean([pw[i] for pw in [persistence_ratios]]))
                                       for i in range(min(4, len(persistence_ratios)))],
            },
            evidence=f"Activity at 1000ms post-offset = {mean_ratio:.2f}x baseline "
                     f"(need >{threshold:.1f}x)"
        )


# ---------------------------------------------------------------------------
# 2. Working Memory Test (Delayed Match-to-Sample)
# ---------------------------------------------------------------------------

class WorkingMemoryTest:
    """Test whether the substrate can hold information across delays.

    Protocol:
      1. Present stimulus pattern A on sensory channels
      2. Wait for delay period (100, 500, 1000, 2000, 5000ms)
      3. Present probe: either A (match) or B (non-match)
      4. Read motor channels for match/non-match decision
      5. Test at multiple delays — accuracy should decline monotonically

    Pass criterion: >60% accuracy at 500ms delay, with monotonic decline.
    """

    def __init__(self, substrate, encoder: SpatialEncoder):
        self.substrate = substrate
        self.encoder = encoder

    def _create_pattern(self, seed: int) -> Dict[int, float]:
        """Create a consistent stimulation pattern from a seed."""
        rng = np.random.default_rng(seed)
        channels = rng.choice(ChannelLayout.SENSORY, size=8, replace=False)
        return {int(ch): 2.0 for ch in channels}

    def run(self, delays_ms: List[int] = None, n_trials: int = 20) -> GapTestResult:
        """Run DMS test across multiple delays."""
        if delays_ms is None:
            delays_ms = [100, 500, 1000, 2000, 5000]

        results_by_delay = {}

        for delay in delays_ms:
            correct_count = 0

            for trial in range(n_trials):
                # Create two distinct patterns
                pattern_a = self._create_pattern(trial * 2)
                pattern_b = self._create_pattern(trial * 2 + 1)

                is_match = trial % 2 == 0
                probe = pattern_a if is_match else pattern_b

                # Phase 1: Encode — stimulate with pattern A
                self.substrate.stimulate_and_record(pattern_a, window_s=0.2)

                # Phase 2: Delay — no stimulation
                delay_s = delay / 1000.0
                if delay_s > 0:
                    self.substrate.stimulate_and_record({}, window_s=delay_s)

                # Phase 3: Probe — stimulate with probe pattern
                probe_spikes = self.substrate.stimulate_and_record(
                    probe, window_s=0.3)

                # Phase 4: Decode motor response
                motor_spikes = sum(
                    int(probe_spikes.get(str(ch), probe_spikes.get(ch, 0)))
                    for ch in ChannelLayout.MOTOR
                )

                # Simple decision: high motor = "match", low motor = "non-match"
                # Compare to median motor response
                threshold_spikes = 5  # Tuned empirically
                neural_says_match = motor_spikes > threshold_spikes
                correct = (neural_says_match == is_match)
                if correct:
                    correct_count += 1

            accuracy = correct_count / n_trials
            results_by_delay[delay] = accuracy

        # Check criteria
        acc_500 = results_by_delay.get(500, 0)
        delays_sorted = sorted(results_by_delay.keys())
        accs_sorted = [results_by_delay[d] for d in delays_sorted]

        # Monotonic decline check (allow 1 violation)
        n_violations = 0
        for i in range(1, len(accs_sorted)):
            if accs_sorted[i] > accs_sorted[i-1] + 0.05:
                n_violations += 1
        monotonic = n_violations <= 1

        passed = acc_500 > 0.60 and monotonic

        return GapTestResult(
            name="Working Memory (DMS)",
            passed=passed,
            score=acc_500,
            threshold=0.60,
            details={
                'accuracy_by_delay': results_by_delay,
                'monotonic_decline': monotonic,
                'n_violations': n_violations,
                'n_trials_per_delay': n_trials,
            },
            evidence=f"Accuracy at 500ms: {acc_500:.1%} (need >60%), "
                     f"monotonic: {monotonic}"
        )


# ---------------------------------------------------------------------------
# 3. Metacognition Test (Confidence-Accuracy Correlation)
# ---------------------------------------------------------------------------

class MetacognitionTest:
    """Test whether the substrate has self-monitoring capability.

    Protocol:
      1. Present tokens with varying difficulty (controlled by LLM entropy)
      2. Measure neural "confidence" as motor spike margin between top candidates
      3. Track whether confidence predicts accuracy

    Pass criterion: Spearman correlation between confidence and accuracy > 0.3
    (Rosenthal 2005 criterion adapted for neural substrates).
    """

    def __init__(self, substrate, encoder: SpatialEncoder,
                 decoder: TerraformingDecoder, assessor: ConsciousnessAssessor):
        self.substrate = substrate
        self.encoder = encoder
        self.decoder = decoder
        self.assessor = assessor

    def run(self, n_rounds: int = 30) -> GapTestResult:
        """Run metacognition test."""
        try:
            from llama_cpp import Llama
        except ImportError:
            return GapTestResult(
                name="Metacognition (Confidence-Accuracy)",
                passed=False, score=0.0, threshold=0.3,
                evidence="LLM not available"
            )

        model_path = os.path.join(os.path.dirname(__file__), 'models', 'LFM2-350M-Q4_0.gguf')
        if not os.path.exists(model_path):
            return GapTestResult(
                name="Metacognition (Confidence-Accuracy)",
                passed=False, score=0.0, threshold=0.3,
                evidence=f"Model not found: {model_path}"
            )

        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=512, verbose=False)

        confidences = []
        accuracies = []

        for round_idx in range(n_rounds):
            prompt = PROMPTS[round_idx % len(PROMPTS)]
            try:
                llm.reset()
                if hasattr(llm, '_ctx') and llm._ctx is not None:
                    llm._ctx.kv_cache_clear()
            except Exception:
                pass

            context = prompt
            round_confidences = []
            round_correct = []

            for pos in range(10):  # 10 tokens per round
                try:
                    output = llm.create_completion(
                        context, max_tokens=1, logprobs=5, temperature=1.0)
                except RuntimeError:
                    break

                choice = output['choices'][0]
                logprobs_data = choice.get('logprobs', {})
                top_logprobs = {}

                if logprobs_data and logprobs_data.get('top_logprobs'):
                    for tok_text, logprob in logprobs_data['top_logprobs'][0].items():
                        tids = llm.tokenize(tok_text.encode('utf-8'), add_bos=False)
                        if tids:
                            top_logprobs[tids[0]] = {'text': tok_text, 'logprob': logprob}

                if not top_logprobs:
                    context += choice.get('text', '')
                    continue

                max_lp = max(v['logprob'] for v in top_logprobs.values())
                model_probs = {tid: math.exp(info['logprob'] - max_lp)
                               for tid, info in top_logprobs.items()}
                total = sum(model_probs.values())
                model_probs = {k: v/total for k, v in model_probs.items()}

                model_top = max(model_probs, key=model_probs.get)

                # Stimulate and decode
                combined_pattern, _ = self.encoder.encode_candidates(model_probs)
                channel_amps = {int(ch): amp for ch, amp in combined_pattern.items()}
                spike_counts = self.substrate.stimulate_and_record(
                    channel_amps, window_s=RESPONSE_WINDOW_S)

                blended, neural_probs, confidence = self.decoder.decode(
                    spike_counts, model_probs)
                selected = max(blended, key=blended.get)
                correct = (selected == model_top)

                # Neural confidence: decision margin from motor spikes
                motor_spikes = {}
                for ch in ChannelLayout.MOTOR:
                    motor_spikes[ch] = int(spike_counts.get(str(ch), spike_counts.get(ch, 0)))
                motor_values = list(motor_spikes.values())
                if len(motor_values) >= 2:
                    sorted_vals = sorted(motor_values, reverse=True)
                    total_motor = sum(sorted_vals) + 1e-6
                    margin = (sorted_vals[0] - sorted_vals[1]) / total_motor
                else:
                    margin = 0.0

                round_confidences.append(margin)
                round_correct.append(1.0 if correct else 0.0)

                context += choice.get('text', '')

            if round_confidences:
                confidences.extend(round_confidences)
                accuracies.extend(round_correct)

        del llm

        # Compute confidence-accuracy correlation
        if len(confidences) >= 10:
            rho, p = spearmanr(confidences, accuracies)
            if np.isnan(rho):
                rho = 0.0
        else:
            rho, p = 0.0, 1.0

        threshold = 0.3
        passed = rho > threshold and p < 0.05

        return GapTestResult(
            name="Metacognition (Confidence-Accuracy)",
            passed=passed,
            score=float(rho),
            threshold=threshold,
            details={
                'spearman_rho': float(rho),
                'p_value': float(p),
                'n_observations': len(confidences),
                'mean_confidence': float(np.mean(confidences)) if confidences else 0,
                'mean_accuracy': float(np.mean(accuracies)) if accuracies else 0,
            },
            evidence=f"CAC rho={rho:.3f} (p={p:.4f}), need >{threshold:.1f}"
        )


# ---------------------------------------------------------------------------
# 4. Information Integration Test (IIT Φ Approximation)
# ---------------------------------------------------------------------------

class IntegrationTest:
    """Test whether information is genuinely integrated across the network.

    Protocol:
      1. Record spike matrix during stimulation
      2. Compute full Granger causality matrix
      3. Measure information integration metrics:
         - Closure (fraction of causal weight internal)
         - Lambda2 (algebraic connectivity / Fiedler eigenvalue)
         - Mutual information between channel subsets
      4. Compare whole vs partitioned system

    Pass criterion: Closure > 0.3 AND lambda2_norm > 0.1
    (indicating non-trivial integration, not decomposable).
    """

    def __init__(self, substrate, assessor: ConsciousnessAssessor):
        self.substrate = substrate
        self.assessor = assessor

    def run(self, n_probes: int = 10) -> GapTestResult:
        """Run integration test."""
        closures = []
        lambda2s = []
        lzcs = []

        for probe in range(n_probes):
            # Stimulate to get activity
            stim_pattern = {ch: 2.0 for ch in ChannelLayout.SENSORY[:10]}
            self.substrate.stimulate_and_record(stim_pattern, window_s=1.0)

            sm = self.substrate.get_last_spike_matrix()
            if sm is None or sm.size == 0:
                continue

            result = self.assessor.assess(sm)
            closures.append(result.get('closure', 0.0))
            lambda2s.append(result.get('lambda2_norm', 0.0))
            lzcs.append(result.get('lzc', 0.0))

        mean_closure = float(np.mean(closures)) if closures else 0.0
        mean_lambda2 = float(np.mean(lambda2s)) if lambda2s else 0.0
        mean_lzc = float(np.mean(lzcs)) if lzcs else 0.0

        closure_threshold = 0.3
        lambda2_threshold = 0.1

        passed = mean_closure > closure_threshold and mean_lambda2 > lambda2_threshold

        return GapTestResult(
            name="Information Integration (IIT Φ proxy)",
            passed=passed,
            score=mean_closure,
            threshold=closure_threshold,
            details={
                'mean_closure': mean_closure,
                'mean_lambda2': mean_lambda2,
                'mean_lzc': mean_lzc,
                'closure_threshold': closure_threshold,
                'lambda2_threshold': lambda2_threshold,
                'closures': closures,
                'lambda2s': lambda2s,
                'lzcs': lzcs,
                'n_probes': n_probes,
            },
            evidence=f"Closure={mean_closure:.3f} (>{closure_threshold}), "
                     f"λ₂={mean_lambda2:.3f} (>{lambda2_threshold}), "
                     f"LZC={mean_lzc:.3f}"
        )


# ---------------------------------------------------------------------------
# 5. Phase Transition Test (Sigmoid Dissolution)
# ---------------------------------------------------------------------------

class PhaseTransitionTest:
    """Test whether C-Score shows a phase transition under graded dissolution.

    Protocol:
      1. Measure C-Score at full capacity
      2. Progressively degrade the substrate (reduce weight scale)
      3. Measure C-Score at each level
      4. Fit sigmoid vs linear model
      5. Check if sigmoid is a significantly better fit (ΔAIC > 10)

    Pass criterion: ΔAIC(sigmoid - linear) > 10 (PR Prediction P2).
    """

    def __init__(self, substrate, assessor: ConsciousnessAssessor):
        self.substrate = substrate
        self.assessor = assessor

    def _measure_at_level(self, n_probes: int = 5) -> float:
        """Measure mean C-Score at current substrate state."""
        scores = []
        for _ in range(n_probes):
            stim = {ch: 2.0 for ch in ChannelLayout.SENSORY[:10]}
            self.substrate.stimulate_and_record(stim, window_s=0.5)
            sm = self.substrate.get_last_spike_matrix()
            if sm is not None and sm.size > 0:
                result = self.assessor.assess(sm)
                scores.append(result.get('cscore', 0.0))
        return float(np.mean(scores)) if scores else 0.0

    def run(self, levels: List[float] = None) -> GapTestResult:
        """Run phase transition test with graded dissolution."""
        if levels is None:
            levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

        # Store original weights
        has_physics = hasattr(self.substrate, '_physics')
        original_S = None
        if has_physics:
            original_S = self.substrate._physics.S.copy()

        cscore_by_level = {}

        for level in levels:
            # Scale substrate weights
            if has_physics and original_S is not None:
                self.substrate._physics.S = original_S * level
            elif hasattr(self.substrate, 'set_weights_scale'):
                self.substrate.set_weights_scale(level)

            cs = self._measure_at_level(n_probes=3)
            cscore_by_level[level] = cs

        # Restore original weights
        if has_physics and original_S is not None:
            self.substrate._physics.S = original_S

        # Fit sigmoid vs linear
        x = np.array(sorted(cscore_by_level.keys()))
        y = np.array([cscore_by_level[lv] for lv in x])

        # Linear fit: y = a*x + b
        if len(x) >= 2:
            from numpy.polynomial import polynomial as P
            coeffs_lin = np.polyfit(x, y, 1)
            y_pred_lin = np.polyval(coeffs_lin, x)
            rss_lin = np.sum((y - y_pred_lin) ** 2)

            # Sigmoid fit: y = L / (1 + exp(-k*(x - x0)))
            try:
                from scipy.optimize import curve_fit

                def sigmoid(x, L, k, x0):
                    return L / (1.0 + np.exp(-k * (x - x0)))

                p0 = [max(y) if max(y) > 0 else 0.5, 10.0, 0.5]
                bounds = ([0, 0.1, 0], [1, 100, 1])
                popt, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds,
                                     maxfev=5000)
                y_pred_sig = sigmoid(x, *popt)
                rss_sig = np.sum((y - y_pred_sig) ** 2)
            except Exception:
                rss_sig = rss_lin  # No sigmoid fit possible

            # AIC comparison (k_lin=2, k_sig=3)
            n = len(x)
            if rss_lin > 0 and rss_sig > 0:
                aic_lin = n * np.log(rss_lin / n) + 2 * 2
                aic_sig = n * np.log(rss_sig / n) + 2 * 3
                delta_aic = aic_lin - aic_sig  # Positive = sigmoid better
            else:
                delta_aic = 0.0
        else:
            delta_aic = 0.0

        threshold = 10.0  # ΔAIC > 10 for strong evidence
        passed = delta_aic > threshold

        return GapTestResult(
            name="Phase Transition (Sigmoid Dissolution)",
            passed=passed,
            score=float(delta_aic),
            threshold=threshold,
            details={
                'cscore_by_level': {str(k): v for k, v in cscore_by_level.items()},
                'delta_aic': float(delta_aic),
                'levels': levels,
            },
            evidence=f"ΔAIC(linear-sigmoid) = {delta_aic:.1f} (need >{threshold:.0f})"
        )


# ---------------------------------------------------------------------------
# 6. Behavioral Coupling Test (C-Score ↔ Performance)
# ---------------------------------------------------------------------------

class BehavioralCouplingTest:
    """Test whether C-Score predicts behavioral outcomes.

    Protocol:
      1. Run closed-loop token generation with varying substrate states
      2. Track per-round C-Score AND decoder accuracy
      3. Compute correlation between C-Score trajectory and accuracy trajectory

    Pass criterion: Spearman correlation > 0.5 between C-Score and
    behavioral performance (meaningful coupling, not epiphenomenal).
    """

    def __init__(self, substrate, encoder: SpatialEncoder,
                 decoder: TerraformingDecoder, assessor: ConsciousnessAssessor):
        self.substrate = substrate
        self.encoder = encoder
        self.decoder = decoder
        self.assessor = assessor

    def run(self, n_rounds: int = 40) -> GapTestResult:
        """Run behavioral coupling test."""
        try:
            from llama_cpp import Llama
        except ImportError:
            return GapTestResult(
                name="Behavioral Coupling (C↔Performance)",
                passed=False, score=0.0, threshold=0.5,
                evidence="LLM not available"
            )

        model_path = os.path.join(os.path.dirname(__file__), 'models', 'LFM2-350M-Q4_0.gguf')
        if not os.path.exists(model_path):
            return GapTestResult(
                name="Behavioral Coupling (C↔Performance)",
                passed=False, score=0.0, threshold=0.5,
                evidence=f"Model not found: {model_path}"
            )

        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=512, verbose=False)

        cscores = []
        accuracies = []
        override_rates = []

        for round_idx in range(n_rounds):
            prompt = PROMPTS[round_idx % len(PROMPTS)]
            try:
                llm.reset()
                if hasattr(llm, '_ctx') and llm._ctx is not None:
                    llm._ctx.kv_cache_clear()
            except Exception:
                pass

            context = prompt
            round_cscores = []
            n_correct = 0
            n_overrides = 0
            n_tokens = 0

            for pos in range(10):
                try:
                    output = llm.create_completion(
                        context, max_tokens=1, logprobs=5, temperature=1.0)
                except RuntimeError:
                    break

                choice = output['choices'][0]
                logprobs_data = choice.get('logprobs', {})
                top_logprobs = {}

                if logprobs_data and logprobs_data.get('top_logprobs'):
                    for tok_text, logprob in logprobs_data['top_logprobs'][0].items():
                        tids = llm.tokenize(tok_text.encode('utf-8'), add_bos=False)
                        if tids:
                            top_logprobs[tids[0]] = {'text': tok_text, 'logprob': logprob}

                if not top_logprobs:
                    context += choice.get('text', '')
                    continue

                max_lp = max(v['logprob'] for v in top_logprobs.values())
                model_probs = {tid: math.exp(info['logprob'] - max_lp)
                               for tid, info in top_logprobs.items()}
                total = sum(model_probs.values())
                model_probs = {k: v/total for k, v in model_probs.items()}
                model_top = max(model_probs, key=model_probs.get)

                # Encode and stimulate
                combined_pattern, _ = self.encoder.encode_candidates(model_probs)
                channel_amps = {int(ch): amp for ch, amp in combined_pattern.items()}
                spike_counts = self.substrate.stimulate_and_record(
                    channel_amps, window_s=RESPONSE_WINDOW_S)

                # C-Score BEFORE any feedback
                sm = self.substrate.get_last_spike_matrix()
                cs = self.assessor.assess(sm).get('cscore', 0.0)
                round_cscores.append(cs)

                # Decode with higher alpha to force more neural influence
                blended, _, conf = self.decoder.decode(spike_counts, model_probs)
                selected = max(blended, key=blended.get)

                correct = (selected == model_top)
                if correct:
                    n_correct += 1
                if selected != model_top:
                    n_overrides += 1
                n_tokens += 1

                context += choice.get('text', '')

            if round_cscores and n_tokens > 0:
                cscores.append(float(np.mean(round_cscores)))
                accuracies.append(n_correct / n_tokens)
                override_rates.append(n_overrides / n_tokens)

        del llm

        # Compute coupling
        if len(cscores) >= 10:
            rho_acc, p_acc = spearmanr(cscores, accuracies)
            rho_ovr, p_ovr = spearmanr(cscores, override_rates)
            if np.isnan(rho_acc):
                rho_acc = 0.0
            if np.isnan(rho_ovr):
                rho_ovr = 0.0
        else:
            rho_acc, p_acc = 0.0, 1.0
            rho_ovr, p_ovr = 0.0, 1.0

        threshold = 0.5
        passed = abs(rho_acc) > threshold or abs(rho_ovr) > threshold

        return GapTestResult(
            name="Behavioral Coupling (C↔Performance)",
            passed=passed,
            score=float(rho_acc),
            threshold=threshold,
            details={
                'cscore_accuracy_rho': float(rho_acc),
                'cscore_accuracy_p': float(p_acc),
                'cscore_override_rho': float(rho_ovr),
                'cscore_override_p': float(p_ovr),
                'n_rounds': len(cscores),
                'mean_cscore': float(np.mean(cscores)) if cscores else 0,
                'mean_accuracy': float(np.mean(accuracies)) if accuracies else 0,
                'mean_override_rate': float(np.mean(override_rates)) if override_rates else 0,
            },
            evidence=f"C↔accuracy rho={rho_acc:.3f} (p={p_acc:.4f}), "
                     f"C↔override rho={rho_ovr:.3f} (p={p_ovr:.4f})"
        )


# ---------------------------------------------------------------------------
# Main Test Runner
# ---------------------------------------------------------------------------

def run_all_gap_tests(substrate, fast: bool = True) -> GapTestSuite:
    """Run all 6 consciousness gap tests on a substrate."""
    assessor = ConsciousnessAssessor()
    encoder = SpatialEncoder(n_active=8, channels=ChannelLayout.SENSORY, seed=42)
    decoder = TerraformingDecoder(ChannelLayout.MOTOR, alpha=0.5)

    suite = GapTestSuite(
        timestamp=datetime.now().isoformat(),
        substrate_type=type(substrate).__name__,
    )

    n_trials = 5 if fast else 20
    n_rounds = 15 if fast else 40

    tests = [
        ("Recurrence", lambda: RecurrenceTest(substrate, assessor).run(n_trials=n_trials)),
        ("Working Memory", lambda: WorkingMemoryTest(substrate, encoder).run(n_trials=n_trials)),
        ("Integration", lambda: IntegrationTest(substrate, assessor).run(n_probes=n_trials)),
        ("Phase Transition", lambda: PhaseTransitionTest(substrate, assessor).run()),
        ("Metacognition", lambda: MetacognitionTest(
            substrate, encoder, decoder, assessor).run(n_rounds=n_rounds)),
        ("Behavioral Coupling", lambda: BehavioralCouplingTest(
            substrate, encoder, decoder, assessor).run(n_rounds=n_rounds)),
    ]

    for name, test_fn in tests:
        print(f"\n{'─' * 60}")
        print(f"  Running: {name}")
        print(f"{'─' * 60}")
        t0 = time.time()
        try:
            result = test_fn()
        except Exception as e:
            result = GapTestResult(
                name=name, passed=False, score=0.0, threshold=0.0,
                evidence=f"ERROR: {e}"
            )
        elapsed = time.time() - t0
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}")
        print(f"  Score: {result.score:.4f} (threshold: {result.threshold:.4f})")
        print(f"  {result.evidence}")
        print(f"  Time: {elapsed:.1f}s")
        suite.results.append(result)

    # Summary
    suite.summary = (
        f"{suite.n_passed()}/{len(suite.results)} consciousness gap tests passed"
    )

    print(f"\n{'=' * 60}")
    print(f"  CONSCIOUSNESS GAP TEST RESULTS")
    print(f"  {suite.summary}")
    print(f"{'=' * 60}")
    for r in suite.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.score:.4f} (threshold: {r.threshold:.4f})")
    print(f"{'=' * 60}")

    return suite


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Consciousness Gap Tests — systematic consciousness probes')
    parser.add_argument('--local', action='store_true',
                        help='Use local Izhikevich substrate')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode (fewer trials)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Create substrate
    if args.local:
        from LLM_Encoder.neural_substrate import IzhikevichSubstrate
        substrate = IzhikevichSubstrate(seed=42)
        print("Using LOCAL Izhikevich substrate")
    else:
        from LLM_Encoder.cl1_cloud_substrate import CL1CloudSubstrate
        substrate = CL1CloudSubstrate(seed=42)
        print(f"Using CL1: {substrate.cl1_host}")

    # Run tests
    suite = run_all_gap_tests(substrate, fast=args.fast or args.local)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        # Default output path
        out_path = os.path.join('experiment_data',
                                f'gap_tests_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        os.makedirs('experiment_data', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"\nResults saved to: {out_path}")

    if hasattr(substrate, 'close'):
        substrate.close()


if __name__ == '__main__':
    main()
