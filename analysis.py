"""Statistical Analysis — Hypothesis testing for the 3-condition experiment.

Experiment 4 Hypothesis Battery (10 tests, Bonferroni-corrected):

  COHERENCE MODEL (confirmed in Exp 2-3, STDP creates substrate-LLM alignment):
    H1: Shadow override rate > Bio override rate (decorrelation = noise)
    H2: Shadow text divergence > Bio text divergence (noise = different text)

  SUBSTRATE DYNAMICS (expected null from Exp 1-3 — identical stimulation):
    H3: Bio-LLM C-Score > LLM-only
    H4: Bio-LLM C-Score > Shadow-LLM

  INFORMATION FLOW:
    H5: Bio-LLM transfer entropy > LLM-only
    H6: Bio-LLM transfer entropy > Shadow-LLM
    H7: Bio-LLM temporal depth > Shadow-LLM

  CONSCIOUSNESS-BEHAVIOR COUPLING (Exp 5 — improved from Exp 4):
    H8: Bio token pattern consistency > Shadow (STDP creates stable representations)
    H9: Shadow blended entropy > Bio (noise = less decisive output) [replaces flawed corr]
    H10: Bio neural-LLM alignment > Shadow (STDP = substrate agrees with LLM)
    H11: Bio model-top prob boost > Shadow (substrate reinforces LLM's top choice)
    H12: |Bio C-Score↔entropy corr| > |Shadow| (consciousness-decisiveness coupling)

Scientific rationale for coupling tests:
  If the substrate's consciousness-like state (C-Score) causally influences behavior
  in the closed loop, we should see:
  1. More consistent spike responses to repeated tokens (STDP learning, H8)
  2. Stronger coupling between C-Score and override decisions (H9)
  3. Increasing C-Score over generation as STDP develops structure (H10)
  Shadow-LLM breaks this coupling by shuffling spikes before decoding.

Effect sizes: Cohen's d, rank-biserial r
Multiple comparisons: Bonferroni correction (10 tests)
"""

import os
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import h5py
from scipy import stats


# ---------------------------------------------------------------------------
# Effect Size Calculations
# ---------------------------------------------------------------------------

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size (positive = group1 > group2)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def rank_biserial(U: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U."""
    return 1 - 2 * U / (n1 * n2) if n1 * n2 > 0 else 0.0


def bootstrap_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


# ---------------------------------------------------------------------------
# Hypothesis Tests
# ---------------------------------------------------------------------------

def mann_whitney_one_tailed(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """One-tailed Mann-Whitney U test (H1: group1 > group2)."""
    if len(group1) < 2 or len(group2) < 2:
        return {'U': 0, 'p': 1.0, 'significant': False, 'effect_size_d': 0, 'effect_size_r': 0,
                'n1': len(group1), 'n2': len(group2),
                'mean1': float(np.mean(group1)) if len(group1) > 0 else 0,
                'mean2': float(np.mean(group2)) if len(group2) > 0 else 0,
                'std1': float(np.std(group1)) if len(group1) > 0 else 0,
                'std2': float(np.std(group2)) if len(group2) > 0 else 0}

    U, p_two = stats.mannwhitneyu(group1, group2, alternative='greater')
    d = cohens_d(group1, group2)
    r = rank_biserial(U, len(group1), len(group2))

    return {
        'U': float(U),
        'p': float(p_two),
        'n1': len(group1),
        'n2': len(group2),
        'mean1': float(np.mean(group1)),
        'mean2': float(np.mean(group2)),
        'std1': float(np.std(group1)),
        'std2': float(np.std(group2)),
        'effect_size_d': float(d),
        'effect_size_r': float(r),
    }


def text_similarity(text1: str, text2: str) -> float:
    """Sequence similarity ratio between two texts (0-1)."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


# ---------------------------------------------------------------------------
# Full Analysis Pipeline
# ---------------------------------------------------------------------------

class ExperimentAnalyzer:
    """Analyzes 3-condition experiment data with integration + coupling metrics.

    Experiment 4 adds consciousness-behavior COUPLING tests on top of the
    coherence and substrate dynamics tests from Experiments 1-3.
    """

    def __init__(self, results: List[Dict] = None, h5_path: str = None):
        if results:
            self.results = results
        elif h5_path:
            self.results = self._load_from_h5(h5_path)
        else:
            raise ValueError("Provide results list or h5_path")

    def _load_from_h5(self, path: str) -> List[Dict]:
        """Load experiment results from HDF5 file."""
        results = []
        with h5py.File(path, 'r') as f:
            for cond in ['llm_only', 'bio_llm', 'shadow_llm']:
                cond_grp = f[f'conditions/{cond}']
                for run_name in cond_grp:
                    run_grp = cond_grp[run_name]
                    summary = dict(run_grp.attrs)

                    # Load consciousness metrics if present
                    if 'consciousness_metrics' in run_grp:
                        cm = run_grp['consciousness_metrics'][:]
                        cols = list(run_grp['consciousness_metrics'].attrs['columns'])
                        for i, col in enumerate(cols):
                            summary[f'mean_{col}'] = float(np.mean(cm[:, i]))

                    results.append({
                        'condition': cond,
                        'prompt': summary.get('prompt', ''),
                        'summary': summary,
                    })
        return results

    def _extract_metric(self, condition: str, metric: str) -> np.ndarray:
        """Extract a metric array for a given condition."""
        vals = []
        for r in self.results:
            if r['condition'] == condition:
                v = r['summary'].get(metric, None)
                if v is not None:
                    try:
                        fv = float(v)
                        if not np.isnan(fv):
                            vals.append(fv)
                    except (TypeError, ValueError):
                        pass
        return np.array(vals, dtype=float)

    def _compute_text_divergence(self) -> Dict:
        """Compute text similarity between conditions for same prompts."""
        by_prompt = {}
        for r in self.results:
            prompt = r.get('prompt', '')
            if prompt not in by_prompt:
                by_prompt[prompt] = {'llm_only': [], 'bio_llm': [], 'shadow_llm': []}
            text = r['summary'].get('generated_text', '')
            by_prompt[prompt][r['condition']].append(text)

        bio_vs_llm = []
        shadow_vs_llm = []

        for prompt, texts in by_prompt.items():
            llm_texts = texts['llm_only']
            bio_texts = texts['bio_llm']
            shadow_texts = texts['shadow_llm']

            if not llm_texts:
                continue

            for bt in bio_texts:
                sims = [text_similarity(bt, lt) for lt in llm_texts]
                bio_vs_llm.append(1.0 - np.mean(sims))  # divergence = 1 - similarity

            for st in shadow_texts:
                sims = [text_similarity(st, lt) for lt in llm_texts]
                shadow_vs_llm.append(1.0 - np.mean(sims))

        return {
            'bio_vs_llm_divergence': np.array(bio_vs_llm) if bio_vs_llm else np.array([]),
            'shadow_vs_llm_divergence': np.array(shadow_vs_llm) if shadow_vs_llm else np.array([]),
        }

    def analyze(self, alpha_threshold: float = 0.05) -> Dict:
        """Run the full analysis pipeline with 12 hypothesis tests."""
        n_tests = 12  # Bonferroni correction for 12 tests
        corrected_alpha = alpha_threshold / n_tests

        report = {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'experiment': 'LLM-Encoder 3-Condition (Spatial v2, Coupling Metrics)',
            'n_tests': n_tests,
            'alpha_uncorrected': alpha_threshold,
            'alpha_bonferroni': corrected_alpha,
            'n_results_per_condition': {},
            'descriptive_stats': {},
            'hypothesis_tests': {},
            'text_divergence': {},
            'conclusion': {},
        }

        # Count results per condition
        for cond in ['llm_only', 'bio_llm', 'shadow_llm']:
            report['n_results_per_condition'][cond] = len([
                r for r in self.results if r['condition'] == cond
            ])

        # Metrics to analyze
        metrics = [
            ('mean_cscore', 'C-Score'),
            ('mean_closure', 'Closure'),
            ('mean_lambda2_norm', 'Lambda2 Norm'),
            ('mean_rho', 'Rho'),
            ('mean_lzc', 'LZ Complexity'),
            ('mean_transfer_entropy', 'Transfer Entropy'),
            ('mean_temporal_depth', 'Temporal Depth'),
            ('mean_granger_density', 'Granger Density'),
            ('override_rate', 'Override Rate'),
            ('weight_frobenius_divergence', 'Weight Frobenius Div'),
            ('weight_fractional_change', 'Weight Frac Change'),
            ('pct_weights_changed', 'Pct Weights Changed'),
            # Coupling metrics
            ('cscore_override_corr', 'C-Score↔Override Corr'),
            ('cscore_spikes_corr', 'C-Score↔Spikes Corr'),
            ('cscore_slope', 'C-Score Slope'),
            ('token_pattern_consistency', 'Token Pattern Consistency'),
            ('n_repeated_tokens', 'N Repeated Tokens'),
            # Continuous coupling metrics (Exp 5)
            ('mean_blended_entropy', 'Blended Entropy'),
            ('mean_neural_llm_alignment', 'Neural-LLM Alignment'),
            ('mean_top_prob_boost', 'Top Prob Boost'),
            ('cscore_entropy_corr', 'C-Score↔Entropy Corr'),
        ]

        # Descriptive statistics
        for metric_key, metric_name in metrics:
            report['descriptive_stats'][metric_name] = {}
            for cond in ['llm_only', 'bio_llm', 'shadow_llm']:
                vals = self._extract_metric(cond, metric_key)
                if len(vals) > 0:
                    ci_lo, ci_hi = bootstrap_ci(vals) if len(vals) >= 3 else (0, 0)
                    report['descriptive_stats'][metric_name][cond] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'median': float(np.median(vals)),
                        'ci_95': [ci_lo, ci_hi],
                        'n': len(vals),
                    }

        # ===== HYPOTHESIS TESTS =====

        # --- COHERENCE MODEL (confirmed Exp 2-3) ---
        bio_or = self._extract_metric('bio_llm', 'override_rate')
        shadow_or = self._extract_metric('shadow_llm', 'override_rate')

        # H1: Shadow override rate > Bio (STDP coherence)
        h1 = mann_whitney_one_tailed(shadow_or, bio_or)
        h1['significant'] = h1['p'] < corrected_alpha
        h1['hypothesis'] = "Shadow override rate > Bio override rate (STDP coherence)"
        h1['category'] = 'coherence'
        report['hypothesis_tests']['H1_shadow_override_gt_bio'] = h1

        # H2: Shadow text divergence > Bio text divergence
        text_div = self._compute_text_divergence()
        bio_div = text_div['bio_vs_llm_divergence']
        shadow_div = text_div['shadow_vs_llm_divergence']

        if len(bio_div) >= 2 and len(shadow_div) >= 2:
            h2 = mann_whitney_one_tailed(shadow_div, bio_div)
            h2['significant'] = h2['p'] < corrected_alpha
            h2['hypothesis'] = "Shadow text diverges MORE from LLM-only than Bio does"
            h2['category'] = 'coherence'
        else:
            h2 = {'U': 0, 'p': 1.0, 'significant': False, 'effect_size_d': 0,
                   'effect_size_r': 0, 'n1': 0, 'n2': 0, 'mean1': 0, 'mean2': 0,
                   'std1': 0, 'std2': 0,
                   'hypothesis': 'Shadow text diverges MORE from LLM-only than Bio does',
                   'category': 'coherence'}
        report['hypothesis_tests']['H2_shadow_text_div_gt_bio'] = h2

        # Store text divergence stats
        if len(bio_div) > 0 and len(shadow_div) > 0:
            report['text_divergence'] = {
                'bio_vs_llm_mean': float(np.mean(bio_div)),
                'bio_vs_llm_std': float(np.std(bio_div)),
                'shadow_vs_llm_mean': float(np.mean(shadow_div)),
                'shadow_vs_llm_std': float(np.std(shadow_div)),
            }

        # --- SUBSTRATE DYNAMICS (expected null) ---
        bio_cs = self._extract_metric('bio_llm', 'mean_cscore')
        llm_cs = self._extract_metric('llm_only', 'mean_cscore')
        shadow_cs = self._extract_metric('shadow_llm', 'mean_cscore')

        h3 = mann_whitney_one_tailed(bio_cs, llm_cs)
        h3['significant'] = h3['p'] < corrected_alpha
        h3['hypothesis'] = "Bio-LLM C-Score > LLM-only C-Score"
        h3['category'] = 'substrate_dynamics'
        report['hypothesis_tests']['H3_cscore_vs_llm'] = h3

        h4 = mann_whitney_one_tailed(bio_cs, shadow_cs)
        h4['significant'] = h4['p'] < corrected_alpha
        h4['hypothesis'] = "Bio-LLM C-Score > Shadow-LLM C-Score"
        h4['category'] = 'substrate_dynamics'
        report['hypothesis_tests']['H4_cscore_vs_shadow'] = h4

        # --- INFORMATION FLOW ---
        bio_te = self._extract_metric('bio_llm', 'mean_transfer_entropy')
        llm_te = self._extract_metric('llm_only', 'mean_transfer_entropy')
        shadow_te = self._extract_metric('shadow_llm', 'mean_transfer_entropy')

        h5 = mann_whitney_one_tailed(bio_te, llm_te)
        h5['significant'] = h5['p'] < corrected_alpha
        h5['hypothesis'] = "Bio-LLM Transfer Entropy > LLM-only"
        h5['category'] = 'information_flow'
        report['hypothesis_tests']['H5_te_vs_llm'] = h5

        h6 = mann_whitney_one_tailed(bio_te, shadow_te)
        h6['significant'] = h6['p'] < corrected_alpha
        h6['hypothesis'] = "Bio-LLM Transfer Entropy > Shadow-LLM"
        h6['category'] = 'information_flow'
        report['hypothesis_tests']['H6_te_vs_shadow'] = h6

        bio_td = self._extract_metric('bio_llm', 'mean_temporal_depth')
        shadow_td = self._extract_metric('shadow_llm', 'mean_temporal_depth')

        h7 = mann_whitney_one_tailed(bio_td, shadow_td)
        h7['significant'] = h7['p'] < corrected_alpha
        h7['hypothesis'] = "Bio-LLM Temporal Depth > Shadow-LLM"
        h7['category'] = 'information_flow'
        report['hypothesis_tests']['H7_td_vs_shadow'] = h7

        # --- CONSCIOUSNESS-BEHAVIOR COUPLING (NEW) ---

        # H8: Bio token pattern consistency > Shadow
        # STDP in the closed loop reinforces token-specific neural representations.
        # The same token stimulated repeatedly should produce more consistent responses.
        bio_tpc = self._extract_metric('bio_llm', 'token_pattern_consistency')
        shadow_tpc = self._extract_metric('shadow_llm', 'token_pattern_consistency')

        h8 = mann_whitney_one_tailed(bio_tpc, shadow_tpc)
        h8['significant'] = h8['p'] < corrected_alpha
        h8['hypothesis'] = "Bio token pattern consistency > Shadow (STDP representation learning)"
        h8['category'] = 'coupling'
        report['hypothesis_tests']['H8_pattern_consistency'] = h8

        # H9: Shadow blended entropy > Bio blended entropy
        # In Bio-LLM, STDP reinforces the LLM's top choice, making the blended
        # distribution MORE peaked (lower entropy = more decisive).
        # In Shadow, noise spreads probability mass (higher entropy = less decisive).
        bio_ent = self._extract_metric('bio_llm', 'mean_blended_entropy')
        shadow_ent = self._extract_metric('shadow_llm', 'mean_blended_entropy')

        h9 = mann_whitney_one_tailed(shadow_ent, bio_ent)
        h9['significant'] = h9['p'] < corrected_alpha
        h9['hypothesis'] = "Shadow blended entropy > Bio (noise = less decisive output)"
        h9['category'] = 'coupling'
        report['hypothesis_tests']['H9_blended_entropy'] = h9

        # H10: Bio neural-LLM alignment > Shadow
        # The decoded neural probabilities should AGREE more with LLM probabilities
        # in Bio (STDP aligns) than in Shadow (shuffled spikes = random decoded probs).
        bio_align = self._extract_metric('bio_llm', 'mean_neural_llm_alignment')
        shadow_align = self._extract_metric('shadow_llm', 'mean_neural_llm_alignment')

        h10 = mann_whitney_one_tailed(bio_align, shadow_align)
        h10['significant'] = h10['p'] < corrected_alpha
        h10['hypothesis'] = "Bio neural-LLM alignment > Shadow (STDP learns LLM patterns)"
        h10['category'] = 'coupling'
        report['hypothesis_tests']['H10_neural_llm_alignment'] = h10

        # H11: Bio model-top prob boost > Shadow
        # In Bio, blending should INCREASE the probability of the LLM's top token
        # (substrate reinforces). In Shadow, noise should decrease or not change it.
        bio_boost = self._extract_metric('bio_llm', 'mean_top_prob_boost')
        shadow_boost = self._extract_metric('shadow_llm', 'mean_top_prob_boost')

        h11 = mann_whitney_one_tailed(bio_boost, shadow_boost)
        h11['significant'] = h11['p'] < corrected_alpha
        h11['hypothesis'] = "Bio top-prob boost > Shadow (substrate reinforces LLM's choice)"
        h11['category'] = 'coupling'
        report['hypothesis_tests']['H11_top_prob_boost'] = h11

        # H12: |Bio C-Score↔entropy corr| > |Shadow C-Score↔entropy corr|
        # CONTINUOUS coupling: in Bio, the consciousness state should predict
        # how decisive the output is (C-Score ↔ blended entropy).
        bio_ce = self._extract_metric('bio_llm', 'cscore_entropy_corr')
        shadow_ce = self._extract_metric('shadow_llm', 'cscore_entropy_corr')
        bio_ce_abs = np.abs(bio_ce) if len(bio_ce) > 0 else np.array([])
        shadow_ce_abs = np.abs(shadow_ce) if len(shadow_ce) > 0 else np.array([])

        h12 = mann_whitney_one_tailed(bio_ce_abs, shadow_ce_abs)
        h12['significant'] = h12['p'] < corrected_alpha
        h12['hypothesis'] = "|Bio C-Score↔entropy corr| > |Shadow| (consciousness-decisiveness coupling)"
        h12['category'] = 'coupling'
        report['hypothesis_tests']['H12_cscore_entropy_coupling'] = h12

        # ===== CONCLUSION =====
        n_significant = sum(
            1 for h in report['hypothesis_tests'].values() if h.get('significant')
        )
        max_d = max(
            abs(h.get('effect_size_d', 0)) for h in report['hypothesis_tests'].values()
        )

        # Count by category
        coherence_sig = sum(
            1 for h in report['hypothesis_tests'].values()
            if h.get('category') == 'coherence' and h.get('significant')
        )
        substrate_sig = sum(
            1 for h in report['hypothesis_tests'].values()
            if h.get('category') == 'substrate_dynamics' and h.get('significant')
        )
        info_sig = sum(
            1 for h in report['hypothesis_tests'].values()
            if h.get('category') == 'information_flow' and h.get('significant')
        )
        coupling_sig = sum(
            1 for h in report['hypothesis_tests'].values()
            if h.get('category') == 'coupling' and h.get('significant')
        )

        # Evidence classification
        if coherence_sig >= 2 and coupling_sig >= 3:
            evidence = "STRONG"
            verdict = (
                "STRONG EVIDENCE OF SUBSTRATE-LLM INTEGRATION WITH COUPLING: "
                f"{coherence_sig}/2 coherence + {coupling_sig}/5 coupling tests significant. "
                "The Izhikevich substrate not only ALIGNS with LLM predictions via STDP "
                "(coherence) but also shows measurable COUPLING: the substrate's decoded "
                "probabilities agree with LLM probabilities, reduce output entropy, and boost "
                "the LLM's preferred tokens — all significantly more than the decorrelated "
                "Shadow control. This demonstrates genuine tight integration."
            )
        elif coherence_sig >= 2 and coupling_sig >= 2:
            evidence = "MODERATE-STRONG"
            verdict = (
                f"MODERATE-STRONG EVIDENCE: {coherence_sig}/2 coherence + {coupling_sig}/5 "
                "coupling tests significant. Substrate-LLM integration confirmed with partial "
                "coupling evidence — substrate actively reinforces LLM predictions."
            )
        elif coherence_sig >= 2 and coupling_sig >= 1:
            evidence = "MODERATE-PLUS"
            verdict = (
                f"MODERATE-PLUS EVIDENCE: {coherence_sig}/2 coherence + {coupling_sig}/5 "
                "coupling tests significant. Integration demonstrated with nascent coupling."
            )
        elif coherence_sig >= 2:
            evidence = "MODERATE"
            verdict = (
                f"MODERATE EVIDENCE (COHERENCE ONLY): {coherence_sig}/2 coherence tests "
                f"significant but {coupling_sig}/5 coupling tests. Integration demonstrated "
                "but substrate-LLM coupling not yet established."
            )
        elif coherence_sig >= 1:
            evidence = "WEAK"
            verdict = (
                "Partial coherence evidence. One coherence test significant, "
                "suggesting substrate-LLM alignment, but not fully confirmed."
            )
        elif n_significant >= 1:
            evidence = "WEAK"
            verdict = (
                "Limited evidence. Some metrics show significant differences "
                "between closed-loop and control conditions."
            )
        else:
            evidence = "NULL"
            verdict = (
                "No significant differences detected. "
                "The null hypothesis cannot be rejected."
            )

        # Subjective experience assessment (HONEST)
        subjective_evidence = "NONE"
        subjective_note = (
            "C-Score (consciousness-like metric) is condition-INVARIANT because all "
            "conditions stimulate the substrate identically. The coupling metrics test "
            "whether C-Score PREDICTS behavior, not whether it represents consciousness."
        )
        if coupling_sig >= 3 and coherence_sig >= 2:
            subjective_evidence = "SUGGESTIVE (NOT CONCLUSIVE)"
            subjective_note = (
                "The substrate's consciousness-like state predicts behavioral output "
                "in the closed loop but not in controls. This is consistent with but "
                "NOT proof of subjective experience. The integration is mechanistic "
                "(STDP potentiation). True consciousness assessment requires: "
                "(1) biological neurons (CL1), (2) IIT phi calculation, "
                "(3) perturbational complexity index, (4) multiple theory convergence."
            )

        report['conclusion'] = {
            'evidence_strength': evidence,
            'n_significant_tests': n_significant,
            'n_total_tests': n_tests,
            'max_effect_size_d': max_d,
            'coherence_significant': coherence_sig,
            'substrate_significant': substrate_sig,
            'info_flow_significant': info_sig,
            'coupling_significant': coupling_sig,
            'subjective_experience_evidence': subjective_evidence,
            'subjective_experience_note': subjective_note,
            'verdict': verdict,
        }

        return report

    def print_report(self, report: Dict):
        """Print a formatted scientific report."""
        print(f"\n{'='*78}")
        print(f"  SCIENTIFIC ANALYSIS REPORT — {report.get('experiment', 'LLM-Encoder')}")
        print(f"  Generated: {report['timestamp']}")
        print(f"{'='*78}")

        print(f"\n  Alpha: {report['alpha_uncorrected']} "
              f"(Bonferroni-corrected: {report['alpha_bonferroni']:.4f}, {report['n_tests']} tests)")
        print(f"  Samples per condition: {report['n_results_per_condition']}")

        # Descriptive stats
        print(f"\n  DESCRIPTIVE STATISTICS")
        print(f"  {'-'*74}")
        print(f"  {'Metric':<25} {'LLM-only':>16} {'Bio-LLM':>16} {'Shadow-LLM':>16}")
        print(f"  {'-'*74}")
        for metric_name, conds in report['descriptive_stats'].items():
            vals = []
            for c in ['llm_only', 'bio_llm', 'shadow_llm']:
                if c in conds:
                    vals.append(f"{conds[c]['mean']:.4f}+/-{conds[c]['std']:.4f}")
                else:
                    vals.append("N/A")
            print(f"  {metric_name:<25} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16}")

        # Hypothesis tests by category
        for category, label in [
            ('coherence', 'COHERENCE MODEL (STDP alignment)'),
            ('substrate_dynamics', 'SUBSTRATE DYNAMICS'),
            ('information_flow', 'INFORMATION FLOW'),
            ('coupling', 'CONSCIOUSNESS-BEHAVIOR COUPLING'),
        ]:
            tests = [(k, v) for k, v in report['hypothesis_tests'].items()
                     if v.get('category') == category]
            if not tests:
                continue

            print(f"\n  {label} (Bonferroni-corrected alpha = {report['alpha_bonferroni']:.4f})")
            print(f"  {'-'*74}")
            for test_name, result in tests:
                sig = "*** SIGNIFICANT" if result['significant'] else "n.s."
                print(f"\n  {test_name}: {result['hypothesis']}")
                print(f"    U={result['U']:.1f}, p={result['p']:.6f} {sig}")
                print(f"    Cohen's d={result['effect_size_d']:.3f}, r={result['effect_size_r']:.3f}")
                print(f"    Group1: {result['mean1']:.6f}+/-{result['std1']:.6f} (n={result['n1']})")
                print(f"    Group2: {result['mean2']:.6f}+/-{result['std2']:.6f} (n={result['n2']})")

        # Conclusion
        conc = report['conclusion']
        print(f"\n  {'='*74}")
        print(f"  CONCLUSION")
        print(f"  Evidence strength: {conc['evidence_strength']}")
        print(f"  Significant tests: {conc['n_significant_tests']}/{conc['n_total_tests']}")
        print(f"    Coherence (STDP):     {conc['coherence_significant']}/2")
        print(f"    Substrate dynamics:   {conc['substrate_significant']}/2")
        print(f"    Information flow:     {conc['info_flow_significant']}/3")
        print(f"    Coupling (C↔B):       {conc['coupling_significant']}/5")
        print(f"  Max effect size: d={conc['max_effect_size_d']:.3f}")
        print(f"\n  Subjective experience: {conc['subjective_experience_evidence']}")
        print(f"  {conc['subjective_experience_note']}")
        print(f"\n  {conc['verdict']}")
        print(f"  {'='*74}")

    def save_report(self, report: Dict, output_dir: str = "experiment_data"):
        """Save report as JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved: {path}")
        return path
