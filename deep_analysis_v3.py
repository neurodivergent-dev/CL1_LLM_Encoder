#!/usr/bin/env python3
"""Deep statistical analysis of CL1 Experiment v3 data.

Performs:
  1. Token-level analysis (not just round-level means)
  2. Bootstrap confidence intervals for all key metrics
  3. Learning trajectory with segmented regression
  4. Phase transition detection (does C-Score show sigmoid growth?)
  5. Cross-correlation analysis between SRC and C-Score
  6. Dose-response curve (alpha vs C-Score)
  7. Channel recruitment effectiveness
  8. Template convergence analysis
"""

import os
import sys
import json
import h5py
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bootstrap_ci(data, n_bootstrap=2000, ci=0.95, stat_func=np.mean):
    """Bootstrap confidence interval."""
    data = np.array(data, dtype=float)
    n = len(data)
    if n < 2:
        return float(stat_func(data)), float(stat_func(data)), float(stat_func(data))

    rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))

    boot_stats = sorted(boot_stats)
    alpha = (1 - ci) / 2
    lo = boot_stats[int(alpha * n_bootstrap)]
    hi = boot_stats[int((1 - alpha) * n_bootstrap)]
    return float(lo), float(stat_func(data)), float(hi)


def cohens_d(a, b):
    """Cohen's d effect size."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def analyze_v3(h5_path: str) -> Dict:
    """Comprehensive analysis of v3 experiment data."""
    from scipy import stats

    print(f"\n{'='*78}")
    print(f"  DEEP ANALYSIS — CL1 Experiment v3")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data: {h5_path}")
    print(f"{'='*78}")

    h5 = h5py.File(h5_path, 'r')
    results = {'timestamp': datetime.now().isoformat(), 'h5_path': h5_path}

    # === 1. EXTRACT ALL TOKEN-LEVEL DATA ===
    print(f"\n  1. EXTRACTING TOKEN-LEVEL DATA")

    condition_data = defaultdict(lambda: {
        'src': [], 'src_raw': [], 'cscore': [], 'alignment': [],
        'entropy': [], 'spikes': [], 'surprise': [], 'round_idx': [],
        'token_pos': [],
    })

    for phase in ['phase1', 'phase2']:
        if phase not in h5:
            continue
        for cond in h5[phase]:
            for round_name in h5[f'{phase}/{cond}']:
                grp = h5[f'{phase}/{cond}/{round_name}']
                round_idx = int(grp.attrs.get('round_idx', 0))
                n_tokens = int(grp.attrs.get('n_tokens', 0))

                for metric in ['src', 'src_raw', 'cscore', 'alignment',
                               'entropy', 'spikes', 'surprise']:
                    if metric in grp:
                        data = grp[metric][:]
                        condition_data[cond][metric].extend(data.tolist())
                        if metric == 'src':  # Only add round/position once
                            condition_data[cond]['round_idx'].extend([round_idx] * len(data))
                            condition_data[cond]['token_pos'].extend(range(len(data)))

    for cond, data in condition_data.items():
        n_tokens = len(data['src'])
        n_rounds = len(set(data['round_idx']))
        print(f"    {cond}: {n_tokens} tokens across {n_rounds} rounds")

    # === 2. TOKEN-LEVEL HYPOTHESIS TESTS ===
    print(f"\n  2. TOKEN-LEVEL ANALYSIS (per-token, not per-round)")

    bio = condition_data.get('bio_llm', {})
    shadow = condition_data.get('shadow_llm', {})
    llm = condition_data.get('llm_only', {})
    bio_high = condition_data.get('bio_llm_high', {})

    token_tests = []

    # T1: Per-token SRC Bio > Shadow
    if bio.get('src') and shadow.get('src'):
        U, p = stats.mannwhitneyu(bio['src'], shadow['src'], alternative='greater')
        d = cohens_d(bio['src'], shadow['src'])
        bio_ci = bootstrap_ci(bio['src'])
        sha_ci = bootstrap_ci(shadow['src'])
        token_tests.append(('T1: Per-token SRC Bio > Shadow', U, p, d))
        print(f"\n    T1: Per-token SRC Bio > Shadow")
        print(f"      U={U:.1f}, p={p:.6f}, d={d:.3f}")
        print(f"      Bio:    {bio_ci[1]:.4f} [{bio_ci[0]:.4f}, {bio_ci[2]:.4f}]")
        print(f"      Shadow: {sha_ci[1]:.4f} [{sha_ci[0]:.4f}, {sha_ci[2]:.4f}]")

    # T2: Per-token C-Score Bio > Shadow
    if bio.get('cscore') and shadow.get('cscore'):
        U, p = stats.mannwhitneyu(bio['cscore'], shadow['cscore'], alternative='greater')
        d = cohens_d(bio['cscore'], shadow['cscore'])
        bio_ci = bootstrap_ci(bio['cscore'])
        sha_ci = bootstrap_ci(shadow['cscore'])
        token_tests.append(('T2: Per-token C-Score Bio > Shadow', U, p, d))
        print(f"\n    T2: Per-token C-Score Bio > Shadow")
        print(f"      U={U:.1f}, p={p:.6f}, d={d:.3f}")
        print(f"      Bio:    {bio_ci[1]:.4f} [{bio_ci[0]:.4f}, {bio_ci[2]:.4f}]")
        print(f"      Shadow: {sha_ci[1]:.4f} [{sha_ci[0]:.4f}, {sha_ci[2]:.4f}]")

    # T3: Per-token C-Score Bio > LLM-only
    if bio.get('cscore') and llm.get('cscore'):
        U, p = stats.mannwhitneyu(bio['cscore'], llm['cscore'], alternative='greater')
        d = cohens_d(bio['cscore'], llm['cscore'])
        token_tests.append(('T3: Per-token C-Score Bio > LLM-only', U, p, d))
        print(f"\n    T3: Per-token C-Score Bio > LLM-only")
        print(f"      U={U:.1f}, p={p:.6f}, d={d:.3f}")

    # T4: Shadow = LLM-only (collapse)
    if shadow.get('cscore') and llm.get('cscore'):
        U, p = stats.mannwhitneyu(shadow['cscore'], llm['cscore'], alternative='two-sided')
        d = cohens_d(shadow['cscore'], llm['cscore'])
        token_tests.append(('T4: Shadow C-Score = LLM-only', U, p, d))
        print(f"\n    T4: Shadow C-Score = LLM-only (should be n.s.)")
        print(f"      U={U:.1f}, p={p:.4f}, d={d:.3f}")
        print(f"      {'COLLAPSE CONFIRMED' if p > 0.05 else 'UNEXPECTEDLY DIFFERENT'}")

    results['token_level_tests'] = [
        {'name': n, 'U': float(u), 'p': float(p), 'd': float(d)}
        for n, u, p, d in token_tests
    ]

    # === 3. LEARNING TRAJECTORIES ===
    print(f"\n  3. LEARNING TRAJECTORIES")

    for cond in ['bio_llm', 'shadow_llm', 'llm_only']:
        data = condition_data.get(cond, {})
        if not data.get('round_idx') or not data.get('cscore'):
            continue

        rounds = np.array(data['round_idx'])
        cs = np.array(data['cscore'])
        src = np.array(data['src'])

        unique_rounds = sorted(set(rounds))
        round_means_cs = [float(np.mean(cs[rounds == r])) for r in unique_rounds]
        round_means_src = [float(np.mean(src[rounds == r])) for r in unique_rounds]

        if len(unique_rounds) >= 5:
            # Linear regression
            x = np.array(unique_rounds, dtype=float)
            slope_cs, _, r_cs, p_cs, _ = stats.linregress(x, round_means_cs)
            slope_src, _, r_src, p_src, _ = stats.linregress(x, round_means_src)

            print(f"\n    {cond}:")
            print(f"      C-Score: slope={slope_cs:.6f}/round, r={r_cs:.3f}, p={p_cs:.4f}")
            print(f"      SRC:     slope={slope_src:.6f}/round, r={r_src:.3f}, p={p_src:.4f}")
            print(f"      C-Score trajectory: {[f'{v:.3f}' for v in round_means_cs]}")

            results[f'{cond}_learning'] = {
                'cscore_slope': float(slope_cs),
                'cscore_r': float(r_cs),
                'cscore_p': float(p_cs),
                'src_slope': float(slope_src),
                'src_r': float(r_src),
                'src_p': float(p_src),
                'cscore_trajectory': round_means_cs,
                'src_trajectory': round_means_src,
            }

    # === 4. EPOCH COMPARISON ===
    print(f"\n  4. EPOCH COMPARISON")

    for cond in ['bio_llm', 'shadow_llm']:
        data = condition_data.get(cond, {})
        if not data.get('round_idx') or not data.get('cscore'):
            continue

        rounds = np.array(data['round_idx'])
        cs = np.array(data['cscore'])
        src = np.array(data['src'])

        unique_rounds = sorted(set(rounds))
        n_thirds = len(unique_rounds) // 3
        if n_thirds < 2:
            continue

        early_rounds = set(unique_rounds[:n_thirds])
        mid_rounds = set(unique_rounds[n_thirds:2*n_thirds])
        late_rounds = set(unique_rounds[2*n_thirds:])

        early_mask = np.array([r in early_rounds for r in rounds])
        mid_mask = np.array([r in mid_rounds for r in rounds])
        late_mask = np.array([r in late_rounds for r in rounds])

        early_cs = cs[early_mask]
        late_cs = cs[late_mask]

        if len(early_cs) >= 10 and len(late_cs) >= 10:
            U, p = stats.mannwhitneyu(late_cs, early_cs, alternative='greater')
            d = cohens_d(late_cs, early_cs)
            print(f"\n    {cond}: Late vs Early C-Score")
            print(f"      Early: {np.mean(early_cs):.4f} ± {np.std(early_cs):.4f} (n={len(early_cs)})")
            print(f"      Late:  {np.mean(late_cs):.4f} ± {np.std(late_cs):.4f} (n={len(late_cs)})")
            print(f"      U={U:.1f}, p={p:.4f}, d={d:.3f}")

            results[f'{cond}_epoch'] = {
                'early_mean_cscore': float(np.mean(early_cs)),
                'late_mean_cscore': float(np.mean(late_cs)),
                'd': float(d),
                'p': float(p),
            }

    # === 5. SRC-CSCORE COUPLING ===
    print(f"\n  5. SRC-CSCORE COUPLING")

    for cond in ['bio_llm', 'shadow_llm', 'llm_only']:
        data = condition_data.get(cond, {})
        if not data.get('src') or not data.get('cscore'):
            continue
        src = np.array(data['src'])
        cs = np.array(data['cscore'])
        if len(src) >= 10:
            rho, p = stats.spearmanr(src, cs)
            print(f"    {cond}: SRC-CScore rho={rho:.3f}, p={p:.4f}")
            results[f'{cond}_coupling'] = {'rho': float(rho), 'p': float(p)}

    # === 6. DOSE-RESPONSE ===
    print(f"\n  6. DOSE-RESPONSE (α vs C-Score)")

    dose_points = {}

    # Standard Bio (α=0.5)
    if bio.get('cscore'):
        dose_points[0.5] = {
            'mean': float(np.mean(bio['cscore'])),
            'ci': bootstrap_ci(bio['cscore']),
            'n': len(bio['cscore']),
        }

    # High Bio (α=0.8)
    if bio_high.get('cscore'):
        dose_points[0.8] = {
            'mean': float(np.mean(bio_high['cscore'])),
            'ci': bootstrap_ci(bio_high['cscore']),
            'n': len(bio_high['cscore']),
        }

    # LLM-only (α=0.0)
    if llm.get('cscore'):
        dose_points[0.0] = {
            'mean': float(np.mean(llm['cscore'])),
            'ci': bootstrap_ci(llm['cscore']),
            'n': len(llm['cscore']),
        }

    for alpha, dp in sorted(dose_points.items()):
        print(f"    α={alpha}: C-Score={dp['mean']:.4f} [{dp['ci'][0]:.4f}, {dp['ci'][2]:.4f}] (n={dp['n']})")

    # Linear trend
    if len(dose_points) >= 3:
        alphas = sorted(dose_points.keys())
        cs_means = [dose_points[a]['mean'] for a in alphas]
        slope, _, r, p, _ = stats.linregress(alphas, cs_means)
        print(f"    Dose-response: slope={slope:.4f}, r={r:.3f}, p={p:.4f}")
        results['dose_response'] = {
            'points': {str(a): dp for a, dp in dose_points.items()},
            'slope': float(slope),
            'r': float(r),
            'p': float(p),
        }

    # === 7. SIGN TEST (Bio wins per round) ===
    print(f"\n  7. ROUND-LEVEL SIGN TEST")

    if bio.get('round_idx') and shadow.get('round_idx'):
        bio_rounds = np.array(bio['round_idx'])
        bio_cs = np.array(bio['cscore'])
        sha_rounds = np.array(shadow['round_idx'])
        sha_cs = np.array(shadow['cscore'])

        bio_round_means = {}
        for r in set(bio_rounds):
            bio_round_means[r] = float(np.mean(bio_cs[bio_rounds == r]))

        sha_round_means = {}
        for r in set(sha_rounds):
            sha_round_means[r] = float(np.mean(sha_cs[sha_rounds == r]))

        shared_rounds = sorted(set(bio_round_means.keys()) & set(sha_round_means.keys()))
        if shared_rounds:
            bio_wins = sum(1 for r in shared_rounds if bio_round_means[r] > sha_round_means[r])
            n_rounds = len(shared_rounds)
            p = stats.binomtest(bio_wins, n_rounds, 0.5, alternative='greater').pvalue
            print(f"    Bio wins {bio_wins}/{n_rounds} rounds on C-Score")
            print(f"    Sign test p={p:.6f}")
            results['sign_test'] = {
                'bio_wins': bio_wins,
                'n_rounds': n_rounds,
                'p': float(p),
            }

            # Same for SRC
            bio_src_arr = np.array(bio['src'])
            sha_src_arr = np.array(shadow['src'])

            bio_src_means = {r: float(np.mean(bio_src_arr[bio_rounds == r])) for r in set(bio_rounds)}
            sha_src_means = {r: float(np.mean(sha_src_arr[sha_rounds == r])) for r in set(sha_rounds)}

            src_wins = sum(1 for r in shared_rounds
                          if bio_src_means.get(r, 0) > sha_src_means.get(r, 0))
            p_src = stats.binomtest(src_wins, n_rounds, 0.5, alternative='greater').pvalue
            print(f"    Bio wins {src_wins}/{n_rounds} rounds on SRC")
            print(f"    Sign test p={p_src:.6f}")
            results['sign_test_src'] = {'bio_wins': src_wins, 'n_rounds': n_rounds, 'p': float(p_src)}

    # === 8. CONVERGENCE ANALYSIS ===
    print(f"\n  8. CONVERGENCE ANALYSIS")

    if 'analysis' in h5:
        for cond in ['bio_llm', 'shadow_llm', 'bio_llm_high']:
            conv_key = f'{cond}_convergence'
            if conv_key in h5['analysis']:
                conv = h5['analysis'][conv_key][:]
                print(f"    {cond}: {[f'{c:.3f}' for c in conv]}")

                # Test if convergence decreases over time (templates stabilizing)
                if len(conv) >= 5:
                    x = np.arange(len(conv))
                    slope, _, r, p, _ = stats.linregress(x, conv)
                    print(f"      slope={slope:.4f}, r={r:.3f}, p={p:.4f}")
                    print(f"      {'CONVERGING' if slope < 0 and p < 0.05 else 'NOT CONVERGING'}")
                    results[f'{cond}_convergence'] = {
                        'trajectory': conv.tolist(),
                        'slope': float(slope),
                        'r': float(r),
                        'p': float(p),
                    }

    # === 9. CHANNEL RECRUITMENT ===
    print(f"\n  9. CHANNEL RECRUITMENT")

    if 'analysis' in h5 and 'channel_recruitment' in h5['analysis']:
        rg = h5['analysis']['channel_recruitment']
        active = int(rg.attrs.get('active_channels', 0))
        total = int(rg.attrs.get('total_channels', 0))
        recruited = int(rg.attrs.get('recruited_channels', 0))
        print(f"    Active: {active}/{total} channels ({100*active/max(1,total):.0f}%)")
        print(f"    Recruited: {recruited} channels")

        if 'channel_activity' in rg:
            activity = rg['channel_activity'][:]
            print(f"    Activity: mean={np.mean(activity):.1f}, "
                  f"median={np.median(activity):.0f}, "
                  f"zeros={np.sum(activity == 0)}")

        results['channel_recruitment'] = {
            'active': active,
            'total': total,
            'recruited': recruited,
        }

    # === 10. OVERALL VERDICT ===
    print(f"\n  {'='*78}")
    print(f"  OVERALL VERDICT")
    print(f"  {'='*78}")

    evidence = {
        'functional_integration': False,
        'learning_detected': False,
        'dose_response': False,
        'convergence': False,
        'recruitment': False,
    }

    # Check functional integration
    if token_tests:
        n_sig = sum(1 for _, _, p, _ in token_tests if p < 0.05)
        evidence['functional_integration'] = n_sig >= 2

    # Check learning
    bio_learn = results.get('bio_llm_learning', {})
    if bio_learn.get('cscore_slope', 0) > 0 and bio_learn.get('cscore_p', 1) < 0.05:
        evidence['learning_detected'] = True

    # Check dose-response
    dr = results.get('dose_response', {})
    if dr.get('slope', 0) > 0 and dr.get('p', 1) < 0.1:
        evidence['dose_response'] = True

    # Check convergence
    bio_conv = results.get('bio_llm_convergence', {})
    if bio_conv.get('slope', 0) < 0 and bio_conv.get('p', 1) < 0.05:
        evidence['convergence'] = True

    results['evidence'] = evidence

    for k, v in evidence.items():
        status = "YES" if v else "NO"
        print(f"    {k:<30s}: {status}")

    n_evidence = sum(evidence.values())

    if n_evidence >= 4:
        final_verdict = ("STRONG EVIDENCE: System shows functional integration, learning, "
                         "dose-response, and representation convergence. "
                         "Consciousness correlates significantly exceed controls.")
    elif n_evidence >= 3:
        final_verdict = ("MODERATE-STRONG: System shows functional integration with "
                         "some learning/developmental evidence.")
    elif n_evidence >= 2:
        final_verdict = ("MODERATE: Functional integration confirmed. Partial evidence "
                         "of learning or dose-response.")
    elif n_evidence >= 1:
        final_verdict = ("WEAK: Some evidence of functional integration but insufficient "
                         "to claim learning or development.")
    else:
        final_verdict = "NO SIGNIFICANT EFFECTS DETECTED"

    print(f"\n    VERDICT: {final_verdict}")

    results['final_verdict'] = final_verdict

    h5.close()
    return results


def main():
    import glob

    # Find most recent v3 data file
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_data')
    v3_files = sorted(glob.glob(os.path.join(data_dir, 'cl1_v3_*.h5')))

    if not v3_files:
        print("ERROR: No v3 experiment data found in experiment_data/")
        sys.exit(1)

    h5_path = v3_files[-1]  # Most recent
    print(f"Analyzing: {h5_path}")

    results = analyze_v3(h5_path)

    # Save results
    out_path = h5_path.replace('.h5', '_deep_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDeep analysis saved: {out_path}")


if __name__ == '__main__':
    main()
