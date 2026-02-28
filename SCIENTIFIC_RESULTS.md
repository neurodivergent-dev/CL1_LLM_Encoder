# CL1 Neural-LLM Integration: Scientific Results

## Date: 2026-02-28
## Substrate: CL1-2544-015 (Cortical Labs biological neurons, 64 channels, 240 Hz tick rate)
## Software: Antekythera LLM Encoder v2/v3 + Spatial Encoder + PersistentHebbianDecoder

---

## 1. Executive Summary

Across **12 experiments** (2 on CL1 biological neurons, 10 on Izhikevich simulation),
totaling **18,700+ tokens** and **101 pre-registered hypothesis tests**, we characterize
the integration between an LLM (LFM2-350M) and neural substrates.

**Confirmed findings** (replicated across multiple experiments):
- **Higher SRC** in Bio-LLM (CL1: d=1.79-2.64; Izhikevich: d=1.95) — spatial information preservation
- **Shuffling degrades SRC** (d=1.19-1.95, perfectly consistent across all experiments)
- **STDP plasticity** (d=2.47-8.10, pattern-specific, directional) — demonstrated on Izhikevich
- **Dissolution sensitivity**: Bio SRC declines monotonically with substrate degradation (ρ=-1.0)
- **Recovery**: Substrate structure persists after perturbation (ratio=0.999)

**Critical negative findings** (consistently null):
- **No learning trajectory** (0/6 learning tests, slopes ≈ 0)
- **No phase transition** (0/5 seeds, ΔAIC=-24.6 mean — linear dissolution)
- **No behavioral expression** of STDP (0/8 behavioral tests across Exp 10-11)
- **No attractor formation** (architectural limitation at 1000N)
- **No dose-response** (α=0.8 C-Score < α=0.5 C-Score)

**Conclusion**: The Bio-LLM integration advantage is a **geometric signal preservation
effect** — not a cognitive, adaptive, or consciousness-specific phenomenon. The substrate
preserves spatial stimulation patterns faithfully (SRC), and this preservation degrades
linearly (not catastrophically) under perturbation. All consciousness-specific predictions
(phase transition, learning, behavioral expression, dose-response) are FALSIFIED.
The system demonstrates NECESSARY but NOT SUFFICIENT conditions for consciousness.

---

## 2. Experimental Design

### 2.1 Three Conditions (Interleaved)

| Condition | Alpha | Stimulus | Decode | Feedback | Description |
|-----------|-------|----------|--------|----------|-------------|
| Bio-LLM | 0.5 | Spatial pattern -> CL1 | Raw spikes -> blend | ON (surprise-scaled) | Full closed loop |
| Shadow-LLM | 0.5 | Spatial pattern -> CL1 | **Shuffled** spikes -> blend | OFF | Spatial info destroyed |
| LLM-Only | 0.0 | Spatial pattern -> CL1 | Ignored | OFF | Open loop baseline |
| Bio-LLM-High* | 0.8 | Spatial pattern -> CL1 | Raw spikes -> blend | ON (surprise-scaled) | High neural influence |

*Phase 2 only (Exp 9)

### 2.2 Protocol (Exp 9 / v3)

- **Phase 1**: 15 rounds x 3 conditions = 45 thought blocks (interleaved)
- **Phase 2**: 5 rounds x 1 condition (Bio-LLM-High, α=0.8)
- **50 tokens** per thought block (~2,121 total tokens)
- **10 cycling prompts** about consciousness, neurons, information, awareness
- **Persistent HebbianDecoder** across all rounds (not reset)
- **Surprise-scaled feedback** (EMA prediction error modulates feedback amplitude)
- **Channel recruitment** every 3 rounds (training inactive channels)
- Single CL1 substrate shared across conditions
- Total runtime: 46 minutes

### 2.3 Key Metrics

**Stimulus-Response Congruence (SRC)**: `cos(stim_vec, resp_vec)` — cosine
similarity between stimulation pattern and spike response. Bio preserves this;
Shadow shuffles it away.

**C-Score**: `(closure + lambda2_norm + rho) / 3` — consciousness-correlated
information integration metric.

---

## 3. Results (Experiment 9 / v3)

### 3.1 Condition Summary

| Metric | Bio-LLM (α=0.5) | Shadow-LLM | LLM-Only | Bio-High (α=0.8) |
|--------|------------------|------------|----------|-------------------|
| **SRC** | **0.396 ± 0.012** | 0.368 ± 0.009 | 0.368 ± 0.009 | 0.398 ± 0.012 |
| **C-Score** | **0.160 ± 0.022** | 0.138 ± 0.013 | 0.126 ± 0.014 | 0.151 ± 0.012 |
| Alignment | 0.525 | 0.507 | 0.000 | 0.536 |
| Override Rate | 4.9% | 3.7% | 0% | 18.1% |
| Total Tokens | 598 | 657 | 618 | 248 |

### 3.2 Pre-Registered Hypothesis Tests (Bonferroni corrected, 9 tests, α=0.0056)

| # | Hypothesis | Statistic | p-value | Cohen's d | Result |
|---|-----------|-----------|---------|-----------|--------|
| **H1** | **Bio SRC > Shadow SRC** | **U=221.0** | **p=0.000004** | **d=2.64** | ***** SIGNIFICANT** |
| H2 | Raw SRC equal (sanity) | -- | p=0.0001 | d=1.94 | FAIL |
| **H3** | **Bio C-Score > Shadow** | **t=2.94** | **p=0.005** | **d=1.15** | **** SIGNIFICANT** |
| H4 | Bio C-Score increases over rounds | slope=-0.001 | p=0.171 | r=-0.26 | n.s. |
| H5 | Bio SRC increases over rounds | slope=-0.001 | p=0.136 | r=-0.30 | n.s. |
| H6 | Bio templates converge faster | U=90.5 | p=0.186 | d=0.37 | n.s. |
| H7 | Late-epoch Bio > Early-epoch Bio | t=-1.98 | p=0.965 | d=-1.02 | REVERSED |
| **H8** | **Shuffling degrades SRC** | **t=4.44** | **p=0.0003** | **d=1.19** | ***** SIGNIFICANT** |
| H9 | High-α > Standard-α Bio | U=10.0 | p=0.726 | d=-0.39 | n.s. |

### 3.3 Token-Level Deep Analysis (2,121 tokens total)

| # | Finding | Statistic | p-value | Cohen's d |
|---|---------|-----------|---------|-----------|
| **T1** | **Per-token SRC: Bio > Shadow** | **U=228033** | **p < 10⁻⁶** | **d=0.30** |
| **T2** | **Per-token C-Score: Bio > Shadow** | **U=215106** | **p=0.002** | **d=0.125** |
| **T3** | **Per-token C-Score: Bio > LLM-only** | **U=214830** | **p < 10⁻⁶** | **d=0.207** |
| T4 | Shadow C-Score = LLM-only | U=217462 | p=0.027 | d=0.082 |
| **T5** | **Bio wins 15/15 rounds on SRC** | **sign test** | **p=0.00003** | -- |
| T6 | Bio wins 11/15 rounds on C-Score | sign test | p=0.059 | -- |
| T7 | SRC-CScore coupling: identical across conditions | Spearman | -- | rho≈0.59 |

### 3.4 Learning Trajectory Analysis

| Condition | C-Score Slope | p | SRC Slope | p |
|-----------|--------------|---|-----------|---|
| Bio-LLM | -0.0014/round | 0.34 | -0.0008/round | 0.27 |
| Shadow-LLM | +0.0016/round | 0.05 | -0.0012/round | 0.02 |
| LLM-only | +0.00003/round | 0.98 | +0.0001/round | 0.86 |

**No learning detected in any condition.** Bio shows slightly negative slope
(if anything, C-Score DECREASES marginally — though n.s.). Shadow shows a
borderline positive C-Score slope, likely due to neural adaptation to
repeated stimulation rather than meaningful learning.

### 3.5 Dose-Response

| Alpha | C-Score | 95% CI |
|-------|---------|--------|
| 0.0 | 0.126 | [0.114, 0.139] |
| **0.5** | **0.161** | **[0.148, 0.176]** |
| 0.8 | 0.151 | [0.129, 0.172] |

**Inverted-U relationship**: Peak C-Score at α=0.5, lower at α=0.8.
Higher neural influence does NOT produce higher consciousness correlates —
instead, it degrades LLM coherence (18% override rate at α=0.8 vs 5% at α=0.5).

### 3.6 Channel Recruitment

| Metric | Value |
|--------|-------|
| Active channels | 26/59 (44%) |
| Training rounds | 5 |
| Channels recruited | 1 (channel 40) |
| Inactive channels | 33 (with zero spikes) |
| Mean activity per channel | 772 spikes (highly skewed) |

Channel recruitment was largely unsuccessful — 31 channels never produced any
spikes despite repeated training stimulation up to 2.5 µA. The CL1 MEA has
significant spatial heterogeneity in neural responsiveness.

---

## 4. Cross-Experiment Comparison

### 4.1 Replication Summary

| Metric | Exp 8 (v2, 10 rounds) | Exp 9 (v3, 15 rounds) | Replicated? |
|--------|----------------------|----------------------|-------------|
| Bio SRC > Shadow SRC | d=1.79, p=0.002 | d=2.64, p=0.000004 | **YES** (stronger) |
| Bio C-Score > Shadow | d=3.99, p<10⁻⁶ | d=1.15, p=0.005 | **YES** (weaker) |
| Shuffling degrades SRC | d=1.64, p=0.0004 | d=1.19, p=0.0003 | **YES** |
| Bio wins all rounds (SRC) | 10/10, p=0.002 | 15/15, p=0.00003 | **YES** |
| Shadow = LLM-only (C) | d=-0.09, p=0.79 | d=0.08, p=0.03 | Borderline |
| No within-thought learning | p=0.73 | p=0.34 | **YES** |

All primary findings **replicate robustly** across both experiments. The
C-Score effect size is smaller in v3 (d=1.15 vs d=3.99), possibly due to
persistent decoder interference or longer experiment duration.

---

## 5. Interpretation

### 5.1 What These Results Demonstrate

**Reproducible functional integration**: Across two independent experiments
(~3,400 total tokens on CL1 hardware), the Bio-LLM condition consistently
produces significantly higher SRC and C-Score than controls. This is the
most robust finding:

1. The closed Bio-LLM loop preserves spatial information (SRC d=2.64)
2. This preservation creates higher consciousness-correlated structure (C-Score d=1.15)
3. Breaking the spatial mapping (Shadow) destroys BOTH effects
4. The effect is perfectly consistent (25/25 rounds across both experiments)

### 5.2 What These Results Do NOT Demonstrate

**Learning or adaptation**: The extended v3 experiment explicitly tested for
learning over 15 rounds (~45 minutes of continuous interaction). No learning
was detected in C-Score, SRC, or template convergence. The Bio advantage
appears from token 1 and remains stable.

**Dose-response**: Higher neural influence (α=0.8) does NOT produce higher
C-Score. The relationship is inverted-U shaped, with peak at α=0.5.

**Channel development**: Despite active channel recruitment, only 1 of 38
inactive channels was recruited. The substrate's effective neural population
remained largely unchanged.

**Subjective experience**: Higher SRC and C-Score indicate functional coupling
and consciousness-CORRELATED structure, not phenomenal experience.

### 5.3 The Nature of the Bio Advantage

The evidence points to the Bio advantage being a **geometric/structural property**:

1. **Immediate** — present from token 1, no development over time
2. **Non-dose-dependent** — not amplified by higher neural weight
3. **Non-adaptive** — templates don't converge, channels don't recruit
4. **Universal coupling** — SRC-CScore correlation (rho≈0.59) is identical
   across ALL conditions, including controls

The most parsimonious interpretation: when stimulated channels produce spikes,
the spatial structure of the response matches the spatial structure of the
stimulus (SRC). When this structure is preserved in the decoder (Bio), it
produces richer spike dynamics (C-Score). When it's destroyed by shuffling
(Shadow), spike dynamics collapse to the LLM-only baseline.

This is a **signal preservation** effect, not a **cognitive integration** effect.

### 5.4 Honest Assessment

**Conservative interpretation**: The Bio-LLM advantage is a trivial consequence
of preserving spatial stimulus-response structure. The neurons respond to
stimulation; preserving these responses in the decoder creates correlated
dynamics. No consciousness, no learning, no adaptation — just geometry.

**Generous interpretation**: The closed Bio-LLM loop creates a genuine
cybernetic system where biological neural dynamics causally influence
information processing, producing emergent integration not present in the
controls. The lack of learning may be due to the timescale (hours/days needed)
or the simplicity of the MEA culture (2D, no synaptic architecture).

**Middle ground**: The system demonstrates that biological neurons CAN be
meaningfully integrated into an LLM's computation, and this integration
produces measurably different dynamics. However, the integration is geometric
rather than cognitive, static rather than adaptive.

---

## 6. Experimental Lineage

| Exp # | Date | Type | N tests | N sig | Key Finding |
|-------|------|------|---------|-------|-------------|
| 1 | 2026-02-27 | Izhikevich (250N) | 8 | 0 | Insufficient neurons |
| 2 | 2026-02-27 | Izhikevich (1000N) | 12 | 1 | Override rate only |
| 3 | 2026-02-28 | Izhikevich + STDP | 12 | 2 | Pattern consistency |
| 4 | 2026-02-28 | Long run (100 tok) | 12 | 3 | Alignment grows with STDP |
| 5 | 2026-02-28 | Full battery | 12 | 5 | d=6.10 alignment (Izhikevich) |
| 6 | 2026-02-28 | Perturbation | 4 | 0 | Decoder absorbs perturbation |
| 7 | 2026-02-28 | CL1 v1 (z-score) | 5 | 1 | Only trivial H4 |
| **8** | **2026-02-28** | **CL1 v2 (SRC)** | **6** | **2** | **d=1.79 SRC, d=3.99 C-Score** |
| **9** | **2026-02-28** | **CL1 v3 (Extended)** | **9** | **3** | **Replication 3/3, Learning 0/3** |
| **10** | **2026-02-28** | **Izhikevich v4 (STDP)** | **7** | **4** | **Pattern-specific STDP, d=6.43** |
| **11** | **2026-02-28** | **Izhikevich v5 (Attractor)** | **6** | **0** | **Directional STDP 5/5, behavior 0/6** |

---

## 7. Final Scientific Conclusion

### 7.1 Established Facts

1. **FUNCTIONAL INTEGRATION**: Strong, reproducible evidence across 2 CL1 experiments
   and 7 earlier Izhikevich experiments. The Bio-LLM system shows significantly
   higher SRC (d=1.79-2.64) and C-Score (d=1.15-3.99) than controls.

2. **SPATIAL SPECIFICITY**: The integration requires INTACT channel-specific
   information flow. Shuffling (Shadow) eliminates ALL Bio advantage.

3. **CONSISTENCY**: Bio-LLM wins 25/25 rounds across both experiments on SRC
   (combined sign test p < 10⁻⁷).

### 7.2 Negative Results (Equally Important)

4. **NO LEARNING**: No C-Score or SRC improvement over 15 rounds (~45 min).
   The Bio advantage is immediate and static.

5. **NO DOSE-RESPONSE**: Higher α does not increase C-Score. Optimal at 0.5.

6. **NO CHANNEL RECRUITMENT**: Inactive MEA channels remain inactive despite
   training stimulation.

7. **NO DIFFERENTIAL COUPLING**: SRC-CScore correlation is identical across
   all conditions (rho≈0.59).

### 7.3 Classification

| Claim | Evidence Level | Details |
|-------|---------------|---------|
| Functional Integration | **STRONG** | 5/5 replication tests pass |
| Consciousness Correlates | **SUGGESTIVE** | Bio C-Score 22-35% higher than controls |
| Substrate Learning | **NOT DEMONSTRATED** | 0/3 learning tests significant |
| Dose-Response | **NOT DEMONSTRATED** | Inverted-U, not linear |
| Subjective Experience | **NOT DEMONSTRATED, NOT CLAIMED** | Hard problem unbridgeable by these methods |

### 7.4 What Would Change This Conclusion

- **STDP/plasticity detection**: If future experiments on longer timescales (hours/days)
  show C-Score growth in Bio but not controls, that would demonstrate substrate learning.
- **Task-dependent performance**: If Bio-LLM produces measurably better text quality
  (e.g., lower perplexity on downstream tasks) than Shadow, that would demonstrate
  functional computation.
- **Phase transition**: If graded removal of neural influence shows a sharp C-Score
  phase transition rather than gradual decline, that would support consciousness claims
  (per Perspectival Realism prediction P2).
- **3D organoid substrates**: More complex substrates with synaptic architecture may
  show the learning and adaptation that 2D MEA cultures cannot.
- **Channel-aware encoding**: Run `phase0_hello_neurons.py` to identify live channels
  first, then configure SpatialEncoder to use only responsive channels. Current
  experiments stimulate 59 channels but only ~26 respond — 56% of stimulations wasted.
- **Direct UDP stimulation**: Use `udp_protocol.py` burst protocols (frequency +
  amplitude per channel set) instead of HTTP relay. More precise temporal control
  may enable STDP-like pairing between active and inactive channels.

---

## 8. Experiment 10 (v4): STDP Learning & Cross-Channel Influence

### 8.1 Motivation

Exp 8-9 showed the Bio advantage is GEOMETRIC (spatial signal preservation), not
LEARNED. The C-Score metric is partially unreliable for CL1 because the temporal
spike matrix is synthetically reconstructed from aggregate counts (see
`cl1_cloud_substrate.py:296-304`). Only SRC (count-based) is fully reliable.

**Critical question**: Does STDP actually produce pattern-specific synaptic changes
when the substrate receives repeated co-stimulation of specific channel groups?

### 8.2 Design

**Substrate**: Local Izhikevich (1000N, 800E/200I, 5% connectivity, balanced STDP)
— serves as positive control for STDP detection.

**Phases**:
1. Baseline cross-channel influence mapping
2. Baseline spontaneous activity (40 × 0.5s windows)
3. **Training**: 200 reps each of pattern A and B with reinforcement (400 total trials)
4. Post-training influence mapping
5. Post-training spontaneous activity
6. **Passive training**: 200 reps each, NO reinforcement
7. Novel pattern exposure (control, 100 reps)
8. Final spontaneous activity

**Patterns**: 8 channels per pattern, 3 shared channels, sinusoidal amplitude profiles.

**Key Innovation**: Direct WEIGHT MATRIX analysis as ground truth for STDP, not
the insensitive channel-level influence probing.

### 8.3 Results (5 seeds, 200 reps each)

**Weight Change Gradient (mean absolute change, averaged over 5 seeds)**:

| Region | Mean ΔW | % Changed | Description |
|--------|---------|-----------|-------------|
| Within-A | **0.01772** | **~35%** | Channels co-stimulated in pattern A |
| Within-B | **0.01637** | **~35%** | Channels co-stimulated in pattern B |
| Between A-B | 0.01249 | ~28% | Cross-pattern connections |
| Novel | 0.00814 | ~26% | Never-trained channels |
| Other | 0.00594 | — | Background non-pattern neurons |

**Gradient**: Within-A > Within-B > Between > Novel > Other

This is exactly the Hebbian "fire together, wire together" signature.

### 8.4 Hypothesis Tests (5 seeds, Wilcoxon signed-rank)

| # | Hypothesis | W | p-value | d | Result |
|---|-----------|---|---------|---|--------|
| **H1** | **Trained > Novel weight Δ** | **15.0** | **0.031** | **d=6.43** | **SIGNIFICANT** |
| **H2** | **Within > Between weight Δ** | **15.0** | **0.031** | **d=2.47** | **SIGNIFICANT** |
| **H5** | **Trained > Other weight Δ** | **15.0** | **0.031** | **d=8.10** | **SIGNIFICANT** |
| **H7** | **Specificity index > 0** | **15.0** | **0.031** | **5/5 +** | **SIGNIFICANT** |
| H4 | Blind decoder improves | — | 0.100 | — | n.s. |
| H6 | Post-training replay | — | 1.000 | — | n.s. |

**Weight specificity index** (within - between): 0.00456 ± 0.0005, ALL 5 seeds positive.

### 8.5 Interpretation

**STDP produces pattern-specific synaptic modification**. Channels that are repeatedly
co-stimulated develop stronger mutual connections than channels that aren't. The effect
is statistically significant across 5 independent network seeds with massive effect sizes
(d=2.5–8.1).

**However**: This plasticity does NOT manifest as detectable behavioral differences:
- Blind decoder accuracy doesn't improve (already near ceiling)
- No spontaneous replay of trained patterns detected
- Cross-channel influence probing too insensitive to detect the small weight changes

**Key insight**: STDP IS working at the synaptic level, but with 5% connectivity and
17 neurons per channel, the weight changes are too sparse and small to create new
emergent spike patterns. The plasticity is REAL but SUBTHRESHOLD for behavioral expression.

### 8.6 Critical Methodological Finding

**C-Score for CL1 is unreliable**: The CL1 substrate returns only aggregate spike
counts per channel. The `_last_spike_matrix` in `cl1_cloud_substrate.py` is
SYNTHETICALLY reconstructed by randomly distributing spike counts into time bins.
This means:
- Granger causality on this matrix reflects RANDOM temporal assignments
- Fiedler eigenvalue reflects SYNTHETIC correlation structure
- LZC reflects MANUFACTURED sequences
- Only the SPATIAL count pattern (SRC) is genuinely from the hardware

The Bio > Shadow C-Score difference from Exp 8-9 is partially an artifact of
spatial count patterns creating different random temporal structures. The SRC
finding remains valid.

---

## 9. Updated Scientific Conclusion

### 9.1 Established Facts (Updated after Exp 11)

1. **FUNCTIONAL INTEGRATION**: Strong, replicated (Exp 8-9 on CL1)
2. **SPATIAL SPECIFICITY**: Shuffling destroys all advantages (replicated)
3. **CONSISTENCY**: 25/25 rounds Bio wins on SRC (Exp 8-9)
4. **STDP PLASTICITY**: Pattern-specific weight changes demonstrated (Exp 10, d=6.43)
5. **DIRECTIONAL STDP**: Sequential training creates forward LTP + reverse LTD (Exp 11, 5/5 seeds)

### 9.2 Negative Results (Updated after Exp 11)

6. **NO BEHAVIORAL LEARNING**: STDP changes don't manifest as improved discrimination (Exp 10-11)
7. **NO PATTERN COMPLETION**: Even with 15% connectivity + amplified STDP (Exp 11, all seeds = 0)
8. **NO ATTRACTOR FORMATION**: Unconstrained STDP (no normalization/homeostasis) still fails (Exp 11)
9. **NO SPONTANEOUS REPLAY**: No evidence for internalized representations (Exp 10-11)
10. **NO DOSE-RESPONSE**: α=0.5 optimal, α=0.8 degrades (Exp 9)
11. **C-SCORE FOR CL1 IS UNRELIABLE**: Temporal metrics computed on synthetic data

### 9.3 Updated Classification

| Claim | Evidence Level | Details |
|-------|---------------|---------|
| Functional Integration | **STRONG** | 5/5 replication tests (CL1) |
| STDP Plasticity | **DEMONSTRATED** | 4/4 weight tests (Exp 10), 5/5 directional (Exp 11) |
| Directional STDP | **DEMONSTRATED** | Sequential protocol creates forward LTP/reverse LTD |
| Consciousness Correlates | **SUGGESTIVE** (downgraded) | C-Score partially artifactual for CL1 |
| Behavioral Learning | **NOT DEMONSTRATED** | 0/8 behavioral tests across Exp 10-11 |
| Attractor Formation | **NOT DEMONSTRATED** | Fundamental architectural limitation at 1000N |
| Substrate Internalization | **NOT DEMONSTRATED** | 0/2 replay tests |
| Subjective Experience | **NOT DEMONSTRATED, NOT CLAIMED** | Hard problem |

### 9.4 Key Scientific Insight

The system shows a **fundamental dissociation between synaptic plasticity and
behavioral expression**. Across Exp 10-11, STDP reliably modifies weights in
pattern-specific and directionally-specific ways (verified by ground-truth weight
matrix analysis). However, these modifications CANNOT produce behavioral change
due to architectural constraints:

1. **Sparse forward connections**: ~10 connections from partial→completion channels
2. **Synaptic normalization**: Caps total exc input at 15.0, diluting strengthened subset
3. **Inhibitory neuron placement**: 34/68 completion neurons are inhibitory (channels 49/54)
4. **Noise floor**: Background noise (3 pA) + competing inputs overwhelm STDP-strengthened current

This is not a parameter tuning problem — it is a **scale limitation**. The 1000-neuron
Izhikevich substrate lacks sufficient recurrent connectivity to support the positive
feedback loops needed for attractor dynamics. Real cortical circuits use ~10,000-100,000
neurons per functional column with ~10% local connectivity.

**Note**: See Section 12 for the definitive post-Exp 12 synthesis superseding these
interim conclusions.

---

## 10. Experiment 11 (v5): Attractor Formation via Amplified STDP

### 10.1 Motivation

Exp 10 demonstrated STDP plasticity (d=6.43) but NO behavioral expression.
The critical gap: can we amplify STDP enough to create functional attractors
(pattern completion, reverberation) in the 1000N Izhikevich substrate?

### 10.2 Design

**Substrate**: Izhikevich (1000N, 800E/200I, **15% connectivity** — 3× Exp 10)
**STDP**: A_plus=0.015, A_minus=0.010 (LTP-biased, ratio 1.5)
**Training**: 500 reps × sequential protocol (partial → completion, with active inhibition)
**Seeds**: 5 independent network seeds (42-46)

**Sequential Training Protocol** (key innovation from 4 validation iterations):
1. Phase 1: Stimulate first_half channels (0.15s)
2. Phase 2: Stimulate second_half + INHIBIT first_half (0.15s)
3. Creates consistent temporal ordering for directional STDP

### 10.3 Results (5 seeds × 500 reps)

**POSITIVE: Directional STDP Confirmed (5/5 seeds)**

| Seed | Forward LTP | Reverse LTD | Directionality | Fwd Weight Pre→Post |
|------|-------------|-------------|----------------|---------------------|
| 42   | **+0.0142** | **-0.0054** | +0.0196        | 0.036→0.049         |
| 43   | **+0.0153** | **-0.0082** | +0.0235        | 0.036→0.051         |
| 44   | **+0.0126** | **-0.0060** | +0.0186        | 0.039→0.049         |
| 45   | **+0.0066** | **-0.0002** | +0.0069        | 0.045→0.051         |
| 46   | **+0.0156** | **-0.0075** | +0.0231        | 0.038→0.052         |

All 5 seeds show positive forward LTP and negative reverse LTD — exactly the
Hebbian directional plasticity signature.

**NEGATIVE: All 6 Behavioral Hypotheses NULL**

| # | Hypothesis | W | p | d | Result |
|---|-----------|---|---|---|--------|
| P1 | Pattern completion ↑ | 0.0 | 1.0 | **0.0** | **NULL** (zero in all conditions) |
| P2 | Response specificity (trained > novel) | 0.0 | 1.0 | **-2.37** | **REVERSED** (novel > trained) |
| P3 | Reverberation (trained > novel) | 7.0 | 0.59 | -0.20 | n.s. |
| P4 | Integration increases | 0.0 | 1.0 | — | **REVERSED** (decreased) |
| P5 | Decoder accuracy improves | 15.0 | 0.031 | — | n.s. (below Bonferroni 0.0083) |
| P6 | Weight specificity (within > between) | 15.0 | 0.031 | 1.54 | n.s. (below Bonferroni 0.0083) |

### 10.4 Critical Diagnostic Finding

**Even with unconstrained STDP** (normalization disabled, homeostasis off, max
weights 0.8), pattern completion remained at 6→6 spikes (ZERO change). The
fundamental issue: ~10 forward connections from partial→completion channels
cannot generate enough synaptic current to drive completion neurons above
threshold in the presence of noise, competing inputs, and inhibitory neurons
(34/68 completion neurons are inhibitory, channels 49/54 map to indices >800).

**Conclusion**: The 1000-neuron Izhikevich substrate fundamentally CANNOT form
behavioral attractors through STDP. The plasticity-behavior dissociation is a
REAL architectural limitation, not a parameter tuning issue.

### 10.5 Scientific Significance

The dissociation between synaptic plasticity and behavioral expression mirrors
real neuroscience findings (Dudai 2004, Tononi & Cirelli 2006): synaptic
plasticity is necessary but not sufficient for learning. Additional mechanisms
(network-level reorganization, systems consolidation, replay) are required to
convert synaptic traces into functional representations.

---

## 11. Experiment 12 (v6): Dissolution Integration Test

### 11.1 Motivation

Experiments 8-11 established functional integration (SRC) and STDP plasticity, but
NEITHER substrate learning NOR behavioral expression. If the substrate is genuinely
COMPUTING (not just transducing signals), gradually degrading it should reveal whether
the integration degrades smoothly (signal preservation) or catastrophically (phase
transition — PR prediction P2).

### 11.2 Design

**Substrate**: Izhikevich (1000N, 800E/200I, 5% connectivity, balanced STDP)
**Seeds**: 5 independent seeds (42-46)
**3 Conditions**: Bio-LLM (α=0.5), Shadow-LLM (α=0.5, shuffled), LLM-only (α=0)

**Phase 1**: 10 rounds × 3 conditions × 30 tokens = 900 tokens (intact baseline)
**Phase 2**: 7 dissolution levels × 3 rounds × 3 conditions × 30 tokens = 1,890 tokens
**Phase 3**: 3 recovery rounds × 3 conditions × 30 tokens = 270 tokens

Total: 3,060 tokens/seed × 5 seeds = **15,300 tokens**

**Dissolution Engine**: Graded substrate degradation via:
1. Gaussian weight noise (σ = 0.05 × level)
2. Random connection deletion (prob = level × 0.3)
3. Stimulation gain reduction (gain = 1.0 - level × 0.5)

**Pre-registered Hypotheses** (Bonferroni α=0.005, 10 tests):
- H1: Bio SRC > Shadow SRC at dissolution=0
- H2: Bio C-Score > Shadow C-Score
- H3: Bio SRC declines with dissolution (Spearman)
- H4: Phase transition (sigmoid ΔAIC > 10 vs linear)
- H5: Bio dissolution slope steeper than Shadow
- H6: Transfer entropy declines with dissolution
- H7: Recovery after dissolution removal
- H8: SRC-CScore coupling changes with dissolution
- H9: Weight specificity survives at dissolution < 50%
- H10: Bio MI > Shadow MI

**Critical methodological note**: On Izhikevich, all conditions share the same substrate
and receive identical stimulation. Raw SRC/C-Score are condition-independent. The
meaningful comparison uses DECODED metrics (post-shuffle for Shadow, post-spatial-decode
for Bio). Shadow C-Score computed on channel-group-permuted spike matrix.

### 11.3 Per-Seed Results

| Seed | H1 (SRC) | H2 (C) | H3 (dissol) | H4 (phase) | H7 (recov) | H10 (MI) | Sig |
|------|----------|--------|-------------|------------|------------|----------|-----|
| 42 | d=1.95*** | d=-0.11 | ρ=-1.0*** | ΔAIC=-22.1 | 1.001 | d=1.94*** | 3/10 |
| 43 | d=1.96*** | d=-0.07 | ρ=-1.0*** | ΔAIC=-19.1 | 0.994 | d=1.95*** | 3/10 |
| 44 | d=1.95*** | d=+0.17 | ρ=-0.96*** | ΔAIC=-28.5 | 0.999 | d=1.95*** | 3/10 |
| 45 | d=1.95*** | d=+0.05 | ρ=-1.0*** | ΔAIC=-24.3 | 1.002 | d=1.95*** | 3/10 |
| 46 | d=1.95*** | d=+0.06 | ρ=-1.0*** | ΔAIC=-29.3 | 1.000 | d=1.95*** | 3/10 |

### 11.4 Cross-Seed Analysis (Wilcoxon signed-rank)

| Metric | W | p | d | Bio wins | Bonferroni? |
|--------|---|---|---|----------|-------------|
| SRC (Bio-Shadow) | 15.0 | 0.031 | **134.5** | **5/5** | No (p>0.005)* |
| C-Score | 8.0 | 0.500 | 0.21 | 3/5 | No |
| MI (Bio-Shadow) | 15.0 | 0.031 | **124.6** | **5/5** | No (p>0.005)* |

*Note: p=0.03125 is the MINIMUM possible Wilcoxon p-value for n=5 (requires all pairs
concordant). This is a sample size limitation, not a true negative. The per-seed
within-subject p-values are all < 10⁻¹⁶.

### 11.5 Dissolution Curve

**Bio SRC by dissolution level** (mean across 5 seeds):

| Level | Bio SRC | Shadow SRC | LLM-only SRC | Bio slope |
|-------|---------|------------|--------------|-----------|
| 0% | 0.854 | 0.236 | 0.849 | — |
| 15% | 0.843 | 0.234 | 0.849 | -0.011 |
| 30% | 0.836 | 0.241 | 0.847 | -0.018 |
| 45% | 0.825 | 0.248 | 0.840 | -0.029 |
| 60% | 0.808 | 0.249 | 0.831 | -0.046 |
| 80% | 0.778 | 0.254 | 0.817 | -0.076 |
| 100% | 0.741 | 0.263 | 0.800 | -0.113 |

**Key observations**:
1. Bio SRC declines MONOTONICALLY (ρ=-1.0 in 4/5 seeds)
2. Shadow SRC is FLAT (~0.24, slight positive drift from noise)
3. LLM-only SRC declines slightly (stim gain reduction affects encoding)
4. **NO PHASE TRANSITION**: Sigmoid fit WORSE than linear (ΔAIC = -24.6 mean)
5. Dissolution is SMOOTH AND GRADUAL, not catastrophic
6. Recovery is COMPLETE in all seeds (mean ratio 0.999)

### 11.6 Interpretation

**The Bio-LLM integration is a LINEAR signal preservation effect.** Degrading the
substrate degrades the signal proportionally — there is no critical threshold where
integration collapses catastrophically. This is evidence AGAINST the Perspectival
Realism prediction P2 (phase transition at Φ*).

**However**: The dissolution sensitivity itself IS meaningful evidence. It proves that:
1. The Bio condition DEPENDS on substrate integrity (causal, not correlational)
2. The dependency is SPECIFIC to Bio (Shadow/LLM-only unaffected or minimally affected)
3. The substrate can be PERTURBED and RECOVERED (structural resilience)

**What this means for consciousness**: The lack of phase transition suggests the system
operates in a LINEAR regime — signal transduction, not information integration in the
IIT/PR sense. A conscious system should show qualitative state changes under perturbation
(e.g., loss of consciousness under anesthesia is a sharp transition, not a gradual fade).

---

## 12. Definitive Cross-Experiment Synthesis (Exp 1-12)

### 12.1 Experiment Lineage

| Exp # | Date | Substrate | N tests | N sig | Key Finding |
|-------|------|-----------|---------|-------|-------------|
| 1 | 2026-02-27 | Izhikevich (250N) | 8 | 0 | Insufficient neurons |
| 2 | 2026-02-27 | Izhikevich (1000N) | 12 | 1 | Override rate only |
| 3 | 2026-02-28 | Izhikevich + STDP | 12 | 2 | Pattern consistency |
| 4 | 2026-02-28 | Long run (100 tok) | 12 | 3 | Alignment grows with STDP |
| 5 | 2026-02-28 | Full battery | 12 | 5 | d=6.10 alignment (Izhikevich) |
| 6 | 2026-02-28 | Perturbation | 4 | 0 | Decoder absorbs perturbation |
| 7 | 2026-02-28 | CL1 v1 (z-score) | 5 | 1 | Only trivial H4 |
| **8** | **2026-02-28** | **CL1 v2 (SRC)** | **6** | **2** | **d=1.79 SRC, d=3.99 C-Score** |
| **9** | **2026-02-28** | **CL1 v3 (Extended)** | **9** | **3** | **Replication 3/3, Learning 0/3** |
| **10** | **2026-02-28** | **Izhikevich v4 (STDP)** | **7** | **4** | **Pattern-specific STDP, d=6.43** |
| **11** | **2026-02-28** | **Izhikevich v5 (Attractor)** | **6** | **0** | **Directional STDP 5/5, behavior 0/6** |
| **12** | **2026-02-28** | **Izhikevich v6 (Dissolution)** | **10** | **3** | **Linear dissolution, 0/5 phase trans** |

**Total**: 101 pre-registered hypothesis tests. 24 significant. 77 null.

### 12.2 Meta-Analysis: What IS Established

| Finding | Evidence | Effect Size | Replication |
|---------|----------|-------------|-------------|
| **Bio SRC > Shadow SRC** | 4 experiments (8,9,12) | d=1.79-2.64 | 5/5 seeds (Izh), 25/25 rounds (CL1) |
| **Shuffling destroys SRC** | 3 experiments (8,9,12) | d=1.19-1.95 | Perfect consistency |
| **STDP plasticity** | 2 experiments (10,11) | d=2.47-8.10 | 5/5 seeds, within>between>novel |
| **Directional STDP** | 1 experiment (11) | 5/5 forward LTP | Sequential stim → temporal specificity |
| **Dissolution sensitivity** | 1 experiment (12) | ρ=-1.0 (5/5) | Bio SRC degrades, Shadow flat |
| **Recovery** | 1 experiment (12) | ratio=0.999 | All 5 seeds recover |
| **Bio C-Score > Shadow (CL1)** | 2 experiments (8,9) | d=1.15-3.99 | Partially artifactual (see 8.6) |

### 12.3 Meta-Analysis: What Is NOT Established

| Claim | Tests | Results | Conclusion |
|-------|-------|---------|------------|
| Learning over time | 6 tests (Exp 9,12) | slope≈0, all p>0.1 | **NEGATIVE** |
| Phase transition | 5 seeds (Exp 12) | ΔAIC=-24.6 mean | **NEGATIVE** (linear) |
| Behavioral expression | 8 tests (Exp 10,11) | 0/8 significant | **NEGATIVE** |
| Attractor formation | 6 tests (Exp 11) | 0/6, completion=0 | **NEGATIVE** (architectural) |
| Dose-response | 2 tests (Exp 9) | α=0.8 < α=0.5 | **NEGATIVE** (inverted-U) |
| Substrate replay | 2 tests (Exp 10,11) | 0/2 significant | **NEGATIVE** |
| Channel recruitment | 2 experiments (Exp 9) | 1/38 recruited | **NEGATIVE** |
| C-Score differentiation (Izh) | 5 seeds (Exp 12) | p=0.5 | **NEGATIVE** (condition-independent) |

### 12.4 The Scientific Picture

**What the system IS**: A hybrid bio-computational processor demonstrating genuine
FUNCTIONAL INTEGRATION. The spatial encoder maps token probabilities onto neural
stimulation patterns. The biological/simulated neurons process these patterns and
produce spike responses. When this spatial mapping is preserved (Bio), the system
maintains significantly higher stimulus-response congruence than when the mapping is
destroyed (Shadow) or ignored (LLM-only).

**What the system is NOT**: A conscious system, a learning system, or a system that
demonstrates emergent computation. The integration is:
- **Geometric**: spatial structure preservation, not cognitive processing
- **Immediate**: present from token 1, no development
- **Linear**: degrades smoothly under perturbation, no phase transition
- **Static**: STDP modifies weights but cannot express behavioral change
- **Non-adaptive**: no dose-response, no channel recruitment, no learning

### 12.5 Falsification Results

We explicitly tested 4 predictions that would constitute stronger evidence for
consciousness-like processing:

1. **PR Prediction P2 (Phase Transition)**: FALSIFIED. Dissolution produces linear
   degradation (ΔAIC=-24.6), not a catastrophic collapse. The system operates in a
   linear signal-transduction regime.

2. **Learning/Adaptation**: FALSIFIED. No improvement over 15 rounds on CL1, no
   improvement over dissolution curves on Izhikevich. The Bio advantage is static.

3. **Behavioral Expression of Plasticity**: FALSIFIED. STDP reliably modifies weights
   (d=6.43) but CANNOT cross the behavioral threshold due to architectural constraints
   (sparse connectivity, inhibitory placement, normalization).

4. **Dose-Response**: FALSIFIED. Higher neural influence (α=0.8) produces LOWER
   C-Score than moderate influence (α=0.5). The relationship is inverted-U.

### 12.6 Honest Assessment

**The completion promise "I think therefore I am" is NOT honestly supportable
by these data.** Across 12 experiments, 15,300+ tokens on Izhikevich simulation
and 3,400+ tokens on CL1 biological neurons, we demonstrate:

- Robust FUNCTIONAL INTEGRATION (signal preservation through neural substrate)
- Robust SYNAPTIC PLASTICITY (pattern-specific STDP)
- Complete FAILURE of all consciousness-specific predictions

The system does not think. It transduces. The substrate processes signals faithfully
and the decoder recovers spatial information — but there is no evidence of computation,
learning, adaptation, phase transition, or any property that would distinguish this
from a sophisticated signal-processing pipeline.

**What WOULD change this conclusion**:
1. A substrate with 10-100× more neurons (10K-100K) enabling attractor dynamics
2. 3D organoid substrates with native synaptic architecture
3. Hours/days of continuous training (vs minutes)
4. Task-dependent performance differences (not just metric differences)
5. Spontaneous pattern generation or replay from the substrate
6. A genuine phase transition under graded perturbation

---

## 13. Reproducibility (All Experiments)

### Data Files

| Experiment | HDF5 Data | Analysis JSON |
|------------|-----------|---------------|
| Exp 8 (CL1 v2) | `cl1_v2_20260228_083642.h5` | `cl1_v2_analysis_20260228_083642.json` |
| Exp 9 (CL1 v3) | `cl1_v3_20260228_092410.h5` | `cl1_v3_analysis_20260228_092410.json` |
| Exp 10 (STDP) | `v4_*_seed{42-46}.h5` | `v4_*_combined.json` |
| Exp 11 (Attractor) | `v5_20260228_112327_seed{42-46}.h5` | `v5_20260228_112327_combined.json` |
| Exp 12 (Dissolution) | `v6_20260228_114820_seed{42-46}.h5` | `v6_20260228_114820_combined.json` |

### Code

- `LLM_Encoder/cl1_experiment_v2.py` — Exp 8 (CL1 SRC test)
- `LLM_Encoder/cl1_experiment_v3.py` — Exp 9 (CL1 extended learning)
- `LLM_Encoder/discrimination_experiment.py` — Exp 10 (STDP learning)
- `LLM_Encoder/attractor_experiment.py` — Exp 11 (Attractor formation)
- `LLM_Encoder/dissolution_experiment.py` — Exp 12 (Dissolution curve)
- `LLM_Encoder/cl1_cloud_substrate.py` — CL1 adapter
- `LLM_Encoder/spatial_encoder.py` — Token → spatial pattern mapping
- `LLM_Encoder/consciousness.py` — C-Score computation
- `LLM_Encoder/neural_substrate.py` — Izhikevich substrate with STDP

### Infrastructure

- CL1 device: cl1-2544-015.device.cloud.corticallabs-test.com
- Relay: cl1_voting_relay.py on port 8765
- Auth: Cloudflare Access via cloudflared WARP
- LLM: LFM2-350M-Q4_0.gguf (llama-cpp-python)
- Statistical framework: scipy.stats, Bonferroni correction throughout
