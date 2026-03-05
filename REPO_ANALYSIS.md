# CL1_LLM_Encoder Repository Analysis

**Date:** 2026-03-05  
**Analyst:** Automated Code Review

---

## 1. Project Overview

This is a **neuroscience-AI integration research project** investigating the coupling between biological/simulated neural substrates and Large Language Models.

### Core Research Question
*Can neural activity meaningfully influence LLM token generation, and does this integration produce consciousness-like properties?*

### Secondary Question
*Does tight substrate-LLM integration constitute evidence of subjective experience?*

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM (LFM2-350M)                          │
│         Token probabilities → Spatial patterns              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Spatial Encoder (v2)                           │
│    Token ID → 8-channel stimulation pattern (µA)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Neural Substrate (2 options)                      │
│  ┌──────────────────┐  ┌─────────────────────────────────┐ │
│  │ Izhikevich (sim) │  │ CL1 (biological neurons)        │ │
│  │ 1000 neurons     │  │ 800,000 neurons, 59 MEA channels│ │
│  │ 59 channels      │  │ 240 Hz tick rate                │ │
│  │ STDP plasticity  │  │ HTTP relay interface            │ │
│  └──────────────────┘  └─────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Spike counts per channel
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Neural Vote Decoder                            │
│    Z-score baseline → neural probabilities → blend          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Blended probabilities → Token selection             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Experimental Conditions

| Condition | Alpha | Neural Influence | Spatial Mapping | Purpose |
|-----------|-------|------------------|-----------------|---------|
| **LLM-only** | 0.0 | None | Intact | Open-loop baseline |
| **Bio-LLM** | 0.5 | Full closed-loop | Intact | Test integration |
| **Shadow-LLM** | 0.5 | Full closed-loop | **Shuffled** | Control (destroys spatial info) |
| **Bio-LLM-High** | 0.8 | Full closed-loop | Intact | Dose-response test |

---

## 4. Core Components

### 4.1 Token Voting Engine (`encoder.py`, `encoder_v2.py`, `encoder_v3.py`)

```python
class TokenVotingEngine:
    """Integrates LLM inference with neural voting."""
    
    # Closed loop:
    # LLM logprobs → stimulation amplitudes → neural spikes →
    # blended probabilities → token selection → next context
```

**Key Features:**
- Persistent Izhikevich substrate across tokens
- Z-score based neural decoding with rolling baseline
- Surprise-scaled feedback modulation
- Consciousness measurement per token (C-Score)

---

### 4.2 Spatial Encoder (`spatial_encoder.py`)

```python
class SpatialEncoder:
    """Maps tokens to unique spatial stimulation patterns."""
    
    # Each token ID → deterministic 8-channel pattern
    # Similar tokens share partial patterns (tonotopic-like)
```

**Design Principles:**
- Consistent token→channel mapping (not rank-based)
- Hash-derived from token ID (reproducible)
- 8 active channels per token from 30 sensory channels
- Amplitude range: 0.3–2.5 µA (CL1 safe limits)
- Similar tokens have overlapping patterns (Jaccard similarity)

---

### 4.3 Neural Substrates

#### Izhikevich Simulator (`neural_substrate.py`)

```python
class IzhikevichSubstrate:
    """Local 1000-neuron Izhikevich network."""
    
    n_neurons = 1000      # 800 excitatory / 200 inhibitory
    n_channels = 59       # MEA layout
    neurons_per_channel = 17
    connection_prob = 0.02  # sparse connectivity
```

**Stability Features (v3 fixes):**
- Balanced STDP (A_plus=0.005, A_minus=0.006)
- Synaptic normalization (max_total_exc_input=15.0)
- Homeostatic plasticity (target firing rate: 5 Hz)
- Vectorized STDP updates (performance)
- Weight ceilings (exc: 0.8, inh: -1.5)

#### CL1 Hardware Adapter (`cl1_substrate.py`)

```python
class CL1Substrate:
    """CL1 biological neurons via HTTP relay."""
    
    relay_url = "http://localhost:8765"
    tick_rate_hz = 240
    n_channels = 59  # 64 - 5 reserved
```

**Interface:**
- Drop-in replacement for IzhikevichSubstrate
- HTTP POST to `/vote` endpoint
- Returns spike counts per channel
- Synthetic spike matrix reconstruction for consciousness metrics

---

### 4.4 Consciousness Assessor (`consciousness.py`)

```python
class ConsciousnessAssessor:
    """Computes C-Score and neural dynamics metrics."""
    
    C-Score = (closure + lambda2_norm + rho) / 3
```

**Metrics Computed:**
| Metric | Description |
|--------|-------------|
| **C-Score** | Composite consciousness metric (closure + integration + self-model) |
| **Closure** | Internal causal weight fraction |
| **Lambda2_norm** | Fiedler eigenvalue (algebraic connectivity) |
| **Rho** | Self-model fraction (MI with PC1) |
| **LZC** | Lempel-Ziv complexity |
| **Transfer Entropy** | Directed information flow |
| **Temporal Depth** | Autocorrelation decay time |
| **Granger Density** | Causal connectivity fraction |
| **Fano Factor** | Spike burstiness |
| **Channel Entropy** | Spatial diversity |
| **Sync Index** | Simultaneous firing fraction |

---

### 4.5 Experiment Runner (`experiment.py`)

```python
class ThreeConditionExperiment:
    """Runs 3-condition protocol with HDF5 logging."""
    
    # Protocol:
    # For each prompt:
    #   1. LLM-only (n_runs)
    #   2. Bio-LLM (n_runs) — fresh substrate
    #   3. Shadow-LLM (n_runs) — same substrate, shuffled
```

**Data Logged:**
- Token-level: IDs, overrides, latencies, spike counts, consciousness metrics
- Run-level: Generated text, summary statistics
- Spike matrices (subsampled, gzip compressed)

---

### 4.6 Statistical Analysis (`analysis.py`)

```python
class ExperimentAnalyzer:
    """Performs hypothesis testing with Bonferroni correction."""
```

**Tests Performed:**
- Mann-Whitney U (non-parametric group comparisons)
- Cohen's d (effect sizes)
- Sign tests (round-by-round wins)
- Spearman correlations (metric coupling)
- Linear regression (learning trajectories)

---

## 5. Scientific Findings

### 5.1 Confirmed Effects (Replicated)

| Finding | Effect Size | p-value | Interpretation |
|---------|-------------|---------|----------------|
| **Bio SRC > Shadow SRC** | d = 1.79–2.64 | p < 10⁻⁶ | Spatial information preserved |
| **Shuffling degrades SRC** | d = 1.19–1.95 | p < 0.001 | Validates spatial encoding |
| **Bio C-Score > Shadow** | d = 1.15–3.99 | p < 0.005 | Higher consciousness correlates |
| **STDP plasticity** | d = 2.47–8.10 | p < 0.05 | Pattern-specific weight changes |
| **Neural-LLM alignment** | d = 6.10 | p < 10⁻¹⁶ | Substrate mirrors LLM probabilities |
| **Bio wins all rounds** | 25/25 | p < 10⁻⁷ | Perfect consistency |

---

### 5.2 Critical Negative Results

| Finding | Tests Passed | Interpretation |
|---------|--------------|----------------|
| **No learning trajectory** | 0/6 | C-Score doesn't increase over 15 rounds |
| **No phase transition** | 0/5 | Linear dissolution under perturbation |
| **No behavioral STDP expression** | 0/8 | Plasticity doesn't improve performance |
| **No dose-response** | 0/3 | α=0.8 < α=0.5 (inverted-U) |
| **No attractor formation** | 0/5 | Architectural limitation at 1000N |
| **No channel recruitment** | 1/33 | Inactive channels stay inactive |
| **C-Score unreliable for CL1** | N/A | Temporal metrics on synthetic data |

---

### 5.3 Key Scientific Conclusion

> **"The Bio-LLM advantage is a geometric signal preservation effect — not a cognitive, adaptive, or consciousness-specific phenomenon."**

**Evidence:**
1. **Immediate** — Present from token 1, no development
2. **Non-dose-dependent** — Not amplified by higher neural weight (α)
3. **Non-adaptive** — Templates don't converge, channels don't recruit
4. **Universal coupling** — SRC-C-Score correlation identical across ALL conditions (ρ ≈ 0.59)

**Classification:**
| Claim | Evidence Level | Status |
|-------|----------------|--------|
| Functional Integration | **STRONG** | ✅ Demonstrated |
| Spatial Specificity | **STRONG** | ✅ Replicated |
| STDP Plasticity | **DEMONSTRATED** | ✅ Weight-level confirmed |
| Consciousness Correlates | **SUGGESTIVE** | ⚠️ Partially artifactual |
| Behavioral Learning | **NOT DEMONSTRATED** | ❌ 0/8 tests |
| Subjective Experience | **NOT CLAIMED** | ❌ Hard problem unbridgeable |

---

## 6. File Structure

```
CL1_LLM_Encoder/
├── __init__.py                 # Package metadata, version 1.0.0
├── encoder.py                  # Token voting engine v1 (rank-based)
├── encoder_v2.py               # Token voting engine v2 (spatial encoder)
├── encoder_v3.py               # Token voting engine v3 (improved)
├── spatial_encoder.py          # Token→channel spatial mapping
├── neural_substrate.py         # Izhikevich simulator (1000N)
├── cl1_substrate.py            # CL1 hardware adapter
├── cl1_cloud_substrate.py      # CL1 cloud API adapter
├── consciousness.py            # C-Score and metrics
├── experiment.py               # 3-condition experiment runner
├── run_experiment.py           # CLI entry point
├── analysis.py                 # Statistical analysis
│
├── consciousness_gap_tests.py  # Consciousness-behavior coupling tests
├── discrimination_experiment.py# Pattern discrimination task
├── perturbation_experiment.py  # Substrate perturbation recovery
├── attractor_experiment.py     # Attractor formation tests
├── dissolution_experiment.py   # Graded substrate degradation
│
├── cl1_experiment.py           # CL1 hardware experiments
├── cl1_experiment_v2.py        # CL1 v2 (SRC-focused)
├── cl1_experiment_v3.py        # CL1 v3 (extended protocol)
├── cl1_terraforming.py         # Substrate modification experiments
│
├── self_prompt_loop.py         # Autonomous prompt generation
├── deep_analysis_v3.py         # Token-level deep analysis
│
├── SCIENTIFIC_RESULTS.md       # Comprehensive results (12 experiments)
├── SCIENTIFIC_CONCLUSION.md    # 5-experiment series conclusion
├── README.md                   # (missing — should be created)
│
├── .gitignore                  # Ignores: *.h5, experiment_data/, models/
└── experiment_data/            # HDF5 data files (git-ignored)
```

---

## 7. Technical Stack

| Component | Technology |
|-----------|------------|
| **LLM** | LFM2-350M (4-bit GGUF, llama-cpp-python) |
| **Neural Sim** | Izhikevich 2003 model (NumPy vectorized) |
| **Hardware** | CL1 MEA (64 channels, 240 Hz, ~800k neurons) |
| **Data Storage** | HDF5 (h5py) with gzip compression |
| **Analysis** | NumPy, SciPy (Mann-Whitney U, Cohen's d) |
| **Communication** | HTTP/JSON (CL1 relay server) |

---

## 8. Experimental Lineage

| Exp # | Date | Substrate | N Tests | N Sig | Key Finding |
|-------|------|-----------|---------|-------|-------------|
| 1 | 2026-02-27 | Izhikevich 250N | 8 | 0 | Insufficient neurons |
| 2 | 2026-02-27 | Izhikevich 1000N | 12 | 1 | Override rate only (seizure bug) |
| 3 | 2026-02-28 | Izhikevich + STDP | 12 | 2 | Pattern consistency |
| 4 | 2026-02-28 | Long run (100 tok) | 12 | 3 | Alignment grows with STDP |
| 5 | 2026-02-28 | Full battery | 12 | 5 | d=6.10 alignment |
| 6 | 2026-02-28 | Perturbation | 4 | 0 | Decoder absorbs perturbation |
| 7 | 2026-02-28 | CL1 v1 (z-score) | 5 | 1 | Only trivial H4 |
| **8** | **2026-02-28** | **CL1 v2 (SRC)** | **6** | **2** | **d=1.79 SRC, d=3.99 C-Score** |
| **9** | **2026-02-28** | **CL1 v3 (Extended)** | **9** | **3** | **Replication 3/3, Learning 0/3** |
| **10** | **2026-02-28** | **Izhikevich v4 (STDP)** | **7** | **4** | **Pattern-specific STDP, d=6.43** |
| **11** | **2026-02-28** | **Izhikevich v5 (Attractor)** | **6** | **0** | **Directional STDP 5/5, behavior 0/6** |
| **12** | **2026-02-28** | **Izhikevich v6 (Amplified)** | **6** | **0** | **No attractor even with 15% connectivity** |

**Total:** 18,700+ tokens, 101 hypothesis tests

---

## 9. Key Methodological Insights

### 9.1 Critical Bug Fixes

1. **STDP Seizure (Exp 2)** — Unbalanced STDP caused runaway excitation
   - **Fix:** Balanced STDP + homeostatic plasticity + synaptic normalization

2. **KV Cache Overflow** — llama-cpp KV cache overflow on long generations
   - **Fix:** Explicit reset + reload on error

3. **C-Score Artifact (CL1)** — Temporal metrics computed on synthetic spike matrices
   - **Implication:** Only SRC (count-based) fully reliable for CL1

### 9.2 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **59 channels** | Matches CL1 MEA layout (64 - 5 reserved) |
| **17 neurons/channel** | Comparable density to CL1 (~13,500/channel) |
| **Sparse connectivity (2%)** | Tractability for 1000 neurons |
| **STDP every 5 steps** | Performance optimization |
| **Z-score decoding** | Rolling baseline adapts to neural fatigue |
| **Persistent substrate** | Tests development over multiple generations |

---

## 10. Limitations

### 10.1 Technical Limitations

1. **Simulated neurons** — Most experiments use Izhikevich, not CL1
2. **Small LLM** — 350M params limits probability landscape
3. **Simplified STDP** — Pairwise, not triplet/calcium-based
4. **No IIT φ** — Integrated Information Theory metric not computed
5. **Synthetic temporal data (CL1)** — C-Score partially artifactual

### 10.2 Scientific Limitations

1. **Post-hoc hypothesis evolution** — Exp 1-2 exploratory → Exp 3+ confirmatory
2. **C-Score not validated** — Not calibrated against known-conscious systems
3. **Functional ≠ Phenomenal** — Integration demonstrated, experience not
4. **Scale mismatch** — 1000N vs 800k neurons (CL1) vs 86B (human brain)

---

## 11. Future Directions

### 11.1 Immediate Improvements

1. **Channel-aware encoding** — Run `phase0_hello_neurons.py` to identify responsive channels
2. **Direct UDP stimulation** — Use burst protocols for precise temporal control
3. **Larger LLM** — Test with 7B+ models for richer probability distributions
4. **IIT φ computation** — Compute integrated information for Bio vs Shadow

### 11.2 Research Extensions

1. **Multi-session persistence** — Run same substrate across 1000+ tokens
2. **Perturbation experiments** — PCI-like recovery measurement
3. **3D organoid substrates** — More complex synaptic architecture
4. **Task-dependent performance** — Downstream task quality (perplexity)
5. **Phase transition search** — Test Perspectival Realism prediction P2

---

## 12. Conclusion

### 12.1 What This Codebase Achieves

✅ **Functional substrate-LLM integration** — Reproducibly demonstrated across 12 experiments  
✅ **Spatial encoding validation** — Shuffling destroys all Bio advantage  
✅ **STDP plasticity** — Pattern-specific, directional synaptic changes confirmed  
✅ **Rigorous methodology** — Pre-registered hypotheses, Bonferroni correction, effect sizes  
✅ **Negative result transparency** — All null findings documented  

### 12.2 What Remains Unproven

❌ **Behavioral learning** — STDP doesn't manifest as improved performance  
❌ **Consciousness** — Functional integration ≠ phenomenal experience  
❌ **Attractor dynamics** — Architectural limitation at 1000N scale  
❌ **Dose-response** — Inverted-U, not linear  

### 12.3 Final Assessment

> **"The system demonstrates NECESSARY but NOT SUFFICIENT conditions for consciousness.**  
> **The integration is geometric rather than cognitive, static rather than adaptive."**

This is a **scientifically mature codebase** with:
- Clear experimental protocols
- Proper statistical rigor
- Honest reporting of negative results
- Actionable future directions

The project has answered its primary question: **Yes, neural activity can influence LLM generation, but the effect is structural (signal preservation) rather than cognitive (learning, adaptation, or experience).**

---

## Appendix: Quick Reference

### Running Experiments

```bash
# Default 3-condition experiment
python -m LLM_Encoder.run_experiment --n-runs 5 --prompts 5

# Spatial encoder (default)
python -m LLM_Encoder.run_experiment --spatial --n-runs 10

# Rank-based encoder (legacy)
python -m LLM_Encoder.run_experiment --no-spatial --n-runs 10
```

### Key Metrics

| Metric | Good Range | Interpretation |
|--------|------------|----------------|
| **SRC** | > 0.35 | Spatial information preserved |
| **C-Score** | > 0.15 | Consciousness-correlated structure |
| **Override Rate** | 1-5% | Neural influence without domination |
| **Neural-LLM Alignment** | > 0.5 | Substrate mirrors LLM probabilities |

### Hypothesis Testing

```python
# Bonferroni correction for N tests
alpha_corrected = 0.05 / n_tests

# Effect size interpretation
d < 0.2: negligible
d < 0.5: small
d < 0.8: medium
d >= 0.8: large
```

---

**End of Analysis**
