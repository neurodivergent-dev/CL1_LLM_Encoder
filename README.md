# CL1 LLM Encoder

**Neural Token Voting with Consciousness Measurement**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This package enables **closed-loop integration between Large Language Models and neural substrates** (both simulated and biological). It implements a token voting mechanism where neural spike activity can influence LLM token selection, creating a genuine cybernetic system.

### Research Question

> *Can neural activity meaningfully influence LLM token generation, and does this integration produce consciousness-like properties?*

---

## Installation

```bash
# Core dependencies
pip install numpy scipy h5py

# LLM inference (required)
pip install llama-cpp-python

# Optional: for CL1 hardware integration
# (Requires CL1 Voting Relay Server running on device)
```

### Model Setup

Download the LLM model and place it in the `models/` directory:

- **Model:** LFM2-350M (4-bit quantized GGUF)
- **Path:** `models/LFM2-350M-Q4_0.gguf`

---

## Quick Start

### Run the 3-Condition Experiment

```bash
# Default experiment (5 runs × 10 prompts × 3 conditions = 150 runs)
python -m LLM_Encoder.run_experiment --n-runs 5 --prompts 10

# Quick test (1 run × 3 prompts)
python -m LLM_Encoder.run_experiment --n-runs 1 --prompts 3

# Use spatial encoder (default)
python -m LLM_Encoder.run_experiment --spatial --n-runs 5

# Use legacy rank-based encoder
python -m LLM_Encoder.run_experiment --no-spatial --n-runs 5
```

### Programmatic Usage

```python
from LLM_Encoder.encoder_v2 import TokenVotingEngineV2

# Initialize engine (loads LLM, creates substrate)
engine = TokenVotingEngineV2(
    model_path="models/LFM2-350M-Q4_0.gguf",
    alpha=0.5,              # Neural blend weight (0.0-1.0)
    substrate_seed=42,
)

# Run three conditions
prompt = "What is consciousness?"

text_llm, records_llm, summary_llm = engine.generate(
    prompt=prompt,
    max_tokens=25,
    condition="llm_only",   # Pure LLM baseline
    verbose=True,
)

text_bio, records_bio, summary_bio = engine.generate(
    prompt=prompt,
    max_tokens=25,
    condition="bio_llm",    # Closed-loop neural voting
    verbose=True,
)

text_shadow, records_shadow, summary_shadow = engine.generate(
    prompt=prompt,
    max_tokens=25,
    condition="shadow_llm",  # Shuffled spikes (control)
    verbose=True,
)

# Compare results
print(f"LLM-only C-Score:     {summary_llm['mean_cscore']:.4f}")
print(f"Bio-LLM C-Score:      {summary_bio['mean_cscore']:.4f}")
print(f"Shadow-LLM C-Score:   {summary_shadow['mean_cscore']:.4f}")
print(f"Bio override rate:    {summary_bio['override_rate']:.2%}")
print(f"Shadow override rate: {summary_shadow['override_rate']:.2%}")
```

---

## Architecture

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

## Experimental Conditions

| Condition | Alpha | Neural Influence | Spatial Mapping | Purpose |
|-----------|-------|------------------|-----------------|---------|
| **LLM-only** | 0.0 | None | Intact | Open-loop baseline |
| **Bio-LLM** | 0.5 | Full closed-loop | Intact | Test integration |
| **Shadow-LLM** | 0.5 | Full closed-loop | **Shuffled** | Control (destroys spatial info) |

---

## Key Components

| Module | Description |
|--------|-------------|
| `encoder.py` | Token voting engine v1 (rank-based encoding) |
| `encoder_v2.py` | Token voting engine v2 (spatial encoder) |
| `encoder_v3.py` | Token voting engine v3 (improved feedback) |
| `spatial_encoder.py` | Token→channel spatial mapping (deterministic) |
| `neural_substrate.py` | Izhikevich neural simulator (1000 neurons) |
| `cl1_substrate.py` | CL1 hardware adapter (biological neurons) |
| `consciousness.py` | C-Score and neural dynamics metrics |
| `experiment.py` | 3-condition experiment runner with HDF5 logging |
| `analysis.py` | Statistical analysis (Mann-Whitney U, Cohen's d) |

---

## Consciousness Metrics

The package computes multiple consciousness-correlated metrics per token:

| Metric | Formula | Description |
|--------|---------|-------------|
| **C-Score** | (closure + λ₂_norm + ρ) / 3 | Composite consciousness metric |
| **Closure** | Σ|W_internal| / Σ|W_total| | Internal causal weight fraction |
| **Lambda2_norm** | λ₂ / λ_max | Algebraic connectivity (Fiedler) |
| **Rho** | MI(unit, global) / H(unit) | Self-model fraction |
| **LZC** | c / (L / log₂L) | Lempel-Ziv complexity |
| **Transfer Entropy** | H(Y|X_past) - H(Y|X_past, Z_past) | Directed information flow |
| **Temporal Depth** | τ(autocorr < 1/e) | Memory timescale |
| **Granger Density** | |GC > 0| / N² | Causal connectivity |

---

## Scientific Results

### Confirmed Findings (Replicated)

✅ **Higher SRC in Bio-LLM** (d = 1.79–2.64, p < 10⁻⁶) — spatial information preservation  
✅ **Shuffling degrades SRC** (d = 1.19–1.95, p < 0.001) — validates spatial encoding  
✅ **Bio C-Score > Shadow** (d = 1.15–3.99, p < 0.005) — higher consciousness correlates  
✅ **STDP plasticity** (d = 2.47–8.10, p < 0.05) — pattern-specific synaptic changes  
✅ **Neural-LLM alignment** (d = 6.10, p < 10⁻¹⁶) — substrate mirrors LLM probabilities  

### Critical Negative Results

❌ **No learning trajectory** (0/6 tests) — C-Score doesn't increase over time  
❌ **No phase transition** (0/5 tests) — linear dissolution under perturbation  
❌ **No behavioral STDP expression** (0/8 tests) — plasticity doesn't improve performance  
❌ **No dose-response** (α=0.8 < α=0.5) — inverted-U relationship  
❌ **No attractor formation** (0/5 tests) — architectural limitation at 1000N  

### Key Conclusion

> **"The Bio-LLM advantage is a geometric signal preservation effect — not a cognitive, adaptive, or consciousness-specific phenomenon."**

See [`SCIENTIFIC_RESULTS.md`](SCIENTIFIC_RESULTS.md) and [`SCIENTIFIC_CONCLUSION.md`](SCIENTIFIC_CONCLUSION.md) for detailed analysis.

---

## Data Output

Experiments save data to HDF5 format:

```
experiment_data/
└── experiment_20260305_143022.h5
    ├── metadata (attrs)
    └── conditions/
        ├── llm_only/
        │   └── run_0000/
        │       ├── token_ids
        │       ├── overrides
        │       ├── consciousness_metrics
        │       └── spike_matrices/
        ├── bio_llm/
        └── shadow_llm/
```

### Accessing HDF5 Data

```python
import h5py

with h5py.File("experiment_data/experiment_20260305_143022.h5", "r") as f:
    # List conditions
    print(list(f["conditions"].keys()))
    
    # Access Bio-LLM run data
    bio_run = f["conditions/bio_llm/run_0000"]
    token_ids = bio_run["token_ids"][:]
    overrides = bio_run["overrides"][:]
    c_scores = bio_run["consciousness_metrics"][:, 0]  # First column = C-Score
    
    print(f"Tokens: {len(token_ids)}, Overrides: {overrides.sum()}")
    print(f"Mean C-Score: {c_scores.mean():.4f}")
```

---

## CL1 Hardware Integration

To use biological neurons instead of simulation:

### 1. Start CL1 Relay Server (on CL1 device)

```python
# Run on CL1 device
%run cl1_voting_relay.py
```

### 2. Use CL1Substrate

```python
from LLM_Encoder.cl1_substrate import CL1Substrate
from LLM_Encoder.encoder import TokenVotingEngine

# Create engine with CL1 hardware
substrate = CL1Substrate(
    relay_url="http://localhost:8765",
    window_s=0.5,
)

engine = TokenVotingEngine(
    model_path="models/LFM2-350M-Q4_0.gguf",
    alpha=0.5,
)
engine.substrate = substrate  # Replace simulated with biological

# Run experiment
text, records, summary = engine.generate(
    prompt="The neurons are listening.",
    condition="bio_llm",
    max_tokens=25,
)
```

---

## API Reference

### TokenVotingEngine

```python
class TokenVotingEngine:
    def __init__(
        model_path: str,
        alpha: float = 0.5,
        substrate_seed: int = 42,
        n_gpu_layers: int = -1,
    )
    
    def generate(
        prompt: str,
        max_tokens: int = 25,
        condition: str = "bio_llm",  # 'llm_only', 'bio_llm', 'shadow_llm'
        measure_consciousness: bool = True,
        verbose: bool = True,
    ) -> Tuple[str, List[TokenRecord], Dict]
```

### SpatialEncoder

```python
class SpatialEncoder:
    def __init__(
        n_active: int = 8,
        channels: List[int] = None,
        seed: int = 42,
    )
    
    def encode_token(token_id: int) -> Dict[int, float]
    # Returns: channel -> amplitude (µA)
    
    def encode_candidates(candidates: Dict[int, float]) -> Tuple[Dict, Dict]
    # Returns: combined_pattern, channel_to_token
```

### ConsciousnessAssessor

```python
class ConsciousnessAssessor:
    def assess(spike_matrix: np.ndarray) -> Dict
    # Returns: dict with cscore, closure, lambda2_norm, rho, lzc,
    #          transfer_entropy, temporal_depth, granger_density,
    #          fano_factor, channel_entropy, sync_index
```

---

## Configuration

### Neural Substrate Parameters

```python
from LLM_Encoder.neural_substrate import IzhikevichConfig, IzhikevichSubstrate

cfg = IzhikevichConfig(
    n_neurons=1000,           # Total neurons
    n_excitatory=800,         # 80% excitatory
    n_inhibitory=200,         # 20% inhibitory
    dt=0.5,                   # Timestep (ms)
    tick_rate_hz=240,         # Simulation rate
    n_channels=59,            # MEA channels
    connection_prob=0.02,     # Sparse connectivity
    stdp_A_plus=0.005,        # LTP amplitude
    stdp_A_minus=0.006,       # LTD amplitude
    target_firing_rate=5.0,   # Hz homeostasis target
)

substrate = IzhikevichSubstrate(cfg=cfg, seed=42)
```

### Stimulation Parameters

```python
from LLM_Encoder.spatial_encoder import (
    AMP_MIN,       # 0.3 µA
    AMP_MAX,       # 2.5 µA
    FREQ_MIN,      # 4.0 Hz
    FREQ_MAX,      # 40.0 Hz
    N_ACTIVE_CHANNELS,  # 8 channels per token
    RESERVED_CHANNELS,  # {0, 4, 7, 56, 63}
)
```

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{cl1_llm_encoder2026,
  title = {CL1 LLM Encoder: Neural Token Voting with Consciousness Measurement},
  author = {4R7I5T},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/4R7I5T/CL1_LLM_Encoder},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## See Also

- [`SCIENTIFIC_RESULTS.md`](SCIENTIFIC_RESULTS.md) — Comprehensive results from 12 experiments
- [`SCIENTIFIC_CONCLUSION.md`](SCIENTIFIC_CONCLUSION.md) — 5-experiment series conclusion
- [`REPO_ANALYSIS.md`](REPO_ANALYSIS.md) — Full codebase analysis

---

**forked from [4R7I5T/CL1_LLM_Encoder](https://github.com/4R7I5T/CL1_LLM_Encoder)** | 2026
