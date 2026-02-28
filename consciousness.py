"""Consciousness Measurement — C-Score and neural dynamics metrics.

Computes the Perspectival Analysis C-Score on spike data from the
neural substrate, plus additional consciousness-relevant metrics:
  - Fano factor (burstiness)
  - Channel entropy (diversity)
  - Synchrony index
  - Transfer entropy (directed information flow)
  - Temporal mutual information (closed-loop coupling)
  - Lempel-Ziv complexity (temporal structure)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# Lempel-Ziv Complexity
# ---------------------------------------------------------------------------

def _lz_complexity(sequence: np.ndarray) -> int:
    """LZ76 complexity count (Kaspar-Schuster 1987)."""
    s = ''.join(map(str, sequence))
    n = len(s)
    if n == 0:
        return 0
    i, k, l = 0, 1, 1
    c = 1
    while l + k <= n:
        sub = s[l: l + k]
        source = s[0: l + k - 1]
        if sub in source:
            k += 1
        else:
            c += 1
            l += k
            k = 1
    return c


def lempel_ziv_complexity(states: np.ndarray) -> float:
    """Normalized LZ complexity of (T, N) array."""
    T, N = states.shape
    means = np.mean(states, axis=0)
    binary = (states > means).astype(int)
    seq = binary.ravel()
    c = _lz_complexity(seq)
    L = len(seq)
    if L < 2:
        return 0.0
    return c / (L / np.log2(L))


# ---------------------------------------------------------------------------
# Granger Causality
# ---------------------------------------------------------------------------

def compute_granger_causality(states: np.ndarray, max_lag: int = 5) -> np.ndarray:
    """Pairwise Granger causality matrix. W[i,j] = GC from j to i."""
    T, N = states.shape
    W = np.zeros((N, N))
    if T <= 2 * max_lag + 1:
        return W

    for i in range(N):
        y = states[max_lag:, i]
        X_r = np.column_stack([
            states[max_lag - lag - 1: T - lag - 1, i] for lag in range(max_lag)
        ])
        try:
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            rss_r = np.sum((y - X_r @ beta_r) ** 2)
        except np.linalg.LinAlgError:
            continue

        for j in range(N):
            if i == j:
                continue
            X_f = np.column_stack([
                X_r,
                *[states[max_lag - lag - 1: T - lag - 1, j: j + 1] for lag in range(max_lag)]
            ])
            try:
                beta_f = np.linalg.lstsq(X_f, y, rcond=None)[0]
                rss_f = np.sum((y - X_f @ beta_f) ** 2)
            except np.linalg.LinAlgError:
                continue
            if rss_f > 0 and rss_r > rss_f:
                k_diff = max_lag
                df = T - max_lag - 2 * max_lag
                if df > 0:
                    F = ((rss_r - rss_f) / k_diff) / (rss_f / df)
                    W[i, j] = max(0.0, F)
    return W


# ---------------------------------------------------------------------------
# C-Score Components
# ---------------------------------------------------------------------------

def compute_closure(W: np.ndarray) -> float:
    """Operational closure: internal causal weight fraction."""
    internal = np.sum(np.abs(W))
    return internal / (internal + 1e-10)


def compute_lambda2(W: np.ndarray) -> Tuple[float, float]:
    """Fiedler eigenvalue (algebraic connectivity) + normalized."""
    W_sym = (np.abs(W) + np.abs(W.T)) / 2
    degrees = W_sym.sum(axis=1)
    L = np.diag(degrees) - W_sym
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L)))
    lambda2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    lambda_max = float(eigenvalues[-1]) if eigenvalues[-1] > 0 else 1.0
    lambda2_norm = lambda2 / lambda_max if lambda_max > 1e-10 else 0.0
    return lambda2, lambda2_norm


def _histogram_mi(X: np.ndarray, Y: np.ndarray, bins: int = 20) -> float:
    """Mutual information via histograms."""
    c_xy, _, _ = np.histogram2d(X, Y, bins=bins)
    c_xy = c_xy / (c_xy.sum() + 1e-10)
    c_x = c_xy.sum(axis=1)
    c_y = c_xy.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if c_xy[i, j] > 0 and c_x[i] > 0 and c_y[j] > 0:
                mi += c_xy[i, j] * np.log2(c_xy[i, j] / (c_x[i] * c_y[j]))
    return max(0.0, mi)


def compute_rho(states: np.ndarray, bins: int = 20) -> float:
    """Self-model fraction: MI between each unit and global state (PC1)."""
    T, N = states.shape
    if T < 5 or N < 2:
        return 0.0
    centered = states - states.mean(axis=0)
    try:
        cov = centered.T @ centered / (T - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        pc1 = centered @ eigenvectors[:, -1]
    except np.linalg.LinAlgError:
        return 0.0

    from collections import deque as _dq
    mi_values = []
    for n in range(N):
        h_x = 0.0
        counts, _ = np.histogram(states[:, n], bins=bins)
        probs = counts / (counts.sum() + 1e-10)
        for p in probs:
            if p > 0:
                h_x -= p * np.log2(p)
        if h_x < 1e-10:
            mi_values.append(0.0)
            continue
        mi = _histogram_mi(states[:, n], pc1, bins)
        mi_values.append(mi / h_x)

    rho_raw = float(np.mean(mi_values))
    mi_arr = np.array(mi_values)
    if len(mi_arr) > 1 and np.mean(mi_arr) > 1e-10:
        cv = np.std(mi_arr) / np.mean(mi_arr)
        if cv < 0.05:
            rho_raw *= cv / 0.05
    return min(1.0, max(0.0, rho_raw))


def compute_cscore(
    spike_matrix: np.ndarray,
    max_lag: int = 3,
    subsample_n: int = 15,
    downsample_factor: int = 10,
) -> Dict:
    """Compute PA composite C-Score from spike matrix (n_neurons, n_ticks).

    Uses fast mode (no surrogates) for experiment loop speed.
    """
    n_elec, n_ticks = spike_matrix.shape
    states = spike_matrix.T.astype(np.float64)

    # Subsample channels
    if states.shape[1] > subsample_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(states.shape[1], size=subsample_n, replace=False)
        idx.sort()
        states = states[:, idx]

    # Downsample temporally
    if downsample_factor > 1:
        n_bins = states.shape[0] // downsample_factor
        if n_bins >= 10:
            states = states[:n_bins * downsample_factor].reshape(
                n_bins, downsample_factor, -1
            ).mean(axis=1)

    # Components
    W = compute_granger_causality(states, max_lag=max_lag)
    closure = compute_closure(W)
    lambda2, lambda2_norm = compute_lambda2(W)
    rho = compute_rho(states)
    lzc = lempel_ziv_complexity(states)

    cscore = (closure + lambda2_norm + rho) / 3.0
    cscore = float(np.clip(cscore, 0, 1))

    return {
        'cscore': cscore,
        'closure': float(closure),
        'lambda2': float(lambda2),
        'lambda2_norm': float(lambda2_norm),
        'rho': float(rho),
        'lzc': float(lzc),
        'granger_mean': float(np.mean(W[W > 0])) if np.any(W > 0) else 0.0,
        'granger_density': float(np.sum(W > 0)) / max(1, W.size),
    }


# ---------------------------------------------------------------------------
# Transfer Entropy (directed information flow)
# ---------------------------------------------------------------------------

def transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int = 1, bins: int = 10) -> float:
    """Transfer entropy from source to target time series."""
    T = len(source)
    if T < lag + 2:
        return 0.0

    target_future = target[lag:]
    target_past = target[:T - lag]
    source_past = source[:T - lag]

    # Discretize
    def discretize(x, b):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-10:
            return np.zeros_like(x, dtype=int)
        return np.clip(((x - mn) / (mx - mn + 1e-10) * b).astype(int), 0, b - 1)

    tf = discretize(target_future, bins)
    tp = discretize(target_past, bins)
    sp = discretize(source_past, bins)

    # Joint and conditional entropies
    n = len(tf)
    # H(target_future | target_past)
    joint_tp_tf = np.zeros((bins, bins))
    for i in range(n):
        joint_tp_tf[tp[i], tf[i]] += 1
    joint_tp_tf /= n + 1e-10

    # H(target_future | target_past, source_past)
    joint_all = np.zeros((bins, bins, bins))
    for i in range(n):
        joint_all[tp[i], sp[i], tf[i]] += 1
    joint_all /= n + 1e-10

    # TE = H(tf|tp) - H(tf|tp,sp)
    h_tf_tp = 0.0
    p_tp = joint_tp_tf.sum(axis=1)
    for i in range(bins):
        for j in range(bins):
            if joint_tp_tf[i, j] > 0 and p_tp[i] > 0:
                h_tf_tp -= joint_tp_tf[i, j] * np.log2(joint_tp_tf[i, j] / p_tp[i])

    h_tf_tp_sp = 0.0
    p_tp_sp = joint_all.sum(axis=2)
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                if joint_all[i, j, k] > 0 and p_tp_sp[i, j] > 0:
                    h_tf_tp_sp -= joint_all[i, j, k] * np.log2(
                        joint_all[i, j, k] / p_tp_sp[i, j]
                    )

    te = h_tf_tp - h_tf_tp_sp
    return max(0.0, te)


# ---------------------------------------------------------------------------
# Full Consciousness Assessment
# ---------------------------------------------------------------------------

class ConsciousnessAssessor:
    """Computes comprehensive consciousness metrics from spike data.

    Designed for the 3-condition experiment: computes identical metrics
    for all conditions to enable fair comparison.
    """

    def __init__(self):
        self._spike_history: List[np.ndarray] = []
        self._cscore_history: List[float] = []

    def assess(self, spike_matrix: np.ndarray) -> Dict:
        """Full consciousness assessment on a spike matrix (n_neurons, n_ticks).

        Returns dict with all metrics needed for condition comparison.
        """
        n_neurons, n_ticks = spike_matrix.shape
        self._spike_history.append(spike_matrix)

        # C-Score (main consciousness metric)
        cscore_result = compute_cscore(spike_matrix)
        self._cscore_history.append(cscore_result['cscore'])

        # Per-channel spike statistics
        channel_spikes = spike_matrix.sum(axis=1)
        total_spikes = int(channel_spikes.sum())

        # Fano factor (population level)
        if len(self._spike_history) >= 3:
            totals = [m.sum() for m in self._spike_history[-20:]]
            arr = np.array(totals, dtype=float)
            fano = float(np.var(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0.0
        else:
            fano = 0.0

        # Channel entropy
        if total_spikes > 0:
            probs = channel_spikes / total_spikes
            probs = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        else:
            entropy = 0.0

        # Synchrony: fraction of time bins where >10% of neurons fire
        bin_activity = spike_matrix.sum(axis=0)
        sync_threshold = 0.1 * n_neurons
        sync_index = float(np.mean(bin_activity > sync_threshold))

        # LZ complexity
        lzc = cscore_result['lzc']

        # Inter-channel transfer entropy (sample pairs)
        te_values = []
        if n_ticks > 20 and n_neurons >= 4:
            rng = np.random.default_rng(42)
            pairs = [(i, j) for i in range(min(8, n_neurons))
                     for j in range(min(8, n_neurons)) if i != j]
            if len(pairs) > 10:
                pairs = [pairs[k] for k in rng.choice(len(pairs), 10, replace=False)]
            for i, j in pairs:
                te = transfer_entropy(
                    spike_matrix[i].astype(float),
                    spike_matrix[j].astype(float),
                    lag=1, bins=5
                )
                te_values.append(te)
        mean_te = float(np.mean(te_values)) if te_values else 0.0

        # Temporal autocorrelation (network-level)
        pop_rate = spike_matrix.sum(axis=0).astype(float)
        if len(pop_rate) > 10 and np.std(pop_rate) > 0:
            pop_centered = pop_rate - np.mean(pop_rate)
            autocorr = np.correlate(pop_centered, pop_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr /= autocorr[0] + 1e-10
            # Decay time constant (time to drop below 1/e)
            decay_idx = np.where(autocorr < 1/np.e)[0]
            temporal_depth = float(decay_idx[0]) if len(decay_idx) > 0 else float(len(autocorr))
        else:
            temporal_depth = 0.0

        return {
            # Core C-Score
            'cscore': cscore_result['cscore'],
            'closure': cscore_result['closure'],
            'lambda2_norm': cscore_result['lambda2_norm'],
            'rho': cscore_result['rho'],
            'lzc': lzc,
            'granger_mean': cscore_result['granger_mean'],
            'granger_density': cscore_result['granger_density'],

            # Neural dynamics
            'total_spikes': total_spikes,
            'fano_factor': round(fano, 4),
            'channel_entropy': round(entropy, 4),
            'sync_index': round(sync_index, 4),

            # Information flow
            'transfer_entropy': round(mean_te, 6),
            'temporal_depth': round(temporal_depth, 2),

            # Network state
            'mean_firing_rate': round(total_spikes / (n_neurons * n_ticks + 1e-10) * 240, 2),
            'active_channels': int(np.sum(channel_spikes > 0)),
            'n_neurons': n_neurons,
            'n_ticks': n_ticks,
        }

    def get_cscore_trajectory(self) -> List[float]:
        """Return C-Score over time for trajectory analysis."""
        return list(self._cscore_history)
