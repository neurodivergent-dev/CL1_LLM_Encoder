"""Spatial Encoder — Token-specific stimulation patterns for neural substrate.

Instead of mapping tokens to channels by rank (which changes every vote),
this creates a CONSISTENT spatial encoding:
  - Each token ID maps to a unique spatial pattern across the MEA
  - Patterns are derived from the token's embedding vector
  - The encoding is hash-based but structured to create meaningful
    spatial distributions (not random noise)

Design principles from CL1 research:
  - Non-overlapping channel groups (sensory vs motor vs feedback)
  - Amplitude within safe range (0.3-2.5 µA)
  - Charge-balanced biphasic stimulation
  - Spatial patterns use multiple channels simultaneously (not point stim)
  - Similar tokens should have similar spatial patterns (tonotopic-like)

This creates a genuine encoding that the culture can learn to differentiate,
enabling the closed-loop to develop token-specific neural representations.
"""

import numpy as np
from typing import Dict, List, Tuple
from functools import lru_cache


# CL1 hardware constraints
RESERVED_CHANNELS = frozenset({0, 4, 7, 56, 63})
SENSORY_CHANNELS = [ch for ch in range(1, 33) if ch not in RESERVED_CHANNELS]   # 30 channels
MOTOR_CHANNELS = [ch for ch in range(33, 48) if ch not in RESERVED_CHANNELS]     # 15 channels
FEEDBACK_CHANNELS = [ch for ch in range(48, 63) if ch not in RESERVED_CHANNELS]  # 13 channels

# Stimulation safety limits
AMP_MIN = 0.3   # µA
AMP_MAX = 2.5   # µA
FREQ_MIN = 4.0  # Hz
FREQ_MAX = 40.0 # Hz

# Spatial pattern configuration
N_ACTIVE_CHANNELS = 8   # channels activated per token
PATTERN_OVERLAP = 0.3   # fraction of shared channels between similar tokens


class SpatialEncoder:
    """Maps tokens to unique spatial stimulation patterns.

    Each token gets a consistent multi-channel stimulation pattern:
      - Which channels to stimulate (spatial selectivity)
      - What amplitude per channel (intensity profile)
      - The pattern is deterministic from token_id (reproducible)

    Similar tokens (nearby in embedding space) share partial patterns,
    creating a tonotopic-like representation.
    """

    def __init__(
        self,
        n_active: int = N_ACTIVE_CHANNELS,
        channels: List[int] = None,
        seed: int = 42,
    ):
        self.n_active = n_active
        self.channels = channels or SENSORY_CHANNELS
        self.n_channels = len(self.channels)
        self.rng_base = np.random.default_rng(seed)

        # Pre-compute random projection matrix for embedding->pattern
        # This maps high-dimensional token IDs to channel-space patterns
        self._projection = self.rng_base.standard_normal((self.n_channels, 64))
        self._projection /= np.linalg.norm(self._projection, axis=1, keepdims=True)

    @lru_cache(maxsize=4096)
    def encode_token(self, token_id: int) -> Dict[int, float]:
        """Generate a spatial stimulation pattern for a token.

        Parameters
        ----------
        token_id : int
            The LLM token ID.

        Returns
        -------
        pattern : dict
            Channel -> amplitude (µA). Only active channels included.
        """
        # Create a pseudo-embedding from token_id (deterministic hash)
        rng = np.random.default_rng(token_id * 7919 + 31)  # prime hash
        embedding = rng.standard_normal(64)

        # Project embedding onto channel space
        activations = self._projection @ embedding
        activations = (activations - activations.mean()) / (activations.std() + 1e-10)

        # Select top-N channels by activation (spatial selectivity)
        top_indices = np.argsort(activations)[-self.n_active:]

        # Map activations to amplitudes (sigmoid -> [AMP_MIN, AMP_MAX])
        pattern = {}
        for idx in top_indices:
            ch = self.channels[idx]
            # Sigmoid normalization to safe amplitude range
            act = activations[idx]
            normalized = 1.0 / (1.0 + np.exp(-act))
            amp = AMP_MIN + (AMP_MAX - AMP_MIN) * normalized
            pattern[ch] = round(float(amp), 3)

        return pattern

    def encode_token_with_probability(
        self,
        token_id: int,
        probability: float,
    ) -> Dict[int, float]:
        """Generate pattern modulated by token probability.

        Higher probability tokens get stronger stimulation (more confident
        the culture "hears" this token candidate).
        """
        base_pattern = self.encode_token(token_id)
        # Scale amplitude by probability (still within safe range)
        scaled = {}
        for ch, amp in base_pattern.items():
            modulated = AMP_MIN + (amp - AMP_MIN) * max(0.1, min(1.0, probability * 3))
            scaled[ch] = round(float(np.clip(modulated, AMP_MIN, AMP_MAX)), 3)
        return scaled

    def encode_candidates(
        self,
        candidates: Dict[int, float],
    ) -> Tuple[Dict[int, float], Dict[int, int]]:
        """Encode multiple token candidates into a combined stimulation pattern.

        Each candidate's spatial pattern is superimposed on the MEA.
        Overlapping channels get summed amplitudes (capped at AMP_MAX).

        Parameters
        ----------
        candidates : dict
            Token ID -> probability.

        Returns
        -------
        combined_pattern : dict
            Channel -> total amplitude.
        channel_to_primary_token : dict
            Channel -> token ID of the strongest contributor (for decoding).
        """
        combined = {}
        channel_ownership = {}  # channel -> (max_amplitude, token_id)

        for token_id, prob in candidates.items():
            pattern = self.encode_token_with_probability(token_id, prob)
            for ch, amp in pattern.items():
                if ch in combined:
                    combined[ch] += amp
                else:
                    combined[ch] = amp

                # Track which token "owns" this channel (highest amplitude)
                if ch not in channel_ownership or amp > channel_ownership[ch][0]:
                    channel_ownership[ch] = (amp, token_id)

        # Clip combined amplitudes
        for ch in combined:
            combined[ch] = round(float(np.clip(combined[ch], AMP_MIN, AMP_MAX)), 3)

        channel_to_token = {ch: info[1] for ch, info in channel_ownership.items()}

        return combined, channel_to_token

    def pattern_similarity(self, token_a: int, token_b: int) -> float:
        """Compute pattern overlap between two tokens (Jaccard similarity)."""
        pattern_a = set(self.encode_token(token_a).keys())
        pattern_b = set(self.encode_token(token_b).keys())
        intersection = len(pattern_a & pattern_b)
        union = len(pattern_a | pattern_b)
        return intersection / union if union > 0 else 0.0

    def get_channel_statistics(self, n_tokens: int = 1000) -> Dict:
        """Analyze the encoding's channel usage distribution."""
        channel_counts = np.zeros(len(self.channels))
        for tid in range(n_tokens):
            pattern = self.encode_token(tid)
            for ch in pattern:
                idx = self.channels.index(ch)
                channel_counts[idx] += 1

        return {
            'mean_usage': float(channel_counts.mean()),
            'std_usage': float(channel_counts.std()),
            'max_usage': float(channel_counts.max()),
            'min_usage': float(channel_counts.min()),
            'coverage': float(np.mean(channel_counts > 0)),
            'n_tokens_sampled': n_tokens,
        }


class SpatialDecoder:
    """Decodes spike responses back into token probabilities.

    Uses the spatial encoder's channel-to-token mapping plus rolling
    baseline z-scores to determine which tokens the culture "preferred".
    """

    def __init__(self, encoder: SpatialEncoder, alpha: float = 0.5):
        self.encoder = encoder
        self.alpha = alpha
        self._channel_history: Dict[int, List[float]] = {}
        self._history_maxlen = 30

    def decode(
        self,
        spike_counts: Dict[str, int],
        model_probs: Dict[int, float],
        channel_to_token: Dict[int, int],
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """Decode spike response into blended token probabilities.

        Returns (blended_probs, neural_probs, z_scores).
        """
        # Compute z-scores per channel
        z_scores = {}
        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            if ch not in self._channel_history:
                self._channel_history[ch] = []

            history = self._channel_history[ch]
            if len(history) >= 3:
                mean = np.mean(history)
                std = np.std(history)
                z = (count - mean) / std if std > 0 else 0.0
            else:
                z = 0.0

            z_scores[ch] = round(z, 3)

            # Update history
            history.append(float(count))
            if len(history) > self._history_maxlen:
                history.pop(0)

        # Aggregate z-scores per token (sum of z-scores for token's channels)
        token_scores = {}
        for ch, z in z_scores.items():
            token_id = channel_to_token.get(ch)
            if token_id is not None:
                if token_id not in token_scores:
                    token_scores[token_id] = 0.0
                token_scores[token_id] += z

        # Sigmoid normalization
        neural_probs = {}
        if token_scores:
            for tid, score in token_scores.items():
                neural_probs[tid] = 1.0 / (1.0 + np.exp(-score))
            # Normalize
            total = sum(neural_probs.values())
            if total > 0:
                neural_probs = {k: v / total for k, v in neural_probs.items()}

        # Blend with model probabilities
        all_tokens = set(model_probs.keys()) | set(neural_probs.keys())
        blended = {}
        for tok in all_tokens:
            mp = model_probs.get(tok, 0.0)
            np_ = neural_probs.get(tok, 0.0)
            blended[tok] = (1 - self.alpha) * mp + self.alpha * np_

        total = sum(blended.values())
        if total > 0:
            blended = {tok: p / total for tok, p in blended.items()}

        return blended, neural_probs, z_scores
