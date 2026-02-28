"""CL1 Hardware Substrate — Replaces local Izhikevich with real neurons.

Communicates with the CL1 Voting Relay Server (cl1_voting_relay.py) running
on the CL1 device via HTTP. Provides the same interface as IzhikevichSubstrate
so it can be used as a drop-in replacement in TokenVotingEngineV2.

The relay must be running on the CL1 device:
    %run cl1_voting_relay.py

Then this adapter sends stimulation patterns and receives spike counts via HTTP.
"""

import time
import json
import numpy as np
from typing import Dict, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

RESERVED_CHANNELS = frozenset({0, 4, 7, 56, 63})
N_CHANNELS = 59  # 64 - 5 reserved


class CL1Substrate:
    """CL1 hardware substrate adapter.

    Drop-in replacement for IzhikevichSubstrate. Routes stimulation to
    real biological neurons via the CL1 Voting Relay HTTP server.
    """

    def __init__(
        self,
        relay_url: str = "http://localhost:8765",
        window_s: float = 0.5,
        seed: int = 42,
        timeout_s: float = 10.0,
    ):
        self.relay_url = relay_url.rstrip('/')
        self.window_s = window_s
        self.timeout_s = timeout_s
        self.seed = seed

        # Track state for compatibility with IzhikevichSubstrate interface
        self._last_spike_counts = {}
        self._last_spike_matrix = None
        self._initial_weights = None  # CL1 doesn't expose weights
        self._stim_count = 0
        self._total_spikes = 0

        # Check connection
        self._connected = False
        self._check_connection()

    def _check_connection(self):
        """Verify the CL1 relay is reachable."""
        try:
            resp = urlopen(f"{self.relay_url}/health", timeout=self.timeout_s)
            data = json.loads(resp.read().decode())
            if data.get('status') == 'ok':
                self._connected = True
                print(f"  [CL1] Connected to relay at {self.relay_url}")
                print(f"  [CL1] Neurons: {data.get('neurons_connected', '?')}")
                print(f"  [CL1] Tick rate: {data.get('tick_rate_hz', '?')} Hz")
                return True
        except (URLError, TimeoutError, ConnectionRefusedError) as e:
            print(f"  [CL1] WARNING: Cannot reach relay at {self.relay_url}: {e}")
            print(f"  [CL1] Make sure cl1_voting_relay.py is running on the CL1 device")
            self._connected = False
        return False

    def _post_json(self, endpoint: str, data: dict) -> dict:
        """POST JSON to the relay and return response."""
        url = f"{self.relay_url}{endpoint}"
        body = json.dumps(data).encode('utf-8')
        req = Request(url, data=body, headers={'Content-Type': 'application/json'})
        try:
            resp = urlopen(req, timeout=self.timeout_s)
            return json.loads(resp.read().decode())
        except (URLError, TimeoutError) as e:
            print(f"  [CL1] HTTP error on {endpoint}: {e}")
            return {'spike_counts': {}, 'error': str(e)}

    def stimulate_and_record(
        self,
        channel_amplitudes: Dict[int, float],
        window_s: float = None,
    ) -> Dict[str, int]:
        """Stimulate CL1 channels and record spike response.

        Parameters match IzhikevichSubstrate.stimulate_and_record().
        """
        if window_s is None:
            window_s = self.window_s

        # Build candidates list for the relay
        candidates = []
        for ch, amp in channel_amplitudes.items():
            ch = int(ch)
            if ch in RESERVED_CHANNELS:
                continue
            # Clamp amplitude to safe range
            amp = max(0.3, min(2.5, float(amp)))
            candidates.append({'channel': ch, 'amplitude': amp})

        if not candidates:
            return {}

        # Send to CL1 relay
        result = self._post_json('/vote', {
            'candidates': candidates,
            'window_s': window_s,
        })

        spike_counts = result.get('spike_counts', {})
        self._last_spike_counts = spike_counts
        self._stim_count += 1
        self._total_spikes += sum(spike_counts.values())

        # Build a fake spike matrix for consciousness assessment
        # Shape: (n_timesteps, n_channels) — we approximate from counts
        n_bins = max(20, int(window_s * 200))  # 200 Hz effective
        matrix = np.zeros((n_bins, N_CHANNELS), dtype=float)

        # Distribute spikes across time bins (Poisson approximation)
        rng = np.random.default_rng(self.seed + self._stim_count)
        usable_channels = [i for i in range(64) if i not in RESERVED_CHANNELS]
        ch_to_idx = {ch: idx for idx, ch in enumerate(usable_channels)}

        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            idx = ch_to_idx.get(ch)
            if idx is not None and idx < N_CHANNELS:
                # Distribute count spikes across time bins
                if count > 0:
                    spike_times = rng.choice(n_bins, size=min(count, n_bins * 2), replace=True)
                    for t in spike_times:
                        matrix[t, idx] += 1.0

        self._last_spike_matrix = matrix

        return spike_counts

    def get_last_spike_matrix(self) -> np.ndarray:
        """Return the last spike matrix (T, N) for consciousness assessment."""
        if self._last_spike_matrix is not None:
            return self._last_spike_matrix
        return np.zeros((20, N_CHANNELS), dtype=float)

    def get_weight_divergence(self) -> Dict[str, float]:
        """CL1 doesn't expose synaptic weights — return placeholder."""
        return {
            'frobenius_divergence': 0.0,
            'fractional_change': 0.0,
            'pct_weights_changed': 0.0,
        }

    def get_state_snapshot(self) -> Dict:
        """Return current substrate state."""
        return {
            'type': 'CL1_hardware',
            'relay_url': self.relay_url,
            'connected': self._connected,
            'total_stim': self._stim_count,
            'total_spikes': self._total_spikes,
        }

    def collect_baseline(self, duration_s: float = 5.0) -> Dict:
        """Collect spontaneous baseline activity from CL1."""
        return self._post_json('/baseline', {'duration_s': duration_s})

    def scan_channels(self) -> Dict:
        """Scan for active channels on CL1."""
        try:
            resp = urlopen(f"{self.relay_url}/channels", timeout=self.timeout_s)
            return json.loads(resp.read().decode())
        except (URLError, TimeoutError) as e:
            return {'error': str(e)}

    @property
    def is_connected(self) -> bool:
        return self._connected
