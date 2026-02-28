"""CL1 Cloud Substrate — Routes stimulations through Cloudflare Access + Jupyter kernel.

The CL1 relay runs on localhost:8765 inside the CL1 device, but our machine
can only reach it through the Cloudflare-protected Jupyter WebSocket. This
adapter executes Python code on a CL1 Jupyter kernel to call the relay locally.

Architecture:
    [Local machine] --WSS--> [Cloudflare Access] --WSS--> [Jupyter kernel on CL1]
    Jupyter kernel executes: requests.post("http://localhost:8765/vote", ...)

Latency: ~300-800ms per vote (WebSocket roundtrip), acceptable for real-time demo.
"""

import json
import ssl
import subprocess
import time
import uuid
import numpy as np
from typing import Dict, Optional

try:
    import websocket
except ImportError:
    raise ImportError("pip install websocket-client")

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# Create unverified SSL context for Cloudflare proxy
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

RESERVED_CHANNELS = frozenset({0, 4, 7, 56, 63})
N_CHANNELS = 59  # 64 - 5 reserved


class CL1CloudSubstrate:
    """CL1 substrate via Cloudflare Access + Jupyter kernel proxy.

    Drop-in replacement for IzhikevichSubstrate. Sends stimulation to real
    biological neurons on CL1 hardware via Jupyter kernel execution.
    """

    def __init__(
        self,
        cl1_host: str = "cl1-2544-015.device.cloud.corticallabs-test.com",
        kernel_id: str = None,
        window_s: float = 0.5,
        seed: int = 42,
        timeout_s: float = 15.0,
    ):
        self.cl1_host = cl1_host
        self.window_s = window_s
        self.timeout_s = timeout_s
        self.seed = seed

        # State tracking
        self._last_spike_counts = {}
        self._last_spike_matrix = None
        self._stim_count = 0
        self._total_spikes = 0

        # Get CF Access token
        print(f"  [CL1-Cloud] Authenticating via Cloudflare Access...")
        self._cf_token = self._get_cf_token()
        self._xsrf = self._get_xsrf_token()

        # Find or use specified kernel
        if kernel_id is None:
            kernel_id = self._find_kernel()
        self._kernel_id = kernel_id
        print(f"  [CL1-Cloud] Using kernel: {kernel_id}")

        # Connect to kernel WebSocket
        self._ws = None
        self._connect_kernel()

        # Verify relay is reachable from CL1
        self._connected = self._verify_relay()

    def _get_cf_token(self) -> str:
        """Get Cloudflare Access JWT token."""
        try:
            token = subprocess.check_output(
                ["cloudflared", "access", "token", "--app", f"https://{self.cl1_host}"],
                stderr=subprocess.DEVNULL, timeout=10
            ).decode().strip()
            print(f"  [CL1-Cloud] CF token obtained ({len(token)} chars)")
            return token
        except Exception as e:
            raise RuntimeError(f"Failed to get CF Access token: {e}")

    def _get_xsrf_token(self) -> str:
        """Get Jupyter XSRF token via curl (reliable through Cloudflare)."""
        result = subprocess.run(
            ["curl", "-L", "-c", "-", "-s",
             "-H", f"CF-Access-Token: {self._cf_token}",
             f"https://{self.cl1_host}/_/jupyter/"],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.split('\n'):
            if '_xsrf' in line:
                parts = line.split('\t')
                if len(parts) >= 7:
                    return parts[-1].strip()
        raise RuntimeError("Could not obtain XSRF token")

    def _find_kernel(self) -> str:
        """Find a running Jupyter kernel via curl."""
        result = subprocess.run(
            ["curl", "-s",
             "-H", f"CF-Access-Token: {self._cf_token}",
             f"https://{self.cl1_host}/_/jupyter/api/kernels"],
            capture_output=True, text=True, timeout=15
        )
        kernels = json.loads(result.stdout)
        if not kernels:
            raise RuntimeError("No Jupyter kernels running on CL1")
        for k in kernels:
            if k.get('execution_state') in ('idle', 'busy'):
                return k['id']
        return kernels[0]['id']

    def _connect_kernel(self):
        """Connect to Jupyter kernel via WebSocket."""
        ws_url = f"wss://{self.cl1_host}/_/jupyter/api/kernels/{self._kernel_id}/channels"
        headers = {
            "CF-Access-Token": self._cf_token,
            "Cookie": f"_xsrf={self._xsrf}",
        }
        print(f"  [CL1-Cloud] Connecting to kernel WebSocket...")
        self._ws = websocket.create_connection(
            ws_url,
            header=headers,
            sslopt={"cert_reqs": ssl.CERT_NONE},
            timeout=self.timeout_s,
        )
        print(f"  [CL1-Cloud] Kernel WebSocket connected!")

    def _execute_on_kernel(self, code: str, timeout: float = None) -> str:
        """Execute Python code on the CL1 Jupyter kernel and return stdout."""
        if timeout is None:
            timeout = self.timeout_s

        msg_id = str(uuid.uuid4())
        msg = {
            "header": {
                "msg_id": msg_id,
                "username": "antekythera",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "buffers": [],
            "channel": "shell",
        }

        self._ws.send(json.dumps(msg))

        # Collect output
        stdout_parts = []
        error_parts = []
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                self._ws.settimeout(max(0.1, deadline - time.time()))
                raw = self._ws.recv()
                reply = json.loads(raw)

                parent_id = reply.get("parent_header", {}).get("msg_id")
                if parent_id != msg_id:
                    continue

                msg_type = reply.get("header", {}).get("msg_type", "")

                if msg_type == "stream":
                    stdout_parts.append(reply["content"].get("text", ""))
                elif msg_type == "execute_result":
                    data = reply["content"].get("data", {})
                    stdout_parts.append(data.get("text/plain", ""))
                elif msg_type == "error":
                    tb = reply["content"].get("traceback", [])
                    error_parts.append("\n".join(tb))
                elif msg_type == "execute_reply":
                    # Execution complete
                    status = reply["content"].get("status", "")
                    if status == "error" and not error_parts:
                        error_parts.append(reply["content"].get("evalue", "unknown error"))
                    break

            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                error_parts.append(str(e))
                break

        if error_parts:
            return f"ERROR: {''.join(error_parts)}"
        return "".join(stdout_parts)

    def _verify_relay(self) -> bool:
        """Verify the relay is running on the CL1."""
        result = self._execute_on_kernel("""
import requests
try:
    r = requests.get("http://localhost:8765/health", timeout=5)
    d = r.json()
    print(f"RELAY_OK|{d.get('total_votes',0)}|{d.get('total_spikes',0)}|{d.get('tick_rate_hz',0)}")
except Exception as e:
    print(f"RELAY_FAIL|{e}")
""")
        if "RELAY_OK" in result:
            parts = result.strip().split("|")
            print(f"  [CL1-Cloud] Relay verified! Votes: {parts[1]}, Spikes: {parts[2]}, Rate: {parts[3]} Hz")
            return True
        else:
            print(f"  [CL1-Cloud] WARNING: Relay not reachable: {result}")
            return False

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

        # Build candidates list
        candidates = []
        for ch, amp in channel_amplitudes.items():
            ch = int(ch)
            if ch in RESERVED_CHANNELS:
                continue
            amp = max(0.3, min(2.5, float(amp)))
            candidates.append({'channel': ch, 'amplitude': amp})

        if not candidates:
            return {}

        # Execute vote on CL1 via kernel
        cand_json = json.dumps(candidates)
        code = f"""
import requests, json
try:
    r = requests.post("http://localhost:8765/vote",
        json={{"candidates": {cand_json}, "window_s": {window_s}}},
        timeout=10)
    d = r.json()
    sc = d.get("spike_counts", {{}})
    print("VOTE_OK|" + json.dumps(sc))
except Exception as e:
    print(f"VOTE_FAIL|{{e}}")
"""
        result = self._execute_on_kernel(code, timeout=window_s + 10)

        spike_counts = {}
        if "VOTE_OK|" in result:
            json_part = result.split("VOTE_OK|", 1)[1].strip()
            try:
                spike_counts = json.loads(json_part)
            except json.JSONDecodeError:
                pass

        self._last_spike_counts = spike_counts
        self._stim_count += 1
        self._total_spikes += sum(spike_counts.values())

        # Build spike matrix for consciousness assessment
        n_bins = max(20, int(window_s * 200))
        matrix = np.zeros((n_bins, N_CHANNELS), dtype=float)

        usable_channels = [i for i in range(64) if i not in RESERVED_CHANNELS]
        ch_to_idx = {ch: idx for idx, ch in enumerate(usable_channels)}

        rng = np.random.default_rng(self.seed + self._stim_count)
        for ch_str, count in spike_counts.items():
            ch = int(ch_str)
            idx = ch_to_idx.get(ch)
            if idx is not None and idx < N_CHANNELS:
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
        """CL1 doesn't expose weights — return placeholder."""
        return {
            'frobenius_divergence': 0.0,
            'fractional_change': 0.0,
            'pct_weights_changed': 0.0,
        }

    def get_state_snapshot(self) -> Dict:
        """Return current substrate state."""
        return {
            'type': 'CL1_cloud_hardware',
            'cl1_host': self.cl1_host,
            'kernel_id': self._kernel_id,
            'connected': self._connected,
            'total_stim': self._stim_count,
            'total_spikes': self._total_spikes,
        }

    def close(self):
        """Close the WebSocket connection."""
        if self._ws:
            try:
                self._ws.close()
            except:
                pass
            self._ws = None

    @property
    def is_connected(self) -> bool:
        return self._connected
