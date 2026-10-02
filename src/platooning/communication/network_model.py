"""
Network model for V2V communication simulation.

Encapsulates channel parameters (delay distribution, loss model, bandwidth)
that the ``V2VNetwork`` uses to decide whether a message is delivered.

.. todo:: Phase 2 — implement probabilistic channel models (e.g.
   log-distance path loss, Nakagami fading).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NetworkModel:
    """Parameters describing the simulated communication channel.

    Attributes
    ----------
    range_meters : float
        Maximum communication range (metres).
    base_latency_ms : float
        Minimum one-way latency (milliseconds).
    packet_loss_rate : float
        Probability that a message is dropped [0.0, 1.0].
    bandwidth_mbps : float
        Simulated channel bandwidth (Mbps).  Informational for Phase 0.
    """

    range_meters: float = 100.0
    base_latency_ms: float = 50.0
    packet_loss_rate: float = 0.0
    bandwidth_mbps: float = 10.0
