"""
Network model for V2V communication simulation.

Encapsulates channel parameters that the ``V2VNetwork`` uses to decide
whether a message is delivered, delayed, or dropped.

Terminology
-----------
- **Configured latency**: The simulated delay applied to a message.
- **Measured latency**: Actual ``receive_time - send_time``.
- **Configured packet loss**: The probability used by the simulation.
- **Measured packet loss**: ``dropped_messages / attempted_messages``.
- **Message age**: ``current_simulation_time - original_vehicle_state_timestamp``.
- **Stale message**: A received message whose age exceeds ``stale_threshold_ms``.

.. note::
   All parameter values are **simulation assumptions** for research, not
   claims about actual production V2V systems.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NetworkModel:
    """Parameters describing the simulated communication channel.

    Attributes
    ----------
    range_meters : float
        Maximum communication range (metres).  Vehicles farther apart
        than this value cannot exchange messages.
    base_latency_ms : float
        Simulated one-way message latency (milliseconds).
    packet_loss_rate : float
        Probability that a message is dropped ``[0.0, 1.0]``.
    message_frequency_hz : float
        How often each vehicle broadcasts its state (Hz).
    stale_threshold_ms : float
        A received message is considered *stale* when its age exceeds
        this value (milliseconds).
    bandwidth_mbps : float
        Simulated channel bandwidth (Mbps).  Informational only.
    """

    range_meters: float = 100.0
    base_latency_ms: float = 50.0
    packet_loss_rate: float = 0.0
    message_frequency_hz: float = 10.0
    stale_threshold_ms: float = 200.0
    bandwidth_mbps: float = 10.0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def broadcast_interval(self) -> float:
        """Return the interval between broadcasts (seconds)."""
        if self.message_frequency_hz <= 0:
            return float("inf")
        return 1.0 / self.message_frequency_hz

    @property
    def latency_seconds(self) -> float:
        """Return the base latency in seconds."""
        return self.base_latency_ms / 1000.0

    @property
    def stale_threshold_seconds(self) -> float:
        """Return the stale-message threshold in seconds."""
        return self.stale_threshold_ms / 1000.0

    @classmethod
    def from_config(cls, config: dict) -> NetworkModel:
        """Create a ``NetworkModel`` from the ``communication`` config section.

        Parameters
        ----------
        config : dict
            The ``communication`` section of the project YAML.
        """
        return cls(
            range_meters=float(config.get("range_meters", 100.0)),
            base_latency_ms=float(config.get("latency_ms", 50.0)),
            packet_loss_rate=float(config.get("packet_loss_rate", 0.0)),
            message_frequency_hz=float(config.get("message_frequency_hz", 10.0)),
            stale_threshold_ms=float(config.get("stale_threshold_ms", 200.0)),
        )
