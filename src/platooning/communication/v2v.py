"""
Simulated V2V (Vehicle-to-Vehicle) communication network.

Provides a ``V2VNetwork`` that accepts messages, applies simulated latency
and packet loss, and delivers them to recipients within communication range.

.. todo:: Phase 2 — implement full message routing with position-based
   range checks and configurable network models.
"""

from __future__ import annotations

from typing import Any

from platooning.models.communication import CommunicationMessage
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


class V2VNetwork:
    """Simulated V2V communication layer.

    Parameters
    ----------
    config : dict
        The ``communication`` section of the project configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self.range_meters: int = int(config.get("range_meters", 100))
        self.latency_ms: int = int(config.get("latency_ms", 50))
        self._packet_loss_rate: float = float(config.get("packet_loss_rate", 0.0))
        self._message_freq_hz: float = float(config.get("message_frequency_hz", 10))
        self._message_queue: list[CommunicationMessage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send(self, message: CommunicationMessage) -> None:
        """Enqueue a message for delivery.

        .. todo:: Phase 2 — apply range check, latency model, and
           probabilistic packet loss before delivery.
        """
        # TODO — Phase 2: implement range / loss filtering.
        self._message_queue.append(message)
        logger.debug(
            "Message queued: %s → %s (%s)",
            message.sender_id,
            message.receiver_id,
            message.message_type.value,
        )

    def receive(self, vehicle_id: str) -> list[CommunicationMessage]:
        """Return all pending messages addressed to *vehicle_id*.

        .. todo:: Phase 2 — implement realistic delivery semantics.
        """
        delivered = [
            m
            for m in self._message_queue
            if m.receiver_id == vehicle_id or m.receiver_id == "broadcast"
        ]
        # Remove delivered messages from the queue
        self._message_queue = [m for m in self._message_queue if m not in delivered]
        return delivered

    def clear(self) -> None:
        """Discard all queued messages."""
        self._message_queue.clear()

    @property
    def pending_count(self) -> int:
        """Return the number of messages currently waiting for delivery."""
        return len(self._message_queue)
