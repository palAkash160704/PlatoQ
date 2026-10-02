"""
Simulated V2V (Vehicle-to-Vehicle) communication network.

Provides a ``V2VNetwork`` that accepts vehicle states, generates broadcast
messages, applies communication range, latency, and packet loss, and
delivers messages to recipients based on simulation time.

Architecture::

    VehicleState[]
         │
         ▼
    V2VNetwork.broadcast()
         │
         ├── range check (Euclidean distance)
         ├── packet loss (seeded RNG)
         └── latency scheduling (delivery_time = send_time + latency)
         │
         ▼
    Pending delivery queue
         │
         ▼
    V2VNetwork.deliver()   (called at each sim step)
         │
         ▼
    Delivered messages  →  latest_known_states  →  statistics
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from platooning.communication.network_model import NetworkModel
from platooning.models.communication import CommunicationMessage, MessageType
from platooning.models.vehicle import VehicleState
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================================
# Communication statistics
# ======================================================================
@dataclass
class CommunicationStats:
    """Accumulated V2V communication statistics.

    Definitions
    -----------
    - **messages_sent**: Total transmission attempts.
    - **messages_delivered**: Successfully delivered messages.
    - **messages_dropped**: Messages lost to packet loss.
    - **messages_out_of_range**: Messages rejected due to distance.
    - **messages_stale**: Delivered messages whose age exceeded the
      stale threshold at their most recent access.
    - **total_latency**: Sum of measured latencies (for averaging).
    - **max_latency**: Maximum observed measured latency.
    """

    messages_sent: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    messages_out_of_range: int = 0
    messages_stale: int = 0
    total_latency: float = 0.0
    max_latency: float = 0.0

    @property
    def delivery_rate(self) -> float:
        """Fraction of sent messages that were delivered."""
        if self.messages_sent == 0:
            return 1.0
        return self.messages_delivered / self.messages_sent

    @property
    def measured_packet_loss_rate(self) -> float:
        """Fraction of sent messages that were dropped."""
        if self.messages_sent == 0:
            return 0.0
        return self.messages_dropped / self.messages_sent

    @property
    def average_latency_ms(self) -> float:
        """Average measured latency in milliseconds."""
        if self.messages_delivered == 0:
            return 0.0
        return (self.total_latency / self.messages_delivered) * 1000.0

    @property
    def max_latency_ms(self) -> float:
        """Maximum measured latency in milliseconds."""
        return self.max_latency * 1000.0

    @property
    def stale_message_rate(self) -> float:
        """Fraction of delivered messages marked stale."""
        if self.messages_delivered == 0:
            return 0.0
        return self.messages_stale / self.messages_delivered


# ======================================================================
# V2V Network
# ======================================================================
class V2VNetwork:
    """Simulated V2V communication layer.

    Parameters
    ----------
    config : dict
        The ``communication`` section of the project configuration.
    seed : int
        Random seed for reproducible packet-loss simulation.
    network_model : NetworkModel, optional
        Pre-built network model.  When *None*, one is created from
        *config* automatically.
    """

    def __init__(
        self,
        config: dict[str, Any],
        seed: int = 42,
        network_model: NetworkModel | None = None,
    ) -> None:
        self._config = config
        self._model = network_model or NetworkModel.from_config(config)
        self._rng = random.Random(seed)

        # Delivery queue: messages waiting for their delivery_time
        self._pending: list[CommunicationMessage] = []

        # Delivered inbox: receiver_id → list of delivered messages
        self._inbox: dict[str, list[CommunicationMessage]] = {}

        # Latest known vehicle state per (receiver, sender) pair
        self._latest_states: dict[str, dict[str, VehicleState]] = {}

        # Last broadcast time per vehicle (for frequency control)
        self._last_broadcast: dict[str, float] = {}

        # Accumulated statistics
        self._stats = CommunicationStats()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def model(self) -> NetworkModel:
        """Return the underlying ``NetworkModel``."""
        return self._model

    @property
    def pending_count(self) -> int:
        """Return the number of messages currently waiting for delivery."""
        return len(self._pending)

    # ------------------------------------------------------------------
    # Distance & neighbour helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _distance(a: VehicleState, b: VehicleState) -> float:
        """Euclidean distance between two vehicles."""
        dx = a.position_x - b.position_x
        dy = a.position_y - b.position_y
        return math.sqrt(dx * dx + dy * dy)

    def get_neighbors(
        self,
        vehicle_id: str,
        vehicle_states: list[VehicleState],
    ) -> list[VehicleState]:
        """Return vehicles within communication range of *vehicle_id*.

        The sender itself is never included.

        Parameters
        ----------
        vehicle_id : str
            The vehicle whose neighbours we want.
        vehicle_states : list[VehicleState]
            All current vehicle states.

        Returns
        -------
        list[VehicleState]
            Neighbouring vehicles within ``range_meters``.
        """
        sender_state: VehicleState | None = None
        for s in vehicle_states:
            if s.vehicle_id == vehicle_id:
                sender_state = s
                break

        if sender_state is None:
            return []

        neighbors: list[VehicleState] = []
        for s in vehicle_states:
            if s.vehicle_id == vehicle_id:
                continue
            if self._distance(sender_state, s) <= self._model.range_meters:
                neighbors.append(s)
        return neighbors

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------
    def broadcast(
        self,
        sender_state: VehicleState,
        all_vehicle_states: list[VehicleState],
        simulation_time: float,
    ) -> int:
        """Broadcast *sender_state* to all in-range neighbours.

        Applies communication range, packet loss, and latency scheduling.
        Messages are placed in the pending queue and delivered later
        via :meth:`deliver`.

        Parameters
        ----------
        sender_state : VehicleState
            The broadcasting vehicle's current state.
        all_vehicle_states : list[VehicleState]
            All current vehicle states (used for range calculation).
        simulation_time : float
            Current SUMO simulation time (seconds).

        Returns
        -------
        int
            Number of messages scheduled (before packet-loss drops).
        """
        # Frequency gating: skip if broadcast interval not yet elapsed
        last_t = self._last_broadcast.get(sender_state.vehicle_id, -float("inf"))
        interval = self._model.broadcast_interval
        if (simulation_time - last_t) < interval - 1e-9:
            return 0

        self._last_broadcast[sender_state.vehicle_id] = simulation_time

        neighbors = self.get_neighbors(sender_state.vehicle_id, all_vehicle_states)
        scheduled = 0

        for neighbor in neighbors:
            self._stats.messages_sent += 1

            # Packet loss check
            if self._rng.random() < self._model.packet_loss_rate:
                self._stats.messages_dropped += 1
                logger.debug(
                    "[V2V] %s → %s status=DROPPED reason=PACKET_LOSS",
                    sender_state.vehicle_id,
                    neighbor.vehicle_id,
                )
                continue

            latency = self._model.latency_seconds
            delivery_time = simulation_time + latency

            msg = CommunicationMessage(
                sender_id=sender_state.vehicle_id,
                receiver_id=neighbor.vehicle_id,
                timestamp=simulation_time,
                message_type=MessageType.VEHICLE_STATE,
                payload=sender_state,
                latency=latency,
                delivered=False,
                delivery_time=delivery_time,
            )

            self._pending.append(msg)
            scheduled += 1

            logger.debug(
                "[V2V] t=%.2f %s → %s distance=%.1fm latency=%.0fms "
                "delivery_time=%.2f status=SCHEDULED",
                simulation_time,
                sender_state.vehicle_id,
                neighbor.vehicle_id,
                self._distance(sender_state, neighbor),
                latency * 1000,
                delivery_time,
            )

        # Count out-of-range vehicles (for stats only)
        in_range_ids = {n.vehicle_id for n in neighbors}
        for s in all_vehicle_states:
            if s.vehicle_id == sender_state.vehicle_id:
                continue
            if s.vehicle_id not in in_range_ids:
                self._stats.messages_out_of_range += 1
                logger.debug(
                    "[V2V] %s → %s status=REJECTED reason=OUT_OF_RANGE",
                    sender_state.vehicle_id,
                    s.vehicle_id,
                )

        return scheduled

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------
    def deliver(self, simulation_time: float) -> list[CommunicationMessage]:
        """Deliver all pending messages whose delivery time has arrived.

        Parameters
        ----------
        simulation_time : float
            Current SUMO simulation time (seconds).

        Returns
        -------
        list[CommunicationMessage]
            Newly delivered messages.
        """
        ready: list[CommunicationMessage] = []
        still_pending: list[CommunicationMessage] = []

        for msg in self._pending:
            if msg.delivery_time <= simulation_time:
                msg.delivered = True
                ready.append(msg)

                # Update stats
                self._stats.messages_delivered += 1
                measured_latency = simulation_time - msg.timestamp
                self._stats.total_latency += measured_latency
                if measured_latency > self._stats.max_latency:
                    self._stats.max_latency = measured_latency

                # Update inbox
                self._inbox.setdefault(msg.receiver_id, []).append(msg)

                # Update latest known state
                if isinstance(msg.payload, VehicleState):
                    receiver_map = self._latest_states.setdefault(msg.receiver_id, {})
                    existing = receiver_map.get(msg.sender_id)
                    if existing is None or msg.timestamp > existing.timestamp:
                        receiver_map[msg.sender_id] = msg.payload

                logger.debug(
                    "[V2V] t=%.2f %s → %s status=DELIVERED age=%.0fms",
                    simulation_time,
                    msg.sender_id,
                    msg.receiver_id,
                    measured_latency * 1000,
                )
            else:
                still_pending.append(msg)

        self._pending = still_pending
        return ready

    # ------------------------------------------------------------------
    # Legacy send/receive (Phase 0 backward compatibility)
    # ------------------------------------------------------------------
    def send(self, message: CommunicationMessage) -> None:
        """Enqueue a message for delivery (Phase 0 compatibility).

        For Phase 2+ usage, prefer :meth:`broadcast` and :meth:`deliver`.
        """
        self._pending.append(message)
        logger.debug(
            "Message queued: %s → %s (%s)",
            message.sender_id,
            message.receiver_id,
            message.message_type.value,
        )

    def receive(self, vehicle_id: str) -> list[CommunicationMessage]:
        """Return all pending messages addressed to *vehicle_id*.

        For Phase 2+ usage, prefer :meth:`deliver` and
        :meth:`get_latest_state`.
        """
        delivered = [
            m
            for m in self._pending
            if m.receiver_id == vehicle_id or m.receiver_id == "broadcast"
        ]
        self._pending = [m for m in self._pending if m not in delivered]
        return delivered

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------
    def get_latest_state(
        self,
        receiver_id: str,
        sender_id: str,
    ) -> VehicleState | None:
        """Return the latest known state of *sender_id* as seen by *receiver_id*.

        Returns *None* if no state has been received.
        """
        return self._latest_states.get(receiver_id, {}).get(sender_id)

    def get_all_known_states(self, receiver_id: str) -> dict[str, VehicleState]:
        """Return all latest known states for *receiver_id*.

        Returns
        -------
        dict[str, VehicleState]
            Mapping ``sender_id → latest VehicleState``.
        """
        return dict(self._latest_states.get(receiver_id, {}))

    def is_stale(
        self,
        state: VehicleState,
        current_time: float,
    ) -> bool:
        """Return *True* if *state* is older than the stale threshold.

        Parameters
        ----------
        state : VehicleState
            A previously received vehicle state.
        current_time : float
            Current simulation time (seconds).
        """
        age = current_time - state.timestamp
        return age > self._model.stale_threshold_seconds

    def get_message_age(
        self,
        state: VehicleState,
        current_time: float,
    ) -> float:
        """Return the age of *state* in seconds."""
        return current_time - state.timestamp

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_statistics(self) -> CommunicationStats:
        """Return accumulated communication statistics."""
        return self._stats

    def count_stale_states(self, current_time: float) -> int:
        """Count how many latest-known states are currently stale."""
        count = 0
        for _receiver_id, sender_map in self._latest_states.items():
            for _sender_id, state in sender_map.items():
                if self.is_stale(state, current_time):
                    count += 1
        self._stats.messages_stale = count
        return count

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Discard all queued messages and reset state."""
        self._pending.clear()
        self._inbox.clear()

    def reset(self) -> None:
        """Full reset — clear all state including statistics."""
        self.clear()
        self._latest_states.clear()
        self._last_broadcast.clear()
        self._stats = CommunicationStats()
