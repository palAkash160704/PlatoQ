"""
V2V communication message data model.

Represents a single message exchanged between vehicles (or between a vehicle
and the platoon manager) over the simulated V2V communication channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(Enum):
    """Enumeration of supported V2V message types."""

    HEARTBEAT = "heartbeat"
    JOIN_REQUEST = "join_request"
    JOIN_RESPONSE = "join_response"
    LEAVE_NOTIFICATION = "leave_notification"
    PLATOON_UPDATE = "platoon_update"
    EMERGENCY = "emergency"
    STATE_BROADCAST = "state_broadcast"
    VEHICLE_STATE = "vehicle_state"


@dataclass
class CommunicationMessage:
    """A single V2V communication message.

    Attributes
    ----------
    sender_id : str
        Vehicle (or manager) ID that originated the message.
    receiver_id : str
        Intended recipient vehicle (or manager) ID.  Use ``"broadcast"``
        for messages intended for all neighbours within range.
    timestamp : float
        Simulation time at which the message was created (seconds).
    message_type : MessageType
        Semantic type of the message.
    payload : Any
        Message payload.  For ``VEHICLE_STATE`` messages this is a
        ``VehicleState`` instance; for other types it may be a dict.
    latency : float
        Simulated transmission latency (seconds).
    delivered : bool
        Whether the message was successfully delivered (accounts for
        simulated packet loss).
    delivery_time : float
        Simulation time at which the message becomes deliverable
        (``timestamp + latency``).  Set automatically by the network.
    """

    sender_id: str
    receiver_id: str
    timestamp: float = 0.0
    message_type: MessageType = MessageType.HEARTBEAT
    payload: Any = field(default_factory=dict)
    latency: float = 0.0
    delivered: bool = True
    delivery_time: float = 0.0
