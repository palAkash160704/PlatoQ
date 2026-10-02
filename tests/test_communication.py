"""Tests for the CommunicationMessage model, V2VNetwork, and optimiser/manager interfaces."""

import pytest

from platooning.communication.v2v import V2VNetwork
from platooning.models.communication import CommunicationMessage, MessageType
from platooning.models.vehicle import VehicleState
from platooning.optimization.classical.optimizer import (
    BaseOptimizer,
    ClassicalOptimizer,
)
from platooning.optimization.quantum.quantum_optimizer import QuantumOptimizer
from platooning.optimization.qubo.formulation import QUBOOptimizer
from platooning.platooning.platoon_manager import PlatoonManager


# ======================================================================
# CommunicationMessage
# ======================================================================
class TestCommunicationMessage:
    """Verify CommunicationMessage creation and defaults."""

    def test_minimal_creation(self) -> None:
        msg = CommunicationMessage(sender_id="v1", receiver_id="v2")
        assert msg.sender_id == "v1"
        assert msg.receiver_id == "v2"
        assert msg.delivered is True

    def test_full_creation(self) -> None:
        msg = CommunicationMessage(
            sender_id="v1",
            receiver_id="broadcast",
            timestamp=5.0,
            message_type=MessageType.JOIN_REQUEST,
            payload={"platoon_id": "p1"},
            latency=0.05,
            delivered=False,
        )
        assert msg.message_type == MessageType.JOIN_REQUEST
        assert msg.payload["platoon_id"] == "p1"
        assert msg.delivered is False
        assert msg.latency == 0.05

    def test_default_message_type(self) -> None:
        msg = CommunicationMessage(sender_id="v1", receiver_id="v2")
        assert msg.message_type == MessageType.HEARTBEAT

    def test_message_type_enum_values(self) -> None:
        """All expected message types should be defined."""
        expected = {
            "heartbeat",
            "join_request",
            "join_response",
            "leave_notification",
            "platoon_update",
            "emergency",
            "state_broadcast",
            "vehicle_state",
        }
        actual = {mt.value for mt in MessageType}
        assert expected == actual


# ======================================================================
# V2VNetwork (Phase 2 Simulation Model)
# ======================================================================
class TestV2VNetwork:
    """Verify V2VNetwork simulated network behaviour."""

    @pytest.fixture()
    def network(self) -> V2VNetwork:
        return V2VNetwork(
            {
                "range_meters": 100,
                "latency_ms": 50,
                "packet_loss_rate": 0.0,
                "message_frequency_hz": 10,
                "stale_threshold_ms": 200,
            },
            seed=42,
        )

    @pytest.fixture()
    def v1(self) -> VehicleState:
        return VehicleState("v1", 10.0, 0.0, 0.0, speed=20)

    @pytest.fixture()
    def v2_in_range(self) -> VehicleState:
        return VehicleState("v2", 10.0, 80.0, 0.0, speed=20)

    @pytest.fixture()
    def v3_out_of_range(self) -> VehicleState:
        return VehicleState("v3", 10.0, 150.0, 0.0, speed=20)

    # --- Neighbour Discovery & Range ---
    def test_neighbours_within_range(
        self, network: V2VNetwork, v1, v2_in_range, v3_out_of_range
    ) -> None:
        neighbors = network.get_neighbors(
            v1.vehicle_id, [v1, v2_in_range, v3_out_of_range]
        )
        assert len(neighbors) == 1
        assert neighbors[0].vehicle_id == "v2"

    def test_sender_not_own_neighbour(self, network: V2VNetwork, v1) -> None:
        neighbors = network.get_neighbors(v1.vehicle_id, [v1])
        assert len(neighbors) == 0

    # --- Broadcasting ---
    def test_broadcast_generates_messages(
        self, network: V2VNetwork, v1, v2_in_range
    ) -> None:
        scheduled = network.broadcast(v1, [v1, v2_in_range], 10.0)
        assert scheduled == 1
        assert network.pending_count == 1

    def test_frequency_gating(self, network: V2VNetwork, v1, v2_in_range) -> None:
        # At 10Hz, broadcast interval is 0.1s
        network.broadcast(v1, [v1, v2_in_range], 10.0)
        # 10.05s is too soon
        scheduled = network.broadcast(v1, [v1, v2_in_range], 10.05)
        assert scheduled == 0
        # 10.1s is exactly on time
        scheduled = network.broadcast(v1, [v1, v2_in_range], 10.1)
        assert scheduled == 1

    # --- Latency & Delivery ---
    def test_latency_delay(self, network: V2VNetwork, v1, v2_in_range) -> None:
        # 50ms latency -> delivery time 10.05
        network.broadcast(v1, [v1, v2_in_range], 10.0)

        # At 10.0, not delivered
        delivered = network.deliver(10.0)
        assert len(delivered) == 0
        assert network.pending_count == 1

        # At 10.05, delivered
        delivered = network.deliver(10.05)
        assert len(delivered) == 1
        assert network.pending_count == 0
        assert delivered[0].receiver_id == "v2"

    # --- Packet Loss ---
    def test_packet_loss_drops_messages(self, v1, v2_in_range) -> None:
        net = V2VNetwork({"packet_loss_rate": 1.0, "message_frequency_hz": 10}, seed=42)
        scheduled = net.broadcast(v1, [v1, v2_in_range], 10.0)
        assert scheduled == 0
        assert net.pending_count == 0
        stats = net.get_statistics()
        assert stats.messages_sent == 1
        assert stats.messages_dropped == 1

    # --- Message Age & Latest State ---
    def test_latest_state_and_age(self, network: V2VNetwork, v1, v2_in_range) -> None:
        network.broadcast(v1, [v1, v2_in_range], 10.0)
        network.deliver(10.05)

        latest = network.get_latest_state("v2", "v1")
        assert latest is not None
        assert latest.timestamp == 10.0

        # At t=10.05, age = 0.05
        age = network.get_message_age(latest, 10.05)
        assert age == pytest.approx(0.05)

        # Threshold is 200ms
        assert not network.is_stale(latest, 10.05)
        assert network.is_stale(latest, 10.25)

    # --- Statistics ---
    def test_statistics(
        self, network: V2VNetwork, v1, v2_in_range, v3_out_of_range
    ) -> None:
        network.broadcast(v1, [v1, v2_in_range, v3_out_of_range], 10.0)
        network.deliver(10.05)
        stats = network.get_statistics()

        assert stats.messages_sent == 1
        assert stats.messages_delivered == 1
        assert stats.messages_out_of_range == 1
        assert stats.delivery_rate == 1.0
        assert stats.average_latency_ms == pytest.approx(50.0)

    # --- Legacy Phase 0 API tests ---
    def test_send_and_receive(self, network: V2VNetwork) -> None:
        msg = CommunicationMessage(sender_id="v1", receiver_id="v2")
        network.send(msg)
        assert network.pending_count == 1
        received = network.receive("v2")
        assert len(received) == 1
        assert received[0].sender_id == "v1"

    def test_receive_broadcast(self, network: V2VNetwork) -> None:
        msg = CommunicationMessage(sender_id="v1", receiver_id="broadcast")
        network.send(msg)
        received = network.receive("v99")
        assert len(received) == 1

    def test_no_messages_for_other(self, network: V2VNetwork) -> None:
        msg = CommunicationMessage(sender_id="v1", receiver_id="v2")
        network.send(msg)
        received = network.receive("v3")
        assert len(received) == 0

    def test_clear(self, network: V2VNetwork) -> None:
        network.send(CommunicationMessage(sender_id="v1", receiver_id="v2"))
        network.clear()
        assert network.pending_count == 0


# ======================================================================
# Optimizer interfaces
# ======================================================================
class TestOptimizerInterfaces:
    """Verify that all optimizer placeholders instantiate and share the interface."""

    def test_classical_optimizer_instantiation(self) -> None:
        opt = ClassicalOptimizer({"method": "classical", "random_seed": 42})
        assert isinstance(opt, BaseOptimizer)
        assert opt.get_solution() is None

    def test_qubo_optimizer_instantiation(self) -> None:
        opt = QUBOOptimizer()
        assert isinstance(opt, BaseOptimizer)
        assert opt.get_solution() is None

    def test_quantum_optimizer_instantiation(self) -> None:
        opt = QuantumOptimizer()
        assert isinstance(opt, BaseOptimizer)
        assert opt.get_solution() is None

    def test_classical_optimize_not_implemented(self) -> None:
        opt = ClassicalOptimizer({"method": "classical"})
        with pytest.raises(NotImplementedError):
            opt.optimize([])

    def test_qubo_optimize_not_implemented(self) -> None:
        opt = QUBOOptimizer()
        with pytest.raises(NotImplementedError):
            opt.optimize([])

    def test_quantum_optimize_not_implemented(self) -> None:
        opt = QuantumOptimizer()
        with pytest.raises(NotImplementedError):
            opt.optimize([])


# ======================================================================
# PlatoonManager
# ======================================================================
class TestPlatoonManager:
    """Verify PlatoonManager initialisation and interface."""

    def test_initialization(self) -> None:
        pm = PlatoonManager({"minimum_platoon_size": 2, "maximum_platoon_size": 10})
        assert pm.get_all_platoons() == []

    def test_get_platoon_returns_none_for_unknown(self) -> None:
        pm = PlatoonManager({"minimum_platoon_size": 2, "maximum_platoon_size": 10})
        assert pm.get_platoon("nonexistent") is None

    def test_form_platoon_not_implemented(self) -> None:
        pm = PlatoonManager({"minimum_platoon_size": 2, "maximum_platoon_size": 10})
        with pytest.raises(NotImplementedError):
            pm.form_platoon(["v1", "v2"], "v1")
