"""Tests for the CommunicationMessage model, V2VNetwork, and optimiser/manager interfaces."""

import pytest

from platooning.communication.v2v import V2VNetwork
from platooning.models.communication import CommunicationMessage, MessageType
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
        }
        actual = {mt.value for mt in MessageType}
        assert expected == actual


# ======================================================================
# V2VNetwork
# ======================================================================
class TestV2VNetwork:
    """Verify basic V2VNetwork send/receive behaviour."""

    @pytest.fixture()
    def network(self) -> V2VNetwork:
        return V2VNetwork(
            {
                "range_meters": 100,
                "latency_ms": 50,
                "packet_loss_rate": 0.0,
                "message_frequency_hz": 10,
            }
        )

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
