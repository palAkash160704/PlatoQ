# System Architecture

## Overview

The system is structured as a **layered pipeline** where each layer has a well-defined responsibility and communicates with adjacent layers through explicit interfaces (function calls on data models / abstract base classes).

```
┌────────────────────────────┐
│   SUMO Traffic Simulator   │
└────────────┬───────────────┘
             │  TraCI
┌────────────▼───────────────┐
│     Simulation Layer       │  simulator.py, traci_manager.py
│  (BaseSimulator interface) │
└────────────┬───────────────┘
             │  list[VehicleState]
┌────────────▼───────────────┐
│    Vehicle State Layer     │  models/vehicle.py
└────────────┬───────────────┘
             │
┌────────────▼───────────────┐
│  V2V Communication Layer   │  communication/v2v.py
│  (simulated wireless)      │  communication/network_model.py
└────────────┬───────────────┘
             │  CommunicationMessage
┌────────────▼───────────────┐
│      Platoon Manager       │  platooning/platoon_manager.py
│  (lifecycle orchestrator)  │  platooning/formation.py, rules.py
└────────────┬───────────────┘
             │  list[VehicleState]
┌────────────▼───────────────┐
│    Optimization Layer      │
│  ┌──────────────────────┐  │
│  │ ClassicalOptimizer   │  │  optimization/classical/
│  ├──────────────────────┤  │
│  │ QUBOOptimizer        │  │  optimization/qubo/
│  ├──────────────────────┤  │
│  │ QuantumOptimizer     │  │  optimization/quantum/
│  └──────────────────────┘  │
│  All implement             │
│  BaseOptimizer interface   │
└────────────┬───────────────┘
             │  solution (platoon assignments)
┌────────────▼───────────────┐
│  Cooperative Control Layer │  control/cooperative_controller.py
│  (CACC / vehicle-following)│
└────────────┬───────────────┘
             │  TraCI commands
┌────────────▼───────────────┐
│   SUMO Traffic Simulator   │
└────────────────────────────┘
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `config/settings.py` | Load YAML config, merge environment variables, validate |
| `models/vehicle.py` | `VehicleState` dataclass — per-vehicle snapshot |
| `models/platoon.py` | `Platoon` dataclass — platoon membership and metadata |
| `models/communication.py` | `CommunicationMessage` dataclass + `MessageType` enum |
| `simulation/simulator.py` | `BaseSimulator` ABC + `SUMOSimulator` concrete class |
| `simulation/traci_manager.py` | Low-level TraCI connection lifecycle |
| `communication/v2v.py` | `V2VNetwork` — message queuing, send/receive |
| `communication/network_model.py` | `NetworkModel` — channel parameters |
| `platooning/platoon_manager.py` | `PlatoonManager` — platoon lifecycle orchestration |
| `platooning/formation.py` | Candidate evaluation for platoon formation |
| `platooning/rules.py` | `PlatooningRules` — configurable constraints |
| `optimization/classical/optimizer.py` | `BaseOptimizer` ABC + `ClassicalOptimizer` |
| `optimization/qubo/` | QUBO variable mapping, objective, formulation |
| `optimization/quantum/` | QAOA solver + `QuantumOptimizer` |
| `control/cooperative_controller.py` | `CooperativeController` — vehicle-following |
| `experiments/` | Scenario definition, experiment runner, metrics |
| `visualization/plots.py` | Plotting utilities |
| `utils/logger.py` | Centralised logging setup |
| `utils/helpers.py` | Small shared utility functions |

---

## Key Design Decisions

### 1. Simulator Independence

The `BaseSimulator` abstract class ensures that the rest of the system never depends on SUMO/TraCI directly.  If CARLA or another simulator is introduced later, only a new `BaseSimulator` subclass is needed.

### 2. Optimizer Interchangeability

All optimizers implement the same `BaseOptimizer` interface (`optimize()`, `evaluate()`, `get_solution()`).  This allows the `PlatoonManager` to switch between classical and quantum backends **without code changes** — only a configuration change is needed.

### 3. Separation of Optimization and Control

- **Optimization** answers: *which vehicles should platoon together?*
- **Control** answers: *how does each vehicle follow the one ahead?*

This separation means the QUBO/QAOA formulation is tested against the same control law as the classical baseline, ensuring a fair comparison.

### 4. Data Flow via Dataclasses

All inter-layer data is exchanged as Python dataclasses (`VehicleState`, `Platoon`, `CommunicationMessage`).  This keeps interfaces explicit and testable without requiring a message broker or complex serialization.

---

## Data Flow

1. **SUMO** simulates traffic and advances by `step_length` seconds per step.
2. **TraCIManager** queries SUMO for all vehicle properties.
3. **SUMOSimulator** packages the raw data into `list[VehicleState]`.
4. **V2VNetwork** simulates message exchange between vehicles.
5. **PlatoonManager** receives vehicle states and communication messages, evaluates whether reconfiguration is needed, and invokes the optimizer.
6. **BaseOptimizer.optimize()** receives `list[VehicleState]` and returns a solution (platoon assignments).
7. **PlatoonManager** applies the solution — creating, merging, or splitting platoons.
8. **CooperativeController** computes the acceleration for each follower vehicle.
9. **TraCIManager** sends speed/acceleration commands back to SUMO.
10. Loop continues until the simulation ends.

---

## Phase 3 — Operational Data Flow

The following pipeline is **fully implemented and tested** as of Phase 3:

```text
SUMO (eclipse-sumo 1.27.1)
  │
  ▼
TraCIManager
  │
  ▼
SUMOSimulator
  │
  ▼
VehicleState
  │
  ▼
V2VNetwork
  │   └── NetworkModel (range, latency, packet loss, freq)
  ▼
Neighbour State (Latest Known VehicleState)
  │
  ▼
PlatoonManager
  │   └── Compatibility Evaluation (distance, speed, route, lane, staleness)
  │   └── Deterministic Classical Formation Algorithm
  ▼
Platoon Configuration + Formation Metrics
  │
  ▼
Application Logger (compact periodic output)
```

Steps 6, 8, 9 (Optimizer, Controller) remain as stubs and will be connected in subsequent phases.

