# Hybrid Quantum-Classical Optimization for Cooperative Vehicle Platooning

## Project Overview

This project investigates whether **hybrid quantum-classical optimization** can improve cooperative vehicle platoon management in connected vehicle environments.

**Key clarifications:**

- **Fully autonomous vehicles are NOT assumed.** The project focuses on **connected / cooperative vehicles** that communicate via simulated V2V (Vehicle-to-Vehicle) messaging.
- **SUMO** is the primary traffic simulation environment.
- Quantum optimization is applied to the **higher-level platoon management** problem (e.g. vehicle-to-platoon assignment, platoon merging/splitting), **not** to low-level vehicle control.
- Conventional cooperative control (e.g. CACC) handles vehicle-following and gap maintenance.
- Classical optimization serves as the **baseline** for comparison.

---

## Research Problem

Cooperative vehicle platooning — where groups of connected vehicles travel together at coordinated speeds and close headways — offers benefits in fuel efficiency, road capacity, and safety.  However, the **combinatorial** nature of platoon formation, vehicle assignment, merging, and reconfiguration decisions grows challenging as the number of vehicles and constraints increases.

This project explores whether formulating these decisions as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem and solving them with the **Quantum Approximate Optimization Algorithm (QAOA)** can offer competitive or superior solutions compared to classical methods.

---

## Planned System Architecture

```
SUMO Traffic Simulator
        ↓
  Simulation Layer (TraCI)
        ↓
  Vehicle State Layer
        ↓
  V2V Communication Layer
        ↓
  Platoon Manager
        ↓
  Optimization Layer
  ├── Classical Optimizer (baseline)
  ├── QUBO Formulation
  └── Quantum / Hybrid Optimizer (QAOA)
        ↓
  Cooperative Control Layer (CACC)
        ↓
  SUMO Traffic Simulator
```

---

## Planned Optimization Approaches

| Approach | Phase | Description |
|---|---|---|
| **Classical baseline** | Phase 4 | Greedy / ILP / OR-Tools solver |
| **QUBO formulation** | Phase 5 | Map platoon assignment to binary optimization |
| **QAOA** | Phase 6 | Solve QUBO on a simulated quantum backend |
| **Hybrid quantum-classical** | Phase 6 | Combine classical pre/post-processing with QAOA |

> **Note:** QUBO and QAOA are planned for **future phases**. Phase 0 establishes only the project skeleton and interfaces.

---

## Evaluation Plan

The classical and hybrid quantum-classical approaches will eventually be compared on:

| Metric | Description |
|---|---|
| Solution quality | Objective function value |
| Computation time | Wall-clock optimiser runtime |
| Scalability | Performance as vehicle count grows |
| Platoon formation time | Steps until stable platoons form |
| Safety metrics | Gap violations, emergency braking events |
| Communication robustness | Sensitivity to packet loss / latency |
| Efficiency proxy | Fuel consumption estimate based on headway |

---

## Technology Stack

- **Python 3.11+**
- **SUMO** (traffic simulation) + **TraCI** (vehicle state extraction)
- **NumPy / SciPy / pandas** (scientific computing)
- **OR-Tools** (classical optimization — Phase 4)
- **Qiskit** (quantum optimization — Phase 6)
- **matplotlib** (visualization)
- **pytest** (testing)
- **ruff + black** (code quality)
- **YAML** (configuration)

---

## Current Status

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Project setup & software architecture | ✅ Complete |
| **Phase 1** | SUMO simulation & TraCI integration | ✅ Complete |
| **Phase 2** | V2V communication simulation | ✅ Complete |
| Phase 3 | Platoon management & formation | 🔲 Planned |
| Phase 4 | Classical optimization | 🔲 Planned |
| Phase 5 | QUBO formulation | 🔲 Planned |
| Phase 6 | Quantum / hybrid optimization (QAOA) | 🔲 Planned |
| Phase 7 | Cooperative control (CACC) | 🔲 Planned |
| Phase 8 | Experiments & evaluation | 🔲 Planned |
| Phase 9 | Visualization & dashboard | 🔲 Planned |

---

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url> && cd quantum-platooning

# 2. Create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements-dev.txt

# 4. Install SUMO (pip method — recommended)
pip install eclipse-sumo traci sumolib

# 5. Run the simulation
python -m platooning.main

# 6. Run the simulation with SUMO-GUI
python -m platooning.main --gui

# 7. Run the test suite
pytest
```

---

## Phase 1 — SUMO Simulation

SUMO is the traffic simulator. TraCI connects Python to SUMO.
Vehicle states are extracted from SUMO into `VehicleState` objects.

**Current state:**
- 5 connected/cooperative vehicles (V1–V5) run on a 2 km straight road
- Vehicle states (position, speed, acceleration, lane, route) are extracted in real-time
- Vehicles are **not** yet forming optimised platoons — that belongs to later phases
- Periodic compact logging shows vehicle state snapshots

**Example output:**
```
[SIM] t=20.0s | vehicles=5
  V1   | x=  443.54 | y= -4.80 | speed=31.61 | accel= +0.00 | lane=main_road_0
  V2   | x=  376.41 | y= -4.80 | speed=33.33 | accel= +0.00 | lane=main_road_0
  V3   | x=  296.42 | y= -4.80 | speed=33.33 | accel= +0.00 | lane=main_road_0
  V4   | x=  213.68 | y= -4.80 | speed=29.37 | accel= +0.00 | lane=main_road_0
  V5   | x=  144.36 | y= -4.80 | speed=26.78 | accel= +2.60 | lane=main_road_0
```

---

## Phase 2 — V2V Communication

Phase 2 introduces a simulated Vehicle-to-Vehicle (V2V) communication layer.

- **Information Only:** Vehicles exchange state information (position, speed, etc.). They do not change their physical behaviour based on these messages yet.
- **Communication Range:** Only vehicles within `range_meters` can communicate.
- **Latency:** Configurable simulated transmission delay (`latency_ms`).
- **Packet Loss:** Probabilistic message dropping (`packet_loss_rate`).
- **Message Frequency:** Configurable broadcast rate (e.g. 10 Hz).
- **Stale Data:** Messages exceeding `stale_threshold_ms` are flagged.
- **Statistics:** Real-time tracking of sent, delivered, dropped, and stale messages.

**Example output:**
```
[SIM] t=40.0s | vehicles=5
...
[V2V] sent=2810 | delivered=2804 | dropped=0
[V2V] delivery_rate=99.8% | avg_latency=100.0ms | stale=12
```

---

## Project Structure

```
quantum-platooning/
├── config/              # YAML configuration files
├── src/platooning/      # Main Python package
│   ├── config/          # Configuration loading & validation
│   ├── models/          # Data models (VehicleState, Platoon, Message)
│   ├── simulation/      # SUMO / TraCI abstraction
│   ├── communication/   # Simulated V2V network
│   ├── platooning/      # Platoon manager & formation logic
│   ├── optimization/    # Classical, QUBO, and Quantum optimisers
│   ├── control/         # Cooperative vehicle controller
│   ├── experiments/     # Scenario runner & metrics
│   ├── visualization/   # Plotting utilities
│   └── utils/           # Logging & helpers
├── tests/               # pytest test suite
├── simulation/sumo/     # SUMO networks, routes, configs, output
├── experiments/         # Scenario definitions & results
├── docs/                # Architecture & development documentation
└── data/                # Raw, processed, and result data
```

---

## License

This project is developed as a college major project for academic/research purposes.

---

## References

- SUMO — https://sumo.dlr.de/
- Qiskit — https://qiskit.org/
- QAOA — Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
- QUBO — Glover et al., "Quantum Bridge Analytics I: a tutorial on formulating and using QUBO models" (2018)
