# Research Scope

This document defines what is **included** and what is **not included** in the initial project scope.  It serves as a reference to prevent scope creep and to set clear expectations for academic evaluation.

---

## Included in Scope

### Connected / Cooperative Vehicle Platooning

- Vehicles communicate through **simulated V2V (Vehicle-to-Vehicle)** messaging.
- A centralised **Platoon Manager** coordinates platoon formation, merging, splitting, and reconfiguration.
- Vehicles are **not required to be fully autonomous**; a human driver is assumed to be present, with cooperative driving assistance features.

### SUMO Simulation

- **SUMO** (Simulation of Urban Mobility) is the primary traffic simulation environment.
- Vehicle states are extracted via **TraCI**.
- The simulation provides realistic traffic dynamics including lane-changing, acceleration, and deceleration.
- **Phase 1 status:** SUMO integration is fully operational. Five vehicles run on a 2 km straight road, and their states (position, speed, acceleration, lane, route) are extracted in real-time via TraCI into `VehicleState` objects. Phase 1 establishes only the simulation foundation — no platoon formation, optimization, or cooperative control is implemented yet.

### Platoon Formation & Assignment

- Dynamic formation of platoons based on vehicle proximity, destination compatibility, and speed alignment.
- Vehicles can join and leave platoons during the simulation.

### Classical Optimization (Baseline)

- A conventional optimisation approach (e.g. greedy, ILP, or OR-Tools) will serve as the **baseline** against which quantum methods are compared.

### QUBO Formulation

- The platoon assignment / reconfiguration problem will be formulated as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem.
- This formulation is necessary to interface with quantum solvers.

### QAOA (Quantum Approximate Optimization Algorithm)

- The QUBO will be solved using **QAOA** on a **simulated quantum backend** (Qiskit Aer or equivalent).
- The project will investigate whether QAOA-based hybrid optimization provides competitive results compared to classical methods.

### Hybrid Quantum-Classical Optimization

- Classical pre-processing (candidate filtering) and post-processing (solution decoding/repair) will wrap the quantum solver.

### Experimental Evaluation

- Solution quality, computation time, scalability, and safety metrics will be compared between classical and hybrid approaches.

### V2V Communication Simulation

- A software-simulated V2V network with configurable range, latency, and packet loss.
- Communication affects platoon management decisions (e.g. a vehicle that cannot communicate is ineligible for platooning).

---

## NOT Included in Initial Scope

| Exclusion | Rationale |
|---|---|
| **Real autonomous vehicle deployment** | This is a simulation-based research project. |
| **Physical vehicles** | No hardware is involved. |
| **Real-world V2X hardware** | Communication is software-simulated. |
| **Production-grade autonomous driving** | The project focuses on platoon-level optimization, not perception, planning, or full self-driving. |
| **Mandatory CARLA integration** | CARLA may be explored as a future extension but is not required. |
| **Guaranteed quantum advantage** | The project investigates whether hybrid methods are *competitive*, not that they are provably superior. |
| **Mandatory real quantum hardware** | All quantum experiments use simulated backends (e.g. Qiskit Aer). |
| **Large-scale production deployment** | The system is designed for research experiments, not deployment. |

---

## Future Extensions (Out of Scope for Initial Phases)

- Integration with **CARLA** for 3D visualisation.
- Testing on **real quantum hardware** (IBM Quantum, etc.).
- Multi-lane / highway merging scenarios with complex traffic demand.
- Advanced cooperative manoeuvres (cooperative lane-changing, emergency braking propagation).
- Machine-learning–based platoon management as an additional comparison baseline.
