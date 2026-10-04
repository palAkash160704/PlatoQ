# Phase 5 — QAOA: Hybrid Quantum-Classical Optimization

## Overview

Phase 5 implements the **Quantum Approximate Optimization Algorithm (QAOA)** to
solve the Phase 4 QUBO formulation of the platoon assignment problem.  QAOA is a
hybrid quantum-classical algorithm that uses a parameterised quantum circuit
(executed on a **local simulator**) combined with a classical optimiser to find
approximate solutions to combinatorial optimisation problems.

> **This project does NOT claim quantum advantage.**  QAOA is applied as a
> research exploration to compare against the classical baseline, not as a
> replacement.

---

## 1. What is QAOA?

QAOA (Quantum Approximate Optimization Algorithm) is a variational quantum
algorithm designed for combinatorial optimisation.  It works by:

1. Encoding the optimisation problem as a **cost Hamiltonian** on qubits.
2. Preparing all qubits in an equal superposition state |+⟩.
3. Alternating between a **cost unitary** U_C(γ) and a **mixer unitary** U_B(β).
4. Measuring the qubits to obtain a candidate solution (bitstring).
5. Using a **classical optimiser** to tune the parameters (γ, β) to minimise
   the expected cost.

The circuit depth *p* controls how many layers of cost + mixer unitaries are
applied.  Higher *p* can potentially find better solutions, but at increased
computational cost.

---

## 2. Why QAOA for Platooning?

The platoon assignment problem is naturally **combinatorial**: given *n*
vehicles, we must decide which vehicles join which platoons, subject to
compatibility, distance, speed, and size constraints.

Phase 4 formulated this as a **QUBO** (Quadratic Unconstrained Binary
Optimisation) problem.  QAOA is one of the leading quantum algorithms for
solving QUBO problems, making it a natural candidate for comparison.

**Goal:** Evaluate whether QAOA can find competitive solutions compared to the
exact QUBO solver and the classical greedy baseline on small problem instances.

---

## 3. QUBO → Ising Hamiltonian

The Phase 4 QUBO has the form:

```
E(x) = Σ Q_{ii} x_i  +  Σ Q_{ij} x_i x_j
```

where `x_i ∈ {0, 1}`.

To map this onto qubits, we use the **binary-to-spin transformation**:

```
x_i = (1 - z_i) / 2,  where z_i ∈ {-1, +1}
```

This yields an Ising Hamiltonian:

```
H = offset  +  Σ h_i Z_i  +  Σ J_{ij} Z_i Z_j
```

The transformation preserves the exact energy equivalence — any bitstring
evaluated under the QUBO gives the same energy as the corresponding spin
configuration under the Ising Hamiltonian.  This is verified numerically
in the test suite.

---

## 4. QAOA Circuit Construction

For depth *p*, the QAOA circuit is:

```
|+⟩^n  →  U_C(γ₁) U_B(β₁)  →  ...  →  U_C(γ_p) U_B(β_p)  →  Measure
```

### Cost Unitary U_C(γ)

Implements `exp(-iγ H_C)` where H_C is the Ising cost Hamiltonian:

- **ZZ interactions:** `CNOT(i,j) → Rz(2γ J_{ij}, j) → CNOT(i,j)`
- **Z fields:** `Rz(2γ h_i, i)`

### Mixer Unitary U_B(β)

Implements the standard X-mixer `exp(-iβ Σ X_i)`:

- Each qubit receives `Rx(2β, i)`

---

## 5. Classical Parameter Optimisation

The variational parameters `[γ₁,...,γ_p, β₁,...,β_p]` are optimised using
a classical optimiser (default: **COBYLA**) to minimise the expected QUBO
energy over measurement samples.

The optimisation loop:
1. Propose parameters → build circuit → execute on simulator → measure
2. Calculate average QUBO energy from measurement counts
3. Classical optimiser updates parameters
4. Repeat until convergence

---

## 6. Measurement and Candidate Selection

After optimisation, a final measurement round produces a distribution over
bitstrings.  For each unique bitstring:

1. Calculate its QUBO energy
2. Decode it into platoon assignments
3. Validate against all hard constraints

The **best feasible solution** (lowest energy among constraint-satisfying
bitstrings) is selected.  If no feasible solution exists, the best-energy
solution is returned with `feasible = False`.

---

## 7. Solution Decoding

The decoder converts binary QUBO variables back to platoon assignments using
the exact Phase 4 variable mapping:

```
x[j,i] = 1  means  "vehicle j joins the platoon led by vehicle i"
```

The decoder produces a `PlatoonConfiguration`:
```
P1: leader=V1, members=[V1, V2, V3]
Ungrouped: [V4]
```

---

## 8. Constraint Validation

Every decoded solution is validated against **10 constraint types**:

| # | Constraint | Description |
|---|-----------|-------------|
| 1 | Assignment uniqueness | Each vehicle in at most one platoon |
| 2 | Leader activation | If members assigned to leader i, then x[i,i]=1 |
| 3 | Compatibility | All pairs in a platoon must be compatible |
| 4 | Route compatibility | Same route required |
| 5 | Lane compatibility | Same lane required |
| 6 | Communication freshness | State data not stale |
| 7 | Maximum platoon distance | Within max_formation_distance_m |
| 8 | Maximum speed difference | Within max_speed_difference_mps |
| 9 | Minimum platoon size | At least minimum_platoon_size vehicles |
| 10 | Maximum platoon size | At most maximum_platoon_size vehicles |

A low QUBO energy alone is NOT sufficient — the solution must pass all
constraint checks.

---

## 9. Comparison Framework

Phase 5 compares three approaches on identical input scenarios:

| Approach | Description |
|----------|-------------|
| **Classical Greedy** | Phase 3 deterministic greedy formation |
| **Exact QUBO** | Phase 4 exhaustive enumeration (reference optimal) |
| **QAOA** | Phase 5 parameterised quantum circuit on simulator |

### Metrics computed:

- Objective value (Phase 3 objective function)
- QUBO energy
- Number of platoons formed
- Average / largest platoon size
- Grouped / ungrouped vehicles
- Constraint violations
- Runtime (wall-clock)
- Objective gap (exact − QAOA)
- Relative objective gap
- Feasibility rate (feasible samples / total samples)

---

## 10. Why This Project Does NOT Claim Quantum Advantage

1. **Simulator-based:** All QAOA runs use a local statevector simulator, not
   real quantum hardware.  Simulator runtime is O(2^n) in the worst case.

2. **Classical overhead:** The QAOA runtime includes the classical parameter
   optimisation loop (many circuit evaluations), which dominates the total cost.

3. **Small problem sizes:** The current QUBO for 5 vehicles has ~18 binary
   variables.  At this scale, exact classical enumeration is trivial.

4. **No noise model:** Real quantum hardware introduces noise, decoherence,
   and gate errors that would degrade solution quality.

5. **Research exploration:** The purpose is to establish the full QAOA pipeline
   (QUBO → Ising → Circuit → Measure → Decode → Validate → Compare) and
   evaluate its behaviour, not to demonstrate superiority.

---

## Worked Example: 2-Vehicle QAOA

**Input:** 2 vehicles, same route, same lane, 5m apart, same speed.

1. **QUBO:** 4 binary variables: x[0,0], x[1,0], x[1,1], s[0,2]
2. **Ising:** 4-qubit Hamiltonian with h and J coefficients
3. **QAOA p=1:** 4-qubit circuit with 2 parameters (γ₁, β₁)
4. **Optimiser:** COBYLA finds optimal (γ*, β*)
5. **Measurement:** 1024 shots → bitstring distribution
6. **Best bitstring:** [1, 1, 0, 1] → V1 leads, V2 follows, size slack = 2
7. **Decoded:** P1: leader=v1, members=[v1, v2]
8. **Validated:** Feasible ✓, 0 violations
9. **Energy:** Matches exact solver → QAOA found the optimal solution

---

## Configuration

```yaml
quantum:
  backend: "aer_simulator"
  shots: 1024
  reps: 1           # QAOA depth (p)
  optimizer: "COBYLA"
  seed: 42
```

Supported values:
- `reps`: 1, 2, 3
- `shots`: 512, 1024, 2048
- `optimizer`: "COBYLA", "Nelder-Mead"

---

## Running the Experiment

```bash
# Run the Phase 5 experiment (does not affect normal simulation)
python experiments/phase5_qaoa_experiment.py
```

Results are saved to `experiments/results/phase5/`.
