"""
QAOA (Quantum Approximate Optimization Algorithm) solver for Phase 5.

Implements a complete QAOA pipeline:
    1. Accept Phase 4 QUBO matrix
    2. Convert to Ising Hamiltonian
    3. Construct parameterised QAOA circuit
    4. Optimise variational parameters (gamma, beta) with a classical optimizer
    5. Execute on a local simulator (Qiskit Aer)
    6. Collect measurement results
    7. Select the best observed bitstring
    8. Return structured results

This implementation runs entirely on a local simulator — no API key or
real quantum hardware is required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from platooning.optimization.quantum.ising import (
    IsingHamiltonian,
    qubo_to_ising,
)
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QAOAResult:
    """Structured result from a QAOA optimisation run.

    Attributes
    ----------
    best_bitstring : list[int]
        The binary vector with the lowest observed QUBO energy.
    best_energy : float
        QUBO energy of the best bitstring.
    expectation_value : float
        Expected (average) QUBO energy across all measurement samples.
    optimal_parameters : list[float]
        Optimised QAOA parameters [gamma_1,...,gamma_p, beta_1,...,beta_p].
    reps : int
        QAOA circuit depth (number of layers).
    shots : int
        Number of measurement shots.
    optimizer : str
        Name of the classical optimiser used.
    runtime_seconds : float
        Wall-clock time for the full QAOA pipeline.
    counts : dict[str, int]
        Raw measurement counts {bitstring: count}.
    success : bool
        True if the solver completed without error.
    num_qubits : int
        Number of qubits in the circuit.
    num_optimizer_evals : int
        Number of classical optimiser function evaluations.
    """

    best_bitstring: list[int] = field(default_factory=list)
    best_energy: float = float("inf")
    expectation_value: float = float("inf")
    optimal_parameters: list[float] = field(default_factory=list)
    reps: int = 1
    shots: int = 1024
    optimizer: str = "COBYLA"
    runtime_seconds: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    success: bool = False
    num_qubits: int = 0
    num_optimizer_evals: int = 0


class QAOASolver:
    """QAOA-based solver for QUBO instances.

    Parameters
    ----------
    reps : int
        QAOA circuit depth (p), default 1.
    shots : int
        Number of measurement shots, default 1024.
    optimizer : str
        Classical optimiser name (``"COBYLA"``, ``"SPSA"``, ``"Nelder-Mead"``).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        reps: int = 1,
        shots: int = 1024,
        optimizer: str = "COBYLA",
        seed: int | None = 42,
    ) -> None:
        self._reps = reps
        self._shots = shots
        self._optimizer_name = optimizer
        self._seed = seed

    def solve(
        self,
        q_matrix: dict[tuple[int, int], float],
        num_vars: int,
        reps: int | None = None,
        shots: int | None = None,
        optimizer: str | None = None,
        seed: int | None = None,
    ) -> QAOAResult:
        """Run the full QAOA pipeline on a QUBO instance.

        Parameters
        ----------
        q_matrix : dict[tuple[int, int], float]
            Phase 4 QUBO matrix (upper-triangular).
        num_vars : int
            Number of binary variables.
        reps : int, optional
            Override QAOA depth.
        shots : int, optional
            Override measurement shots.
        optimizer : str, optional
            Override classical optimiser.
        seed : int, optional
            Override random seed.

        Returns
        -------
        QAOAResult
            Structured result containing the best bitstring, energy,
            measurement counts, and solver metadata.
        """
        start_time = time.perf_counter()

        p = reps if reps is not None else self._reps
        n_shots = shots if shots is not None else self._shots
        opt_name = optimizer if optimizer is not None else self._optimizer_name
        rng_seed = seed if seed is not None else self._seed

        result = QAOAResult(
            reps=p,
            shots=n_shots,
            optimizer=opt_name,
            num_qubits=num_vars,
        )

        if num_vars == 0:
            result.best_bitstring = []
            result.best_energy = 0.0
            result.expectation_value = 0.0
            result.success = True
            result.runtime_seconds = time.perf_counter() - start_time
            return result

        # 1. Convert QUBO → Ising
        logger.info("[QAOA] Building cost Hamiltonian...")
        ising = qubo_to_ising(q_matrix, num_vars)
        logger.info("[QAOA] Variables: %d", num_vars)
        logger.info("[QAOA] QAOA depth: %d", p)
        logger.info("[QAOA] Shots: %d", n_shots)
        logger.info("[QAOA] Classical optimizer: %s", opt_name)

        # 2. Set up simulator
        simulator = AerSimulator(method="statevector")
        if rng_seed is not None:
            simulator.set_options(seed_simulator=rng_seed)

        # 3. Classical optimisation of QAOA parameters
        eval_count = 0

        def objective(params: np.ndarray) -> float:
            """Evaluate the expectation value of the cost Hamiltonian."""
            nonlocal eval_count
            eval_count += 1

            gammas = params[:p]
            betas = params[p:]

            qc = self._build_qaoa_circuit(ising, gammas, betas)
            qc.measure_all()

            job = simulator.run(qc, shots=n_shots)
            counts = job.result().get_counts()

            # Calculate expectation value
            total_energy = 0.0
            for bitstr, count in counts.items():
                bits = [int(b) for b in reversed(bitstr)]
                energy = _qubo_energy(q_matrix, bits)
                total_energy += energy * count

            return total_energy / n_shots

        # Initial parameters
        rng = np.random.default_rng(rng_seed)
        initial_params = rng.uniform(0, np.pi, size=2 * p)

        logger.info("[QAOA] Starting parameter optimization...")

        opt_result = minimize(
            objective,
            initial_params,
            method=opt_name,
            options={"maxiter": 200, "rhobeg": 0.5},
        )

        optimal_params = opt_result.x
        result.optimal_parameters = optimal_params.tolist()
        result.num_optimizer_evals = eval_count

        logger.info("[QAOA] Optimization completed after %d evaluations", eval_count)

        # 4. Final measurement with optimal parameters
        gammas = optimal_params[:p]
        betas = optimal_params[p:]

        qc = self._build_qaoa_circuit(ising, gammas, betas)
        qc.measure_all()

        job = simulator.run(qc, shots=n_shots)
        counts = job.result().get_counts()

        # 5. Process results
        result.counts = dict(counts)

        best_energy = float("inf")
        best_bits: list[int] = []
        total_energy = 0.0

        for bitstr, count in counts.items():
            bits = [int(b) for b in reversed(bitstr)]
            energy = _qubo_energy(q_matrix, bits)
            total_energy += energy * count

            if energy < best_energy:
                best_energy = energy
                best_bits = bits

        result.best_bitstring = best_bits
        result.best_energy = best_energy
        result.expectation_value = total_energy / n_shots
        result.success = True

        elapsed = time.perf_counter() - start_time
        result.runtime_seconds = elapsed

        logger.info("[QAOA] Best energy: %.3f", best_energy)
        logger.info("[QAOA] Expectation value: %.3f", result.expectation_value)
        logger.info("[QAOA] Runtime: %.2f s", elapsed)

        return result

    def _build_qaoa_circuit(
        self,
        ising: IsingHamiltonian,
        gammas: np.ndarray,
        betas: np.ndarray,
    ) -> QuantumCircuit:
        """Build a QAOA circuit for the given Ising Hamiltonian and parameters.

        The circuit structure for depth p is:

            |+>^n  →  U_C(γ₁) U_B(β₁) → ... → U_C(γ_p) U_B(β_p)

        where:
            U_C(γ) = exp(-i γ H_C)  is the cost unitary
            U_B(β) = exp(-i β H_B)  is the mixer unitary (X-mixer)

        Parameters
        ----------
        ising : IsingHamiltonian
            The Ising cost Hamiltonian.
        gammas : ndarray
            Cost layer parameters, shape (p,).
        betas : ndarray
            Mixer layer parameters, shape (p,).

        Returns
        -------
        QuantumCircuit
            The parameterised QAOA circuit (without measurement).
        """
        n = ising.num_qubits
        p = len(gammas)

        qc = QuantumCircuit(n)

        # Initial state: |+>^n
        for i in range(n):
            qc.h(i)

        # QAOA layers
        for layer in range(p):
            gamma = gammas[layer]
            beta = betas[layer]

            # Cost unitary U_C(γ)
            self._apply_cost_unitary(qc, ising, gamma)

            # Mixer unitary U_B(β)
            self._apply_mixer_unitary(qc, n, beta)

        return qc

    def _apply_cost_unitary(
        self,
        qc: QuantumCircuit,
        ising: IsingHamiltonian,
        gamma: float,
    ) -> None:
        """Apply the cost unitary exp(-i γ H_C) to the circuit.

        For ZZ terms:  exp(-i γ J_{ij} Z_i Z_j)
            = CNOT(i,j) → Rz(2γJ_{ij}, j) → CNOT(i,j)

        For Z terms:   exp(-i γ h_i Z_i)
            = Rz(2γh_i, i)
        """
        # ZZ interactions
        for (i, j), jij in ising.J.items():
            angle = 2.0 * gamma * jij
            qc.cx(i, j)
            qc.rz(angle, j)
            qc.cx(i, j)

        # Z fields
        for i, hi in ising.h.items():
            angle = 2.0 * gamma * hi
            qc.rz(angle, i)

    def _apply_mixer_unitary(self, qc: QuantumCircuit, n: int, beta: float) -> None:
        """Apply the X-mixer unitary exp(-i β Σ X_i).

        Each qubit gets Rx(2β).
        """
        for i in range(n):
            qc.rx(2.0 * beta, i)


def _qubo_energy(q_matrix: dict[tuple[int, int], float], bits: list[int]) -> float:
    """Evaluate QUBO energy for a bitstring."""
    energy = 0.0
    for (i, j), coef in q_matrix.items():
        if bits[i] == 1 and bits[j] == 1:
            energy += coef
    return energy
