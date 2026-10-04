"""
QUBO → Ising Hamiltonian conversion for Phase 5 QAOA.

Converts the Phase 4 QUBO matrix into an Ising Hamiltonian suitable for
QAOA circuit construction using the binary-to-spin transformation:

    x_i = (1 - z_i) / 2,  where z_i ∈ {-1, +1}

The resulting Ising Hamiltonian has the form:

    H = constant + Σ h_i z_i + Σ J_ij z_i z_j

This module preserves the exact energy equivalence between the QUBO and
Ising representations up to floating-point precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IsingHamiltonian:
    """Ising model representation of a QUBO problem.

    Attributes
    ----------
    h : dict[int, float]
        Linear coefficients (external fields) for each spin variable.
    J : dict[tuple[int, int], float]
        Quadratic coupling coefficients between spin pairs (i < j).
    offset : float
        Constant energy offset from the QUBO→Ising transformation.
    num_qubits : int
        Number of qubits (same as QUBO variables).
    """

    h: dict[int, float] = field(default_factory=dict)
    J: dict[tuple[int, int], float] = field(default_factory=dict)
    offset: float = 0.0
    num_qubits: int = 0

    def ising_energy(self, bitstring: list[int]) -> float:
        """Evaluate the Ising energy for a given bitstring (0/1 encoding).

        The bitstring is converted to spin values z_i = 1 - 2*x_i:
            x_i = 0 → z_i = +1
            x_i = 1 → z_i = -1

        Parameters
        ----------
        bitstring : list[int]
            Binary vector (0/1) of length num_qubits.

        Returns
        -------
        float
            The Ising energy for this configuration.
        """
        # Convert bitstring to spins: z_i = 1 - 2*x_i
        spins = [1 - 2 * b for b in bitstring]

        energy = self.offset

        for i, hi in self.h.items():
            energy += hi * spins[i]

        for (i, j), jij in self.J.items():
            energy += jij * spins[i] * spins[j]

        return energy


def qubo_to_ising(
    q_matrix: dict[tuple[int, int], float], num_vars: int
) -> IsingHamiltonian:
    """Convert a QUBO matrix to an Ising Hamiltonian.

    The QUBO energy is:
        E(x) = Σ_{i<=j} Q_{ij} x_i x_j

    where diagonal terms Q_{ii} are linear coefficients and off-diagonal
    terms Q_{ij} (i<j) are quadratic coefficients.

    Using x_i = (1 - z_i) / 2:
        x_i * x_j = (1 - z_i)(1 - z_j) / 4
                   = (1 - z_i - z_j + z_i*z_j) / 4

    For diagonal (i == j):
        x_i^2 = x_i = (1 - z_i) / 2

    Parameters
    ----------
    q_matrix : dict[tuple[int, int], float]
        Upper-triangular QUBO matrix from Phase 4.
    num_vars : int
        Number of binary variables.

    Returns
    -------
    IsingHamiltonian
        Equivalent Ising representation.
    """
    h: dict[int, float] = {}
    j_couplings: dict[tuple[int, int], float] = {}
    offset = 0.0

    for (i, j), coef in q_matrix.items():
        if coef == 0.0:
            continue

        if i == j:
            # Diagonal: Q_{ii} * x_i = Q_{ii} * (1 - z_i) / 2
            #         = Q_{ii}/2  -  Q_{ii}/2 * z_i
            offset += coef / 2.0
            h[i] = h.get(i, 0.0) - coef / 2.0
        else:
            # Off-diagonal (i < j): Q_{ij} * x_i * x_j
            #   = Q_{ij} * (1 - z_i - z_j + z_i*z_j) / 4
            #   = Q_{ij}/4  -  Q_{ij}/4 * z_i  -  Q_{ij}/4 * z_j
            #     + Q_{ij}/4 * z_i * z_j
            offset += coef / 4.0
            h[i] = h.get(i, 0.0) - coef / 4.0
            h[j] = h.get(j, 0.0) - coef / 4.0

            key = (min(i, j), max(i, j))
            j_couplings[key] = j_couplings.get(key, 0.0) + coef / 4.0

    # Remove zero entries
    h = {k: v for k, v in h.items() if abs(v) > 1e-15}
    j_couplings = {k: v for k, v in j_couplings.items() if abs(v) > 1e-15}

    return IsingHamiltonian(h=h, J=j_couplings, offset=offset, num_qubits=num_vars)


def verify_energy_equivalence(
    q_matrix: dict[tuple[int, int], float],
    ising: IsingHamiltonian,
    bitstring: list[int],
    tol: float = 1e-10,
) -> tuple[float, float, bool]:
    """Verify that QUBO and Ising energies agree for a given bitstring.

    Parameters
    ----------
    q_matrix : dict
        The QUBO matrix.
    ising : IsingHamiltonian
        The Ising Hamiltonian.
    bitstring : list[int]
        Binary vector to evaluate.
    tol : float
        Numerical tolerance for comparison.

    Returns
    -------
    tuple[float, float, bool]
        (qubo_energy, ising_energy, energies_match)
    """
    # QUBO energy
    qubo_energy = 0.0
    for (i, j), coef in q_matrix.items():
        if bitstring[i] == 1 and bitstring[j] == 1:
            qubo_energy += coef

    # Ising energy
    ising_energy = ising.ising_energy(bitstring)

    match = abs(qubo_energy - ising_energy) < tol

    return qubo_energy, ising_energy, match
