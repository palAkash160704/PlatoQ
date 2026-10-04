# QUBO Formulation for Cooperative Vehicle Platoon Formation

## 1. Problem Definition
The objective is to group a set of $N$ vehicles into up to $K$ candidate platoons such that the total formation cost is minimized. The classical objective maximizes grouped vehicles, minimizes intra-platoon distance, and minimizes speed differences against the leader, while respecting strict physical and network constraints.

To formulate this as a Quadratic Unconstrained Binary Optimization (QUBO) problem, we invert the classical objective to a minimization form and embed all hard constraints into the objective function as quadratic penalty terms.

## 2. Decision Variables
### Assignment Variables
Let $x_{i,k} \in \{0, 1\}$ be a binary variable where:
- $x_{i,k} = 1$ if vehicle $i$ is assigned to candidate platoon $k$.
- $x_{i,k} = 0$ otherwise.
where $i \in \{0, \dots, N-1\}$ and $k \in \{0, \dots, K-1\}$.

### Ungrouped Representation
A vehicle is considered **ungrouped** if it is not assigned to any platoon (i.e., $\sum_k x_{i,k} = 0$). By formulating the constraint as $\sum_k x_{i,k} \le 1$, we explicitly allow vehicles to remain ungrouped without needing dedicated $u_i$ variables, reducing the total variable count by $N$.

### Size Slack Variables
Let $s_{k,v} \in \{0, 1\}$ be slack variables for candidate platoon $k$ matching a specific valid size $v$. 
Valid sizes $V = \{2, \dots, N_{max}\}$.
- $s_{k,v} = 1$ if platoon $k$ has exactly $v$ members.
- $s_{k,v} = 0$ otherwise.

## 3. Mathematical Adaptations from Phase 3
Phase 3 uses some metrics that are difficult to encode purely quadratically. We make two mathematically sound adaptations:
1. **Pairwise Distance Penalty**: Phase 3 penalizes the sum of distances between adjacent members. The QUBO formulation adapts this to a symmetric pairwise distance penalty $W_{dist}^{qubo} \cdot D_{ij}$ for ALL pairs in a candidate platoon.
2. **Pairwise Speed Penalty**: Phase 3 calculates speed difference against a designated platoon leader. The QUBO avoids explicitly encoding a leader (which would require auxiliary ternary logic) by adapting this to a symmetric pairwise speed difference penalty between all members of the platoon.

## 4. Objective Function & Penalties
The total QUBO energy $E$ to minimize is the sum of five components:
$E = E_{linear} + E_{assign} + E_{incompat} + E_{metric} + E_{size}$

### 4.1. Linear Cost ($E_{linear}$)
Assigning a vehicle to a platoon provides a membership benefit $W_{membership}$ and avoids the baseline ungrouped penalty $W_{ungrouped}$.
$E_{linear} = \sum_{i=0}^{N-1} \sum_{k=0}^{K-1} -(W_{membership} + W_{ungrouped}) x_{i,k}$

### 4.2. Exact Assignment Constraint ($E_{assign}$)
A vehicle can be in at most ONE candidate platoon. 
$E_{assign} = P_{assign} \sum_{i=0}^{N-1} \sum_{k < m} x_{i,k} x_{i,m}$

### 4.3. Incompatibility Constraint ($E_{incompat}$)
If vehicles $i$ and $j$ are incompatible (too far, stale V2V message, different route/lane, too much speed diff), they cannot share a platoon.
$E_{incompat} = P_{incompat} \sum_{k=0}^{K-1} \sum_{i < j \in \text{Incompatible}} x_{i,k} x_{j,k}$

### 4.4. Pairwise Distance & Speed Penalties ($E_{metric}$)
$E_{metric} = \sum_{k=0}^{K-1} \sum_{i < j} x_{i,k} x_{j,k} \left( W_{dist}^{qubo} D_{ij} + W_{speed}^{qubo} |S_i - S_j| \right)$

### 4.5. Platoon Size Constraints ($E_{size}$)
Candidate platoons must be either empty (size 0) or contain $v \in V$ vehicles. Size 1 is invalid.
$E_{size} = P_{size} \sum_{k=0}^{K-1} \left[ \left( \sum_{i} x_{i,k} - \sum_{v \in V} v \cdot s_{k,v} \right)^2 + M \sum_{u < v} s_{k,u} s_{k,v} \right]$
Expanded quadratically:
- Linear $x_{i,k}$: $+ P_{size}$
- Quadratic $x_{i,k}, x_{j,k}$: $+ 2 P_{size}$
- Linear $s_{k,v}$: $+ P_{size} \cdot v^2$
- Quadratic $s_{k,u}, s_{k,v}$: $+ 2 P_{size} \cdot u \cdot v + M \cdot P_{size}$
- Cross $x_{i,k}, s_{k,v}$: $- 2 P_{size} \cdot v$

Here $M=10$ is a strong mutual-exclusion multiplier for slacks.

## 5. QUBO Matrix Representation
The matrix $Q$ is strictly upper-triangular where variables are deterministically mapped:
- **Indices 0 to N·K - 1**: Assignment variables $x_{i,k}$
- **Indices N·K to N·K + K·|V| - 1**: Slack variables $s_{k,v}$

Total variables: $N \cdot K + K \cdot (N_{max} - 1)$

## 6. Classical Solver Strategy
For small test instances (e.g., $N=5$), the matrix size is $18 \times 18$. The solution space of $2^{18} = 262,144$ is exhaustively enumerated using a classical solver. This establishes the absolute ground-truth optimum to validate the formulation before scaling up in Phase 5 with QAOA.
