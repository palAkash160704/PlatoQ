"""
QUBO formulation and classical exact solver for Phase 4.
This version uses a Leader-Based representation to EXACTLY reproduce the Phase 3 speed penalty.
"""

import itertools
import math
from typing import Any

from platooning.models.vehicle import VehicleState
from platooning.models.platoon import Platoon
from platooning.platooning.formation import build_compatibility_graph
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


def build_qubo(
    states: list[VehicleState], config: dict[str, Any], current_time: float
) -> tuple[dict[tuple[int, int], float], int]:
    """
    Construct the QUBO matrix using a Leader-Based formulation.
    This exactly models the Phase 3 leader-relative speed penalty without surrogates.
    
    Returns:
        q_matrix: Upper triangular dictionary mapping (u, v) to the QUBO coefficient.
        num_vars: Total number of binary variables.
    """
    plat_conf = config.get("platooning", {})
    qubo_conf = config.get("qubo", {})

    n_veh = len(states)
    if n_veh == 0:
        return {}, 0

    min_size = int(plat_conf.get("minimum_platoon_size", 2))
    max_size = int(plat_conf.get("maximum_platoon_size", 5))

    w_mem = float(plat_conf.get("weight_membership", 10.0))
    w_ung = float(plat_conf.get("weight_ungrouped_penalty", 5.0))
    w_dist_pair = float(qubo_conf.get("weight_pairwise_distance", 0.1))
    
    # EXACT Phase 3 Speed Weight
    w_speed_exact = float(plat_conf.get("weight_speed_penalty", 1.0))

    p_assign = float(qubo_conf.get("penalty_assignment", 1000.0))
    p_incompat = float(qubo_conf.get("penalty_incompatibility", 1000.0))
    p_size = float(qubo_conf.get("penalty_size", 1000.0))
    p_leader = 1000.0 # Strict enforcement for logical implication

    # Sort states exactly as Phase 3 to establish deterministic leader precedence
    sorted_states = sorted(states, key=lambda v: (v.route_id, -v.position_x, v.vehicle_id))
    
    # 1. Variable Mapping
    x_indices = {}
    idx = 0
    # x_{j,i}: vehicle j is assigned to the platoon led by vehicle i (where i <= j)
    for i in range(n_veh):
        for j in range(i, n_veh):
            x_indices[(j, i)] = idx
            idx += 1
            
    s_indices = {}
    valid_sizes_per_i = {}
    for i in range(n_veh):
        max_possible_size = min(max_size, n_veh - i)
        valid_sizes_per_i[i] = []
        if max_possible_size >= min_size:
            for v in range(min_size, max_possible_size + 1):
                s_indices[(i, v)] = idx
                valid_sizes_per_i[i].append(v)
                idx += 1
                
    num_vars = idx
    q_matrix: dict[tuple[int, int], float] = {}

    def add_term(u: int, v: int, val: float):
        if val == 0.0:
            return
        if u > v:
            u, v = v, u
        q_matrix[(u, v)] = q_matrix.get((u, v), 0.0) + val

    comp_graph = build_compatibility_graph(states, config, current_time)

    # 2. Linear Reward & Exact Assignment (sum_{i<=j} x_{j,i} <= 1)
    for j in range(n_veh):
        # Linear assignment reward (avoiding ungrouped penalty)
        for i in range(j + 1):
            idx_x = x_indices[(j, i)]
            add_term(idx_x, idx_x, -(w_mem + w_ung))
            
        # Constraint: At most one assignment
        for i in range(j + 1):
            for k in range(i + 1, j + 1):
                idx1 = x_indices[(j, i)]
                idx2 = x_indices[(j, k)]
                add_term(idx1, idx2, 2.0 * p_assign)

    # 3. Leader Enforcement: If j is in i's platoon (j>i), i must be the leader (x_{i,i}=1)
    for i in range(n_veh):
        idx_ii = x_indices[(i, i)]
        for j in range(i + 1, n_veh):
            idx_ji = x_indices[(j, i)]
            # Penalty: P * x_{j,i} * (1 - x_{i,i})
            add_term(idx_ji, idx_ji, p_leader)
            add_term(idx_ji, idx_ii, -p_leader)

    # 4. Incompatibility, Speed (Exact), and Distance (Surrogate)
    for i in range(n_veh):
        v_leader = sorted_states[i]
        
        for j in range(i + 1, n_veh):
            v_member = sorted_states[j]
            idx_ji = x_indices[(j, i)]
            
            # Leader to Member compatibility
            is_compat = False
            for comp in comp_graph.get(v_leader.vehicle_id, []):
                if comp.vehicle_b_id == v_member.vehicle_id:
                    is_compat = comp.compatible
                    break
                    
            if not is_compat:
                add_term(idx_ji, idx_ji, p_incompat)
            else:
                # OPTION A: Exact Representation of Phase 3 Speed Penalty
                # Phase 3 exactly calculates speed_diff against the leader
                speed_diff = abs(v_leader.speed - v_member.speed)
                add_term(idx_ji, idx_ji, w_speed_exact * speed_diff)
                
                # Pairwise Distance Surrogate (documented in audit)
                dist_ji = ((v_leader.position_x - v_member.position_x)**2 + 
                           (v_leader.position_y - v_member.position_y)**2)**0.5
                add_term(idx_ji, idx_ji, w_dist_pair * dist_ji)
                
            # Member to Member compatibility (cross terms)
            for k in range(j + 1, n_veh):
                v_member2 = sorted_states[k]
                idx_ki = x_indices[(k, i)]
                
                is_compat_cross = False
                for comp in comp_graph.get(v_member.vehicle_id, []):
                    if comp.vehicle_b_id == v_member2.vehicle_id:
                        is_compat_cross = comp.compatible
                        break
                        
                if not is_compat_cross:
                    add_term(idx_ji, idx_ki, p_incompat)
                else:
                    dist_jk = ((v_member.position_x - v_member2.position_x)**2 + 
                               (v_member.position_y - v_member2.position_y)**2)**0.5
                    add_term(idx_ji, idx_ki, w_dist_pair * dist_jk)
                    
    # 5. Platoon Size Constraints
    for i in range(n_veh):
        valid_v = valid_sizes_per_i[i]
        
        for j in range(i, n_veh):
            idx = x_indices[(j, i)]
            add_term(idx, idx, p_size)
            
        for j in range(i, n_veh):
            for k in range(j + 1, n_veh):
                idx1 = x_indices[(j, i)]
                idx2 = x_indices[(k, i)]
                add_term(idx1, idx2, 2.0 * p_size)
                
        for v in valid_v:
            idx_s = s_indices[(i, v)]
            add_term(idx_s, idx_s, p_size * (v**2))
            
        for v_idx, u in enumerate(valid_v):
            for v in valid_v[v_idx + 1:]:
                idx_su = s_indices[(i, u)]
                idx_sv = s_indices[(i, v)]
                # Mutual exclusion penalty multiplier 10.0
                add_term(idx_su, idx_sv, 2.0 * p_size * u * v + 10.0 * p_size)
                
        for j in range(i, n_veh):
            for v in valid_v:
                idx_x = x_indices[(j, i)]
                idx_s = s_indices[(i, v)]
                add_term(idx_x, idx_s, -2.0 * p_size * v)

    return q_matrix, num_vars


def evaluate_energy(q_matrix: dict[tuple[int, int], float], x: list[int]) -> float:
    """Evaluate the QUBO energy for a given binary vector x."""
    energy = 0.0
    for (i, j), coef in q_matrix.items():
        if x[i] == 1 and x[j] == 1:
            energy += coef
    return energy


def solve_qubo_exact(q_matrix: dict[tuple[int, int], float], num_vars: int, max_vars: int = 25) -> tuple[list[int], float]:
    """
    Solve the QUBO using exact enumeration.
    Only allows up to max_vars to prevent simulation freezing.
    """
    if num_vars > max_vars:
        raise ValueError(f"QUBO variables ({num_vars}) exceed exact solver limit ({max_vars}).")
        
    if num_vars == 0:
        return [], 0.0
        
    best_x = None
    best_energy = float('inf')
    
    for combo in itertools.product([0, 1], repeat=num_vars):
        e = 0.0
        for (i, j), coef in q_matrix.items():
            if combo[i] and combo[j]:
                e += coef
                
        if e < best_energy:
            best_energy = e
            best_x = list(combo)
            
    return best_x or [0]*num_vars, best_energy


def decode_solution(binary_solution: list[int], states: list[VehicleState], config: dict[str, Any]) -> list[Platoon]:
    """
    Decode the binary QUBO solution back into physical Platoon objects.
    """
    n_veh = len(states)
    if n_veh == 0 or len(binary_solution) == 0:
        return []

    sorted_states = sorted(states, key=lambda v: (v.route_id, -v.position_x, v.vehicle_id))
    
    x_indices = {}
    idx = 0
    for i in range(n_veh):
        for j in range(i, n_veh):
            x_indices[(j, i)] = idx
            idx += 1
            
    platoons = []
    pid_counter = 1
    
    for i in range(n_veh):
        idx_ii = x_indices[(i, i)]
        if binary_solution[idx_ii] == 1:
            leader = sorted_states[i]
            members = [leader]
            
            for j in range(i + 1, n_veh):
                idx_ji = x_indices[(j, i)]
                if binary_solution[idx_ji] == 1:
                    members.append(sorted_states[j])
                    
            if len(members) >= int(config.get("platooning", {}).get("minimum_platoon_size", 2)):
                p = Platoon(
                    platoon_id=f"QUBO_P{pid_counter}",
                    vehicle_ids=[v.vehicle_id for v in members],
                    leader_id=leader.vehicle_id
                )
                p.route_id = leader.route_id
                p.lane_id = leader.lane_id
                p.status = "FORMED"
                platoons.append(p)
                pid_counter += 1
                
    return platoons


def validate_solution(binary_solution: list[int], states: list[VehicleState], config: dict[str, Any], current_time: float) -> dict[str, Any]:
    """
    Validate the decoded binary solution against hard constraints.
    Returns {"valid": bool, "violations": list[str]}.
    """
    result: dict[str, Any] = {"valid": True, "violations": []}
    
    n_veh = len(states)
    if n_veh == 0 or len(binary_solution) == 0:
        return result
        
    sorted_states = sorted(states, key=lambda v: (v.route_id, -v.position_x, v.vehicle_id))
    
    x_indices = {}
    idx = 0
    for i in range(n_veh):
        for j in range(i, n_veh):
            x_indices[(j, i)] = idx
            idx += 1
            
    assigned_count = {i: 0 for i in range(n_veh)}
    platoon_members: dict[int, list[int]] = {i: [] for i in range(n_veh)}
    
    for i in range(n_veh):
        for j in range(i, n_veh):
            if binary_solution[x_indices[(j, i)]] == 1:
                assigned_count[j] += 1
                platoon_members[i].append(j)
                
    for j, count in assigned_count.items():
        if count > 1:
            result["valid"] = False
            result["violations"].append(f"Vehicle {sorted_states[j].vehicle_id} assigned to {count} platoons.")
            
    min_size = int(config.get("platooning", {}).get("minimum_platoon_size", 2))
    max_size = int(config.get("platooning", {}).get("maximum_platoon_size", 5))
    comp_graph = build_compatibility_graph(states, config, current_time)
            
    for i, members in platoon_members.items():
        size = len(members)
        if size == 0:
            continue
            
        if size < min_size or size > max_size:
            result["valid"] = False
            result["violations"].append(f"Candidate platoon {i} has invalid size {size}.")
            
        if i not in members:
            result["valid"] = False
            result["violations"].append(f"Candidate platoon {i} has members but not the leader {i}.")
            
        for u_idx in range(len(members)):
            for v_idx in range(u_idx + 1, len(members)):
                u = members[u_idx]
                v = members[v_idx]
                v1_id = sorted_states[u].vehicle_id
                v2_id = sorted_states[v].vehicle_id
                
                is_compat = False
                for comp in comp_graph.get(v1_id, []):
                    if comp.vehicle_b_id == v2_id:
                        is_compat = comp.compatible
                        break
                
                if not is_compat:
                    result["valid"] = False
                    result["violations"].append(f"Incompatible pair {v1_id} and {v2_id} in platoon {i}.")
                    
    return result
