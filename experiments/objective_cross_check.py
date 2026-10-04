"""
Cross check Phase 3 vs QUBO formulations for numerical equivalence.
"""

import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from platooning.models.vehicle import VehicleState
from platooning.config.settings import load_config
from platooning.platooning.platoon_manager import PlatoonManager
from platooning.optimization.qubo_builder import build_qubo, evaluate_energy, solve_qubo_exact
from platooning.models.platoon import Platoon

def main():
    config = load_config()
    # Explicitly set weights for clarity
    config["platooning"]["weight_membership"] = 10.0
    config["platooning"]["weight_ungrouped_penalty"] = 5.0
    config["platooning"]["weight_speed_penalty"] = 1.0
    config["platooning"]["weight_distance_penalty"] = 0.5
    
    config["qubo"]["weight_pairwise_speed"] = 1.0
    config["qubo"]["weight_pairwise_distance"] = 0.1
    
    states = [
        VehicleState(vehicle_id="V1", timestamp=10.0, position_x=100.0, position_y=0.0, speed=30.0, route_id="r1", lane_id="l1"),
        VehicleState(vehicle_id="V2", timestamp=10.0, position_x=90.0,  position_y=0.0, speed=29.0, route_id="r1", lane_id="l1"),
        VehicleState(vehicle_id="V3", timestamp=10.0, position_x=80.0,  position_y=0.0, speed=28.0, route_id="r1", lane_id="l1"),
    ]
    
    # 5. SAME OBJECTIVE CROSS-CHECK
    # Manual Phase 3:
    # All in 1 platoon (V1 leader). Members = 3.
    # Grouped reward = 3 * 10 = +30. Ungrouped penalty = 0 * 5 = 0.
    # Distances adjacent: (100 - 90) + (90 - 80) = 10 + 10 = 20.
    # Distance Penalty = 20 * 0.5 = -10.0.
    # Speed Differences from leader (V1=30): V1=0, V2=1, V3=2. Sum = 3.
    # Speed Penalty = 3 * 1.0 = -3.0.
    # Total Phase 3 Obj = 30 - 10 - 3 = +17.0.
    
    # QUBO Component Objective (Linear Cost + Speed Penalty + Distance Surrogate):
    # Base offset (invisible to QUBO but mathematically present): N * W_ungrouped = 3 * 5 = +15.0
    # Grouped variable x_{0,0}=1, x_{1,0}=1, x_{2,0}=1. 
    # Linear cost applied for each: -(W_mem + W_ung) = -(10 + 5) = -15.
    # Sum Linear Cost = 3 * -15 = -45.
    # Speed EXACT: x_{1,0}: |29-30|*1 = +1. x_{2,0}: |28-30|*1 = +2. Sum = +3.0.
    # Distance Surrogate: x_{1,0}: 10*0.1=1.0, x_{2,0}: 20*0.1=2.0, x_{1,0}x_{2,0}: 10*0.1=1.0. Sum = 4.0.
    # Total QUBO Variable Cost (without constraints) = -45 + 3 + 4.0 = -38.0.
    # QUBO Total Ground Energy with constraints: Platoon size is 3 -> Slack s_{0,3} = 1.
    # Platoon constraint cost = 0.
    # Energy = -38.0.
    # 
    # Equivalent Phase 3 Objective derived from QUBO Energy:
    # Phase3_Obj = -(Energy + Base_Offset) 
    # BUT Phase 3 uses adjacent distances, QUBO uses pairwise distances.
    # We must explicitly adjust the derived objective for the distance difference.
    
    pm = PlatoonManager(config)
    p = Platoon(platoon_id="P1", vehicle_ids=["V1", "V2", "V3"], leader_id="V1")
    p.route_id = "r1"
    p.lane_id = "l1"
    p.status = "FORMED"
    pm._platoons = {"P1": p}
    pm._calculate_metrics(states, [p], 10.0)
    m = pm.get_formation_metrics()
    
    q_matrix, n = build_qubo(states, config, 10.0)
    sol, e = solve_qubo_exact(q_matrix, n)
    
    print("=" * 50)
    print("OBJECTIVE CROSS CHECK")
    print(f"Phase 3 Objective (All Grouped): {m.objective_value}")
    print(f"QUBO Energy (All Grouped): {e}")
    print("=" * 50)

    # 6. QUBO ENERGY VS ORIGINAL OBJECTIVE
    scenarios = [
        ("Ungrouped", []),
        ("Size 2 (V1, V2)", [Platoon("P1", ["V1", "V2"], "V1")]),
        ("Size 3 (V1, V2, V3)", [Platoon("P1", ["V1", "V2", "V3"], "V1")])
    ]
    
    for name, plats in scenarios:
        pm._platoons = {p.platoon_id: p for p in plats}
        pm._calculate_metrics(states, plats, 10.0)
        obj = pm.get_formation_metrics().objective_value
        
        # Build binary vector for this scenario
        sol_vec = [0] * n
        
        if name == "Size 2 (V1, V2)":
            # Leader V1 (index 0). Members V1(0), V2(1).
            # Indices for x_{0,0}, x_{1,0}
            sol_vec[0] = 1 # x_{0,0}
            sol_vec[1] = 1 # x_{1,0}
            # Slack s_{0,2}
            sol_vec[9] = 1 
        elif name == "Size 3 (V1, V2, V3)":
            # Leader V1 (index 0). Members V1(0), V2(1), V3(2).
            sol_vec[0] = 1 # x_{0,0}
            sol_vec[1] = 1 # x_{1,0}
            sol_vec[2] = 1 # x_{2,0}
            # Slack s_{0,3}
            sol_vec[10] = 1
            
        qubo_e = evaluate_energy(q_matrix, sol_vec)
        
        print(f"Scenario: {name}")
        print(f"  Phase 3 Obj: {obj}")
        print(f"  QUBO Energy: {qubo_e}")
        print(f"  Derived Obj Phase 3 (ignoring dist diff): {-(qubo_e + 15.0)}")
        print()

if __name__ == "__main__":
    main()
