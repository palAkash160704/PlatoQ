
"""
Phase 6 Dynamic Experiment.

Evaluates Classical, Exact QUBO, and QAOA management over specific
dynamic traffic scenarios.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from platooning.models.vehicle import VehicleState
from platooning.platooning.dynamic_manager import DynamicPlatoonManager

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase6"


# ======================================================================
# Scenarios
# ======================================================================
def build_config() -> dict:
    """Return a standard test configuration."""
    return {
        "communication": {"stale_threshold_ms": 200.0},
        "platooning": {
            "enabled": True,
            "formation_interval_ms": 500,
            "max_formation_distance_m": 50.0,
            "max_speed_difference_mps": 3.0,
            "minimum_platoon_size": 2,
            "maximum_platoon_size": 5,
            "weight_membership": 10.0,
            "weight_distance_penalty": 0.5,
            "weight_speed_penalty": 1.0,
            "weight_ungrouped_penalty": 5.0,
        },
        "qubo": {
            "enabled": True,
            "solver": "exact",
            "max_exact_variables": 25,
            "weight_pairwise_distance": 0.1,
            "weight_pairwise_speed": 0.5,
            "penalty_assignment": 1000.0,
            "penalty_incompatibility": 1000.0,
            "penalty_size": 1000.0,
        },
        "quantum": {
            "backend": "aer_simulator",
            "shots": 512,  # lowered slightly for faster dynamic experiments
            "reps": 1,
            "optimizer": "COBYLA",
            "seed": 42,
        },
        "dynamic": {
            "reconfiguration_interval_ms": 2000,
            "reconfiguration_cooldown_ms": 1000,
            "minimum_improvement": 0.5,
        },
    }


def make_states(
    time: float,
    v1_x: float,
    v2_x: float,
    v3_x: float,
    v1_s: float = 30.0,
    v2_s: float = 30.0,
    v3_s: float = 30.0,
    v3_route: str = "r1",
) -> list[VehicleState]:
    return [
        VehicleState("V1", time, v1_x, 0.0, v1_s, route_id="r1", lane_id="l1"),
        VehicleState("V2", time, v2_x, 0.0, v2_s, route_id="r1", lane_id="l1"),
        VehicleState("V3", time, v3_x, 0.0, v3_s, route_id=v3_route, lane_id="l1"),
    ]


def scenario_a_separation() -> list[tuple[float, list[VehicleState]]]:
    """Scenario A: Vehicle 3 gradually separates and falls behind."""
    seq = []
    seq.append((0.0, make_states(0.0, 100.0, 95.0, 90.0)))
    seq.append((1.0, make_states(1.0, 130.0, 125.0, 110.0)))
    seq.append((2.0, make_states(2.0, 160.0, 155.0, 100.0)))
    seq.append((3.0, make_states(3.0, 190.0, 185.0, 130.0)))
    return seq


def scenario_b_merge() -> list[tuple[float, list[VehicleState]]]:
    """Scenario B: Vehicles are apart, then come together to merge."""
    seq = []
    seq.append((0.0, make_states(0.0, 100.0, 95.0, 30.0)))
    seq.append((1.0, make_states(1.0, 130.0, 125.0, 90.0)))
    seq.append((2.0, make_states(2.0, 160.0, 155.0, 150.0)))
    seq.append((3.0, make_states(3.0, 190.0, 185.0, 180.0)))
    return seq


def scenario_c_speed_divergence() -> list[tuple[float, list[VehicleState]]]:
    """Scenario C: Vehicle 3 gradually increases speed difference."""
    seq = []
    # All close and same speed (diff = 0)
    seq.append((0.0, make_states(0.0, 100.0, 95.0, 90.0, v1_s=30.0, v2_s=29.5, v3_s=29.0)))
    # V3 speed drops further (diff = 2 <= 3.0)
    seq.append((1.0, make_states(1.0, 130.0, 124.5, 118.0, v1_s=30.0, v2_s=29.5, v3_s=28.0)))
    # V3 speed drops to 25m/s (diff = 5 > 3.0)
    seq.append((2.0, make_states(2.0, 160.0, 154.0, 143.0, v1_s=30.0, v2_s=29.5, v3_s=25.0)))
    # V3 stays slow
    seq.append((3.0, make_states(3.0, 190.0, 183.5, 168.0, v1_s=30.0, v2_s=29.5, v3_s=25.0)))
    return seq


def scenario_d_v2v_degradation() -> list[tuple[float, list[VehicleState]]]:
    """Scenario D: Communication state becomes stale."""
    seq = []
    # t=0: Normal
    seq.append((0.0, [
        VehicleState("V1", 0.0, 100.0, 0.0, 30.0, "r1", "l1"),
        VehicleState("V2", 0.0, 95.0, 0.0, 30.0, "r1", "l1"),
    ]))
    # t=1: Normal
    seq.append((1.0, [
        VehicleState("V1", 1.0, 130.0, 0.0, 30.0, "r1", "l1"),
        VehicleState("V2", 1.0, 125.0, 0.0, 30.0, "r1", "l1"),
    ]))
    # t=2: V2 drops packet, its timestamp is still 1.0
    seq.append((2.0, [
        VehicleState("V1", 2.0, 160.0, 0.0, 30.0, "r1", "l1"),
        VehicleState("V2", 1.0, 125.0, 0.0, 30.0, "r1", "l1"),
    ]))
    # t=3: V2 comes back
    seq.append((3.0, [
        VehicleState("V1", 3.0, 190.0, 0.0, 30.0, "r1", "l1"),
        VehicleState("V2", 3.0, 185.0, 0.0, 30.0, "r1", "l1"),
    ]))
    return seq


def scenario_e_split() -> list[tuple[float, list[VehicleState]]]:
    """Scenario E: Platoon Split."""
    seq = []
    def ms(time, x1, x2, x3, x4):
        return [
            VehicleState("V1", time, x1, 0.0, 30.0, "r1", "l1"),
            VehicleState("V2", time, x2, 0.0, 30.0, "r1", "l1"),
            VehicleState("V3", time, x3, 0.0, 30.0, "r1", "l1"),
            VehicleState("V4", time, x4, 0.0, 30.0, "r1", "l1"),
        ]
    seq.append((0.0, ms(0.0, 100, 95, 90, 85)))
    seq.append((1.0, ms(1.0, 150, 145, 90, 85)))
    return seq


def scenario_f_join_leave() -> list[tuple[float, list[VehicleState]]]:
    """Scenario F: Join and Leave."""
    seq = []
    # V1, V2 together. V3 far ahead.
    seq.append((0.0, [
        VehicleState("V1", 0.0, 50, 0.0, 30.0, "r1", "l1"),
        VehicleState("V2", 0.0, 45, 0.0, 30.0, "r1", "l1"),
        VehicleState("V3", 0.0, 200, 0.0, 20.0, "r1", "l1"),
    ]))
    # V1, V2 catch up to V3 -> Join
    seq.append((20.0, [
        VehicleState("V1", 20.0, 150, 0.0, 30.0, "r1", "l1"),
        VehicleState("V2", 20.0, 145, 0.0, 30.0, "r1", "l1"),
        VehicleState("V3", 20.0, 155, 0.0, 30.0, "r1", "l1"),
    ]))
    # V2 changes route -> Leave
    seq.append((21.0, [
        VehicleState("V1", 21.0, 180, 0.0, 30.0, "r1", "l1"),
        VehicleState("V2", 21.0, 175, 0.0, 30.0, "r2", "l1"), # different route
        VehicleState("V3", 21.0, 185, 0.0, 30.0, "r1", "l1"),
    ]))
    return seq


# ======================================================================
# Runner
# ======================================================================
def print_event(e: dict):
    print("-" * 60)
    print("RECONFIGURATION EVENT")
    print("-" * 60)
    print(f"Time:              {e['timestamp']:.1f} s")
    print(f"Trigger:           {e['trigger']}")
    print(f"Severity:          HIGH")  # Simplified for display
    print()
    print("Old configuration:")
    for pid, members in e["old_configuration"].items():
        print(f"{pid} = [{', '.join(members)}]")
    if not e["old_configuration"]:
        print("None")
    print()
    print("New configuration:")
    for pid, members in e["new_configuration"].items():
        print(f"{pid} = [{', '.join(members)}]")
    if not e["new_configuration"]:
        print("None")
    print()
    print(f"Optimizer:         {e['optimizer']}")
    print(f"Objective Before:  {e['old_objective']:.2f}")
    print(f"Objective After:   {e['new_objective']:.2f}")
    print(f"Feasible:          {e['feasible']}")
    print(f"Optimizer Runtime: {e['runtime_ms']:.2f} ms")
    print(f"Event Type:        {e['event_type'].upper()}")
    print("-" * 60)
    print()


def run_scenario(
    name: str,
    sequence: list[tuple[float, list[VehicleState]]],
    optimizer_mode: str,
) -> dict:
    """Run a single scenario with the specified optimiser."""
    config = build_config()
    dm = DynamicPlatoonManager(config, optimizer_mode=optimizer_mode)

    print(f"    Running {optimizer_mode}...")

    for t, states in sequence:
        # Wrap states as if they came from V2V
        latest_known = {
            s.vehicle_id: {s.vehicle_id: s} for s in states
        }
        dm.update_vehicle_states(latest_known, t)

    metrics = dm.finalize_metrics(sequence[-1][0])
    events = dm.events_to_dicts()

    for e in events:
        print_event(e)

    return {
        "metrics": metrics.to_dict(),
        "events": events,
        "objective_history": dm._objective_history,
        "platoon_count_history": dm._platoon_count_history,
    }


def run_all():
    """Run all scenarios across all optimisers."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR / "plots", exist_ok=True)

    scenarios = {
        "A_Separation": scenario_a_separation(),
        "B_Merge": scenario_b_merge(),
        "C_SpeedDivergence": scenario_c_speed_divergence(),
        "D_V2VDegradation": scenario_d_v2v_degradation(),
        "E_Split": scenario_e_split(),
        "F_JoinLeave": scenario_f_join_leave(),
    }
    optimizers = ["classical", "exact_qubo", "qaoa"]

    results = defaultdict(dict)

    print("=" * 60)
    print("PHASE 6: DYNAMIC PLATOON MANAGEMENT EXPERIMENTS")
    print("=" * 60)

    for sc_name, seq in scenarios.items():
        print(f"\nScenario: {sc_name}")
        for opt in optimizers:
            res = run_scenario(sc_name, seq, opt)
            results[sc_name][opt] = res

            # Print quick summary
            m = res["metrics"]
            print(
                f"      -> {m['total_reconfigurations']} events, "
                f"churn: {m['churn_rate']:.2f}, "
                f"avg lat: {m['average_reconfiguration_latency_ms']:.1f}ms"
            )

    # Save JSON
    with open(RESULTS_DIR / "phase6_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plotting
    print("\nGenerating plots...")
    for sc_name, data in results.items():
        plt.figure(figsize=(10, 6))
        for opt, res in data.items():
            obj_hist = res["objective_history"]
            if obj_hist:
                times = [t for t, _ in obj_hist]
                objs = [o for _, o in obj_hist]
                # Use step plot for discrete changes
                plt.step(times, objs, where="post", label=opt, linewidth=2, marker="o")

        plt.title(f"Objective Value over Time - {sc_name}")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Objective Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(RESULTS_DIR / "plots" / f"{sc_name}_objective.png")
        plt.close()

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
